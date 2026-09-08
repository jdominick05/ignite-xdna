r"""
Live Interactive Webcam Portrait Matting Demo on AMD Ryzen AI XDNA1 NPU.

Evaluates MODNet Zero-Concat XINT8 (512x512) on the physical Phoenix 4x4 NPU overlay
with 99.1% node placement (533/538 nodes) at ~54 FPS (18.58 ms).

Interactive Controls:
  b       : Toggle Bokeh background blur
  [ / ]   : Decrease / increase blur intensity
  g       : Toggle virtual Green Screen studio mode
  m       : Toggle raw alpha matte / trimap view
  s       : Toggle side-by-side split screen (raw camera vs matted)
  c       : Cycle execution provider: NPU -> iGPU (DirectML) -> CPU
  p       : Save snapshot PNG to results/modnet/
  SPACE   : Pause / unpause video feed
  q / ESC : Quit demo

Requirements:
    conda activate resnet_env17
    $env:RYZEN_AI_INSTALLATION_PATH = 'C:\Program Files\RyzenAI\1.7.1'
    python demos/portrait_matting_demo.py
"""

import os
import sys
from pathlib import Path
import time
import argparse

# Fast camera initialization under MSMF backend on Windows
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from npu.modnet import preprocess
from npu.paths import MODELS, modnet_cache_key
from npu.session import build_session, clear_cache

DEFAULT_MODEL = MODELS / "modnet" / "modnet_zero_concat_xint8.onnx"




def postprocess_logits(raw_logits, orig_shape, bias_shift=4.5):
    orig_h, orig_w = orig_shape
    raw = np.squeeze(raw_logits) + bias_shift
    # Sigmoid in numpy with clipping to avoid overflow
    alpha_small = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    alpha = cv2.resize(alpha_small, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return alpha


def draw_toast_only(frame, msg):
    h, w = frame.shape[:2]
    tsize = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    tx = (w - tsize[0]) // 2
    ty = 42
    pill = frame.copy()
    cv2.rectangle(pill, (tx - 12, ty - 16), (tx + tsize[0] + 12, ty + 8), (18, 18, 18), -1)
    cv2.addWeighted(pill, 0.85, frame, 0.15, 0, frame)
    cv2.rectangle(frame, (tx - 12, ty - 16), (tx + tsize[0] + 12, ty + 8), (60, 60, 60), 1)
    cv2.putText(frame, msg, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 200), 1, cv2.LINE_AA)


def draw_hud(frame, ep_name, latencies, fps, modes_text, is_paused=False, toast=None, calib_progress=None):
    h, w = frame.shape[:2]
    
    # 1. Top status bar: slim (height 26px)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 26), (14, 14, 14), -1)
    
    # 2. Bottom legend bar: slim (height 20px)
    cv2.rectangle(overlay, (0, h - 20), (w, h), (14, 14, 14), -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    # Status colors
    ep_colors = {
        "NPU": (80, 240, 120),      # Neon green
        "DML": (240, 180, 50),      # Amber-cyan
        "CPU": (100, 120, 255),     # Soft red
    }
    ep_color = ep_colors.get(ep_name.upper(), (200, 200, 200))
    
    # Left: Title, EP, Nodes, Telemetry
    t_pre, t_infer, t_post, t_total = latencies
    ep_str = f"[{ep_name.upper()}] 99.1% Silicon" if ep_name.upper() == "NPU" else f"[{ep_name.upper()}]"
    left_str = f"MODNet 512  |  {ep_str}  |  {t_infer:.1f}ms ({fps:.1f} FPS)"
    cv2.putText(frame, left_str, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 225, 230), 1, cv2.LINE_AA)
    # Highlight the EP name in color
    cv2.putText(frame, ep_name.upper(), (85, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, ep_color, 1, cv2.LINE_AA)

    # Right: Active Modes
    pause_str = " [PAUSED]" if is_paused else ""
    right_str = f"{modes_text}{pause_str}"
    text_size = cv2.getTextSize(right_str, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
    cv2.putText(frame, right_str, (w - text_size[0] - 10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 215, 255), 1, cv2.LINE_AA)

    # Bottom: Sleek Keybind Legend
    legend = "[T] Calibrate on Me   [- / +] Outline   [B] Blur   [[ / ]] Radius   [G] Green   [M] Matte   [S] Split   [H] HUD   [P] Snap   [Q] Quit"
    cv2.putText(frame, legend, (10, h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (185, 190, 195), 1, cv2.LINE_AA)

    # Center Toast / Calibration Banner if active
    if calib_progress is not None:
        curr, total = calib_progress
        pct = curr / max(total, 1)
        bar_w = 260
        bx = (w - bar_w) // 2
        by = 38
        pill = frame.copy()
        cv2.rectangle(pill, (bx - 12, by - 6), (bx + bar_w + 12, by + 34), (18, 18, 18), -1)
        cv2.addWeighted(pill, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (bx - 12, by - 6), (bx + bar_w + 12, by + 34), (60, 60, 60), 1)
        
        calib_txt = f"CALIBRATING ON YOU: Frame {curr}/{total} - Hold still..."
        cv2.putText(frame, calib_txt, (bx, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 220), 1, cv2.LINE_AA)
        # Progress bar
        cv2.rectangle(frame, (bx, by + 20), (bx + bar_w, by + 27), (45, 45, 45), -1)
        cv2.rectangle(frame, (bx, by + 20), (bx + int(bar_w * pct), by + 27), (0, 230, 120), -1)
    elif toast is not None and toast[1] > time.perf_counter():
        draw_toast_only(frame, toast[0])


def main():
    ap = argparse.ArgumentParser(description="Live webcam portrait matting demo on Ryzen AI NPU.")
    ap.add_argument("--source", default="0", help="Camera index (e.g. 0) or video file path")
    ap.add_argument("--ep", choices=["npu", "dml", "cpu"], default="npu", help="Initial execution provider")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to MODNet cut XINT8 model")
    ap.add_argument("--blur-radius", type=int, default=35, help="Initial background blur kernel size")
    ap.add_argument("--bias-shift", type=float, default=4.5, help="Baseline logit bias shift for outline sensitivity")
    ap.add_argument("--max-seconds", type=float, default=None, help="Auto-quit after N seconds")
    ap.add_argument("--fresh", action="store_true", help="Clear NPU compile cache on startup")
    args = ap.parse_args()

    # Determine camera source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"[portrait_demo] WARNING: Camera source {args.source} failed to open. Falling back to synthetic stream.")
        cap = None
    else:
        # Request 640x480 at 30 fps
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

    # Load test images for synthetic fallback or testing
    val_files = sorted(list((ROOT / "data" / "modnet_val").glob("*.jpg")))
    val_idx = 0

    active_ep = args.ep
    sessions = {}

    def get_session(ep):
        if ep not in sessions:
            print(f"[portrait_demo] Building session for {ep.upper()}...")
            if ep == "npu" and args.fresh:
                clear_cache(modnet_cache_key(args.model))
            sess = build_session(args.model, ep, modnet_cache_key(args.model))
            sessions[ep] = sess
        return sessions[ep]

    current_sess = get_session(active_ep)
    input_name = current_sess.get_inputs()[0].name
    output_name = current_sess.get_outputs()[0].name

    # Interactive mode flags
    mode_blur = True
    blur_ksize = args.blur_radius if args.blur_radius % 2 == 1 else args.blur_radius + 1
    bias_shift = args.bias_shift
    mode_greenscreen = False
    mode_matte_only = False
    mode_split = True
    is_paused = False
    last_frame = None

    # Sleek HUD & Quick Calibration states
    show_hud = True
    toast = ("Press [T] to Calibrate on You | [-/+] Sensitivity", time.perf_counter() + 4.0)
    calib_active = False
    calib_frames = []
    calib_logits_list = []
    calib_target = 25

    # Performance smoothing
    fps_history = []
    t_start_demo = time.perf_counter()

    window_name = "Ryzen AI XDNA1 Portrait Matting Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 720)

    print("\n=======================================================")
    print(" Ryzen AI XDNA1 Portrait Matting Live Demo")
    print("=======================================================")
    print(" Controls:")
    print("   T       : Quick-Calibrate on you (captures 25 frames)")
    print("   - / +   : Decrease / increase outline sensitivity")
    print("   B       : Toggle background blur")
    print("   [ / ]   : Decrease / increase blur kernel size")
    print("   G       : Toggle virtual green-screen studio")
    print("   M       : Toggle raw alpha matte trimap view")
    print("   S       : Toggle side-by-side split screen")
    print("   H       : Toggle HUD display (hide / show)")
    print("   C       : Cycle execution provider (NPU -> DML -> CPU)")
    print("   P       : Save snapshot PNG to results/modnet/")
    print("   SPACE   : Pause / resume live camera")
    print("   Q / ESC : Quit")
    print("=======================================================\n")

    while True:
        if args.max_seconds and (time.perf_counter() - t_start_demo) >= args.max_seconds:
            print(f"[portrait_demo] Reached max-seconds ({args.max_seconds}s). Exiting.")
            break

        if not is_paused or last_frame is None:
            if cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("[portrait_demo] Camera stream ended. Exiting.")
                    break
            else:
                # Synthetic stream loop
                fpath = val_files[val_idx % len(val_files)]
                val_idx += 1
                frame = cv2.imread(str(fpath))
                frame = cv2.resize(frame, (640, 480))
            last_frame = frame
        else:
            frame = last_frame.copy()

        t_frame_start = time.perf_counter()

        # 1. Preprocessing
        t0 = time.perf_counter()
        tensor, orig_shape = preprocess(frame, target_size=512)
        t_pre = (time.perf_counter() - t0) * 1000

        # 2. Inference
        t0 = time.perf_counter()
        raw_logits = current_sess.run([output_name], {input_name: tensor})[0]
        t_infer = (time.perf_counter() - t0) * 1000

        # Calibration collection (captures 25 frames to tune to the user's lighting and camera)
        if calib_active:
            calib_frames.append(frame.copy())
            calib_logits_list.append(np.squeeze(raw_logits))
            if len(calib_frames) >= calib_target:
                calib_active = False
                all_l = np.array(calib_logits_list)
                H_l, W_l = all_l.shape[1], all_l.shape[2]
                tl = all_l[:, :int(H_l * 0.15), :int(W_l * 0.20)]
                tr = all_l[:, :int(H_l * 0.15), int(W_l * 0.80):]
                bg_l = np.concatenate([tl.flatten(), tr.flatten()])
                bg_p99 = np.percentile(bg_l, 99.0)
                
                # Solves optimal sensitivity: suppresses background peak below -1.5
                calib_shift = float(np.clip(-bg_p99 - 2.0, 2.0, 7.5))
                bias_shift = calib_shift

                # Save calibration frames for offline deep training/re-quantization
                user_calib_dir = ROOT / "data" / "user_calib"
                user_calib_dir.mkdir(parents=True, exist_ok=True)
                for idx_cf, cf in enumerate(calib_frames):
                    cv2.imwrite(str(user_calib_dir / f"user_{idx_cf:02d}.jpg"), cf)

                toast = (f"Calibrated to you! Sens: {bias_shift:+.1f} (25 frames saved)", time.perf_counter() + 3.5)
                print(f"[portrait_demo] User calibration complete: bg_peak={bg_p99:.1f}, tuned bias_shift={bias_shift:+.1f}")
                calib_frames = []
                calib_logits_list = []

        # 3. Postprocessing
        t0 = time.perf_counter()
        alpha = postprocess_logits(raw_logits, orig_shape, bias_shift=bias_shift)
        alpha_3ch = np.repeat(np.expand_dims(alpha, axis=-1), 3, axis=-1)

        # Apply visual effects based on active modes
        if mode_matte_only:
            # Show grayscale trimap (0 to 255)
            rendered = (alpha_3ch * 255.0).astype(np.uint8)
        elif mode_greenscreen:
            # Chroma green background: [0, 235, 0]
            green_bg = np.zeros_like(frame)
            green_bg[:] = (0, 235, 0)
            rendered = (alpha_3ch * frame + (1.0 - alpha_3ch) * green_bg).astype(np.uint8)
        elif mode_blur:
            # Bokeh Gaussian blur background
            blurred = cv2.GaussianBlur(frame, (blur_ksize, blur_ksize), 0)
            rendered = (alpha_3ch * frame + (1.0 - alpha_3ch) * blurred).astype(np.uint8)
        else:
            rendered = frame.copy()

        # Split screen composite (Raw Camera Feed on Left | Matted Output on Right)
        if mode_split:
            split_x = orig_shape[1] // 2
            display = frame.copy()
            display[:, split_x:] = rendered[:, split_x:]
            # Subtle 1px vertical divider
            cv2.line(display, (split_x, 0), (split_x, orig_shape[0]), (160, 160, 160), 1)
            if show_hud:
                # Small subtle badges in corners (not covering face)
                cv2.putText(display, "RAW", (12, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(display, "NPU MATTE", (split_x + 12, 44),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 255), 1, cv2.LINE_AA)
        else:
            display = rendered

        t_post = (time.perf_counter() - t0) * 1000
        t_total = (time.perf_counter() - t_frame_start) * 1000

        # FPS calculation
        live_fps = 1000.0 / max(t_total, 1.0)
        fps_history.append(live_fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        smoothed_fps = np.mean(fps_history)

        # Build mode status label
        active_modes = []
        if mode_split:
            active_modes.append("SPLIT")
        if mode_blur:
            active_modes.append(f"BLUR({blur_ksize})")
        if mode_greenscreen:
            active_modes.append("GREEN_SCREEN")
        if mode_matte_only:
            active_modes.append("TRIMAP")
        active_modes.append(f"SENS({bias_shift:+.1f})")
        mode_text = " | ".join(active_modes) if active_modes else "ORIGINAL"

        # Draw HUD overlay if enabled
        calib_prog = (len(calib_frames), calib_target) if calib_active else None
        if show_hud:
            draw_hud(display, active_ep, (t_pre, t_infer, t_post, t_total), smoothed_fps, mode_text, is_paused, toast, calib_prog)
        elif calib_prog is not None:
            # Always show calibration progress bar even if HUD is hidden
            draw_hud(display, active_ep, (t_pre, t_infer, t_post, t_total), smoothed_fps, mode_text, is_paused, None, calib_prog)
        elif toast is not None and toast[1] > time.perf_counter():
            draw_toast_only(display, toast[0])

        cv2.imshow(window_name, display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord('t') or key == ord('T'):
            calib_active = True
            calib_frames = []
            calib_logits_list = []
            toast = ("Calibrating on you: Hold still...", time.perf_counter() + 2.5)
            print("[portrait_demo] Starting calibration on user...")
        elif key == ord('h') or key == ord('H'):
            show_hud = not show_hud
            toast = ("HUD Enabled" if show_hud else "HUD Hidden (press [H] to restore)", time.perf_counter() + 2.0)
        elif key == ord('b'):
            mode_blur = not mode_blur
            if mode_blur:
                mode_greenscreen = False
                mode_matte_only = False
            toast = (f"Background Blur: {'ON' if mode_blur else 'OFF'}", time.perf_counter() + 1.5)
        elif key == ord('g'):
            mode_greenscreen = not mode_greenscreen
            if mode_greenscreen:
                mode_blur = False
                mode_matte_only = False
            toast = (f"Green Screen: {'ON' if mode_greenscreen else 'OFF'}", time.perf_counter() + 1.5)
        elif key == ord('m'):
            mode_matte_only = not mode_matte_only
            if mode_matte_only:
                mode_blur = False
                mode_greenscreen = False
            toast = (f"Matte Trimap: {'ON' if mode_matte_only else 'OFF'}", time.perf_counter() + 1.5)
        elif key == ord('s'):
            mode_split = not mode_split
            toast = (f"Split View: {'ON' if mode_split else 'OFF'}", time.perf_counter() + 1.5)
        elif key == ord('['):
            blur_ksize = max(5, blur_ksize - 10)
            toast = (f"Blur Radius: {blur_ksize}", time.perf_counter() + 1.5)
        elif key == ord(']'):
            blur_ksize = min(95, blur_ksize + 10)
            toast = (f"Blur Radius: {blur_ksize}", time.perf_counter() + 1.5)
        elif key == ord('+') or key == ord('='):
            bias_shift = min(15.0, bias_shift + 0.5)
            toast = (f"Sensitivity: {bias_shift:+.1f} (wider outline)", time.perf_counter() + 1.5)
            print(f"[portrait_demo] Outline sensitivity: {bias_shift:+.1f}")
        elif key == ord('-') or key == ord('_'):
            bias_shift = max(-5.0, bias_shift - 0.5)
            toast = (f"Sensitivity: {bias_shift:+.1f} (tighter outline)", time.perf_counter() + 1.5)
            print(f"[portrait_demo] Outline sensitivity: {bias_shift:+.1f}")
        elif key == ord(' '):
            is_paused = not is_paused
            toast = ("Video PAUSED" if is_paused else "Video RESUMED", time.perf_counter() + 1.5)
        elif key == ord('c'):
            # Cycle EP: NPU -> DML -> CPU -> NPU
            ep_order = ["npu", "dml", "cpu"]
            next_idx = (ep_order.index(active_ep) + 1) % len(ep_order)
            active_ep = ep_order[next_idx]
            current_sess = get_session(active_ep)
            input_name = current_sess.get_inputs()[0].name
            output_name = current_sess.get_outputs()[0].name
            toast = (f"Switched EP to {active_ep.upper()}", time.perf_counter() + 2.0)
            print(f"[portrait_demo] Switched execution provider to {active_ep.upper()}")
        elif key == ord('p'):
            out_dir = ROOT / "results" / "modnet"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            snap_path = out_dir / f"snapshot_{ts}_{active_ep}.png"
            cv2.imwrite(str(snap_path), display)
            toast = (f"Snapshot saved: {snap_path.name}", time.perf_counter() + 2.0)
            print(f"[portrait_demo] Saved snapshot to {snap_path}")

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("[portrait_demo] Demo session finished cleanly.")


if __name__ == "__main__":
    main()
