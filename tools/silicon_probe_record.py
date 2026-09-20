"""Record a bounded probe in a new log, with idle witnesses around device children."""
import argparse
import datetime
import os
import platform
import subprocess
import sys
from pathlib import Path

SMI = "C:/Windows/System32/AMD/xrt-smi.exe"


def witness():
    result = subprocess.run([SMI, "examine", "-r", "aie-partitions"], capture_output=True, text=True, timeout=20)
    text = result.stdout + result.stderr
    print(text, flush=True)
    if result.returncode or "No hardware contexts running" not in text:
        raise RuntimeError("NPU idle witness failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--seconds", type=int, default=600)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    path = Path(args.log)
    with path.open("x", encoding="utf-8", newline="\n") as log:
        class Tee:
            def write(self, value):
                value = value.replace(str(Path.home()), "C:/Users/<user>").replace(str(Path.home()).replace("\\", "/"), "C:/Users/<user>")
                log.write(value)
                sys.__stdout__.write(value)

            def flush(self):
                log.flush()
                sys.__stdout__.flush()

        sys.stdout = Tee()
        print("UTC:", datetime.datetime.now(datetime.timezone.utc).isoformat())
        print("MACHINE:", platform.node(), platform.processor())
        print("COMMAND:", subprocess.list2cmdline(command))
        print("COMMIT:", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
        if args.device:
            print("PRE_DEVICE_WITNESS")
            witness()
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            output, _ = child.communicate(timeout=args.seconds)
            print(output)
            print("EXIT_CODE:", child.returncode)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True)
            output, _ = child.communicate(timeout=30)
            print(output)
            print("TIMEOUT:", args.seconds)
            raise
        finally:
            if args.device:
                print("POST_DEVICE_WITNESS")
                witness()
            sys.stdout.flush()
            sys.stdout = sys.__stdout__
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
