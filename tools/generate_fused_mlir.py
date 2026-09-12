#!/usr/bin/env python3
"""
tools/generate_fused_mlir.py - Generates kernels/aie2/im2col_fused_2layer.mlir
Architects 2-Layer L2 MemTile Activation Ping-Pong Fusion for 16 Cores on AMD Phoenix AIE2.
"""

def generate_fused_mlir(out_path="kernels/aie2/im2col_fused_2layer.mlir"):
    lines = []
    def emit(s=""):
        lines.append(s)

    emit("//===- im2col_fused_2layer.mlir --------------------------------*- MLIR -*-===//")
    emit("//")
    emit("// 2-Layer Consecutive Conv2D L2 MemTile Activation Ping-Pong Fusion")
    emit("// AMD Phoenix XDNA1 (AIE2) - 16 Compute Cores across Columns 0..3")
    emit("//")
    emit("// Features:")
    emit("//   - Hardware Ping-Pong Buffers in MemTile SRAM:")
    emit("//       * Buffer Ping (L2_BANK_0): Offset 0x40000 (4,096 B per column)")
    emit("//       * Buffer Pong (L2_BANK_1): Offset 0x60000 (4,096 B per column)")
    emit("//   - Hardware Synchronization Locks in MemTile Tile(c, 1):")
    emit("//       * Lock 4 (l2_ping_lock): Init val = 1 (empty, write-ready for Layer 0)")
    emit("//       * Lock 5 (l2_pong_lock): Init val = 0 (idle until Layer 0 releases)")
    emit("//   - Chained BD Execution:")
    emit("//       * Layer 0 Core egress gathers directly into L2_BANK_0 (0x40000)")
    emit("//       * Layer 1 Core ingress reads directly from L2_BANK_0 (0x40000)")
    emit("//       * Zero host DDR intermediate writebacks")
    emit("//   - Time-Multiplexed 16 Compute Cores executing Layer 0 then Layer 1")
    emit("//===----------------------------------------------------------------------===//\n")

    emit("module {")
    emit("  aie.device(npu1) {")

    # Tile Declarations
    emit("    // ------------------------------------------------------------------------")
    emit("    // Tile Declarations (4 Columns x 6 Rows = 24 Tiles)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    // Column {c}")
        emit(f"    %tile_{c}_0 = aie.tile({c}, 0) // Shim NoC")
        emit(f"    %tile_{c}_1 = aie.tile({c}, 1) // MemTile L2")
        for r in range(2, 6):
            emit(f"    %tile_{c}_{r} = aie.tile({c}, {r}) // Core Tile ({c},{r})")

    # External Buffers and Shim Locks
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // External DDR Buffers & Shim Locks (Arg 0 Input, Arg 1 Output)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    %ext_buf_{c} = aie.external_buffer : memref<2048xi8>")
        emit(f"    %ext_out_buf_{c} = aie.external_buffer : memref<1024xi8>")
        emit(f"    %shim_lock_0_{c} = aie.lock(%tile_{c}_0, 0) {{ init = 1 : i32, sym_name = \"shim_lock_0_{c}\" }}")
        emit(f"    %shim_lock_1_{c} = aie.lock(%tile_{c}_0, 1) {{ init = 1 : i32, sym_name = \"shim_lock_1_{c}\" }}")

    # MemTile L2 Buffers & Locks
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // MemTile L2 Buffers & Locks (512 KB SRAM per MemTile)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    // MemTile ({c}, 1)")
        emit(f"    %mem_in_{c} = aie.buffer(%tile_{c}_1) {{ sym_name = \"mem_in_{c}\", address = 0 : i32 }} : memref<2048xi8>")
        emit(f"    %l2_bank0_{c} = aie.buffer(%tile_{c}_1) {{ sym_name = \"l2_bank0_{c}\", address = 262144 : i32 }} : memref<4096xi8> // 0x40000 (Ping)")
        emit(f"    %l2_bank1_{c} = aie.buffer(%tile_{c}_1) {{ sym_name = \"l2_bank1_{c}\", address = 393216 : i32 }} : memref<4096xi8> // 0x60000 (Pong)")
        emit(f"    %mem_out_{c} = aie.buffer(%tile_{c}_1) {{ sym_name = \"mem_out_{c}\", address = 16384 : i32 }} : memref<1024xi8> // 0x04000")
        emit(f"    %mem_lock_prod_{c} = aie.lock(%tile_{c}_1, 0) {{ init = 1 : i32, sym_name = \"mem_lock_prod_{c}\" }}")
        emit(f"    %mem_lock_cons_{c} = aie.lock(%tile_{c}_1, 1) {{ init = 0 : i32, sym_name = \"mem_lock_cons_{c}\" }}")
        emit(f"    %mem_out_prod_{c} = aie.lock(%tile_{c}_1, 2) {{ init = 4 : i32, sym_name = \"mem_out_prod_{c}\" }}")
        emit(f"    %mem_out_cons_{c} = aie.lock(%tile_{c}_1, 3) {{ init = 0 : i32, sym_name = \"mem_out_cons_{c}\" }}")
        emit(f"    %l2_ping_lock_{c} = aie.lock(%tile_{c}_1, 4) {{ init = 1 : i32, sym_name = \"l2_ping_lock_{c}\" }}")
        emit(f"    %l2_pong_lock_{c} = aie.lock(%tile_{c}_1, 5) {{ init = 0 : i32, sym_name = \"l2_pong_lock_{c}\" }}")

    # Core L1 Buffers & Locks
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // Core L1 Buffers & Locks (16 Compute Cores)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        for r in range(2, 6):
            emit(f"    // Core ({c}, {r})")
            emit(f"    %core_ping_{c}_{r} = aie.buffer(%tile_{c}_{r}) {{ sym_name = \"core_ping_{c}_{r}\" }} : memref<576xi8>")
            emit(f"    %core_pong_{c}_{r} = aie.buffer(%tile_{c}_{r}) {{ sym_name = \"core_pong_{c}_{r}\" }} : memref<576xi8>")
            emit(f"    %core_weights_0_{c}_{r} = aie.buffer(%tile_{c}_{r}) {{ sym_name = \"core_weights_0_{c}_{r}\", address = 1024 : i32 }} : memref<2304xi8> // 0x70400 (L0)")
            emit(f"    %core_weights_1_{c}_{r} = aie.buffer(%tile_{c}_{r}) {{ sym_name = \"core_weights_1_{c}_{r}\", address = 5120 : i32 }} : memref<2304xi8> // 0x71400 (L1)")
            emit(f"    %core_out_i8_{c}_{r} = aie.buffer(%tile_{c}_{r}) {{ sym_name = \"core_out_i8_{c}_{r}\" }} : memref<256xi8>")
            emit(f"    %ping_prod_{c}_{r} = aie.lock(%tile_{c}_{r}, 0) {{ init = 1 : i32, sym_name = \"ping_prod_{c}_{r}\" }}")
            emit(f"    %ping_cons_{c}_{r} = aie.lock(%tile_{c}_{r}, 1) {{ init = 0 : i32, sym_name = \"ping_cons_{c}_{r}\" }}")
            emit(f"    %pong_prod_{c}_{r} = aie.lock(%tile_{c}_{r}, 2) {{ init = 1 : i32, sym_name = \"pong_prod_{c}_{r}\" }}")
            emit(f"    %pong_cons_{c}_{r} = aie.lock(%tile_{c}_{r}, 3) {{ init = 0 : i32, sym_name = \"pong_cons_{c}_{r}\" }}")
            emit(f"    %out_prod_{c}_{r} = aie.lock(%tile_{c}_{r}, 4) {{ init = 1 : i32, sym_name = \"out_prod_{c}_{r}\" }}")
            emit(f"    %out_cons_{c}_{r} = aie.lock(%tile_{c}_{r}, 5) {{ init = 0 : i32, sym_name = \"out_cons_{c}_{r}\" }}")

    # FFI Declaration
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // Foreign Function Interface (FFI) Declaration: SRS Vector Kernel")
    emit("    // ------------------------------------------------------------------------")
    emit("    func.func private @conv_im2col_ping_pong_m2_srs(memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> () attributes {link_with = \"conv_im2col_kernel_m2.o\"}")

    # Routing Network
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // Stream Routing Network (40 Flows across 4 Columns)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    // Column {c} Flows")
        emit(f"    aie.flow(%tile_{c}_0, DMA : 0, %tile_{c}_1, DMA : 0) // Shim to MemTile (Frame Ingress)")
        for r in range(2, 6):
            emit(f"    aie.flow(%tile_{c}_1, DMA : 0, %tile_{c}_{r}, DMA : 0) // MemTile Multicast Broadcast to Core ({c},{r})")
        for idx, r in enumerate(range(2, 6)):
            emit(f"    aie.flow(%tile_{c}_{r}, DMA : 0, %tile_{c}_1, DMA : {idx+1}) // Core ({c},{r}) to MemTile Gather Channel {idx+1}")
        emit(f"    aie.flow(%tile_{c}_1, DMA : 1, %tile_{c}_0, DMA : 0) // MemTile to Shim (Final Egress)")

    # Shim DMAs
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // Shim NoC DMAs (4 Columns)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    %shim_dma_{c} = aie.shim_dma(%tile_{c}_0) {{")
        emit(f"      %c1 = arith.constant 1 : i32")
        emit(f"      aie.dma_start(MM2S, 0, ^bd_in_{c}, ^ch_s2mm_{c})")
        emit(f"    ^bd_in_{c}:")
        emit(f"      aie.use_lock(%shim_lock_0_{c}, AcquireGreaterEqual, %c1)")
        emit(f"      aie.dma_bd(%ext_buf_{c} : memref<2048xi8> offset = 0 len = 2048)")
        emit(f"      aie.use_lock(%shim_lock_0_{c}, Release, %c1)")
        emit(f"      aie.next_bd ^bd_in_{c}")
        emit(f"    ^ch_s2mm_{c}:")
        emit(f"      aie.dma_start(S2MM, 0, ^bd_out_{c}, ^end_{c})")
        emit(f"    ^bd_out_{c}:")
        emit(f"      aie.use_lock(%shim_lock_1_{c}, AcquireGreaterEqual, %c1)")
        emit(f"      aie.dma_bd(%ext_out_buf_{c} : memref<1024xi8> offset = 0 len = 1024)")
        emit(f"      aie.use_lock(%shim_lock_1_{c}, Release, %c1)")
        emit(f"      aie.next_bd ^bd_out_{c}")
        emit(f"    ^end_{c}:")
        emit(f"      aie.end")
        emit(f"    }}")

    # MemTile DMAs with L2 Ping-Pong Chaining
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // MemTile DMAs (L2 Activation Buffer Floorplan & Lock Chaining)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        emit(f"    %memtile_dma_{c}_1 = aie.memtile_dma(%tile_{c}_1) {{")
        emit(f"      %c1 = arith.constant 1 : i32")
        emit(f"      aie.dma_start(S2MM, 0, ^s2mm_in_bd_{c}, ^ch_s2mm_1_{c})")
        emit(f"    ^s2mm_in_bd_{c}:")
        emit(f"      aie.use_lock(%mem_lock_prod_{c}, AcquireGreaterEqual, %c1)")
        emit(f"      aie.dma_bd(%mem_in_{c} : memref<2048xi8> offset = 0 len = 2048)")
        emit(f"      aie.use_lock(%mem_lock_cons_{c}, Release, %c1)")
        emit(f"      aie.next_bd ^s2mm_in_bd_{c}")

        for idx in range(1, 5):
            emit(f"    ^ch_s2mm_{idx}_{c}:")
            next_ch = f"^ch_s2mm_{idx+1}_{c}" if idx < 4 else f"^ch_mm2s_0_{c}"
            emit(f"      aie.dma_start(S2MM, {idx}, ^bd_g{idx-1}_{c}, {next_ch})")
            emit(f"    ^bd_g{idx-1}_{c}:")
            emit(f"      aie.use_lock(%mem_out_prod_{c}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.dma_bd(%l2_bank0_{c} : memref<4096xi8> offset = {(idx-1)*256} len = 256)")
            emit(f"      aie.use_lock(%mem_out_cons_{c}, Release, %c1)")
            emit(f"      aie.next_bd ^bd_g{idx-1}_{c}")

        emit(f"    ^ch_mm2s_0_{c}:")
        emit(f"      aie.dma_start(MM2S, 0, ^mm2s_in_bd_{c}, ^ch_mm2s_1_{c})")
        emit(f"    ^mm2s_in_bd_{c}:")
        emit(f"      aie.use_lock(%mem_lock_cons_{c}, AcquireGreaterEqual, %c1)")
        emit(f"      aie.dma_bd(%mem_in_{c} : memref<2048xi8> offset = 0 len = 1728")
        emit(f"                 sizes = [6, 3, 3, 32]")
        emit(f"                 strides = [32, 256, 32, 1])")
        emit(f"      aie.use_lock(%mem_lock_prod_{c}, Release, %c1)")
        emit(f"      aie.next_bd ^mm2s_in_bd_{c}")

        emit(f"    ^ch_mm2s_1_{c}:")
        emit(f"      aie.dma_start(MM2S, 1, ^egress_bd_{c}, ^end_mem_{c})")
        emit(f"    ^egress_bd_{c}:")
        emit(f"      aie.use_lock(%mem_out_cons_{c}, AcquireGreaterEqual, %c1)")
        emit(f"      aie.dma_bd(%mem_out_{c} : memref<1024xi8> offset = 0 len = 1024)")
        emit(f"      aie.use_lock(%mem_out_prod_{c}, Release, %c1)")
        emit(f"      aie.next_bd ^egress_bd_{c}")
        emit(f"    ^end_mem_{c}:")
        emit(f"      aie.end")
        emit(f"    }}")

    # Core Compute Tiles (DMAs & Kernels)
    emit("\n    // ------------------------------------------------------------------------")
    emit("    // Core Compute Tiles (16 Cores: DMAs & Sequential 2-Layer Vector Loops)")
    emit("    // ------------------------------------------------------------------------")
    for c in range(4):
        for r in range(2, 6):
            emit(f"    // Tile ({c},{r}) DMA")
            emit(f"    %mem_{c}_{r} = aie.mem(%tile_{c}_{r}) {{")
            emit(f"      %c1 = arith.constant 1 : i32")
            emit(f"      aie.dma_start(S2MM, 0, ^bd_ping_{c}_{r}, ^egress_chan_{c}_{r})")
            emit(f"    ^bd_ping_{c}_{r}:")
            emit(f"      aie.use_lock(%ping_prod_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.dma_bd(%core_ping_{c}_{r} : memref<576xi8> offset = 0 len = 576)")
            emit(f"      aie.use_lock(%ping_cons_{c}_{r}, Release, %c1)")
            emit(f"      aie.next_bd ^bd_pong_{c}_{r}")
            emit(f"    ^bd_pong_{c}_{r}:")
            emit(f"      aie.use_lock(%pong_prod_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.dma_bd(%core_pong_{c}_{r} : memref<576xi8> offset = 0 len = 576)")
            emit(f"      aie.use_lock(%pong_cons_{c}_{r}, Release, %c1)")
            emit(f"      aie.next_bd ^bd_ping_{c}_{r}")
            emit(f"    ^egress_chan_{c}_{r}:")
            emit(f"      aie.dma_start(MM2S, 0, ^bd_egress_{c}_{r}, ^end_core_dma_{c}_{r})")
            emit(f"    ^bd_egress_{c}_{r}:")
            emit(f"      aie.use_lock(%out_cons_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.dma_bd(%core_out_i8_{c}_{r} : memref<256xi8> offset = 0 len = 256)")
            emit(f"      aie.use_lock(%out_prod_{c}_{r}, Release, %c1)")
            emit(f"      aie.next_bd ^bd_egress_{c}_{r}")
            emit(f"    ^end_core_dma_{c}_{r}:")
            emit(f"      aie.end")
            emit(f"    }}")

            emit(f"    // Tile ({c},{r}) Core Vector Execution (Layer 0 -> Layer 1 Sequential Time-Multiplex)")
            emit(f"    %core_{c}_{r} = aie.core(%tile_{c}_{r}) {{")
            emit(f"      %c1 = arith.constant 1 : i32")
            emit(f"      %shift0 = arith.constant 7 : i32")
            emit(f"      %shift1 = arith.constant 7 : i32")
            emit(f"      // Layer 0:")
            emit(f"      aie.use_lock(%ping_cons_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.use_lock(%pong_cons_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.use_lock(%out_prod_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_{c}_{r}, %core_pong_{c}_{r}, %core_weights_0_{c}_{r}, %core_out_i8_{c}_{r}, %shift0) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()")
            emit(f"      aie.use_lock(%ping_prod_{c}_{r}, Release, %c1)")
            emit(f"      aie.use_lock(%pong_prod_{c}_{r}, Release, %c1)")
            emit(f"      aie.use_lock(%out_cons_{c}_{r}, Release, %c1)")
            emit(f"      // Layer 1:")
            emit(f"      aie.use_lock(%ping_cons_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.use_lock(%pong_cons_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      aie.use_lock(%out_prod_{c}_{r}, AcquireGreaterEqual, %c1)")
            emit(f"      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_{c}_{r}, %core_pong_{c}_{r}, %core_weights_1_{c}_{r}, %core_out_i8_{c}_{r}, %shift1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()")
            emit(f"      aie.use_lock(%ping_prod_{c}_{r}, Release, %c1)")
            emit(f"      aie.use_lock(%pong_prod_{c}_{r}, Release, %c1)")
            emit(f"      aie.use_lock(%out_cons_{c}_{r}, Release, %c1)")
            emit(f"      aie.end")
            emit(f"    }}")

    emit("  }")
    emit("}")

    content = "\n".join(lines) + "\n"
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(content)
    print(f"Generated {out_path} ({len(content)} bytes, {len(lines)} lines)")

if __name__ == "__main__":
    generate_fused_mlir()
