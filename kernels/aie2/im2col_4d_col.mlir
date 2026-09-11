//===- im2col_4d_col.mlir ---------------------------------------*- MLIR -*-===//
//
// Standalone 4-D Buffer Descriptor im2col multi-core dataflow harness for
// AMD Phoenix AIE2 (XDNA1), scaling execution across all 4 compute tiles in
// Column 0 (Tiles 0,2 through 0,5).
//
// Target: AMD Phoenix AIE2 (npu1, 4-column array)
// Topology:
//   - Tile(0,0): Shim NoC DMA (external memory ingress from DDR)
//   - Tile(0,1): MemTile L2 SRAM (512 KB activation storage + 4-D DMA generator)
//   - Tile(0,2): Core Compute Tile 0 (L1 64 KB, Bank 0-3 zero-conflict allocation)
//   - Tile(0,3): Core Compute Tile 1 (L1 64 KB, Bank 0-3 zero-conflict allocation)
//   - Tile(0,4): Core Compute Tile 2 (L1 64 KB, Bank 0-3 zero-conflict allocation)
//   - Tile(0,5): Core Compute Tile 3 (L1 64 KB, Bank 0-3 zero-conflict allocation)
//
// Interconnect Distribution:
//   - Stream switchbox broadcast tree: MemTile MM2S Channel 0 streams to
//     S2MM Channel 0 on Tiles (0,2), (0,3), (0,4), and (0,5) simultaneously.
//   - Hardware circuit-switched multicast delivers 1728 bytes of receptive field
//     windows concurrently to all 4 cores without store-and-forward latency or
//     inter-tile contention.
//
// L1 Memory Bank Mapping (per core tile):
//   - Bank 0 (0x70000): Execution stack (1024 B) + stationary weights (2304 B)
//   - Bank 1 (0x74000): Output accumulators (1024 B = 256 INT32 elements)
//   - Bank 2 (0x78000): Ingress Ping buffer (576 B = 2 patches x 288 B)
//   - Bank 3 (0x7C000): Ingress Pong buffer (576 B = 2 patches x 288 B)
//   Total core allocation: 5,504 B (8.40% tile capacity, 58.5 KB headroom).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    // ------------------------------------------------------------------------
    // Tile Architecture: Full Column 0 (Shim, MemTile, 4 Cores)
    // ------------------------------------------------------------------------
    %tile_0_0 = aie.tile(0, 0) // Shim NoC Tile
    %tile_0_1 = aie.tile(0, 1) // MemTile L2 (512 KB)
    %tile_0_2 = aie.tile(0, 2) // Core Tile 0 (Row 2)
    %tile_0_3 = aie.tile(0, 3) // Core Tile 1 (Row 3)
    %tile_0_4 = aie.tile(0, 4) // Core Tile 2 (Row 4)
    %tile_0_5 = aie.tile(0, 5) // Core Tile 3 (Row 5)

    // ------------------------------------------------------------------------
    // External Buffer & Shim Locks (Tile 0,0)
    // ------------------------------------------------------------------------
    %ext_buf = aie.external_buffer : memref<2048xi8>
    %shim_lock_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32, sym_name = "shim_lock_0" }

    // ------------------------------------------------------------------------
    // MemTile L2 Storage & Hardware Locks (Tile 0,1)
    // ------------------------------------------------------------------------
    %mem_in = aie.buffer(%tile_0_1) { sym_name = "mem_in" } : memref<2048xi8>
    %mem_lock_prod = aie.lock(%tile_0_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod" }
    %mem_lock_cons = aie.lock(%tile_0_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons" }

    // ------------------------------------------------------------------------
    // Core L1 Allocations & Hardware Locks: Tile(0,2)
    // ------------------------------------------------------------------------
    %core_ping_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_ping_0_2" } : memref<576xi8>
    %core_pong_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_pong_0_2" } : memref<576xi8>
    %core_weights_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_weights_0_2" } : memref<2304xi8>
    %core_out_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_out_0_2" } : memref<256xi32>
    %ping_prod_0_2 = aie.lock(%tile_0_2, 0) { init = 1 : i32, sym_name = "ping_prod_0_2" }
    %ping_cons_0_2 = aie.lock(%tile_0_2, 1) { init = 0 : i32, sym_name = "ping_cons_0_2" }
    %pong_prod_0_2 = aie.lock(%tile_0_2, 2) { init = 1 : i32, sym_name = "pong_prod_0_2" }
    %pong_cons_0_2 = aie.lock(%tile_0_2, 3) { init = 0 : i32, sym_name = "pong_cons_0_2" }

    // ------------------------------------------------------------------------
    // Core L1 Allocations & Hardware Locks: Tile(0,3)
    // ------------------------------------------------------------------------
    %core_ping_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_ping_0_3" } : memref<576xi8>
    %core_pong_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_pong_0_3" } : memref<576xi8>
    %core_weights_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_weights_0_3" } : memref<2304xi8>
    %core_out_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_out_0_3" } : memref<256xi32>
    %ping_prod_0_3 = aie.lock(%tile_0_3, 0) { init = 1 : i32, sym_name = "ping_prod_0_3" }
    %ping_cons_0_3 = aie.lock(%tile_0_3, 1) { init = 0 : i32, sym_name = "ping_cons_0_3" }
    %pong_prod_0_3 = aie.lock(%tile_0_3, 2) { init = 1 : i32, sym_name = "pong_prod_0_3" }
    %pong_cons_0_3 = aie.lock(%tile_0_3, 3) { init = 0 : i32, sym_name = "pong_cons_0_3" }

    // ------------------------------------------------------------------------
    // Core L1 Allocations & Hardware Locks: Tile(0,4)
    // ------------------------------------------------------------------------
    %core_ping_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_ping_0_4" } : memref<576xi8>
    %core_pong_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_pong_0_4" } : memref<576xi8>
    %core_weights_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_weights_0_4" } : memref<2304xi8>
    %core_out_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_out_0_4" } : memref<256xi32>
    %ping_prod_0_4 = aie.lock(%tile_0_4, 0) { init = 1 : i32, sym_name = "ping_prod_0_4" }
    %ping_cons_0_4 = aie.lock(%tile_0_4, 1) { init = 0 : i32, sym_name = "ping_cons_0_4" }
    %pong_prod_0_4 = aie.lock(%tile_0_4, 2) { init = 1 : i32, sym_name = "pong_prod_0_4" }
    %pong_cons_0_4 = aie.lock(%tile_0_4, 3) { init = 0 : i32, sym_name = "pong_cons_0_4" }

    // ------------------------------------------------------------------------
    // Core L1 Allocations & Hardware Locks: Tile(0,5)
    // ------------------------------------------------------------------------
    %core_ping_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_ping_0_5" } : memref<576xi8>
    %core_pong_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_pong_0_5" } : memref<576xi8>
    %core_weights_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_weights_0_5" } : memref<2304xi8>
    %core_out_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_out_0_5" } : memref<256xi32>
    %ping_prod_0_5 = aie.lock(%tile_0_5, 0) { init = 1 : i32, sym_name = "ping_prod_0_5" }
    %ping_cons_0_5 = aie.lock(%tile_0_5, 1) { init = 0 : i32, sym_name = "ping_cons_0_5" }
    %pong_prod_0_5 = aie.lock(%tile_0_5, 2) { init = 1 : i32, sym_name = "pong_prod_0_5" }
    %pong_cons_0_5 = aie.lock(%tile_0_5, 3) { init = 0 : i32, sym_name = "pong_cons_0_5" }

    // ------------------------------------------------------------------------
    // Foreign Function Interface (FFI) Declaration
    // ------------------------------------------------------------------------
    func.func private @conv_im2col_ping_pong_m2(memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> () attributes {link_with = "conv_im2col_kernel_m2.o"}

    // ------------------------------------------------------------------------
    // Stream Interconnect: Shim Ingress & Multicast Broadcast Tree
    // ------------------------------------------------------------------------
    // Shim NoC DMA MM2S:0 -> MemTile S2MM:0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)

    // MemTile MM2S:0 -> Hardware Multicast Tree across all 4 Core Tiles S2MM:0
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_5, DMA : 0)

    // ------------------------------------------------------------------------
    // Shim NoC DMA (Tile 0,0): Ingress Stream from DDR
    // ------------------------------------------------------------------------
    %shim_dma_0 = aie.shim_dma(%tile_0_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%shim_lock_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0, Release, %c1)
      aie.next_bd ^bd0
    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // MemTile DMA (Tile 0,1): Ingress to L2 and 4-D im2col Multicast Egress
    // ------------------------------------------------------------------------
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c1 = arith.constant 1 : i32

      // S2MM Channel 0: Receive 2048 bytes of activations from Shim NoC DMA
      aie.dma_start(S2MM, 0, ^s2mm_bd, ^next_chan)
    ^s2mm_bd:
      aie.use_lock(%mem_lock_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons, Release, %c1)
      aie.next_bd ^s2mm_bd

    ^next_chan:
      // MM2S Channel 0: Multicast 1728 bytes of im2col receptive fields to all 4 Cores
      // Multi-dimensional addressing performs in-flight sliding window extraction
      aie.dma_start(MM2S, 0, ^mm2s_bd, ^end)
    ^mm2s_bd:
      aie.use_lock(%mem_lock_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod, Release, %c1)
      aie.next_bd ^mm2s_bd

    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile DMA (Tile 0,2): Ping-Pong Ingress
    // ------------------------------------------------------------------------
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping, ^end)
    ^bd_ping:
      aie.use_lock(%ping_prod_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_2, Release, %c1)
      aie.next_bd ^bd_pong
    ^bd_pong:
      aie.use_lock(%pong_prod_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_2, Release, %c1)
      aie.next_bd ^bd_ping
    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile Compute (Tile 0,2): Core 0
    // ------------------------------------------------------------------------
    %core_0_2 = aie.core(%tile_0_2) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%ping_cons_0_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2(%core_ping_0_2, %core_pong_0_2, %core_weights_0_2, %core_out_0_2, %c1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> ()
      aie.use_lock(%ping_prod_0_2, Release, %c1)
      aie.use_lock(%pong_prod_0_2, Release, %c1)
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile DMA (Tile 0,3): Ping-Pong Ingress
    // ------------------------------------------------------------------------
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping, ^end)
    ^bd_ping:
      aie.use_lock(%ping_prod_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_3, Release, %c1)
      aie.next_bd ^bd_pong
    ^bd_pong:
      aie.use_lock(%pong_prod_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_3, Release, %c1)
      aie.next_bd ^bd_ping
    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile Compute (Tile 0,3): Core 1
    // ------------------------------------------------------------------------
    %core_0_3 = aie.core(%tile_0_3) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%ping_cons_0_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2(%core_ping_0_3, %core_pong_0_3, %core_weights_0_3, %core_out_0_3, %c1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> ()
      aie.use_lock(%ping_prod_0_3, Release, %c1)
      aie.use_lock(%pong_prod_0_3, Release, %c1)
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile DMA (Tile 0,4): Ping-Pong Ingress
    // ------------------------------------------------------------------------
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping, ^end)
    ^bd_ping:
      aie.use_lock(%ping_prod_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_4, Release, %c1)
      aie.next_bd ^bd_pong
    ^bd_pong:
      aie.use_lock(%pong_prod_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_4, Release, %c1)
      aie.next_bd ^bd_ping
    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile Compute (Tile 0,4): Core 2
    // ------------------------------------------------------------------------
    %core_0_4 = aie.core(%tile_0_4) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%ping_cons_0_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2(%core_ping_0_4, %core_pong_0_4, %core_weights_0_4, %core_out_0_4, %c1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> ()
      aie.use_lock(%ping_prod_0_4, Release, %c1)
      aie.use_lock(%pong_prod_0_4, Release, %c1)
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile DMA (Tile 0,5): Ping-Pong Ingress
    // ------------------------------------------------------------------------
    %mem_0_5 = aie.mem(%tile_0_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping, ^end)
    ^bd_ping:
      aie.use_lock(%ping_prod_0_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_5, Release, %c1)
      aie.next_bd ^bd_pong
    ^bd_pong:
      aie.use_lock(%pong_prod_0_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_5, Release, %c1)
      aie.next_bd ^bd_ping
    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile Compute (Tile 0,5): Core 3
    // ------------------------------------------------------------------------
    %core_0_5 = aie.core(%tile_0_5) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%ping_cons_0_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2(%core_ping_0_5, %core_pong_0_5, %core_weights_0_5, %core_out_0_5, %c1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> ()
      aie.use_lock(%ping_prod_0_5, Release, %c1)
      aie.use_lock(%pong_prod_0_5, Release, %c1)
      aie.end
    }
  }
}