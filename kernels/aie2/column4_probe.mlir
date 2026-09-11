//===- column4_probe.mlir --------------------------------------*- MLIR -*-===//
//
// AMD Phoenix AIE2 (XDNA1) 5th Physical Column Probe & Routing Harness
// Target: AMD Phoenix NPU (AIE2 / NPU1, physical 5-column array: Cols 0..4)
//
// Purpose:
//   Test addressability, switchbox pathfinding, BD generation, and transaction
//   synthesis for Column 4 (Tiles 4,0..4,5) on AMD Phoenix XDNA1 silicon.
//
// Dual-Topology Verification:
//   1. Path A (Direct Column 4 Shim NoC DMA):
//      Host DDR -> Tile(4,0) Shim MM2S:0 -> Tile(4,1) MemTile S2MM:0
//      Tile(4,1) MemTile MM2S:0 -> Tile(4,2) Core S2MM:0
//      Tile(4,2) Core MM2S:0 -> Tile(4,1) MemTile S2MM:1
//      Tile(4,1) MemTile MM2S:1 -> Tile(4,0) Shim S2MM:0 -> Host DDR
//   2. Path B (Cross-Column West-to-East Switchbox Bypass):
//      Tile(3,1) MemTile MM2S:2 -> East Switchbox -> Tile(4,2) Core S2MM:1
//      Tile(4,2) Core MM2S:1 -> West Switchbox -> Tile(3,1) MemTile S2MM:2
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2_5col) {
    // ------------------------------------------------------------------------
    // Tile Architecture: Column 3 (West Neighbor) & Column 4 (5th Physical Column)
    // ------------------------------------------------------------------------
    %tile_3_0 = aie.tile(3, 0) // Column 3 Shim NoC Tile
    %tile_3_1 = aie.tile(3, 1) // Column 3 MemTile L2 (512 KB)
    %tile_3_2 = aie.tile(3, 2) // Column 3 Core Tile (Row 2)

    %tile_4_0 = aie.tile(4, 0) // Column 4 Shim NoC Tile (Probe Target)
    %tile_4_1 = aie.tile(4, 1) // Column 4 MemTile L2 (512 KB, Probe Target)
    %tile_4_2 = aie.tile(4, 2) // Column 4 Core Tile (Row 2, 64 KB L1, Probe Target)

    // ------------------------------------------------------------------------
    // External Buffers & Hardware Locks for Column 4 Direct Shim DMA
    // ------------------------------------------------------------------------
    %ext_buf_col4_in = aie.external_buffer : memref<1024xi8>
    %ext_buf_col4_out = aie.external_buffer : memref<1024xi8>
    %shim4_lock_0 = aie.lock(%tile_4_0, 0) { init = 1 : i32, sym_name = "shim4_lock_0" }
    %shim4_lock_1 = aie.lock(%tile_4_0, 1) { init = 1 : i32, sym_name = "shim4_lock_1" }

    // ------------------------------------------------------------------------
    // MemTile L2 Storage & Hardware Locks: Tile(4,1)
    // ------------------------------------------------------------------------
    %mem4_in = aie.buffer(%tile_4_1) { sym_name = "mem4_in" } : memref<1024xi8>
    %mem4_out = aie.buffer(%tile_4_1) { sym_name = "mem4_out" } : memref<1024xi8>
    %mem4_lock_in_prod = aie.lock(%tile_4_1, 0) { init = 1 : i32, sym_name = "mem4_lock_in_prod" }
    %mem4_lock_in_cons = aie.lock(%tile_4_1, 1) { init = 0 : i32, sym_name = "mem4_lock_in_cons" }
    %mem4_lock_out_prod = aie.lock(%tile_4_1, 2) { init = 1 : i32, sym_name = "mem4_lock_out_prod" }
    %mem4_lock_out_cons = aie.lock(%tile_4_1, 3) { init = 0 : i32, sym_name = "mem4_lock_out_cons" }

    // ------------------------------------------------------------------------
    // MemTile L2 Storage & Hardware Locks: Tile(3,1) (Cross-Column Transit)
    // ------------------------------------------------------------------------
    %mem3_cross_in = aie.buffer(%tile_3_1) { sym_name = "mem3_cross_in" } : memref<512xi8>
    %mem3_cross_out = aie.buffer(%tile_3_1) { sym_name = "mem3_cross_out" } : memref<512xi8>
    %mem3_lock_prod = aie.lock(%tile_3_1, 0) { init = 1 : i32, sym_name = "mem3_lock_prod" }
    %mem3_lock_cons = aie.lock(%tile_3_1, 1) { init = 0 : i32, sym_name = "mem3_lock_cons" }

    // ------------------------------------------------------------------------
    // Core L1 Allocations & Hardware Locks: Tile(4,2)
    // ------------------------------------------------------------------------
    %core4_ping = aie.buffer(%tile_4_2) { sym_name = "core4_ping" } : memref<256xi8>
    %core4_pong = aie.buffer(%tile_4_2) { sym_name = "core4_pong" } : memref<256xi8>
    %core4_out = aie.buffer(%tile_4_2) { sym_name = "core4_out" } : memref<256xi8>
    %core4_cross_buf = aie.buffer(%tile_4_2) { sym_name = "core4_cross_buf" } : memref<256xi8>

    %core4_lock_ping_prod = aie.lock(%tile_4_2, 0) { init = 1 : i32, sym_name = "core4_lock_ping_prod" }
    %core4_lock_ping_cons = aie.lock(%tile_4_2, 1) { init = 0 : i32, sym_name = "core4_lock_ping_cons" }
    %core4_lock_pong_prod = aie.lock(%tile_4_2, 2) { init = 1 : i32, sym_name = "core4_lock_pong_prod" }
    %core4_lock_pong_cons = aie.lock(%tile_4_2, 3) { init = 0 : i32, sym_name = "core4_lock_pong_cons" }
    %core4_lock_out_prod = aie.lock(%tile_4_2, 4) { init = 1 : i32, sym_name = "core4_lock_out_prod" }
    %core4_lock_out_cons = aie.lock(%tile_4_2, 5) { init = 0 : i32, sym_name = "core4_lock_out_cons" }
    %core4_lock_cross_prod = aie.lock(%tile_4_2, 6) { init = 1 : i32, sym_name = "core4_lock_cross_prod" }
    %core4_lock_cross_cons = aie.lock(%tile_4_2, 7) { init = 0 : i32, sym_name = "core4_lock_cross_cons" }

    // ------------------------------------------------------------------------
    // Circuit-Switched Stream Flows:
    // ------------------------------------------------------------------------
    // Path A: Direct Column 4 Shim -> MemTile -> Core Roundtrip
    aie.flow(%tile_4_0, DMA : 0, %tile_4_1, DMA : 0) // Ingress Shim -> MemTile
    aie.flow(%tile_4_1, DMA : 0, %tile_4_2, DMA : 0) // MemTile -> Core 4,2 Ping-Pong
    aie.flow(%tile_4_2, DMA : 0, %tile_4_1, DMA : 1) // Core 4,2 Out -> MemTile Gathering
    aie.flow(%tile_4_1, DMA : 1, %tile_4_0, DMA : 0) // Egress MemTile -> Shim Out

    // Path B: Cross-Column Routing (Column 3 MemTile <-> Column 4 Core Tile)
    aie.flow(%tile_3_1, DMA : 2, %tile_4_2, DMA : 1) // Col 3 MemTile -> Col 4 Core Ingress
    aie.flow(%tile_4_2, DMA : 1, %tile_3_1, DMA : 2) // Col 4 Core -> Col 3 MemTile Egress

    // ------------------------------------------------------------------------
    // Core Execution Logic: Tile(4,2)
    // ------------------------------------------------------------------------
    %core_4_2 = aie.core(%tile_4_2) {
      %c1 = arith.constant 1 : i32
      aie.use_lock(%core4_lock_ping_cons, AcquireGreaterEqual, %c1)
      aie.use_lock(%core4_lock_out_prod, AcquireGreaterEqual, %c1)
      // Vector compute placeholder on Column 4 Core
      aie.use_lock(%core4_lock_ping_prod, Release, %c1)
      aie.use_lock(%core4_lock_out_cons, Release, %c1)
      aie.end
    }

    // ------------------------------------------------------------------------
    // DMA Configuration: Tile(4,0) Shim NoC DMA
    // ------------------------------------------------------------------------
    %shim_dma_4_0 = aie.shim_dma(%tile_4_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_shim_in, ^chan_s2mm)
    ^bd_shim_in:
      aie.use_lock(%shim4_lock_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_col4_in : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim4_lock_0, Release, %c1)
      aie.next_bd ^bd_shim_in

    ^chan_s2mm:
      aie.dma_start(S2MM, 0, ^bd_shim_out, ^end_shim)
    ^bd_shim_out:
      aie.use_lock(%shim4_lock_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_col4_out : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim4_lock_1, Release, %c1)
      aie.next_bd ^bd_shim_out

    ^end_shim:
      aie.end
    }

    // ------------------------------------------------------------------------
    // DMA Configuration: Tile(4,1) MemTile L2 DMA
    // ------------------------------------------------------------------------
    %mem_dma_4_1 = aie.memtile_dma(%tile_4_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_mem_in, ^chan_mm2s0)
    ^bd_mem_in:
      aie.use_lock(%mem4_lock_in_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem4_in : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem4_lock_in_cons, Release, %c1)
      aie.next_bd ^bd_mem_in

    ^chan_mm2s0:
      aie.dma_start(MM2S, 0, ^bd_mem_stream, ^chan_s2mm1)
    ^bd_mem_stream:
      aie.use_lock(%mem4_lock_in_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem4_in : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem4_lock_in_prod, Release, %c1)
      aie.next_bd ^bd_mem_stream

    ^chan_s2mm1:
      aie.dma_start(S2MM, 1, ^bd_mem_gather, ^chan_mm2s1)
    ^bd_mem_gather:
      aie.use_lock(%mem4_lock_out_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem4_out : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem4_lock_out_cons, Release, %c1)
      aie.next_bd ^bd_mem_gather

    ^chan_mm2s1:
      aie.dma_start(MM2S, 1, ^bd_mem_egress, ^end_mem)
    ^bd_mem_egress:
      aie.use_lock(%mem4_lock_out_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem4_out : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem4_lock_out_prod, Release, %c1)
      aie.next_bd ^bd_mem_egress

    ^end_mem:
      aie.end
    }

    // ------------------------------------------------------------------------
    // DMA Configuration: Tile(4,2) Core L1 DMA
    // ------------------------------------------------------------------------
    %core_dma_4_2 = aie.mem(%tile_4_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_core_ping, ^chan_core_mm2s0)
    ^bd_core_ping:
      aie.use_lock(%core4_lock_ping_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core4_ping : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%core4_lock_ping_cons, Release, %c1)
      aie.next_bd ^bd_core_ping

    ^chan_core_mm2s0:
      aie.dma_start(MM2S, 0, ^bd_core_out, ^chan_core_s2mm1)
    ^bd_core_out:
      aie.use_lock(%core4_lock_out_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core4_out : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%core4_lock_out_prod, Release, %c1)
      aie.next_bd ^bd_core_out

    ^chan_core_s2mm1:
      aie.dma_start(S2MM, 1, ^bd_core_cross_in, ^chan_core_mm2s1)
    ^bd_core_cross_in:
      aie.use_lock(%core4_lock_cross_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core4_cross_buf : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%core4_lock_cross_cons, Release, %c1)
      aie.next_bd ^bd_core_cross_in

    ^chan_core_mm2s1:
      aie.dma_start(MM2S, 1, ^bd_core_cross_out, ^end_core)
    ^bd_core_cross_out:
      aie.use_lock(%core4_lock_cross_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core4_cross_buf : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%core4_lock_cross_prod, Release, %c1)
      aie.next_bd ^bd_core_cross_out

    ^end_core:
      aie.end
    }

    // ------------------------------------------------------------------------
    // DMA Configuration: Tile(3,1) MemTile L2 DMA (Cross-Column Routing Source)
    // ------------------------------------------------------------------------
    %mem_dma_3_1 = aie.memtile_dma(%tile_3_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 2, ^bd_mem3_east, ^chan_mem3_west)
    ^bd_mem3_east:
      aie.use_lock(%mem3_lock_cons, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem3_cross_in : memref<512xi8> offset = 0 len = 512)
      aie.use_lock(%mem3_lock_prod, Release, %c1)
      aie.next_bd ^bd_mem3_east

    ^chan_mem3_west:
      aie.dma_start(S2MM, 2, ^bd_mem3_west, ^end_mem3)
    ^bd_mem3_west:
      aie.use_lock(%mem3_lock_prod, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem3_cross_out : memref<512xi8> offset = 0 len = 512)
      aie.use_lock(%mem3_lock_cons, Release, %c1)
      aie.next_bd ^bd_mem3_west

    ^end_mem3:
      aie.end
    }
  }
}
