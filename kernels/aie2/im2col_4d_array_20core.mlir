//===- im2col_4d_array_20core.mlir ------------------------------*- MLIR -*-===//
//
// Full 20-Core Array im2col Execution Engine on AMD Phoenix XDNA1 (AIE2).
// Features:
//   - Complete 5x6 Physical Grid: Columns 0..4, Rows 0..5 (30 Physical Tiles)
//   - 5 Shim NoC Tiles: Tile(0,0) through Tile(4,0) (DDR Ingress & Egress)
//   - 5 MemTiles L2: Tile(0,1) through Tile(4,1) (2.56 MB L2 SRAM total)
//   - 20 Compute Cores: Tile(c, r) for c in 0..4 and r in 2..5 (1.28 MB L1 total)
//   - Total On-Chip SRAM: 3.84 MB
//   - 1-to-4 Multicast Broadcast Tree per Column from MemTile MM2S:0
//   - 4-Channel Parallel Gathering per Column to MemTile S2MM:1..4
//   - M=2 Dual-Patch Vector Inner Loop with Native Hardware SRS Requantization
//   - Zero-Conflict 4-Bank SRAM Placement per Core Tile
//
// Target: AMD Phoenix AIE2 (npu2_5col target model)
//===----------------------------------------------------------------------===//

module {
  aie.device(npu2_5col) {
    // ------------------------------------------------------------------------
    // Tile Declarations (5 Columns x 6 Rows = 30 Tiles)
    // ------------------------------------------------------------------------
    // Column 0
    %tile_0_0 = aie.tile(0, 0) // Shim NoC
    %tile_0_1 = aie.tile(0, 1) // MemTile L2
    %tile_0_2 = aie.tile(0, 2) // Core Tile (0,2)
    %tile_0_3 = aie.tile(0, 3) // Core Tile (0,3)
    %tile_0_4 = aie.tile(0, 4) // Core Tile (0,4)
    %tile_0_5 = aie.tile(0, 5) // Core Tile (0,5)
    // Column 1
    %tile_1_0 = aie.tile(1, 0) // Shim NoC
    %tile_1_1 = aie.tile(1, 1) // MemTile L2
    %tile_1_2 = aie.tile(1, 2) // Core Tile (1,2)
    %tile_1_3 = aie.tile(1, 3) // Core Tile (1,3)
    %tile_1_4 = aie.tile(1, 4) // Core Tile (1,4)
    %tile_1_5 = aie.tile(1, 5) // Core Tile (1,5)
    // Column 2
    %tile_2_0 = aie.tile(2, 0) // Shim NoC
    %tile_2_1 = aie.tile(2, 1) // MemTile L2
    %tile_2_2 = aie.tile(2, 2) // Core Tile (2,2)
    %tile_2_3 = aie.tile(2, 3) // Core Tile (2,3)
    %tile_2_4 = aie.tile(2, 4) // Core Tile (2,4)
    %tile_2_5 = aie.tile(2, 5) // Core Tile (2,5)
    // Column 3
    %tile_3_0 = aie.tile(3, 0) // Shim NoC
    %tile_3_1 = aie.tile(3, 1) // MemTile L2
    %tile_3_2 = aie.tile(3, 2) // Core Tile (3,2)
    %tile_3_3 = aie.tile(3, 3) // Core Tile (3,3)
    %tile_3_4 = aie.tile(3, 4) // Core Tile (3,4)
    %tile_3_5 = aie.tile(3, 5) // Core Tile (3,5)
    // Column 4
    %tile_4_0 = aie.tile(4, 0) // Shim NoC
    %tile_4_1 = aie.tile(4, 1) // MemTile L2
    %tile_4_2 = aie.tile(4, 2) // Core Tile (4,2)
    %tile_4_3 = aie.tile(4, 3) // Core Tile (4,3)
    %tile_4_4 = aie.tile(4, 4) // Core Tile (4,4)
    %tile_4_5 = aie.tile(4, 5) // Core Tile (4,5)

    // ------------------------------------------------------------------------
    // External DDR Buffers & Shim Locks (5 Columns)
    // ------------------------------------------------------------------------
    %ext_buf_0 = aie.external_buffer : memref<2048xi8>
    %ext_out_buf_0 = aie.external_buffer : memref<1024xi8>
    %shim_lock_0_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32, sym_name = "shim_lock_0_0" }
    %shim_lock_1_0 = aie.lock(%tile_0_0, 1) { init = 1 : i32, sym_name = "shim_lock_1_0" }
    %ext_buf_1 = aie.external_buffer : memref<2048xi8>
    %ext_out_buf_1 = aie.external_buffer : memref<1024xi8>
    %shim_lock_0_1 = aie.lock(%tile_1_0, 0) { init = 1 : i32, sym_name = "shim_lock_0_1" }
    %shim_lock_1_1 = aie.lock(%tile_1_0, 1) { init = 1 : i32, sym_name = "shim_lock_1_1" }
    %ext_buf_2 = aie.external_buffer : memref<2048xi8>
    %ext_out_buf_2 = aie.external_buffer : memref<1024xi8>
    %shim_lock_0_2 = aie.lock(%tile_2_0, 0) { init = 1 : i32, sym_name = "shim_lock_0_2" }
    %shim_lock_1_2 = aie.lock(%tile_2_0, 1) { init = 1 : i32, sym_name = "shim_lock_1_2" }
    %ext_buf_3 = aie.external_buffer : memref<2048xi8>
    %ext_out_buf_3 = aie.external_buffer : memref<1024xi8>
    %shim_lock_0_3 = aie.lock(%tile_3_0, 0) { init = 1 : i32, sym_name = "shim_lock_0_3" }
    %shim_lock_1_3 = aie.lock(%tile_3_0, 1) { init = 1 : i32, sym_name = "shim_lock_1_3" }
    %ext_buf_4 = aie.external_buffer : memref<2048xi8>
    %ext_out_buf_4 = aie.external_buffer : memref<1024xi8>
    %shim_lock_0_4 = aie.lock(%tile_4_0, 0) { init = 1 : i32, sym_name = "shim_lock_0_4" }
    %shim_lock_1_4 = aie.lock(%tile_4_0, 1) { init = 1 : i32, sym_name = "shim_lock_1_4" }

    // ------------------------------------------------------------------------
    // MemTile L2 Buffers & Locks (5 Columns, 512 KB per MemTile)
    // ------------------------------------------------------------------------
    %mem_in_0 = aie.buffer(%tile_0_1) { sym_name = "mem_in_0" } : memref<2048xi8>
    %mem_out_0 = aie.buffer(%tile_0_1) { sym_name = "mem_out_0" } : memref<1024xi8>
    %mem_lock_prod_0 = aie.lock(%tile_0_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod_0" }
    %mem_lock_cons_0 = aie.lock(%tile_0_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons_0" }
    %mem_out_prod_0 = aie.lock(%tile_0_1, 2) { init = 1 : i32, sym_name = "mem_out_prod_0" }
    %mem_out_cons_0 = aie.lock(%tile_0_1, 3) { init = 0 : i32, sym_name = "mem_out_cons_0" }
    %mem_in_1 = aie.buffer(%tile_1_1) { sym_name = "mem_in_1" } : memref<2048xi8>
    %mem_out_1 = aie.buffer(%tile_1_1) { sym_name = "mem_out_1" } : memref<1024xi8>
    %mem_lock_prod_1 = aie.lock(%tile_1_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod_1" }
    %mem_lock_cons_1 = aie.lock(%tile_1_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons_1" }
    %mem_out_prod_1 = aie.lock(%tile_1_1, 2) { init = 1 : i32, sym_name = "mem_out_prod_1" }
    %mem_out_cons_1 = aie.lock(%tile_1_1, 3) { init = 0 : i32, sym_name = "mem_out_cons_1" }
    %mem_in_2 = aie.buffer(%tile_2_1) { sym_name = "mem_in_2" } : memref<2048xi8>
    %mem_out_2 = aie.buffer(%tile_2_1) { sym_name = "mem_out_2" } : memref<1024xi8>
    %mem_lock_prod_2 = aie.lock(%tile_2_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod_2" }
    %mem_lock_cons_2 = aie.lock(%tile_2_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons_2" }
    %mem_out_prod_2 = aie.lock(%tile_2_1, 2) { init = 1 : i32, sym_name = "mem_out_prod_2" }
    %mem_out_cons_2 = aie.lock(%tile_2_1, 3) { init = 0 : i32, sym_name = "mem_out_cons_2" }
    %mem_in_3 = aie.buffer(%tile_3_1) { sym_name = "mem_in_3" } : memref<2048xi8>
    %mem_out_3 = aie.buffer(%tile_3_1) { sym_name = "mem_out_3" } : memref<1024xi8>
    %mem_lock_prod_3 = aie.lock(%tile_3_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod_3" }
    %mem_lock_cons_3 = aie.lock(%tile_3_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons_3" }
    %mem_out_prod_3 = aie.lock(%tile_3_1, 2) { init = 1 : i32, sym_name = "mem_out_prod_3" }
    %mem_out_cons_3 = aie.lock(%tile_3_1, 3) { init = 0 : i32, sym_name = "mem_out_cons_3" }
    %mem_in_4 = aie.buffer(%tile_4_1) { sym_name = "mem_in_4" } : memref<2048xi8>
    %mem_out_4 = aie.buffer(%tile_4_1) { sym_name = "mem_out_4" } : memref<1024xi8>
    %mem_lock_prod_4 = aie.lock(%tile_4_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod_4" }
    %mem_lock_cons_4 = aie.lock(%tile_4_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons_4" }
    %mem_out_prod_4 = aie.lock(%tile_4_1, 2) { init = 1 : i32, sym_name = "mem_out_prod_4" }
    %mem_out_cons_4 = aie.lock(%tile_4_1, 3) { init = 0 : i32, sym_name = "mem_out_cons_4" }

    // ------------------------------------------------------------------------
    // Core L1 Buffers & Locks (20 Compute Cores, 4 Banks per Core)
    // ------------------------------------------------------------------------
    // Core (0,2)
    %core_ping_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_ping_0_2" } : memref<576xi8>
    %core_pong_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_pong_0_2" } : memref<576xi8>
    %core_weights_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_weights_0_2" } : memref<2304xi8>
    %core_out_i8_0_2 = aie.buffer(%tile_0_2) { sym_name = "core_out_i8_0_2" } : memref<256xi8>
    %ping_prod_0_2 = aie.lock(%tile_0_2, 0) { init = 1 : i32, sym_name = "ping_prod_0_2" }
    %ping_cons_0_2 = aie.lock(%tile_0_2, 1) { init = 0 : i32, sym_name = "ping_cons_0_2" }
    %pong_prod_0_2 = aie.lock(%tile_0_2, 2) { init = 1 : i32, sym_name = "pong_prod_0_2" }
    %pong_cons_0_2 = aie.lock(%tile_0_2, 3) { init = 0 : i32, sym_name = "pong_cons_0_2" }
    %out_prod_0_2 = aie.lock(%tile_0_2, 4) { init = 1 : i32, sym_name = "out_prod_0_2" }
    %out_cons_0_2 = aie.lock(%tile_0_2, 5) { init = 0 : i32, sym_name = "out_cons_0_2" }
    // Core (0,3)
    %core_ping_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_ping_0_3" } : memref<576xi8>
    %core_pong_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_pong_0_3" } : memref<576xi8>
    %core_weights_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_weights_0_3" } : memref<2304xi8>
    %core_out_i8_0_3 = aie.buffer(%tile_0_3) { sym_name = "core_out_i8_0_3" } : memref<256xi8>
    %ping_prod_0_3 = aie.lock(%tile_0_3, 0) { init = 1 : i32, sym_name = "ping_prod_0_3" }
    %ping_cons_0_3 = aie.lock(%tile_0_3, 1) { init = 0 : i32, sym_name = "ping_cons_0_3" }
    %pong_prod_0_3 = aie.lock(%tile_0_3, 2) { init = 1 : i32, sym_name = "pong_prod_0_3" }
    %pong_cons_0_3 = aie.lock(%tile_0_3, 3) { init = 0 : i32, sym_name = "pong_cons_0_3" }
    %out_prod_0_3 = aie.lock(%tile_0_3, 4) { init = 1 : i32, sym_name = "out_prod_0_3" }
    %out_cons_0_3 = aie.lock(%tile_0_3, 5) { init = 0 : i32, sym_name = "out_cons_0_3" }
    // Core (0,4)
    %core_ping_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_ping_0_4" } : memref<576xi8>
    %core_pong_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_pong_0_4" } : memref<576xi8>
    %core_weights_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_weights_0_4" } : memref<2304xi8>
    %core_out_i8_0_4 = aie.buffer(%tile_0_4) { sym_name = "core_out_i8_0_4" } : memref<256xi8>
    %ping_prod_0_4 = aie.lock(%tile_0_4, 0) { init = 1 : i32, sym_name = "ping_prod_0_4" }
    %ping_cons_0_4 = aie.lock(%tile_0_4, 1) { init = 0 : i32, sym_name = "ping_cons_0_4" }
    %pong_prod_0_4 = aie.lock(%tile_0_4, 2) { init = 1 : i32, sym_name = "pong_prod_0_4" }
    %pong_cons_0_4 = aie.lock(%tile_0_4, 3) { init = 0 : i32, sym_name = "pong_cons_0_4" }
    %out_prod_0_4 = aie.lock(%tile_0_4, 4) { init = 1 : i32, sym_name = "out_prod_0_4" }
    %out_cons_0_4 = aie.lock(%tile_0_4, 5) { init = 0 : i32, sym_name = "out_cons_0_4" }
    // Core (0,5)
    %core_ping_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_ping_0_5" } : memref<576xi8>
    %core_pong_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_pong_0_5" } : memref<576xi8>
    %core_weights_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_weights_0_5" } : memref<2304xi8>
    %core_out_i8_0_5 = aie.buffer(%tile_0_5) { sym_name = "core_out_i8_0_5" } : memref<256xi8>
    %ping_prod_0_5 = aie.lock(%tile_0_5, 0) { init = 1 : i32, sym_name = "ping_prod_0_5" }
    %ping_cons_0_5 = aie.lock(%tile_0_5, 1) { init = 0 : i32, sym_name = "ping_cons_0_5" }
    %pong_prod_0_5 = aie.lock(%tile_0_5, 2) { init = 1 : i32, sym_name = "pong_prod_0_5" }
    %pong_cons_0_5 = aie.lock(%tile_0_5, 3) { init = 0 : i32, sym_name = "pong_cons_0_5" }
    %out_prod_0_5 = aie.lock(%tile_0_5, 4) { init = 1 : i32, sym_name = "out_prod_0_5" }
    %out_cons_0_5 = aie.lock(%tile_0_5, 5) { init = 0 : i32, sym_name = "out_cons_0_5" }
    // Core (1,2)
    %core_ping_1_2 = aie.buffer(%tile_1_2) { sym_name = "core_ping_1_2" } : memref<576xi8>
    %core_pong_1_2 = aie.buffer(%tile_1_2) { sym_name = "core_pong_1_2" } : memref<576xi8>
    %core_weights_1_2 = aie.buffer(%tile_1_2) { sym_name = "core_weights_1_2" } : memref<2304xi8>
    %core_out_i8_1_2 = aie.buffer(%tile_1_2) { sym_name = "core_out_i8_1_2" } : memref<256xi8>
    %ping_prod_1_2 = aie.lock(%tile_1_2, 0) { init = 1 : i32, sym_name = "ping_prod_1_2" }
    %ping_cons_1_2 = aie.lock(%tile_1_2, 1) { init = 0 : i32, sym_name = "ping_cons_1_2" }
    %pong_prod_1_2 = aie.lock(%tile_1_2, 2) { init = 1 : i32, sym_name = "pong_prod_1_2" }
    %pong_cons_1_2 = aie.lock(%tile_1_2, 3) { init = 0 : i32, sym_name = "pong_cons_1_2" }
    %out_prod_1_2 = aie.lock(%tile_1_2, 4) { init = 1 : i32, sym_name = "out_prod_1_2" }
    %out_cons_1_2 = aie.lock(%tile_1_2, 5) { init = 0 : i32, sym_name = "out_cons_1_2" }
    // Core (1,3)
    %core_ping_1_3 = aie.buffer(%tile_1_3) { sym_name = "core_ping_1_3" } : memref<576xi8>
    %core_pong_1_3 = aie.buffer(%tile_1_3) { sym_name = "core_pong_1_3" } : memref<576xi8>
    %core_weights_1_3 = aie.buffer(%tile_1_3) { sym_name = "core_weights_1_3" } : memref<2304xi8>
    %core_out_i8_1_3 = aie.buffer(%tile_1_3) { sym_name = "core_out_i8_1_3" } : memref<256xi8>
    %ping_prod_1_3 = aie.lock(%tile_1_3, 0) { init = 1 : i32, sym_name = "ping_prod_1_3" }
    %ping_cons_1_3 = aie.lock(%tile_1_3, 1) { init = 0 : i32, sym_name = "ping_cons_1_3" }
    %pong_prod_1_3 = aie.lock(%tile_1_3, 2) { init = 1 : i32, sym_name = "pong_prod_1_3" }
    %pong_cons_1_3 = aie.lock(%tile_1_3, 3) { init = 0 : i32, sym_name = "pong_cons_1_3" }
    %out_prod_1_3 = aie.lock(%tile_1_3, 4) { init = 1 : i32, sym_name = "out_prod_1_3" }
    %out_cons_1_3 = aie.lock(%tile_1_3, 5) { init = 0 : i32, sym_name = "out_cons_1_3" }
    // Core (1,4)
    %core_ping_1_4 = aie.buffer(%tile_1_4) { sym_name = "core_ping_1_4" } : memref<576xi8>
    %core_pong_1_4 = aie.buffer(%tile_1_4) { sym_name = "core_pong_1_4" } : memref<576xi8>
    %core_weights_1_4 = aie.buffer(%tile_1_4) { sym_name = "core_weights_1_4" } : memref<2304xi8>
    %core_out_i8_1_4 = aie.buffer(%tile_1_4) { sym_name = "core_out_i8_1_4" } : memref<256xi8>
    %ping_prod_1_4 = aie.lock(%tile_1_4, 0) { init = 1 : i32, sym_name = "ping_prod_1_4" }
    %ping_cons_1_4 = aie.lock(%tile_1_4, 1) { init = 0 : i32, sym_name = "ping_cons_1_4" }
    %pong_prod_1_4 = aie.lock(%tile_1_4, 2) { init = 1 : i32, sym_name = "pong_prod_1_4" }
    %pong_cons_1_4 = aie.lock(%tile_1_4, 3) { init = 0 : i32, sym_name = "pong_cons_1_4" }
    %out_prod_1_4 = aie.lock(%tile_1_4, 4) { init = 1 : i32, sym_name = "out_prod_1_4" }
    %out_cons_1_4 = aie.lock(%tile_1_4, 5) { init = 0 : i32, sym_name = "out_cons_1_4" }
    // Core (1,5)
    %core_ping_1_5 = aie.buffer(%tile_1_5) { sym_name = "core_ping_1_5" } : memref<576xi8>
    %core_pong_1_5 = aie.buffer(%tile_1_5) { sym_name = "core_pong_1_5" } : memref<576xi8>
    %core_weights_1_5 = aie.buffer(%tile_1_5) { sym_name = "core_weights_1_5" } : memref<2304xi8>
    %core_out_i8_1_5 = aie.buffer(%tile_1_5) { sym_name = "core_out_i8_1_5" } : memref<256xi8>
    %ping_prod_1_5 = aie.lock(%tile_1_5, 0) { init = 1 : i32, sym_name = "ping_prod_1_5" }
    %ping_cons_1_5 = aie.lock(%tile_1_5, 1) { init = 0 : i32, sym_name = "ping_cons_1_5" }
    %pong_prod_1_5 = aie.lock(%tile_1_5, 2) { init = 1 : i32, sym_name = "pong_prod_1_5" }
    %pong_cons_1_5 = aie.lock(%tile_1_5, 3) { init = 0 : i32, sym_name = "pong_cons_1_5" }
    %out_prod_1_5 = aie.lock(%tile_1_5, 4) { init = 1 : i32, sym_name = "out_prod_1_5" }
    %out_cons_1_5 = aie.lock(%tile_1_5, 5) { init = 0 : i32, sym_name = "out_cons_1_5" }
    // Core (2,2)
    %core_ping_2_2 = aie.buffer(%tile_2_2) { sym_name = "core_ping_2_2" } : memref<576xi8>
    %core_pong_2_2 = aie.buffer(%tile_2_2) { sym_name = "core_pong_2_2" } : memref<576xi8>
    %core_weights_2_2 = aie.buffer(%tile_2_2) { sym_name = "core_weights_2_2" } : memref<2304xi8>
    %core_out_i8_2_2 = aie.buffer(%tile_2_2) { sym_name = "core_out_i8_2_2" } : memref<256xi8>
    %ping_prod_2_2 = aie.lock(%tile_2_2, 0) { init = 1 : i32, sym_name = "ping_prod_2_2" }
    %ping_cons_2_2 = aie.lock(%tile_2_2, 1) { init = 0 : i32, sym_name = "ping_cons_2_2" }
    %pong_prod_2_2 = aie.lock(%tile_2_2, 2) { init = 1 : i32, sym_name = "pong_prod_2_2" }
    %pong_cons_2_2 = aie.lock(%tile_2_2, 3) { init = 0 : i32, sym_name = "pong_cons_2_2" }
    %out_prod_2_2 = aie.lock(%tile_2_2, 4) { init = 1 : i32, sym_name = "out_prod_2_2" }
    %out_cons_2_2 = aie.lock(%tile_2_2, 5) { init = 0 : i32, sym_name = "out_cons_2_2" }
    // Core (2,3)
    %core_ping_2_3 = aie.buffer(%tile_2_3) { sym_name = "core_ping_2_3" } : memref<576xi8>
    %core_pong_2_3 = aie.buffer(%tile_2_3) { sym_name = "core_pong_2_3" } : memref<576xi8>
    %core_weights_2_3 = aie.buffer(%tile_2_3) { sym_name = "core_weights_2_3" } : memref<2304xi8>
    %core_out_i8_2_3 = aie.buffer(%tile_2_3) { sym_name = "core_out_i8_2_3" } : memref<256xi8>
    %ping_prod_2_3 = aie.lock(%tile_2_3, 0) { init = 1 : i32, sym_name = "ping_prod_2_3" }
    %ping_cons_2_3 = aie.lock(%tile_2_3, 1) { init = 0 : i32, sym_name = "ping_cons_2_3" }
    %pong_prod_2_3 = aie.lock(%tile_2_3, 2) { init = 1 : i32, sym_name = "pong_prod_2_3" }
    %pong_cons_2_3 = aie.lock(%tile_2_3, 3) { init = 0 : i32, sym_name = "pong_cons_2_3" }
    %out_prod_2_3 = aie.lock(%tile_2_3, 4) { init = 1 : i32, sym_name = "out_prod_2_3" }
    %out_cons_2_3 = aie.lock(%tile_2_3, 5) { init = 0 : i32, sym_name = "out_cons_2_3" }
    // Core (2,4)
    %core_ping_2_4 = aie.buffer(%tile_2_4) { sym_name = "core_ping_2_4" } : memref<576xi8>
    %core_pong_2_4 = aie.buffer(%tile_2_4) { sym_name = "core_pong_2_4" } : memref<576xi8>
    %core_weights_2_4 = aie.buffer(%tile_2_4) { sym_name = "core_weights_2_4" } : memref<2304xi8>
    %core_out_i8_2_4 = aie.buffer(%tile_2_4) { sym_name = "core_out_i8_2_4" } : memref<256xi8>
    %ping_prod_2_4 = aie.lock(%tile_2_4, 0) { init = 1 : i32, sym_name = "ping_prod_2_4" }
    %ping_cons_2_4 = aie.lock(%tile_2_4, 1) { init = 0 : i32, sym_name = "ping_cons_2_4" }
    %pong_prod_2_4 = aie.lock(%tile_2_4, 2) { init = 1 : i32, sym_name = "pong_prod_2_4" }
    %pong_cons_2_4 = aie.lock(%tile_2_4, 3) { init = 0 : i32, sym_name = "pong_cons_2_4" }
    %out_prod_2_4 = aie.lock(%tile_2_4, 4) { init = 1 : i32, sym_name = "out_prod_2_4" }
    %out_cons_2_4 = aie.lock(%tile_2_4, 5) { init = 0 : i32, sym_name = "out_cons_2_4" }
    // Core (2,5)
    %core_ping_2_5 = aie.buffer(%tile_2_5) { sym_name = "core_ping_2_5" } : memref<576xi8>
    %core_pong_2_5 = aie.buffer(%tile_2_5) { sym_name = "core_pong_2_5" } : memref<576xi8>
    %core_weights_2_5 = aie.buffer(%tile_2_5) { sym_name = "core_weights_2_5" } : memref<2304xi8>
    %core_out_i8_2_5 = aie.buffer(%tile_2_5) { sym_name = "core_out_i8_2_5" } : memref<256xi8>
    %ping_prod_2_5 = aie.lock(%tile_2_5, 0) { init = 1 : i32, sym_name = "ping_prod_2_5" }
    %ping_cons_2_5 = aie.lock(%tile_2_5, 1) { init = 0 : i32, sym_name = "ping_cons_2_5" }
    %pong_prod_2_5 = aie.lock(%tile_2_5, 2) { init = 1 : i32, sym_name = "pong_prod_2_5" }
    %pong_cons_2_5 = aie.lock(%tile_2_5, 3) { init = 0 : i32, sym_name = "pong_cons_2_5" }
    %out_prod_2_5 = aie.lock(%tile_2_5, 4) { init = 1 : i32, sym_name = "out_prod_2_5" }
    %out_cons_2_5 = aie.lock(%tile_2_5, 5) { init = 0 : i32, sym_name = "out_cons_2_5" }
    // Core (3,2)
    %core_ping_3_2 = aie.buffer(%tile_3_2) { sym_name = "core_ping_3_2" } : memref<576xi8>
    %core_pong_3_2 = aie.buffer(%tile_3_2) { sym_name = "core_pong_3_2" } : memref<576xi8>
    %core_weights_3_2 = aie.buffer(%tile_3_2) { sym_name = "core_weights_3_2" } : memref<2304xi8>
    %core_out_i8_3_2 = aie.buffer(%tile_3_2) { sym_name = "core_out_i8_3_2" } : memref<256xi8>
    %ping_prod_3_2 = aie.lock(%tile_3_2, 0) { init = 1 : i32, sym_name = "ping_prod_3_2" }
    %ping_cons_3_2 = aie.lock(%tile_3_2, 1) { init = 0 : i32, sym_name = "ping_cons_3_2" }
    %pong_prod_3_2 = aie.lock(%tile_3_2, 2) { init = 1 : i32, sym_name = "pong_prod_3_2" }
    %pong_cons_3_2 = aie.lock(%tile_3_2, 3) { init = 0 : i32, sym_name = "pong_cons_3_2" }
    %out_prod_3_2 = aie.lock(%tile_3_2, 4) { init = 1 : i32, sym_name = "out_prod_3_2" }
    %out_cons_3_2 = aie.lock(%tile_3_2, 5) { init = 0 : i32, sym_name = "out_cons_3_2" }
    // Core (3,3)
    %core_ping_3_3 = aie.buffer(%tile_3_3) { sym_name = "core_ping_3_3" } : memref<576xi8>
    %core_pong_3_3 = aie.buffer(%tile_3_3) { sym_name = "core_pong_3_3" } : memref<576xi8>
    %core_weights_3_3 = aie.buffer(%tile_3_3) { sym_name = "core_weights_3_3" } : memref<2304xi8>
    %core_out_i8_3_3 = aie.buffer(%tile_3_3) { sym_name = "core_out_i8_3_3" } : memref<256xi8>
    %ping_prod_3_3 = aie.lock(%tile_3_3, 0) { init = 1 : i32, sym_name = "ping_prod_3_3" }
    %ping_cons_3_3 = aie.lock(%tile_3_3, 1) { init = 0 : i32, sym_name = "ping_cons_3_3" }
    %pong_prod_3_3 = aie.lock(%tile_3_3, 2) { init = 1 : i32, sym_name = "pong_prod_3_3" }
    %pong_cons_3_3 = aie.lock(%tile_3_3, 3) { init = 0 : i32, sym_name = "pong_cons_3_3" }
    %out_prod_3_3 = aie.lock(%tile_3_3, 4) { init = 1 : i32, sym_name = "out_prod_3_3" }
    %out_cons_3_3 = aie.lock(%tile_3_3, 5) { init = 0 : i32, sym_name = "out_cons_3_3" }
    // Core (3,4)
    %core_ping_3_4 = aie.buffer(%tile_3_4) { sym_name = "core_ping_3_4" } : memref<576xi8>
    %core_pong_3_4 = aie.buffer(%tile_3_4) { sym_name = "core_pong_3_4" } : memref<576xi8>
    %core_weights_3_4 = aie.buffer(%tile_3_4) { sym_name = "core_weights_3_4" } : memref<2304xi8>
    %core_out_i8_3_4 = aie.buffer(%tile_3_4) { sym_name = "core_out_i8_3_4" } : memref<256xi8>
    %ping_prod_3_4 = aie.lock(%tile_3_4, 0) { init = 1 : i32, sym_name = "ping_prod_3_4" }
    %ping_cons_3_4 = aie.lock(%tile_3_4, 1) { init = 0 : i32, sym_name = "ping_cons_3_4" }
    %pong_prod_3_4 = aie.lock(%tile_3_4, 2) { init = 1 : i32, sym_name = "pong_prod_3_4" }
    %pong_cons_3_4 = aie.lock(%tile_3_4, 3) { init = 0 : i32, sym_name = "pong_cons_3_4" }
    %out_prod_3_4 = aie.lock(%tile_3_4, 4) { init = 1 : i32, sym_name = "out_prod_3_4" }
    %out_cons_3_4 = aie.lock(%tile_3_4, 5) { init = 0 : i32, sym_name = "out_cons_3_4" }
    // Core (3,5)
    %core_ping_3_5 = aie.buffer(%tile_3_5) { sym_name = "core_ping_3_5" } : memref<576xi8>
    %core_pong_3_5 = aie.buffer(%tile_3_5) { sym_name = "core_pong_3_5" } : memref<576xi8>
    %core_weights_3_5 = aie.buffer(%tile_3_5) { sym_name = "core_weights_3_5" } : memref<2304xi8>
    %core_out_i8_3_5 = aie.buffer(%tile_3_5) { sym_name = "core_out_i8_3_5" } : memref<256xi8>
    %ping_prod_3_5 = aie.lock(%tile_3_5, 0) { init = 1 : i32, sym_name = "ping_prod_3_5" }
    %ping_cons_3_5 = aie.lock(%tile_3_5, 1) { init = 0 : i32, sym_name = "ping_cons_3_5" }
    %pong_prod_3_5 = aie.lock(%tile_3_5, 2) { init = 1 : i32, sym_name = "pong_prod_3_5" }
    %pong_cons_3_5 = aie.lock(%tile_3_5, 3) { init = 0 : i32, sym_name = "pong_cons_3_5" }
    %out_prod_3_5 = aie.lock(%tile_3_5, 4) { init = 1 : i32, sym_name = "out_prod_3_5" }
    %out_cons_3_5 = aie.lock(%tile_3_5, 5) { init = 0 : i32, sym_name = "out_cons_3_5" }
    // Core (4,2)
    %core_ping_4_2 = aie.buffer(%tile_4_2) { sym_name = "core_ping_4_2" } : memref<576xi8>
    %core_pong_4_2 = aie.buffer(%tile_4_2) { sym_name = "core_pong_4_2" } : memref<576xi8>
    %core_weights_4_2 = aie.buffer(%tile_4_2) { sym_name = "core_weights_4_2" } : memref<2304xi8>
    %core_out_i8_4_2 = aie.buffer(%tile_4_2) { sym_name = "core_out_i8_4_2" } : memref<256xi8>
    %ping_prod_4_2 = aie.lock(%tile_4_2, 0) { init = 1 : i32, sym_name = "ping_prod_4_2" }
    %ping_cons_4_2 = aie.lock(%tile_4_2, 1) { init = 0 : i32, sym_name = "ping_cons_4_2" }
    %pong_prod_4_2 = aie.lock(%tile_4_2, 2) { init = 1 : i32, sym_name = "pong_prod_4_2" }
    %pong_cons_4_2 = aie.lock(%tile_4_2, 3) { init = 0 : i32, sym_name = "pong_cons_4_2" }
    %out_prod_4_2 = aie.lock(%tile_4_2, 4) { init = 1 : i32, sym_name = "out_prod_4_2" }
    %out_cons_4_2 = aie.lock(%tile_4_2, 5) { init = 0 : i32, sym_name = "out_cons_4_2" }
    // Core (4,3)
    %core_ping_4_3 = aie.buffer(%tile_4_3) { sym_name = "core_ping_4_3" } : memref<576xi8>
    %core_pong_4_3 = aie.buffer(%tile_4_3) { sym_name = "core_pong_4_3" } : memref<576xi8>
    %core_weights_4_3 = aie.buffer(%tile_4_3) { sym_name = "core_weights_4_3" } : memref<2304xi8>
    %core_out_i8_4_3 = aie.buffer(%tile_4_3) { sym_name = "core_out_i8_4_3" } : memref<256xi8>
    %ping_prod_4_3 = aie.lock(%tile_4_3, 0) { init = 1 : i32, sym_name = "ping_prod_4_3" }
    %ping_cons_4_3 = aie.lock(%tile_4_3, 1) { init = 0 : i32, sym_name = "ping_cons_4_3" }
    %pong_prod_4_3 = aie.lock(%tile_4_3, 2) { init = 1 : i32, sym_name = "pong_prod_4_3" }
    %pong_cons_4_3 = aie.lock(%tile_4_3, 3) { init = 0 : i32, sym_name = "pong_cons_4_3" }
    %out_prod_4_3 = aie.lock(%tile_4_3, 4) { init = 1 : i32, sym_name = "out_prod_4_3" }
    %out_cons_4_3 = aie.lock(%tile_4_3, 5) { init = 0 : i32, sym_name = "out_cons_4_3" }
    // Core (4,4)
    %core_ping_4_4 = aie.buffer(%tile_4_4) { sym_name = "core_ping_4_4" } : memref<576xi8>
    %core_pong_4_4 = aie.buffer(%tile_4_4) { sym_name = "core_pong_4_4" } : memref<576xi8>
    %core_weights_4_4 = aie.buffer(%tile_4_4) { sym_name = "core_weights_4_4" } : memref<2304xi8>
    %core_out_i8_4_4 = aie.buffer(%tile_4_4) { sym_name = "core_out_i8_4_4" } : memref<256xi8>
    %ping_prod_4_4 = aie.lock(%tile_4_4, 0) { init = 1 : i32, sym_name = "ping_prod_4_4" }
    %ping_cons_4_4 = aie.lock(%tile_4_4, 1) { init = 0 : i32, sym_name = "ping_cons_4_4" }
    %pong_prod_4_4 = aie.lock(%tile_4_4, 2) { init = 1 : i32, sym_name = "pong_prod_4_4" }
    %pong_cons_4_4 = aie.lock(%tile_4_4, 3) { init = 0 : i32, sym_name = "pong_cons_4_4" }
    %out_prod_4_4 = aie.lock(%tile_4_4, 4) { init = 1 : i32, sym_name = "out_prod_4_4" }
    %out_cons_4_4 = aie.lock(%tile_4_4, 5) { init = 0 : i32, sym_name = "out_cons_4_4" }
    // Core (4,5)
    %core_ping_4_5 = aie.buffer(%tile_4_5) { sym_name = "core_ping_4_5" } : memref<576xi8>
    %core_pong_4_5 = aie.buffer(%tile_4_5) { sym_name = "core_pong_4_5" } : memref<576xi8>
    %core_weights_4_5 = aie.buffer(%tile_4_5) { sym_name = "core_weights_4_5" } : memref<2304xi8>
    %core_out_i8_4_5 = aie.buffer(%tile_4_5) { sym_name = "core_out_i8_4_5" } : memref<256xi8>
    %ping_prod_4_5 = aie.lock(%tile_4_5, 0) { init = 1 : i32, sym_name = "ping_prod_4_5" }
    %ping_cons_4_5 = aie.lock(%tile_4_5, 1) { init = 0 : i32, sym_name = "ping_cons_4_5" }
    %pong_prod_4_5 = aie.lock(%tile_4_5, 2) { init = 1 : i32, sym_name = "pong_prod_4_5" }
    %pong_cons_4_5 = aie.lock(%tile_4_5, 3) { init = 0 : i32, sym_name = "pong_cons_4_5" }
    %out_prod_4_5 = aie.lock(%tile_4_5, 4) { init = 1 : i32, sym_name = "out_prod_4_5" }
    %out_cons_4_5 = aie.lock(%tile_4_5, 5) { init = 0 : i32, sym_name = "out_cons_4_5" }

    // ------------------------------------------------------------------------
    // Foreign Function Interface (FFI) Declaration: SRS Vector Kernel
    // ------------------------------------------------------------------------
    func.func private @conv_im2col_ping_pong_m2_srs(memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> () attributes {link_with = "conv_im2col_kernel_m2.o"}

    // ------------------------------------------------------------------------
    // Stream Routing Network (50 flows across 5 Columns)
    // ------------------------------------------------------------------------
    // Column 0 Flows
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 3)
    aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 4)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    // Column 1 Flows
    aie.flow(%tile_1_0, DMA : 0, %tile_1_1, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_2, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_3, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_4, DMA : 0)
    aie.flow(%tile_1_1, DMA : 0, %tile_1_5, DMA : 0)
    aie.flow(%tile_1_2, DMA : 0, %tile_1_1, DMA : 1)
    aie.flow(%tile_1_3, DMA : 0, %tile_1_1, DMA : 2)
    aie.flow(%tile_1_4, DMA : 0, %tile_1_1, DMA : 3)
    aie.flow(%tile_1_5, DMA : 0, %tile_1_1, DMA : 4)
    aie.flow(%tile_1_1, DMA : 1, %tile_1_0, DMA : 0)
    // Column 2 Flows
    aie.flow(%tile_2_0, DMA : 0, %tile_2_1, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_3, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_4, DMA : 0)
    aie.flow(%tile_2_1, DMA : 0, %tile_2_5, DMA : 0)
    aie.flow(%tile_2_2, DMA : 0, %tile_2_1, DMA : 1)
    aie.flow(%tile_2_3, DMA : 0, %tile_2_1, DMA : 2)
    aie.flow(%tile_2_4, DMA : 0, %tile_2_1, DMA : 3)
    aie.flow(%tile_2_5, DMA : 0, %tile_2_1, DMA : 4)
    aie.flow(%tile_2_1, DMA : 1, %tile_2_0, DMA : 0)
    // Column 3 Flows
    aie.flow(%tile_3_0, DMA : 0, %tile_3_1, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_3_2, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_3_3, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_3_4, DMA : 0)
    aie.flow(%tile_3_1, DMA : 0, %tile_3_5, DMA : 0)
    aie.flow(%tile_3_2, DMA : 0, %tile_3_1, DMA : 1)
    aie.flow(%tile_3_3, DMA : 0, %tile_3_1, DMA : 2)
    aie.flow(%tile_3_4, DMA : 0, %tile_3_1, DMA : 3)
    aie.flow(%tile_3_5, DMA : 0, %tile_3_1, DMA : 4)
    aie.flow(%tile_3_1, DMA : 1, %tile_3_0, DMA : 0)
    // Column 4 Flows
    aie.flow(%tile_4_0, DMA : 0, %tile_4_1, DMA : 0)
    aie.flow(%tile_4_1, DMA : 0, %tile_4_2, DMA : 0)
    aie.flow(%tile_4_1, DMA : 0, %tile_4_3, DMA : 0)
    aie.flow(%tile_4_1, DMA : 0, %tile_4_4, DMA : 0)
    aie.flow(%tile_4_1, DMA : 0, %tile_4_5, DMA : 0)
    aie.flow(%tile_4_2, DMA : 0, %tile_4_1, DMA : 1)
    aie.flow(%tile_4_3, DMA : 0, %tile_4_1, DMA : 2)
    aie.flow(%tile_4_4, DMA : 0, %tile_4_1, DMA : 3)
    aie.flow(%tile_4_5, DMA : 0, %tile_4_1, DMA : 4)
    aie.flow(%tile_4_1, DMA : 1, %tile_4_0, DMA : 0)

    // ------------------------------------------------------------------------
    // Shim NoC DMAs (5 Columns)
    // ------------------------------------------------------------------------
    %shim_dma_0 = aie.shim_dma(%tile_0_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_in_0, ^ch_s2mm_0)
    ^bd_in_0:
      aie.use_lock(%shim_lock_0_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_0 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0_0, Release, %c1)
      aie.next_bd ^bd_in_0
    ^ch_s2mm_0:
      aie.dma_start(S2MM, 0, ^bd_out_0, ^end_0)
    ^bd_out_0:
      aie.use_lock(%shim_lock_1_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_out_buf_0 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim_lock_1_0, Release, %c1)
      aie.next_bd ^bd_out_0
    ^end_0:
      aie.end
    }
    %shim_dma_1 = aie.shim_dma(%tile_1_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_in_1, ^ch_s2mm_1)
    ^bd_in_1:
      aie.use_lock(%shim_lock_0_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_1 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0_1, Release, %c1)
      aie.next_bd ^bd_in_1
    ^ch_s2mm_1:
      aie.dma_start(S2MM, 0, ^bd_out_1, ^end_1)
    ^bd_out_1:
      aie.use_lock(%shim_lock_1_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_out_buf_1 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim_lock_1_1, Release, %c1)
      aie.next_bd ^bd_out_1
    ^end_1:
      aie.end
    }
    %shim_dma_2 = aie.shim_dma(%tile_2_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_in_2, ^ch_s2mm_2)
    ^bd_in_2:
      aie.use_lock(%shim_lock_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_2 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0_2, Release, %c1)
      aie.next_bd ^bd_in_2
    ^ch_s2mm_2:
      aie.dma_start(S2MM, 0, ^bd_out_2, ^end_2)
    ^bd_out_2:
      aie.use_lock(%shim_lock_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_out_buf_2 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim_lock_1_2, Release, %c1)
      aie.next_bd ^bd_out_2
    ^end_2:
      aie.end
    }
    %shim_dma_3 = aie.shim_dma(%tile_3_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_in_3, ^ch_s2mm_3)
    ^bd_in_3:
      aie.use_lock(%shim_lock_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_3 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0_3, Release, %c1)
      aie.next_bd ^bd_in_3
    ^ch_s2mm_3:
      aie.dma_start(S2MM, 0, ^bd_out_3, ^end_3)
    ^bd_out_3:
      aie.use_lock(%shim_lock_1_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_out_buf_3 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim_lock_1_3, Release, %c1)
      aie.next_bd ^bd_out_3
    ^end_3:
      aie.end
    }
    %shim_dma_4 = aie.shim_dma(%tile_4_0) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(MM2S, 0, ^bd_in_4, ^ch_s2mm_4)
    ^bd_in_4:
      aie.use_lock(%shim_lock_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_buf_4 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%shim_lock_0_4, Release, %c1)
      aie.next_bd ^bd_in_4
    ^ch_s2mm_4:
      aie.dma_start(S2MM, 0, ^bd_out_4, ^end_4)
    ^bd_out_4:
      aie.use_lock(%shim_lock_1_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%ext_out_buf_4 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%shim_lock_1_4, Release, %c1)
      aie.next_bd ^bd_out_4
    ^end_4:
      aie.end
    }

    // ------------------------------------------------------------------------
    // MemTile DMAs (5 Columns: 4-D im2col Multicast Engine & Gathering)
    // ------------------------------------------------------------------------
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^s2mm_in_bd_0, ^ch_s2mm_1_0)
    ^s2mm_in_bd_0:
      aie.use_lock(%mem_lock_prod_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_0 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons_0, Release, %c1)
      aie.next_bd ^s2mm_in_bd_0
    ^ch_s2mm_1_0:
      aie.dma_start(S2MM, 1, ^bd_g0_0, ^ch_s2mm_2_0)
    ^bd_g0_0:
      aie.use_lock(%mem_out_prod_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_0 : memref<1024xi8> offset = 0 len = 256)
      aie.use_lock(%mem_out_cons_0, Release, %c1)
      aie.next_bd ^bd_g0_0
    ^ch_s2mm_2_0:
      aie.dma_start(S2MM, 2, ^bd_g1_0, ^ch_s2mm_3_0)
    ^bd_g1_0:
      aie.use_lock(%mem_out_prod_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_0 : memref<1024xi8> offset = 256 len = 256)
      aie.use_lock(%mem_out_cons_0, Release, %c1)
      aie.next_bd ^bd_g1_0
    ^ch_s2mm_3_0:
      aie.dma_start(S2MM, 3, ^bd_g2_0, ^ch_s2mm_4_0)
    ^bd_g2_0:
      aie.use_lock(%mem_out_prod_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_0 : memref<1024xi8> offset = 512 len = 256)
      aie.use_lock(%mem_out_cons_0, Release, %c1)
      aie.next_bd ^bd_g2_0
    ^ch_s2mm_4_0:
      aie.dma_start(S2MM, 4, ^bd_g3_0, ^ch_mm2s_0_0)
    ^bd_g3_0:
      aie.use_lock(%mem_out_prod_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_0 : memref<1024xi8> offset = 768 len = 256)
      aie.use_lock(%mem_out_cons_0, Release, %c1)
      aie.next_bd ^bd_g3_0
    ^ch_mm2s_0_0:
      aie.dma_start(MM2S, 0, ^mm2s_in_bd_0, ^ch_mm2s_1_0)
    ^mm2s_in_bd_0:
      aie.use_lock(%mem_lock_cons_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_0 : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod_0, Release, %c1)
      aie.next_bd ^mm2s_in_bd_0
    ^ch_mm2s_1_0:
      aie.dma_start(MM2S, 1, ^egress_bd_0, ^end_mem_0)
    ^egress_bd_0:
      aie.use_lock(%mem_out_cons_0, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_0 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem_out_prod_0, Release, %c1)
      aie.next_bd ^egress_bd_0
    ^end_mem_0:
      aie.end
    }
    %memtile_dma_1_1 = aie.memtile_dma(%tile_1_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^s2mm_in_bd_1, ^ch_s2mm_1_1)
    ^s2mm_in_bd_1:
      aie.use_lock(%mem_lock_prod_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_1 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons_1, Release, %c1)
      aie.next_bd ^s2mm_in_bd_1
    ^ch_s2mm_1_1:
      aie.dma_start(S2MM, 1, ^bd_g0_1, ^ch_s2mm_2_1)
    ^bd_g0_1:
      aie.use_lock(%mem_out_prod_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_1 : memref<1024xi8> offset = 0 len = 256)
      aie.use_lock(%mem_out_cons_1, Release, %c1)
      aie.next_bd ^bd_g0_1
    ^ch_s2mm_2_1:
      aie.dma_start(S2MM, 2, ^bd_g1_1, ^ch_s2mm_3_1)
    ^bd_g1_1:
      aie.use_lock(%mem_out_prod_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_1 : memref<1024xi8> offset = 256 len = 256)
      aie.use_lock(%mem_out_cons_1, Release, %c1)
      aie.next_bd ^bd_g1_1
    ^ch_s2mm_3_1:
      aie.dma_start(S2MM, 3, ^bd_g2_1, ^ch_s2mm_4_1)
    ^bd_g2_1:
      aie.use_lock(%mem_out_prod_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_1 : memref<1024xi8> offset = 512 len = 256)
      aie.use_lock(%mem_out_cons_1, Release, %c1)
      aie.next_bd ^bd_g2_1
    ^ch_s2mm_4_1:
      aie.dma_start(S2MM, 4, ^bd_g3_1, ^ch_mm2s_0_1)
    ^bd_g3_1:
      aie.use_lock(%mem_out_prod_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_1 : memref<1024xi8> offset = 768 len = 256)
      aie.use_lock(%mem_out_cons_1, Release, %c1)
      aie.next_bd ^bd_g3_1
    ^ch_mm2s_0_1:
      aie.dma_start(MM2S, 0, ^mm2s_in_bd_1, ^ch_mm2s_1_1)
    ^mm2s_in_bd_1:
      aie.use_lock(%mem_lock_cons_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_1 : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod_1, Release, %c1)
      aie.next_bd ^mm2s_in_bd_1
    ^ch_mm2s_1_1:
      aie.dma_start(MM2S, 1, ^egress_bd_1, ^end_mem_1)
    ^egress_bd_1:
      aie.use_lock(%mem_out_cons_1, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_1 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem_out_prod_1, Release, %c1)
      aie.next_bd ^egress_bd_1
    ^end_mem_1:
      aie.end
    }
    %memtile_dma_2_1 = aie.memtile_dma(%tile_2_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^s2mm_in_bd_2, ^ch_s2mm_1_2)
    ^s2mm_in_bd_2:
      aie.use_lock(%mem_lock_prod_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_2 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons_2, Release, %c1)
      aie.next_bd ^s2mm_in_bd_2
    ^ch_s2mm_1_2:
      aie.dma_start(S2MM, 1, ^bd_g0_2, ^ch_s2mm_2_2)
    ^bd_g0_2:
      aie.use_lock(%mem_out_prod_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_2 : memref<1024xi8> offset = 0 len = 256)
      aie.use_lock(%mem_out_cons_2, Release, %c1)
      aie.next_bd ^bd_g0_2
    ^ch_s2mm_2_2:
      aie.dma_start(S2MM, 2, ^bd_g1_2, ^ch_s2mm_3_2)
    ^bd_g1_2:
      aie.use_lock(%mem_out_prod_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_2 : memref<1024xi8> offset = 256 len = 256)
      aie.use_lock(%mem_out_cons_2, Release, %c1)
      aie.next_bd ^bd_g1_2
    ^ch_s2mm_3_2:
      aie.dma_start(S2MM, 3, ^bd_g2_2, ^ch_s2mm_4_2)
    ^bd_g2_2:
      aie.use_lock(%mem_out_prod_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_2 : memref<1024xi8> offset = 512 len = 256)
      aie.use_lock(%mem_out_cons_2, Release, %c1)
      aie.next_bd ^bd_g2_2
    ^ch_s2mm_4_2:
      aie.dma_start(S2MM, 4, ^bd_g3_2, ^ch_mm2s_0_2)
    ^bd_g3_2:
      aie.use_lock(%mem_out_prod_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_2 : memref<1024xi8> offset = 768 len = 256)
      aie.use_lock(%mem_out_cons_2, Release, %c1)
      aie.next_bd ^bd_g3_2
    ^ch_mm2s_0_2:
      aie.dma_start(MM2S, 0, ^mm2s_in_bd_2, ^ch_mm2s_1_2)
    ^mm2s_in_bd_2:
      aie.use_lock(%mem_lock_cons_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_2 : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod_2, Release, %c1)
      aie.next_bd ^mm2s_in_bd_2
    ^ch_mm2s_1_2:
      aie.dma_start(MM2S, 1, ^egress_bd_2, ^end_mem_2)
    ^egress_bd_2:
      aie.use_lock(%mem_out_cons_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_2 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem_out_prod_2, Release, %c1)
      aie.next_bd ^egress_bd_2
    ^end_mem_2:
      aie.end
    }
    %memtile_dma_3_1 = aie.memtile_dma(%tile_3_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^s2mm_in_bd_3, ^ch_s2mm_1_3)
    ^s2mm_in_bd_3:
      aie.use_lock(%mem_lock_prod_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_3 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons_3, Release, %c1)
      aie.next_bd ^s2mm_in_bd_3
    ^ch_s2mm_1_3:
      aie.dma_start(S2MM, 1, ^bd_g0_3, ^ch_s2mm_2_3)
    ^bd_g0_3:
      aie.use_lock(%mem_out_prod_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_3 : memref<1024xi8> offset = 0 len = 256)
      aie.use_lock(%mem_out_cons_3, Release, %c1)
      aie.next_bd ^bd_g0_3
    ^ch_s2mm_2_3:
      aie.dma_start(S2MM, 2, ^bd_g1_3, ^ch_s2mm_3_3)
    ^bd_g1_3:
      aie.use_lock(%mem_out_prod_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_3 : memref<1024xi8> offset = 256 len = 256)
      aie.use_lock(%mem_out_cons_3, Release, %c1)
      aie.next_bd ^bd_g1_3
    ^ch_s2mm_3_3:
      aie.dma_start(S2MM, 3, ^bd_g2_3, ^ch_s2mm_4_3)
    ^bd_g2_3:
      aie.use_lock(%mem_out_prod_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_3 : memref<1024xi8> offset = 512 len = 256)
      aie.use_lock(%mem_out_cons_3, Release, %c1)
      aie.next_bd ^bd_g2_3
    ^ch_s2mm_4_3:
      aie.dma_start(S2MM, 4, ^bd_g3_3, ^ch_mm2s_0_3)
    ^bd_g3_3:
      aie.use_lock(%mem_out_prod_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_3 : memref<1024xi8> offset = 768 len = 256)
      aie.use_lock(%mem_out_cons_3, Release, %c1)
      aie.next_bd ^bd_g3_3
    ^ch_mm2s_0_3:
      aie.dma_start(MM2S, 0, ^mm2s_in_bd_3, ^ch_mm2s_1_3)
    ^mm2s_in_bd_3:
      aie.use_lock(%mem_lock_cons_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_3 : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod_3, Release, %c1)
      aie.next_bd ^mm2s_in_bd_3
    ^ch_mm2s_1_3:
      aie.dma_start(MM2S, 1, ^egress_bd_3, ^end_mem_3)
    ^egress_bd_3:
      aie.use_lock(%mem_out_cons_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_3 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem_out_prod_3, Release, %c1)
      aie.next_bd ^egress_bd_3
    ^end_mem_3:
      aie.end
    }
    %memtile_dma_4_1 = aie.memtile_dma(%tile_4_1) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^s2mm_in_bd_4, ^ch_s2mm_1_4)
    ^s2mm_in_bd_4:
      aie.use_lock(%mem_lock_prod_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_4 : memref<2048xi8> offset = 0 len = 2048)
      aie.use_lock(%mem_lock_cons_4, Release, %c1)
      aie.next_bd ^s2mm_in_bd_4
    ^ch_s2mm_1_4:
      aie.dma_start(S2MM, 1, ^bd_g0_4, ^ch_s2mm_2_4)
    ^bd_g0_4:
      aie.use_lock(%mem_out_prod_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_4 : memref<1024xi8> offset = 0 len = 256)
      aie.use_lock(%mem_out_cons_4, Release, %c1)
      aie.next_bd ^bd_g0_4
    ^ch_s2mm_2_4:
      aie.dma_start(S2MM, 2, ^bd_g1_4, ^ch_s2mm_3_4)
    ^bd_g1_4:
      aie.use_lock(%mem_out_prod_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_4 : memref<1024xi8> offset = 256 len = 256)
      aie.use_lock(%mem_out_cons_4, Release, %c1)
      aie.next_bd ^bd_g1_4
    ^ch_s2mm_3_4:
      aie.dma_start(S2MM, 3, ^bd_g2_4, ^ch_s2mm_4_4)
    ^bd_g2_4:
      aie.use_lock(%mem_out_prod_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_4 : memref<1024xi8> offset = 512 len = 256)
      aie.use_lock(%mem_out_cons_4, Release, %c1)
      aie.next_bd ^bd_g2_4
    ^ch_s2mm_4_4:
      aie.dma_start(S2MM, 4, ^bd_g3_4, ^ch_mm2s_0_4)
    ^bd_g3_4:
      aie.use_lock(%mem_out_prod_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_4 : memref<1024xi8> offset = 768 len = 256)
      aie.use_lock(%mem_out_cons_4, Release, %c1)
      aie.next_bd ^bd_g3_4
    ^ch_mm2s_0_4:
      aie.dma_start(MM2S, 0, ^mm2s_in_bd_4, ^ch_mm2s_1_4)
    ^mm2s_in_bd_4:
      aie.use_lock(%mem_lock_cons_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_in_4 : memref<2048xi8> offset = 0 len = 1728
                 sizes = [6, 3, 3, 32]
                 strides = [32, 256, 32, 1])
      aie.use_lock(%mem_lock_prod_4, Release, %c1)
      aie.next_bd ^mm2s_in_bd_4
    ^ch_mm2s_1_4:
      aie.dma_start(MM2S, 1, ^egress_bd_4, ^end_mem_4)
    ^egress_bd_4:
      aie.use_lock(%mem_out_cons_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%mem_out_4 : memref<1024xi8> offset = 0 len = 1024)
      aie.use_lock(%mem_out_prod_4, Release, %c1)
      aie.next_bd ^egress_bd_4
    ^end_mem_4:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Compute Tiles (20 Cores: DMAs & Vector Execution Units)
    // ------------------------------------------------------------------------
    // Tile (0,2) DMA
    %mem_0_2 = aie.mem(%tile_0_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_0_2, ^egress_chan_0_2)
    ^bd_ping_0_2:
      aie.use_lock(%ping_prod_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_2, Release, %c1)
      aie.next_bd ^bd_pong_0_2
    ^bd_pong_0_2:
      aie.use_lock(%pong_prod_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_2, Release, %c1)
      aie.next_bd ^bd_ping_0_2
    ^egress_chan_0_2:
      aie.dma_start(MM2S, 0, ^bd_egress_0_2, ^end_core_dma_0_2)
    ^bd_egress_0_2:
      aie.use_lock(%out_cons_0_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_0_2 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_0_2, Release, %c1)
      aie.next_bd ^bd_egress_0_2
    ^end_core_dma_0_2:
      aie.end
    }
    // Tile (0,2) Core Vector Execution
    %core_0_2 = aie.core(%tile_0_2) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_0_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_0_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_0_2, %core_pong_0_2, %core_weights_0_2, %core_out_i8_0_2, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_0_2, Release, %c1)
      aie.use_lock(%pong_prod_0_2, Release, %c1)
      aie.use_lock(%out_cons_0_2, Release, %c1)
      aie.end
    }
    // Tile (0,3) DMA
    %mem_0_3 = aie.mem(%tile_0_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_0_3, ^egress_chan_0_3)
    ^bd_ping_0_3:
      aie.use_lock(%ping_prod_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_3, Release, %c1)
      aie.next_bd ^bd_pong_0_3
    ^bd_pong_0_3:
      aie.use_lock(%pong_prod_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_3, Release, %c1)
      aie.next_bd ^bd_ping_0_3
    ^egress_chan_0_3:
      aie.dma_start(MM2S, 0, ^bd_egress_0_3, ^end_core_dma_0_3)
    ^bd_egress_0_3:
      aie.use_lock(%out_cons_0_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_0_3 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_0_3, Release, %c1)
      aie.next_bd ^bd_egress_0_3
    ^end_core_dma_0_3:
      aie.end
    }
    // Tile (0,3) Core Vector Execution
    %core_0_3 = aie.core(%tile_0_3) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_0_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_0_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_0_3, %core_pong_0_3, %core_weights_0_3, %core_out_i8_0_3, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_0_3, Release, %c1)
      aie.use_lock(%pong_prod_0_3, Release, %c1)
      aie.use_lock(%out_cons_0_3, Release, %c1)
      aie.end
    }
    // Tile (0,4) DMA
    %mem_0_4 = aie.mem(%tile_0_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_0_4, ^egress_chan_0_4)
    ^bd_ping_0_4:
      aie.use_lock(%ping_prod_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_4, Release, %c1)
      aie.next_bd ^bd_pong_0_4
    ^bd_pong_0_4:
      aie.use_lock(%pong_prod_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_4, Release, %c1)
      aie.next_bd ^bd_ping_0_4
    ^egress_chan_0_4:
      aie.dma_start(MM2S, 0, ^bd_egress_0_4, ^end_core_dma_0_4)
    ^bd_egress_0_4:
      aie.use_lock(%out_cons_0_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_0_4 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_0_4, Release, %c1)
      aie.next_bd ^bd_egress_0_4
    ^end_core_dma_0_4:
      aie.end
    }
    // Tile (0,4) Core Vector Execution
    %core_0_4 = aie.core(%tile_0_4) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_0_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_0_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_0_4, %core_pong_0_4, %core_weights_0_4, %core_out_i8_0_4, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_0_4, Release, %c1)
      aie.use_lock(%pong_prod_0_4, Release, %c1)
      aie.use_lock(%out_cons_0_4, Release, %c1)
      aie.end
    }
    // Tile (0,5) DMA
    %mem_0_5 = aie.mem(%tile_0_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_0_5, ^egress_chan_0_5)
    ^bd_ping_0_5:
      aie.use_lock(%ping_prod_0_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_0_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_0_5, Release, %c1)
      aie.next_bd ^bd_pong_0_5
    ^bd_pong_0_5:
      aie.use_lock(%pong_prod_0_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_0_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_0_5, Release, %c1)
      aie.next_bd ^bd_ping_0_5
    ^egress_chan_0_5:
      aie.dma_start(MM2S, 0, ^bd_egress_0_5, ^end_core_dma_0_5)
    ^bd_egress_0_5:
      aie.use_lock(%out_cons_0_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_0_5 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_0_5, Release, %c1)
      aie.next_bd ^bd_egress_0_5
    ^end_core_dma_0_5:
      aie.end
    }
    // Tile (0,5) Core Vector Execution
    %core_0_5 = aie.core(%tile_0_5) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_0_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_0_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_0_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_0_5, %core_pong_0_5, %core_weights_0_5, %core_out_i8_0_5, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_0_5, Release, %c1)
      aie.use_lock(%pong_prod_0_5, Release, %c1)
      aie.use_lock(%out_cons_0_5, Release, %c1)
      aie.end
    }
    // Tile (1,2) DMA
    %mem_1_2 = aie.mem(%tile_1_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_1_2, ^egress_chan_1_2)
    ^bd_ping_1_2:
      aie.use_lock(%ping_prod_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_1_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_1_2, Release, %c1)
      aie.next_bd ^bd_pong_1_2
    ^bd_pong_1_2:
      aie.use_lock(%pong_prod_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_1_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_1_2, Release, %c1)
      aie.next_bd ^bd_ping_1_2
    ^egress_chan_1_2:
      aie.dma_start(MM2S, 0, ^bd_egress_1_2, ^end_core_dma_1_2)
    ^bd_egress_1_2:
      aie.use_lock(%out_cons_1_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_1_2 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_1_2, Release, %c1)
      aie.next_bd ^bd_egress_1_2
    ^end_core_dma_1_2:
      aie.end
    }
    // Tile (1,2) Core Vector Execution
    %core_1_2 = aie.core(%tile_1_2) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_1_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_1_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_1_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_1_2, %core_pong_1_2, %core_weights_1_2, %core_out_i8_1_2, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_1_2, Release, %c1)
      aie.use_lock(%pong_prod_1_2, Release, %c1)
      aie.use_lock(%out_cons_1_2, Release, %c1)
      aie.end
    }
    // Tile (1,3) DMA
    %mem_1_3 = aie.mem(%tile_1_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_1_3, ^egress_chan_1_3)
    ^bd_ping_1_3:
      aie.use_lock(%ping_prod_1_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_1_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_1_3, Release, %c1)
      aie.next_bd ^bd_pong_1_3
    ^bd_pong_1_3:
      aie.use_lock(%pong_prod_1_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_1_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_1_3, Release, %c1)
      aie.next_bd ^bd_ping_1_3
    ^egress_chan_1_3:
      aie.dma_start(MM2S, 0, ^bd_egress_1_3, ^end_core_dma_1_3)
    ^bd_egress_1_3:
      aie.use_lock(%out_cons_1_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_1_3 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_1_3, Release, %c1)
      aie.next_bd ^bd_egress_1_3
    ^end_core_dma_1_3:
      aie.end
    }
    // Tile (1,3) Core Vector Execution
    %core_1_3 = aie.core(%tile_1_3) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_1_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_1_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_1_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_1_3, %core_pong_1_3, %core_weights_1_3, %core_out_i8_1_3, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_1_3, Release, %c1)
      aie.use_lock(%pong_prod_1_3, Release, %c1)
      aie.use_lock(%out_cons_1_3, Release, %c1)
      aie.end
    }
    // Tile (1,4) DMA
    %mem_1_4 = aie.mem(%tile_1_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_1_4, ^egress_chan_1_4)
    ^bd_ping_1_4:
      aie.use_lock(%ping_prod_1_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_1_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_1_4, Release, %c1)
      aie.next_bd ^bd_pong_1_4
    ^bd_pong_1_4:
      aie.use_lock(%pong_prod_1_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_1_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_1_4, Release, %c1)
      aie.next_bd ^bd_ping_1_4
    ^egress_chan_1_4:
      aie.dma_start(MM2S, 0, ^bd_egress_1_4, ^end_core_dma_1_4)
    ^bd_egress_1_4:
      aie.use_lock(%out_cons_1_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_1_4 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_1_4, Release, %c1)
      aie.next_bd ^bd_egress_1_4
    ^end_core_dma_1_4:
      aie.end
    }
    // Tile (1,4) Core Vector Execution
    %core_1_4 = aie.core(%tile_1_4) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_1_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_1_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_1_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_1_4, %core_pong_1_4, %core_weights_1_4, %core_out_i8_1_4, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_1_4, Release, %c1)
      aie.use_lock(%pong_prod_1_4, Release, %c1)
      aie.use_lock(%out_cons_1_4, Release, %c1)
      aie.end
    }
    // Tile (1,5) DMA
    %mem_1_5 = aie.mem(%tile_1_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_1_5, ^egress_chan_1_5)
    ^bd_ping_1_5:
      aie.use_lock(%ping_prod_1_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_1_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_1_5, Release, %c1)
      aie.next_bd ^bd_pong_1_5
    ^bd_pong_1_5:
      aie.use_lock(%pong_prod_1_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_1_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_1_5, Release, %c1)
      aie.next_bd ^bd_ping_1_5
    ^egress_chan_1_5:
      aie.dma_start(MM2S, 0, ^bd_egress_1_5, ^end_core_dma_1_5)
    ^bd_egress_1_5:
      aie.use_lock(%out_cons_1_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_1_5 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_1_5, Release, %c1)
      aie.next_bd ^bd_egress_1_5
    ^end_core_dma_1_5:
      aie.end
    }
    // Tile (1,5) Core Vector Execution
    %core_1_5 = aie.core(%tile_1_5) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_1_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_1_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_1_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_1_5, %core_pong_1_5, %core_weights_1_5, %core_out_i8_1_5, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_1_5, Release, %c1)
      aie.use_lock(%pong_prod_1_5, Release, %c1)
      aie.use_lock(%out_cons_1_5, Release, %c1)
      aie.end
    }
    // Tile (2,2) DMA
    %mem_2_2 = aie.mem(%tile_2_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_2_2, ^egress_chan_2_2)
    ^bd_ping_2_2:
      aie.use_lock(%ping_prod_2_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_2_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_2_2, Release, %c1)
      aie.next_bd ^bd_pong_2_2
    ^bd_pong_2_2:
      aie.use_lock(%pong_prod_2_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_2_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_2_2, Release, %c1)
      aie.next_bd ^bd_ping_2_2
    ^egress_chan_2_2:
      aie.dma_start(MM2S, 0, ^bd_egress_2_2, ^end_core_dma_2_2)
    ^bd_egress_2_2:
      aie.use_lock(%out_cons_2_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_2_2 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_2_2, Release, %c1)
      aie.next_bd ^bd_egress_2_2
    ^end_core_dma_2_2:
      aie.end
    }
    // Tile (2,2) Core Vector Execution
    %core_2_2 = aie.core(%tile_2_2) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_2_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_2_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_2_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_2_2, %core_pong_2_2, %core_weights_2_2, %core_out_i8_2_2, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_2_2, Release, %c1)
      aie.use_lock(%pong_prod_2_2, Release, %c1)
      aie.use_lock(%out_cons_2_2, Release, %c1)
      aie.end
    }
    // Tile (2,3) DMA
    %mem_2_3 = aie.mem(%tile_2_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_2_3, ^egress_chan_2_3)
    ^bd_ping_2_3:
      aie.use_lock(%ping_prod_2_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_2_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_2_3, Release, %c1)
      aie.next_bd ^bd_pong_2_3
    ^bd_pong_2_3:
      aie.use_lock(%pong_prod_2_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_2_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_2_3, Release, %c1)
      aie.next_bd ^bd_ping_2_3
    ^egress_chan_2_3:
      aie.dma_start(MM2S, 0, ^bd_egress_2_3, ^end_core_dma_2_3)
    ^bd_egress_2_3:
      aie.use_lock(%out_cons_2_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_2_3 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_2_3, Release, %c1)
      aie.next_bd ^bd_egress_2_3
    ^end_core_dma_2_3:
      aie.end
    }
    // Tile (2,3) Core Vector Execution
    %core_2_3 = aie.core(%tile_2_3) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_2_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_2_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_2_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_2_3, %core_pong_2_3, %core_weights_2_3, %core_out_i8_2_3, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_2_3, Release, %c1)
      aie.use_lock(%pong_prod_2_3, Release, %c1)
      aie.use_lock(%out_cons_2_3, Release, %c1)
      aie.end
    }
    // Tile (2,4) DMA
    %mem_2_4 = aie.mem(%tile_2_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_2_4, ^egress_chan_2_4)
    ^bd_ping_2_4:
      aie.use_lock(%ping_prod_2_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_2_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_2_4, Release, %c1)
      aie.next_bd ^bd_pong_2_4
    ^bd_pong_2_4:
      aie.use_lock(%pong_prod_2_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_2_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_2_4, Release, %c1)
      aie.next_bd ^bd_ping_2_4
    ^egress_chan_2_4:
      aie.dma_start(MM2S, 0, ^bd_egress_2_4, ^end_core_dma_2_4)
    ^bd_egress_2_4:
      aie.use_lock(%out_cons_2_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_2_4 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_2_4, Release, %c1)
      aie.next_bd ^bd_egress_2_4
    ^end_core_dma_2_4:
      aie.end
    }
    // Tile (2,4) Core Vector Execution
    %core_2_4 = aie.core(%tile_2_4) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_2_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_2_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_2_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_2_4, %core_pong_2_4, %core_weights_2_4, %core_out_i8_2_4, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_2_4, Release, %c1)
      aie.use_lock(%pong_prod_2_4, Release, %c1)
      aie.use_lock(%out_cons_2_4, Release, %c1)
      aie.end
    }
    // Tile (2,5) DMA
    %mem_2_5 = aie.mem(%tile_2_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_2_5, ^egress_chan_2_5)
    ^bd_ping_2_5:
      aie.use_lock(%ping_prod_2_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_2_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_2_5, Release, %c1)
      aie.next_bd ^bd_pong_2_5
    ^bd_pong_2_5:
      aie.use_lock(%pong_prod_2_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_2_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_2_5, Release, %c1)
      aie.next_bd ^bd_ping_2_5
    ^egress_chan_2_5:
      aie.dma_start(MM2S, 0, ^bd_egress_2_5, ^end_core_dma_2_5)
    ^bd_egress_2_5:
      aie.use_lock(%out_cons_2_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_2_5 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_2_5, Release, %c1)
      aie.next_bd ^bd_egress_2_5
    ^end_core_dma_2_5:
      aie.end
    }
    // Tile (2,5) Core Vector Execution
    %core_2_5 = aie.core(%tile_2_5) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_2_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_2_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_2_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_2_5, %core_pong_2_5, %core_weights_2_5, %core_out_i8_2_5, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_2_5, Release, %c1)
      aie.use_lock(%pong_prod_2_5, Release, %c1)
      aie.use_lock(%out_cons_2_5, Release, %c1)
      aie.end
    }
    // Tile (3,2) DMA
    %mem_3_2 = aie.mem(%tile_3_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_3_2, ^egress_chan_3_2)
    ^bd_ping_3_2:
      aie.use_lock(%ping_prod_3_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_3_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_3_2, Release, %c1)
      aie.next_bd ^bd_pong_3_2
    ^bd_pong_3_2:
      aie.use_lock(%pong_prod_3_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_3_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_3_2, Release, %c1)
      aie.next_bd ^bd_ping_3_2
    ^egress_chan_3_2:
      aie.dma_start(MM2S, 0, ^bd_egress_3_2, ^end_core_dma_3_2)
    ^bd_egress_3_2:
      aie.use_lock(%out_cons_3_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_3_2 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_3_2, Release, %c1)
      aie.next_bd ^bd_egress_3_2
    ^end_core_dma_3_2:
      aie.end
    }
    // Tile (3,2) Core Vector Execution
    %core_3_2 = aie.core(%tile_3_2) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_3_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_3_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_3_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_3_2, %core_pong_3_2, %core_weights_3_2, %core_out_i8_3_2, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_3_2, Release, %c1)
      aie.use_lock(%pong_prod_3_2, Release, %c1)
      aie.use_lock(%out_cons_3_2, Release, %c1)
      aie.end
    }
    // Tile (3,3) DMA
    %mem_3_3 = aie.mem(%tile_3_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_3_3, ^egress_chan_3_3)
    ^bd_ping_3_3:
      aie.use_lock(%ping_prod_3_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_3_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_3_3, Release, %c1)
      aie.next_bd ^bd_pong_3_3
    ^bd_pong_3_3:
      aie.use_lock(%pong_prod_3_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_3_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_3_3, Release, %c1)
      aie.next_bd ^bd_ping_3_3
    ^egress_chan_3_3:
      aie.dma_start(MM2S, 0, ^bd_egress_3_3, ^end_core_dma_3_3)
    ^bd_egress_3_3:
      aie.use_lock(%out_cons_3_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_3_3 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_3_3, Release, %c1)
      aie.next_bd ^bd_egress_3_3
    ^end_core_dma_3_3:
      aie.end
    }
    // Tile (3,3) Core Vector Execution
    %core_3_3 = aie.core(%tile_3_3) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_3_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_3_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_3_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_3_3, %core_pong_3_3, %core_weights_3_3, %core_out_i8_3_3, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_3_3, Release, %c1)
      aie.use_lock(%pong_prod_3_3, Release, %c1)
      aie.use_lock(%out_cons_3_3, Release, %c1)
      aie.end
    }
    // Tile (3,4) DMA
    %mem_3_4 = aie.mem(%tile_3_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_3_4, ^egress_chan_3_4)
    ^bd_ping_3_4:
      aie.use_lock(%ping_prod_3_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_3_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_3_4, Release, %c1)
      aie.next_bd ^bd_pong_3_4
    ^bd_pong_3_4:
      aie.use_lock(%pong_prod_3_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_3_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_3_4, Release, %c1)
      aie.next_bd ^bd_ping_3_4
    ^egress_chan_3_4:
      aie.dma_start(MM2S, 0, ^bd_egress_3_4, ^end_core_dma_3_4)
    ^bd_egress_3_4:
      aie.use_lock(%out_cons_3_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_3_4 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_3_4, Release, %c1)
      aie.next_bd ^bd_egress_3_4
    ^end_core_dma_3_4:
      aie.end
    }
    // Tile (3,4) Core Vector Execution
    %core_3_4 = aie.core(%tile_3_4) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_3_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_3_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_3_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_3_4, %core_pong_3_4, %core_weights_3_4, %core_out_i8_3_4, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_3_4, Release, %c1)
      aie.use_lock(%pong_prod_3_4, Release, %c1)
      aie.use_lock(%out_cons_3_4, Release, %c1)
      aie.end
    }
    // Tile (3,5) DMA
    %mem_3_5 = aie.mem(%tile_3_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_3_5, ^egress_chan_3_5)
    ^bd_ping_3_5:
      aie.use_lock(%ping_prod_3_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_3_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_3_5, Release, %c1)
      aie.next_bd ^bd_pong_3_5
    ^bd_pong_3_5:
      aie.use_lock(%pong_prod_3_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_3_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_3_5, Release, %c1)
      aie.next_bd ^bd_ping_3_5
    ^egress_chan_3_5:
      aie.dma_start(MM2S, 0, ^bd_egress_3_5, ^end_core_dma_3_5)
    ^bd_egress_3_5:
      aie.use_lock(%out_cons_3_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_3_5 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_3_5, Release, %c1)
      aie.next_bd ^bd_egress_3_5
    ^end_core_dma_3_5:
      aie.end
    }
    // Tile (3,5) Core Vector Execution
    %core_3_5 = aie.core(%tile_3_5) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_3_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_3_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_3_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_3_5, %core_pong_3_5, %core_weights_3_5, %core_out_i8_3_5, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_3_5, Release, %c1)
      aie.use_lock(%pong_prod_3_5, Release, %c1)
      aie.use_lock(%out_cons_3_5, Release, %c1)
      aie.end
    }
    // Tile (4,2) DMA
    %mem_4_2 = aie.mem(%tile_4_2) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_4_2, ^egress_chan_4_2)
    ^bd_ping_4_2:
      aie.use_lock(%ping_prod_4_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_4_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_4_2, Release, %c1)
      aie.next_bd ^bd_pong_4_2
    ^bd_pong_4_2:
      aie.use_lock(%pong_prod_4_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_4_2 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_4_2, Release, %c1)
      aie.next_bd ^bd_ping_4_2
    ^egress_chan_4_2:
      aie.dma_start(MM2S, 0, ^bd_egress_4_2, ^end_core_dma_4_2)
    ^bd_egress_4_2:
      aie.use_lock(%out_cons_4_2, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_4_2 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_4_2, Release, %c1)
      aie.next_bd ^bd_egress_4_2
    ^end_core_dma_4_2:
      aie.end
    }
    // Tile (4,2) Core Vector Execution
    %core_4_2 = aie.core(%tile_4_2) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_4_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_4_2, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_4_2, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_4_2, %core_pong_4_2, %core_weights_4_2, %core_out_i8_4_2, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_4_2, Release, %c1)
      aie.use_lock(%pong_prod_4_2, Release, %c1)
      aie.use_lock(%out_cons_4_2, Release, %c1)
      aie.end
    }
    // Tile (4,3) DMA
    %mem_4_3 = aie.mem(%tile_4_3) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_4_3, ^egress_chan_4_3)
    ^bd_ping_4_3:
      aie.use_lock(%ping_prod_4_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_4_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_4_3, Release, %c1)
      aie.next_bd ^bd_pong_4_3
    ^bd_pong_4_3:
      aie.use_lock(%pong_prod_4_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_4_3 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_4_3, Release, %c1)
      aie.next_bd ^bd_ping_4_3
    ^egress_chan_4_3:
      aie.dma_start(MM2S, 0, ^bd_egress_4_3, ^end_core_dma_4_3)
    ^bd_egress_4_3:
      aie.use_lock(%out_cons_4_3, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_4_3 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_4_3, Release, %c1)
      aie.next_bd ^bd_egress_4_3
    ^end_core_dma_4_3:
      aie.end
    }
    // Tile (4,3) Core Vector Execution
    %core_4_3 = aie.core(%tile_4_3) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_4_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_4_3, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_4_3, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_4_3, %core_pong_4_3, %core_weights_4_3, %core_out_i8_4_3, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_4_3, Release, %c1)
      aie.use_lock(%pong_prod_4_3, Release, %c1)
      aie.use_lock(%out_cons_4_3, Release, %c1)
      aie.end
    }
    // Tile (4,4) DMA
    %mem_4_4 = aie.mem(%tile_4_4) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_4_4, ^egress_chan_4_4)
    ^bd_ping_4_4:
      aie.use_lock(%ping_prod_4_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_4_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_4_4, Release, %c1)
      aie.next_bd ^bd_pong_4_4
    ^bd_pong_4_4:
      aie.use_lock(%pong_prod_4_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_4_4 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_4_4, Release, %c1)
      aie.next_bd ^bd_ping_4_4
    ^egress_chan_4_4:
      aie.dma_start(MM2S, 0, ^bd_egress_4_4, ^end_core_dma_4_4)
    ^bd_egress_4_4:
      aie.use_lock(%out_cons_4_4, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_4_4 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_4_4, Release, %c1)
      aie.next_bd ^bd_egress_4_4
    ^end_core_dma_4_4:
      aie.end
    }
    // Tile (4,4) Core Vector Execution
    %core_4_4 = aie.core(%tile_4_4) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_4_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_4_4, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_4_4, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_4_4, %core_pong_4_4, %core_weights_4_4, %core_out_i8_4_4, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_4_4, Release, %c1)
      aie.use_lock(%pong_prod_4_4, Release, %c1)
      aie.use_lock(%out_cons_4_4, Release, %c1)
      aie.end
    }
    // Tile (4,5) DMA
    %mem_4_5 = aie.mem(%tile_4_5) {
      %c1 = arith.constant 1 : i32
      aie.dma_start(S2MM, 0, ^bd_ping_4_5, ^egress_chan_4_5)
    ^bd_ping_4_5:
      aie.use_lock(%ping_prod_4_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping_4_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_4_5, Release, %c1)
      aie.next_bd ^bd_pong_4_5
    ^bd_pong_4_5:
      aie.use_lock(%pong_prod_4_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong_4_5 : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_4_5, Release, %c1)
      aie.next_bd ^bd_ping_4_5
    ^egress_chan_4_5:
      aie.dma_start(MM2S, 0, ^bd_egress_4_5, ^end_core_dma_4_5)
    ^bd_egress_4_5:
      aie.use_lock(%out_cons_4_5, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_out_i8_4_5 : memref<256xi8> offset = 0 len = 256)
      aie.use_lock(%out_prod_4_5, Release, %c1)
      aie.next_bd ^bd_egress_4_5
    ^end_core_dma_4_5:
      aie.end
    }
    // Tile (4,5) Core Vector Execution
    %core_4_5 = aie.core(%tile_4_5) {
      %c1 = arith.constant 1 : i32
      %shift = arith.constant 0 : i32
      aie.use_lock(%ping_cons_4_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_4_5, AcquireGreaterEqual, %c1)
      aie.use_lock(%out_prod_4_5, AcquireGreaterEqual, %c1)
      func.call @conv_im2col_ping_pong_m2_srs(%core_ping_4_5, %core_pong_4_5, %core_weights_4_5, %core_out_i8_4_5, %shift) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi8>, i32) -> ()
      aie.use_lock(%ping_prod_4_5, Release, %c1)
      aie.use_lock(%pong_prod_4_5, Release, %c1)
      aie.use_lock(%out_cons_4_5, Release, %c1)
      aie.end
    }
  }
}
