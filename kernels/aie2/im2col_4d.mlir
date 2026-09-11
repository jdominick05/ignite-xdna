//===- im2col_4d.mlir -------------------------------------------*- MLIR -*-===//
//
// Standalone 4-D Buffer Descriptor im2col dataflow harness for AMD Phoenix AIE2 (XDNA1).
// Resolves the ObjectFifo overlapping-window timeout (ERT_CMD_STATE_TIMEOUT)
// by configuring direct hardware Buffer Descriptors and explicit locks.
//
// Target: AMD Phoenix AIE2 (npu1, 4-column array)
// Layout:
//   - Tile(0,0): Shim NoC DMA (external memory interface)
//   - Tile(0,1): MemTile L2 SRAM (512 KB)
//   - Tile(0,2): AIE Core L1 Data Memory (64 KB)
//
// Tensor Dimensions:
//   - Input: H=8, W=8, C=32 (INT8 elements, total 8 * 8 * 32 = 2048 bytes)
//   - Kernel: 3x3, unit stride, valid padding
//   - Receptive field patch: 3 * 3 * 32 = 288 bytes
//   - Output spatial width: W_out = W - K + 1 = 8 - 3 + 1 = 6 patches
//   - Total streamed bytes per row: 6 * 288 = 1728 bytes
//
// 4-D DMA Striding (MemTile MM2S Channel 0):
//   - Dim 0: size = 32, stride = 1   (Channel vector chunk, 32 bytes = 8 words)
//   - Dim 1: size = 3,  stride = 32  (Kernel column step, 32 bytes = 8 words)
//   - Dim 2: size = 3,  stride = 256 (Kernel row line stride, 8 cols * 32 bytes = 256 bytes = 64 words)
//   - Dim 3: size = 6,  stride = 32  (Output spatial advancement, 6 patches across row)
//
// MLIR-AIE Dim Ordering (Outermost to Innermost):
//   sizes   = [6, 3, 3, 32]
//   strides = [32, 256, 32, 1]
//
// Hardware Register Mapping (Phoenix AIE2 MemTile):
//   - D0: WRAP = 8 (10-bit),  STEP = 1  (17-bit word stride)
//   - D1: WRAP = 3 (10-bit),  STEP = 8  (17-bit word stride)
//   - D2: WRAP = 3 (10-bit),  STEP = 64 (17-bit word stride)
//   - D3: WRAP = 6 (10-bit),  STEP = 8  (17-bit word stride)
//   All wraps <= 1023 (10 bits) and all strides <= 131072 (17 bits).
//
//===----------------------------------------------------------------------===//

module {
  aie.device(npu1) {
    // ------------------------------------------------------------------------
    // Tile Architecture: Column 0 spanning Shim, MemTile, and Core
    // ------------------------------------------------------------------------
    %tile_0_0 = aie.tile(0, 0) // Shim NoC Tile
    %tile_0_1 = aie.tile(0, 1) // MemTile L2 (512 KB)
    %tile_0_2 = aie.tile(0, 2) // AIE Core L1 (64 KB)

    // ------------------------------------------------------------------------
    // External Buffer & Shim Locks (Tile 0,0)
    // ------------------------------------------------------------------------
    %ext_buf = aie.external_buffer : memref<2048xi8>
    %shim_lock_0 = aie.lock(%tile_0_0, 0) { init = 1 : i32, sym_name = "shim_lock_0" }

    // ------------------------------------------------------------------------
    // MemTile L2 Storage & Hardware Locks (Tile 0,1)
    // ------------------------------------------------------------------------
    // 2048-byte activation buffer for H=8, W=8, C=32 INT8
    %mem_in = aie.buffer(%tile_0_1) { sym_name = "mem_in" } : memref<2048xi8>

    // Dedicated producer/consumer locks in MemTile
    // Lock 0: Producer token (1 = empty buffer ready to fill from Shim)
    // Lock 1: Consumer token (0 = unpopulated, 1 = populated ready to stream)
    %mem_lock_prod = aie.lock(%tile_0_1, 0) { init = 1 : i32, sym_name = "mem_lock_prod" }
    %mem_lock_cons = aie.lock(%tile_0_1, 1) { init = 0 : i32, sym_name = "mem_lock_cons" }

    // ------------------------------------------------------------------------
    // Core L1 Ping-Pong Buffers, Stationary Weights & Output (Tile 0,2)
    // ------------------------------------------------------------------------
    // Two 576-byte L1 buffers (2 patches * 3x3x32 = 576 bytes per transfer)
    %core_ping = aie.buffer(%tile_0_2) { sym_name = "core_ping" } : memref<576xi8>
    %core_pong = aie.buffer(%tile_0_2) { sym_name = "core_pong" } : memref<576xi8>

    // Stationary L1 weights: 9 taps * 4 blocks * 64 bytes = 2304 bytes
    %core_weights = aie.buffer(%tile_0_2) { sym_name = "core_weights" } : memref<2304xi8>

    // Output accumulator buffer (2 patches * 128 INT32 = 256 INT32 = 1024 bytes)
    %core_out = aie.buffer(%tile_0_2) { sym_name = "core_out" } : memref<256xi32>

    // Dedicated Ping hardware locks (locks 0 and 1)
    %ping_prod_lock = aie.lock(%tile_0_2, 0) { init = 1 : i32, sym_name = "ping_prod_lock" }
    %ping_cons_lock = aie.lock(%tile_0_2, 1) { init = 0 : i32, sym_name = "ping_cons_lock" }

    // Dedicated Pong hardware locks (locks 2 and 3)
    %pong_prod_lock = aie.lock(%tile_0_2, 2) { init = 1 : i32, sym_name = "pong_prod_lock" }
    %pong_cons_lock = aie.lock(%tile_0_2, 3) { init = 0 : i32, sym_name = "pong_cons_lock" }

    // ------------------------------------------------------------------------
    // Foreign Function Interface (FFI) Declarations
    // ------------------------------------------------------------------------
    func.func private @conv_im2col_ping_pong_m2(memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> () attributes {link_with = "conv_im2col_kernel_m2.o"}

    // ------------------------------------------------------------------------
    // Stream Interconnect (Circuit-Switched Flows)
    // ------------------------------------------------------------------------
    // Route Shim NoC DMA MM2S:0 -> MemTile S2MM:0
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)

    // Route MemTile MM2S:0 -> Core Tile S2MM:0
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)

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
    // MemTile DMA (Tile 0,1): Ingress to L2 and 4-D im2col Egress to Core
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
      // MM2S Channel 0: Stream 1728 bytes of im2col receptive fields to Core L1
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

      // S2MM Channel 0: Alternates between Core Ping and Core Pong buffers (576 bytes each)
      aie.dma_start(S2MM, 0, ^bd_ping, ^end)
    ^bd_ping:
      aie.use_lock(%ping_prod_lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_ping : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%ping_cons_lock, Release, %c1)
      aie.next_bd ^bd_pong

    ^bd_pong:
      aie.use_lock(%pong_prod_lock, AcquireGreaterEqual, %c1)
      aie.dma_bd(%core_pong : memref<576xi8> offset = 0 len = 576)
      aie.use_lock(%pong_cons_lock, Release, %c1)
      aie.next_bd ^bd_ping

    ^end:
      aie.end
    }

    // ------------------------------------------------------------------------
    // Core Tile Compute (Tile 0,2): Synchronized Ping-Pong Consumption
    // ------------------------------------------------------------------------
    %core_0_2 = aie.core(%tile_0_2) {
      %c1 = arith.constant 1 : i32

      // Synchronize with S2MM DMA: acquire initial ping and pong buffers
      aie.use_lock(%ping_cons_lock, AcquireGreaterEqual, %c1)
      aie.use_lock(%pong_cons_lock, AcquireGreaterEqual, %c1)

      // Call M=2 vectorized compute engine across ping-pong buffers
      // Amortizes stationary L1 weights across dual spatial patches (M=2)
      func.call @conv_im2col_ping_pong_m2(%core_ping, %core_pong, %core_weights, %core_out, %c1) : (memref<576xi8>, memref<576xi8>, memref<2304xi8>, memref<256xi32>, i32) -> ()

      // Release ping and pong buffers back to S2MM DMA
      aie.use_lock(%ping_prod_lock, Release, %c1)
      aie.use_lock(%pong_prod_lock, Release, %c1)

      aie.end
    }
  }
}
