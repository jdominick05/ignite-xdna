# Where yolov8s's frame goes, and which levers are real

AMD's stack beats us on two models. On YOLOv8s the recorded sitting is 16.95 ms against our 17.24, a gap of
0.29 ms on a 17 ms frame - 1.7%. (SESR M7 is the other, 3.64 against 6.67, and is a different problem.) This
file measures where the yolov8s frame goes and sizes every lever found, so that effort goes at something real.

Measured on Device 0 this sitting, both containers 66/66 byte-exact: the flag-off dispatch mean is **16.828 ms**
over 20. The shape of the gap matters as much as its size: AMD's `session.run` is 13.14 ms against our 16.70 ms
dispatch, but their host work is 3.82 ms against our 0.34. **We lose on dispatch and win on host**, so a lever
has to be inside the dispatch.

## The frame, by kind

`tools/engine_stream_report.py`, per frame, per-group schedule:

| kind | tasks | bytes | share of bytes |
|---|---|---|---|
| weight | 389 | 83,618,816 | 24% |
| activation | 5,857 | 237,670,400 | 68% |
| drain | 897 | 27,814,400 | 8% |
| **total** | **7,143** | **349,103,616** | |

At 26.8 GB/s that is 13.03 ms of transport, plus 7,143 tasks x 4 ops x 145 ns = 4.14 ms of sequencing, summing
to about 17.2 ms against the 16.83 measured - close enough to reason with, unlike the ring's costing.

## Lever 1: activation over-read - REAL BUT SMALL, and blocked

`docs/DECISIONS.md` records "fixed packet sizes with over-read into junk planes ... every activation packet is
6,400 B ... the price is extra DMA bytes on shallow layers". Measured per chunk kind, with the packet totals
reconciled against the stream report exactly:

| model | junk bytes | share of activation traffic | at 26.8 GB/s |
|---|---|---|---|
| yolov8n | 2,140,160 | 2.6% | 0.080 ms |
| yolov8s | 4,771,840 | 2.0% | **0.178 ms** |

It is concentrated where `a_pattern` says it would be: `res` packets are four real blocks plus four over-read
(50% junk) and `k1up2` is 20%; `k1`, `k3s1`, `k3s2`, `k5s1` and `pool` waste nothing. Below the 0.29 ms needed,
and collecting it needs per-layer packet sizes, which the same decision rules out because an ObjectFIFO object
is a fixed size and variable ones need MemTile channels the column does not have.

## Lever 2: fill-task saturation - AN ILLUSION

yolov8s carries 37,136 activation packets in 5,857 tasks, a mean of 6.3 against a ceiling of 64 - and **5,406
of those tasks carry exactly 4 packets**. Packing every task full would be 581 tasks, apparently worth 3.06 ms
of instruction ops. It is not available:

- the MemTile split chops the incoming stream into 25,600-byte objects and routes them to cores by offset, so
  every object must be one chunk's four core packets - the order is forced to `[chunk][core]`;
- merging a fixed chunk across rounds would force `[chunk][round][core]`, and `ch.last` retires the
  accumulators, so a chunk moved across a round boundary changes the result.

So a multi-chunk layer's fills are already at their structural floor of one task per (round, chunk). The
64-packet tasks that do exist come from single-chunk layers, where `merge_runs` can flatten every (quad, core)
pair of a run because rows advance uniformly by five across them. Nothing left here.

## Lever 3: weight re-send - THE REAL ONE

yolov8s moves 83,618,816 B of weight packets a frame against a static store of 29,495,808 B: **2.83x
amplification, 54,123,008 B re-sent, 2.020 ms at 26.8 GB/s.** yolov8n is 3.22x, 18,138,880 B, 0.677 ms.

The cause is exact and uniform: **the multiplier equals the layer's spatial tile count**, on every layer of both
models - 64x for a 64-tile layer, 16x for 16, 4x for 4. The coarse schedule re-sends a layer's whole weight run
per round with a stride-0 repeat, because neither the weights nor the partial sums can stay anywhere: core L1 is
65,536 B with 59,392 already used, so a 9,472-byte weight object cannot live there, and the accumulator scratch
holds one tile's sums.

But the MemTile can hold them. With 447,488 B free per column, a layer's **per-column** weight working set -
chunks x groups-owned x 9,472 - fits almost everywhere:

| model | layers that fit | recoverable if fetched once per layer instead of once per tile |
|---|---|---|
| yolov8n | 66 of 66 | 18,014,412 B = **0.672 ms** |
| yolov8s | 60 of 66 | 53,756,412 B = **2.006 ms** |

The six yolov8s layers that do not fit all have **one tile**, so they re-send nothing and there is nothing to
recover from them - the layers too big to hold are exactly the ones that do not need holding. The biggest
winners are small: working sets of 18,944 to 303,104 B, most under 80 KB, against 2.4-4.8 MB of moved bytes
each.

**Why this is not the activation ring again.** The ring fetched a tile once and replayed it across `serves`
groups, at most 8, and its per-layer configuration cost more than the bytes it saved. Weights replay across
`tiles` - up to 64 - and are read-only, so the ratio that killed the ring is inverted here. And the blocker the
record gives for resident weights ("IRON's `init_values` fifos send once then stall on locks, so per-frame
resident weights would need `dma_channel_reset_for`") is exactly the primitive the ring work built and proved on
silicon: reset the channel, re-arm its locks absolutely, re-push.

### The budget a weight buffer would need, read off a flag-off build

The MemTile descriptor and channel budget is what capped the activation ring at six slots, so it is checked
here against `build/conv_engine/yolov8n_flagoff/design.prj/input_with_addresses.mlir` rather than assumed.

**Channels are free.** The flag-off MemTile uses S2MM 0 for the activation split's arrival, MM2S 0-3 to send a
different slice to each core, and the output join takes S2MM 1-4 in and MM2S 4 out. That is five of six in each
direction, leaving **S2MM 5 and MM2S 5 free** - exactly the pair a weight buffer needs, since weights are the
*same* bytes to all four cores and one MM2S can broadcast them where the split needed four.

**Descriptors are free.** The split and join together use **31 of the MemTile's 48**, leaving 17.

**SRAM fits, but it is bank-constrained, which the ring's byte figure hides.** The MemTile holds
`a0_cons_buff_0/1` (25,600 B) at addresses 0 and 65,536 and `o0_buff_0/1` (12,800 B) at 131,072 and 196,608 -
76,800 B of 524,288, one buffer per 65,536-byte bank across banks 0-3. So although 447,488 B are free in total,
an `aie.buffer` must be contiguous: banks 4-7 give 262,144 B, and with the tail of bank 3 the largest
contiguous run is about **314,880 B**. That only just clears the largest per-column working set (303,104 B for
`/model.5/conv/Conv`). Most winning layers are far smaller - 18,944 to 75,776 B - and the two large ones could
hold one group's chunks and re-fetch per group rather than per tile, keeping most of the saving.

**And weights do not touch the MemTile today at all.** `w_of.prod(tile=Tile(c, 0))` places the producer on the
shim and every core consumes it directly: the lowered design allocates `w*_cons_buff_0/1` as `memref<2368xi32>`
- 9,472 B, double-buffered - in each core's L1 at addresses 32,768 and 49,152. So this is new MemTile occupancy
feeding the same core buffers, not a relocation, and the cores' consumption pattern need not change. It is also
why weights cannot simply stay in L1: two of them already take 18,944 B of a core's 65,536, beside the output
and psum buffers.

Nothing here is built. These are byte counts and capacity checks; only a measurement on Device 0 settles
whether the configuration cost stays below the 2.006 ms - and the ring is the standing warning that a
configuration cost can exceed the bytes it saves.

## Lever 4: hardware compression - the mechanism works on this device

The MemTile DMA can compress and decompress, and it is not a dead field. `aie_registers_aie2.json` gives the
MemTile S2MM control register (`0xA0600 + 8i`) a `Decompression_Enable` at bit 4 - "0=no decompression;
1=decompression may be enabled by BD" - and the MM2S control register (`0xA0630 + 8i`) a `Compression_Enable`
at the same bit. Each BD needs its own bit too (MemTile `DMA_BD{n}_4` bit 31, "only effective if channel has
(de)compression enabled"). npu1's reginit table marks `.Compression = XAIE_FEATURE_AVAILABLE`, and aie-rt
implements both halves. The **shim has no compression field at all**, which does not matter: the shim moves
opaque bytes and the codec sits at the MemTile.

Two things are missing rather than broken. **MLIR and IRON expose none of it** - `AIEDmaToNpu.cpp` hardcodes
`words[1] = 0; // Enable_Compression` - so the bits must be written directly, which is the same `npu_write32` /
`npu_maskwrite32` machinery the ring work already proved. And **the format is undocumented**, so no offline
encoder can be written; the compressed form has to be produced by the hardware itself, which is possible
because `Compression_Enable` exists.

**Verified on this device**, not just in source: `programming_examples/basic/dma_compression` is committed and
lit-gated on npu1, and `memtile_both` passes here - `matches=2944 mismatches=0 compressed_to=71.9%(=1.391x)
sha-ok` in 225 ms, arch detected `npu1`. That is the MemTile compressor and decompressor both engaging, with a
byte-exact round trip against the committed golden.

**Measuring our own data is blocked by fixed BD sizing.** `dma_compression()` takes no length parameter, and
the compress-only path comments "asymmetric compress-only: ratio-size shim S2MM to match the compressed stream
length" with the output tap fixed at `RATIOED_N` = 2,944 words. Feeding real weight bytes through it times out:
the compressed length differs, and the README is explicit that a consumer BD whose length does not match the
compressed byte count stalls the DMA. A real measurement needs a design whose output side is sized
independently.

**But the prize is large and the bar is low.** Counted on the CPU over twelve 16,384-byte chunks sampled across
each model's real weight store:

| model | zero bytes per chunk | byte entropy | zlib | lzma | whole store zlib |
|---|---|---|---|---|---|
| yolov8n | 9.6% - 89.9% | 1.15 - 4.75 b | mean **5.20x** | 5.31x | 3.75x |
| yolov8s | 4.3% - 85.8% | 1.32 - 5.84 b | mean **4.67x** | 4.70x | 3.76x |

To clear the 0.29 ms gap on yolov8s needs only **1.102x** - about 7.8 MB of the 83.6 MB of weight traffic. Even
the single worst chunk sampled (1.34x) clears it.

Generic codecs do not predict this one, and **the mismatch points our way**. The hardware managed only 1.391x
on `arange`, which zlib would crush, so the MemTile codec is not a general compressor but almost certainly a
zero-run or sparsity encoder - and `arange` contains **no zeros at all**. That makes 1.391x plausibly its floor
rather than its typical case, while our weight packets are 70-85% zeros by construction: `conv_packet`
zero-fills wherever `cin_avail` or `cout_avail` do not fill a block, and the bias array is 32 int64 mostly zero.

What is still unknown, and each would have to hold: whether a single 9,472-byte packet compresses and
decompresses **standalone** (every lossless demonstration in the tree uses matched BD geometry on both sides,
and the README notes a state-machine warm-up artifact in the first BD, which suggests the codec carries state
across BDs); how to carry the per-packet compressed length, since the consumer BD must match it exactly; and
whether a full CTRL write can clobber bit 4 when out-of-order is enabled on the same channel.
