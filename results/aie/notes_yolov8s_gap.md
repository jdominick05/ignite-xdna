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

### Tested and refuted: an oversized consumer descriptor does not complete short

The obvious way to measure an unknown compressed length is to size the consumer generously and see how much
arrives. `_build_multi_cmp_only` takes its sizing from module constants read at build time - `comp_ty` from
`RATIOED_PER_LINE` and the shim out task from `RATIOED_N` - so both were patched from arange's compressed
length (736 and 2,944) to the raw length (1,024 and 4,096), leaving a consumer descriptor sized for the whole
raw payload.

**It times out, and it times out on `arange`** - the one input whose compressed length is known, where the run
must have returned 2,944 words.

**That experiment is VOID, and the retraction matters more than the result.** It assumed consumer sizing was the
only variable. It was not: `multi_cmp_only` **times out on its own golden `arange` input, unpatched and as
shipped, on this device** - a committed, lit-gated config that simply does not run here. The patched run proves
nothing about oversizing, and the conclusion drawn from it is withdrawn.

What survives is a sound experiment with a different shape. `cmp_only` **passes** on `arange` here (31 ms) and
**times out on our real weight data**, with no patching at all: same config, same sizing, only the data
differing. So a fixed-size consumer does stall when the compressed length differs from what it was sized for -
that much holds. Whether sizing the consumer *generously* rescues it was **reopened** - and then answered.

**An oversized consumer descriptor completes short, with no FoT at all.** Against `cmp_only`, which is
known-good on this part, with `RATIOED_N` patched from 2,944 up to the raw 4,096 and FoT left off: `arange`
returns **2,944 words** - its known compressed length - through a 4,096-word consumer. So `Buffer_Length`
already behaves as a cap on this path, a compressed stream of unknown length *is* receivable, and the original
"must match exactly" claim was an artifact of a broken config rather than a hardware rule.

That looked like it explained the very first failure in this section - `cmp_only` stalling on real weight data
because its consumer was sized to arange's 2,944 words - but **oversizing is necessary and not sufficient**.
With the consumer sized for the raw payload and the arange guard passing at 2,944 words, the first real weight
chunk **still times out**. And that chunk is **81% zero words**, which rules out the explanation being reached
for: a zero-run encoder would compress it far below the consumer, not overflow it.

It also retires an assumption used earlier in this file. **Arange contains no zeros and still compresses to
71.9%**, so this codec finds structure in sequential integers rather than suppressing zeros, and the claim that
1.391x is "its floor because arange has no zeros" was unsupported. Our int8 weights packed into int32 words may
look like high-entropy words to it - and may **expand**. The host output buffer is fixed at N, so expansion past
N would make this path unusable whatever the ratio turns out to be.

What separates the possibilities is synthetic input rather than more real chunks: all-zeros (if even that
stalls, the stall is structural rather than about our data), all-ones and an alternating pattern (trivially
compressible but not monotonic), and uniform random (incompressible - if only that stalls, expansion past the
consumer is the stall mode). One dispatch each, one per invocation, because every miss is a ten-second timeout.

**All-zeros stalls**, with the consumer sized at the full 4,096 words. That is the most trivially compressible
input there is, so the stall is structural: not entropy, not expansion past the descriptor, not anything about
weights.

Two candidate explanations were then checked against the source rather than argued, because the obvious one was
wrong. The consumer sizing *does* derive from `RATIOED_N` for these configs - `out_tap = _linear_tap(RATIOED_N)
if has_mm2s_cmp else None`, used for every non-round-trip config - so the patch really did widen the consumer
and the arange result is not a false positive. And the input side is not the culprit either: `in_tap =
_linear_tap(RATIOED_N) if has_s2mm_dcmp else None`, and `cmp_only` has no decompression, so it is fed the full
4,096 words.

So an oversized consumer completes short on arange, and yet stalls on every other input tried. The mechanism is
something this file has not identified, and three of the last four hypotheses about this codec were wrong -
each time from reasoning past a gap instead of reading. The next step is to **observe** the stall rather than
infer it: `DMA_S2MM_Status` at `0xA0660 + 4*ch` carries `Stalled_Lock_Acq` (bit 2),
`Stalled_Stream_Starvation` (bit 4), `Stalled_TCT_or_Count_FIFO_Full` (bit 5) and `Error_FoT_Length_Exceeded`
(bit 12), and the example's own `regdump` config shows how a core reads such a register back with `read_tm`
into an ObjectFifo once the host enables the processor bus at `0x32038`.

### FoT: the hardware has both halves, and no software uses either

`FoT_Mode` is bits 17:16 of the **S2MM** control register - MemTile `0xA0600 + 8*ch`, core tile
`0x1DE00 + 8*ch`, shim `0x1D200 + 8*ch` - encoded `00` disabled, `01` no_counts, `10` counts_with_task_tokens,
`11` counts_from_mm_register. MM2S has no such field, which is coherent: finishing on TLAST is a receive-side
rule. npu1's reginit marks `.HasFoTMode = XAIE_FEATURE_AVAILABLE` with `.MaxFoTMode =
DMA_FoT_COUNTS_FROM_MM_REG` on all three tile types.

**`Buffer_Length` becomes a cap, not a required count** - inferred from an error bit rather than from prose, but
the inference is tight: `Error_FoT_Length_Exceeded` is *"Channel in FoT mode, Buffer_Length words received but
no TLAST received"*. Running out of buffer **without** TLAST being the error case only makes sense if TLAST
arriving first is the normal path, ending the descriptor short. That is exactly the generously-sized consumer
this file needs.

**The length is readable.** `DMA_S2MM_FoT_Count_FIFO_Pop` at MemTile `0xA06C8 + 4*ch` carries `Valid` (31),
`Last_in_Task` (30), `BD_ID` (29:24) and `Write_Count` (17:0), *"number of words (32-bit) written to memory this
transfer"*. There is also a live `DMA_S2MM_Current_Write_Count` at `0xA06B0 + 4*ch`. Two traps: **`Write_Count`
counts 32-bit words, not bytes**, and **the pop is destructive** - reading it consumes the entry, so a status
dump would steal the count from its real consumer. (The same word-vs-byte care applies to `RATIOED_N` = 2,944,
which the example's README calls a byte count but uses as an int32 element count: 11,776 bytes.)

**No software anywhere uses it.** No aie-rt API touches the count FIFO, no `aiex` op returns a value, the
transaction format's `READ_REGS` opcode is defined but referenced nowhere, and a control-packet read has no
return route. So the bits get written directly, the same way compression must be. The one in-tree mechanism for
getting an arbitrary DMA register back to the host is the `regdump` pattern: a core reads it with `read_tm` into
an ObjectFifo, after the host enables the processor bus at `0x32038`.

**The load-bearing assumption is unsourced.** TLAST is asserted by default at the end of every MM2S BD transfer
(`TLAST_Suppress`, bit 31 of MemTile BD word 2), but nothing in the tree says whether TLAST still arrives at the
end of a *compressed* stream. The whole plan rests on it, and mode `01` tests it for one masked write and no
readback at all: size the consumer generously, set FoT, and see whether `arange` completes at its known 2,944
words instead of stalling.

That makes variable-length receive load-bearing rather than a convenience. Either the `FoT_Mode` field in the
same control register (bits 17:16, "finish on TLAST", with a `FoT_counts_from_mm_register` encoding that
implies a readable count) provides a transfer that ends on the stream rather than on a byte count, or hardware
compression cannot carry per-packet weight data at all - whatever its ratio - because every packet's length
differs and nothing can be sized in advance.

The device is not the variable: the known-good flag-off container verified 66/66 exact immediately after this
timeout, at 7.239 ms, the tenth such verify of the sitting.

### But our weight data does survive the codec, byte for byte

The ratio is blocked; correctness is not, and it can be tested without knowing any length. In
`lossless_roundtrip` both `engage_compress` and `engage_decompress` are set, so `out_tap_rt` is `None` - there
is no ratio-sized tap anywhere, both shim ends are raw-sized, and the compressed form exists only on the
internal leg where the consumer's S2MM expands it back into a raw-sized buffer. Lengths match on the
decompressed side whatever the data does. It is also the right topology: that config's consumer is the
**MemTile**, so it compresses on a core-tile MM2S and decompresses at a MemTile S2MM.

Sixteen chunks, eight from each model's real weight store, after a guard run of `arange` at 4,096/4,096:

| model | chunks sampled | result | zero-word density spanned |
|---|---|---|---|
| yolov8n | 8 of 498 | **8/8 byte-exact**, 4,096/4,096 each | 0.3% - 83.1% |
| yolov8s | 8 of 1,799 | **8/8 byte-exact**, 4,096/4,096 each | 0.6% - 81.5% |

Every one identical, none untouched. The samples deliberately span the range of data character - the
heavily padded chunks and the near-incompressible ones at 0.3% and 0.6% zero words - so the codec is lossless
on our bytes regardless of density, and the hard cases behave exactly like the easy ones.

So the necessary condition holds: **the codec does not corrupt our weight data, and it decompresses correctly at
the MemTile.** What remains is not correctness but plumbing - a compressed stream whose length is data-dependent
cannot be received by a descriptor that must be sized in advance, and that is the whole of the remaining
question.

## What weight residency would actually cost, and one correction

With compression parked, weight residency is the last live lever: **2.006 ms** of re-sent weights against a
**0.29 ms** bar. This is what building it would take, recorded before anyone spends the hours on it.

### The cheap version does not exist

A layer's weights are re-sent once per tile, and the obvious fix is to raise the packet header's counts so one
weight object serves every round. Single-chunk layers already do exactly that - `conv_packet(..., count_out=len(mine))`.
It cannot be extended to a multi-chunk layer. `_core_fn` acquires **one** weight object, loops `range_(n_out)`
emitting and `range_(n_acc)` accumulating into a single `psum`, then releases:

```python
w = w_in.acquire(1)
n_out = memref.load(w, [arith.constant(idx_ty, H_COUNT_OUT)])
n_acc = memref.load(w, [arith.constant(idx_ty, H_COUNT_ACC)])
for _ in range_(n_out):
    a = a_in.acquire(1); o = o_out.acquire(1)
    engine(w, a, o, psum, row); a_in.release(1); o_out.release(1)
for _ in range_(n_acc):
    a = a_in.acquire(1); engine(w, a, scratch_out, psum, row); a_in.release(1)
w_in.release(1)
```

A multi-chunk layer needs each chunk's own weights inside every round, in `ch.last` order, and `psum` retires at
the emit. A larger count would therefore feed one chunk's weights to another chunk's activations - wrong data,
not merely wrong timing. The emulator enforces the same contract (`remaining = count_out + count_acc`, one
weight object per run of activation objects). Note the counts are consumed in `design.py`, not `engine.cc`, so a
valid scheme would need no kernel change; the obstacle is the dataflow, not the one-program rule.

### And no IRON fifo producer can live on a MemTile

`ObjectFifo.prod(tile=)` is documented as *"the shim tile its host-side DMA binds to"*, and `build_program`
already uses it as `Tile(c, 0)`; it does not place a producer on row 1. Combined with the recorded trap that
IRON fifos "send once then stall on locks", a resident weight buffer has to be hand-written on raw locks like
`_activation_ring` - which forces a **third core-function variant**, because cores take weights through
`w_in.acquire(1)` on a fifo handle.

So the real cost is a hand-written MemTile buffer, a third core function, scheduler changes to emit weights once
per layer, and the per-layer reset-and-re-push idiom. That is the activation ring's shape with one more moving
part. The ring needed four protocol revisions and hung silicon twice, and when finally correct was **slower on
both models**. And the 2.006 ms comes from the same derived comparator that predicted yolov8s would win by
0.748 ms when it lost by 3.40 - wrong by 4.1 ms on its most confident case. Whether that is worth the hours is a
judgement to make before starting, not after.

### Correction: sixteen free descriptors, not seventeen

The free ids are **32-47**. The earlier count of seventeen counted `aie.dma_bd` operations rather than distinct
ids. Read off a flag-off build, even channels hold 0-23 (S2MM 0 -> 0-7, MM2S 0 -> 8-9, MM2S 2 -> 10-11,
MM2S 4 -> 12-19, S2MM 2 -> 20-21, S2MM 4 -> 22-23) and odd channels hold 24-31. The range is still usable, but
for a specific reason worth stating: `isBdChannelAccessible` lets an odd channel use only ids >= 24, and both
target channels - S2MM 5 and MM2S 5 - are odd.
