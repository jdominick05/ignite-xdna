# The MemTile activation ring on silicon

Desktop 2 (DESKTOP-CBL5NUA), 2026-09-15. This is a chronological record and later sections supersede earlier
ones: it opens with what the build artifacts say and ends with a root cause established on hardware. Read to
the end before acting on any middle section.

The short version: **the ring computes a byte-exact output tile on silicon, then waits forever for a completion
token that has no route back to the shim.** Three defects were found, fixed and verified before that one, and
none of them was the cause. See "Root cause: the completion token has no route home".

Facts here come from two places and each is marked: reading the probe containers' build artifacts and the
mlir-aie / aie-rt sources, or a dispatch on this machine, named by container. No benchmark figure appears
below, and no ring dispatch has completed.

## Method

`tools/disasm_txn.py` prints only five op categories and leaves most decoded ops unprinted, so its silence is
not evidence. Instead: scan every aligned 32-bit word of a binary and report those that look like a MemTile
register address — column in bits 25+, row bit `1 << 20` set. The scans behind this note were throwaway
scripts, each a few dozen lines and none committed: the BD and lock windows, the channel page, a descriptor's
eight configured words, channel registers with their values, served versus unserved fills per layer, how each
layer would be served, and per-layer byte-exactness against the direct reference. The last is the one worth
rebuilding — it gates a schedule change offline — and `tools/engine_stream_report.py` already covers the
traffic totals.

## Register map, confirmed

| What | Address |
|---|---|
| BD *b*, word *w* | `0xA0000 + 0x20*b + 4*w` |
| Lock *l* | `0xC0000 + 0x10*l` |
| S2MM channel *n* control / START_QUEUE | `0xA0600 + 8*n` / `+4` |
| MM2S channel *n* control / START_QUEUE | `0xA0630 + 8*n` / `+4` |

Bit 1 of a channel control register is its reset. An aie2/aie2p channel has **no enable bit**: the driver masks
that field to zero for all three tile kinds, so a START_QUEUE write is the only way to start or restart one.

## Descriptor layout, confirmed against configured values

| Word | Fields |
|---|---|
| 0 | `ENABLE_PACKET` 31, `PACKET_TYPE` 30:28, `PACKET_ID` 27:23, `OUT_OF_ORDER_BD_ID` 22:17, `BUFFER_LENGTH` 16:0 (32-bit words) |
| 1 | `D0_ZERO_BEFORE` 31:26, `NEXT_BD` 25:20, `USE_NEXT_BD` 19, `BASE_ADDRESS` 18:0 |
| 2-5 | strided-access geometry; zero throughout this build |
| 6 | `ITERATION_CURRENT` 28:23, `ITERATION_WRAP` 22:17, `ITERATION_STEPSIZE` 16:0 — raw, stored off by one |
| 7 | `VALID_BD` 31, `LOCK_REL_VALUE` 30:24, `LOCK_REL_ID` 23:16, `LOCK_ACQ_ENABLE` 15, `LOCK_ACQ_VALUE` 14:8, `LOCK_ACQ_ID` 7:0 |

A MemTile BD is **not** a shim BD: `BdLowering.cpp:205` documents a different word-7 layout for the shim, with
4-bit lock ids. Do not apply it here.

Verified from this build: fill bd 0 word 7 = `0x8140FF42` (acquire lock 2 = `space0`, value `0x7F`; release
lock 0 = `arrived0`, value +1); serve bd 2 = `0x8840F840` (acquire lock 0 value `0x78`, release +8). So
**`LOCK_ACQ_VALUE` is 7-bit two's complement, stored negative** — `0x7F` is −1, `0x78` is −8. Word 6 =
`0x000E18FF` = wrap 8, stepsize 6,400 words, both off by one. Word 1 gives `NEXT_BD` 0, 2, 4, 24, 26 for bds
0, 2, 4, 24, 26 with `USE_NEXT_BD` set — every compiled ring descriptor self-links.

## Semantics that shaped the fix

- **Iteration is an address modifier, not a repeat count.** Execution *k* adds `((current + k) % wrap) * step`.
  One lock acquire and one release fire **per execution**. Repetition comes only from the chain or the queue.
- **`repeat_count` is biased**: a push with `repeat_count = N` performs **N+1** executions.
- **The task queue is 4 deep.** Pushing to a running channel enqueues an independent task; it does not replace
  the running one. Past 4, `Task_Queue_Overflow` is a sticky error bit.
- **`aie.dma_start` arms the channel at configuration time**, unconditionally, and there is no attribute to
  suppress it. A self-chained BD is one task that never completes and never issues a token.
- **The sanctioned re-arm** is reset → restore locks → push, in that order; the reset also freezes the
  channel's bound lock counters, which is why the locks are restored after it.
- **`aiex.npu.writebd` can target a MemTile row** (8 words, one block write) if a full descriptor write is ever
  wanted; maskwrites of the changing fields were enough here.

## The five defects, all found offline

1. **Only amortising layers were served.** The ring replaced the split ObjectFifo for the whole container, but
   only layers that could replay a tile emitted the items that arm and push the serve channels. On an
   eight-layer probe that was 352 activation fills with no serve, starting at `/model.0/conv/Conv`. **Both
   silicon timeouts were this**: the dispatch starved at layer 0 and never reached a ring layer at all.
2. **The window-1 descriptor ids were not the ring's.** Ids 1, 3, 5, 25, 27 sat in blocks unreachable from any
   `dma_start`; CDO generation configures only reachable blocks, so they were dropped and their ids reused by
   the output-join ObjectFifo — bd 1 is configured once, 800 words long, chaining to bd 3. Every window-1
   maskwrite was corrupting a live output descriptor.
3. **The channels were never reset**, so runtime pushes stacked behind the CDO's endless self-chained task.
4. **The acquire count was written positive** where the hardware stores it negated.
5. **The queue push's repeat count was not biased**, and assumed iteration drove the execution count.

## State

Schedule side, committed at `c853b37`: the ring serves every layer with packet geometry; a layer that cannot
replay is served once per group; a tile wider than the ring is filled and served in passes. Gates, both models,
all 66 layers — byte-identical to the direct reference with `schedule_layer_coarse` as a passing control; zero
activation fills nothing serves; flag-off traffic unchanged.

| model | ring | activation | weights | drains | total DDR | DMA tasks |
|---|---|---|---|---|---|---|
| yolov8n | off | 83,072,000 B | 26,303,744 B | 18,112,000 B | 127,487,744 B | 2,972 |
| yolov8n | 16 | 51,737,600 B | 30,736,640 B | 18,112,000 B | 100,586,240 B | 4,496 |
| yolov8s | off | 237,670,400 B | 83,618,816 B | 27,814,400 B | 349,103,616 B | 7,143 |
| yolov8s | 16 | 99,532,800 B | 87,938,048 B | 27,814,400 B | 215,285,248 B | 6,927 |

Total DDR falls 21.1% on yolov8n and 38.3% on yolov8s. Tasks fall on yolov8s (7,143 to 6,927) but **rise** on
yolov8n (2,972 to 4,496), because its 18 non-amortising layers lose the coarse schedule's drain merging and
merged weight runs. Against the recorded cost model — 145 ns per instruction op, 26.8 GB/s of transport — those
1,524 extra tasks cost about 0.22 ms while the 26.9 MB saved is worth about 1.0 ms, so yolov8n keeps most of
its saving but not all of it. Both are derived from a schedule, not measured. Recovering that merging on the
serves-1 path is the next schedule change.

Emitter side, committed at `ba9071b`: defects 3, 4 and 5 are addressed — reset once per column per layer,
negated acquire, cleared `USE_NEXT_BD`, biased repeat counts, single window. Verified in the built stream, both
columns alike: reset `0x2` then `0x0` on each of the five control registers eight times (one per layer);
word 1 value 0 under mask `0x00080000`; word 6 wrap `(chunks - 1) << 17`; word 7 on the four serves only,
`0x01007F00` and `0x02007E00` — acquire −1/−2, release +1/+2 — with the arrival's word 7 never written; fill
pushes repeat {0 ×100, 1 ×48, 3 ×4} and serve pushes {0 ×96, 1 ×52, 7 ×4}, the 4-chunk/2-replay layer showing
3 and 7 where an unbiased emitter would show 4 and 8. Flag-off `insts.bin` hashes `1638da1c474f5b6f2dae9b48`,
identical to a build predating every change here.

## THIRD SILICON TIMEOUT, and the device stopped answering

2026-09-15, after all of the above. `probe8_ring2.ignite` — eight layers, all eight ring-served — dispatched
once with `--iters 0`, NPU confirmed idle beforehand ("No hardware contexts running on device"). Result:
`ERT_CMD_STATE_TIMEOUT` again.

**What is new and matters more than the timeout: the NPU left the bus.** The
`xrt-smi examine -r aie-partitions` run immediately afterwards did not return within 120 seconds; when it
finally did, it reported `ERROR: Please specify a device using --device option` under an **empty**
`Available devices:` list. Minutes earlier the identical query had answered instantly with
`[003d:00:01.1] : NPU Phoenix` and `No hardware contexts running on device`. The two earlier timeouts each
left the device present, clean and answering.

So this is not a stuck hardware context that a later query would drain: the device no longer enumerates at
all. Recovering it needs a driver restart or a reboot, which is a human action. Until it enumerates again,
nothing may be dispatched and no measurement from this machine means anything.

**Correlation, not yet causation, and recorded so it is not lost:** this was the first dispatch carrying the
channel reset. `_arm_ring` pulses the reset bit on S2MM 0 and MM2S 0-3 of every column's MemTile once per
layer — 66 layers in a full container, eight here — mid-dispatch, while the output join is live on S2MM 1-4
and MM2S 4 of the same tiles. A reset drains a channel's queue and freezes its bound lock counters. Nothing
proves it reached beyond the five channels it names, but the first unresponsive device and the first dispatch
to reset anything are the same dispatch.

**The discriminating test, when the device is healthy again:** dispatch `flagoff_now.ignite`, whose blobs are
byte-identical to a pre-change build. If flag-off runs clean, the reset is not poisoning the tile and the fault
is ring-specific. If flag-off also hangs, the device is damaged and no measurement means anything until it
recovers. Run neither until a partition check returns promptly and shows no contexts.

**What the three timeouts have now ruled out.** Starvation is fixed and confirmed: all eight layers are served,
where the first two probes served two of eight and starved at `/model.0/conv/Conv`. The arming values are
right, read back from the stream rather than assumed. So the remaining fault is in how the arming behaves as a
program — the ordering of reset, lock restore, fill push, shim fills and serve push as `run_column_programs`
interleaves four columns — or in the lock protocol under four serve channels contending on one `arrived` lock,
neither of which any offline gate here models. The emulator replays column programs; it does not model a DMA.

## The device recovered, and the fault is ring-specific

The machine was rebooted and the NPU enumerates again: `[003d:00:01.1] : NPU Phoenix`, no hardware contexts.
So the wedge was recoverable and did not outlive a restart.

The discriminating test then ran, and it is decisive. `flagoff_now.ignite` — a full 66-layer container built
from this same worktree with the ring off, whose `insts.bin` hashes `1638da1c474f5b6f2dae9b48`, identical to a
build predating every change in this branch — dispatched clean: **66/66 layers exact, PASS**, first dispatch
9.169 ms (one cold dispatch with `--iters 0`, not a benchmark figure), device still enumerating and idle
afterwards.

That eliminates the broad explanations. The hardware is sound, the build environment is sound, the compiler on
this branch still produces a working container with the ring off, and nothing about the earlier reset pulses
left lasting damage. Whatever hangs is specific to the ring's runtime behaviour.

**Next, and deliberately not the eight-layer probe:** the minimal ring reproducer, one layer. Layer 0
`/model.0/conv/Conv` has one chunk and one replay, so the ring runs at its simplest - iteration wrap 0, fill
repeat 0, serve repeat 0, one arm per column, no re-arm across layers and no multi-pass. If that hangs, the
fault is in the basic arming handshake itself: reset, restore locks, push the fill, stream the shim fills, push
the serves. If it runs, add one dimension at a time - more chunks, then a replay, then more layers - and
whichever addition breaks it names the defect.

## The minimal ring case also times out

`probe1_ring.ignite` — one layer, `/model.0/conv/Conv`, one chunk, one replay, one pass; iteration wrap 0,
fill repeat 0, serve repeat 0 — dispatched once with the NPU confirmed idle. Result:
`ERT_CMD_STATE_TIMEOUT`. The device stayed healthy afterwards (`[003d:00:01.1] : NPU Phoenix`, no hardware
contexts), so the earlier wedge did not repeat; that wedge remains a one-off seen only with the eight-layer
container.

So the ring fails at its simplest. Every feature that could be blamed - replay, multi-pass, re-arming across
layers, differing chunk counts - is absent here, and it still hangs, while the same worktree's flag-off
container passes 66/66 on the same machine minutes earlier.

**What this does not settle.** Layer 0 has 256 rounds, about 64 tiles per column, so it re-arms the ring ~64
times per column. Two candidates survive and this probe cannot separate them:

1. **The re-arm races the replay.** Every `R` unconditionally writes `arrived = 0` and `space = slots`, but the
   serve pushes carry no completion token and nothing ever waits on them. The emitter runs far ahead of the
   cores, so tile N+1's `R` can zero `arrived` while tile N's serves are still streaming; those serves then
   wait forever for tokens that were zeroed and whose fill has already retired.
2. **The basic arming handshake is wrong** in some way that fires on the very first tile.

A single-tile layer would separate them, but `--layers N` builds the first N layers and no early layer has one
round, so the container tooling cannot express that case. The practical discriminator is therefore a
single-variable code experiment: push the serves with `issue_token=True` and wait on them with
`aiex.npu.sync(column, row, MM2S, channel)` before the next re-arm. If the minimal probe then passes, candidate
1 was the cause; if it still hangs, candidate 2 is, and the handshake itself needs instrumenting.

## The MemTile task queue was never bounded

Measured, with the flag-off container as a control that validates the method:

| stream | WRITE | accounted tasks | gap |
|---|---|---|---|
| `flagoff_now` (ring off, 66 layers) | 2,972 | 2,972 | 0 |
| `probe1_ring` (ring on, 1 layer) | 2,560 | 1,280 | 1,280 |

The ring stream decomposes exactly: 768 shim task pushes (matching its 768 `BLOCKWRITE` and 768 `DDR_PATCH`
ops), 1,280 ring queue pushes (256 tiles x one arrival plus 256 x four serves), and 512 lock `write32`s (two
per tile) = 2,560.

`aiex.npu.push_queue` lowers to the same WRITE op a shim task issue does, so half the WRITEs in a ring stream
are queue pushes the emitter never recorded. They occupy no shim slot, so `ensure` was never the thing
deceived - **the queue that overflows is the MemTile's own**. aie-rt sets `XAIE_DMA_MAX_QUEUE_SIZE = 4`, and a
64-tile-per-column layer pushes 64 tasks at each of five 4-deep queues with no token, no sync and nothing
awaited. Past the fourth the hardware sets a sticky `Task_Queue_Overflow` bit rather than blocking, and the
rest are lost. That fires from the first tile, on any layer, whatever its chunk or replay count - which is what
the minimal probe showed.

The fix pushes each serve with `issue_token=True` and awaits all four with `aiex.npu.sync` before returning,
holding a column to one tile in flight. That also closes the re-arm race for free, since `_arm_ring` can no
longer zero an `arrived` that a live replay is waiting on. Offline gate, predicted before measuring and matched
exactly: TCT 384 -> 1,408 (+1,024 = 256 tiles x 4 serves), MASKWRITE 480 -> 1,504 (each sync lowers to a
maskwrite plus a TCT wait), WRITE unchanged at 2,560.

## The fifth attempt never ran: 0xc01e0009 at context creation

`probe1_ring2.ignite` — same shape as the probe that timed out, the sync being the only variable — failed
before dispatching: `RuntimeError: Failed to create context virtual (0xc01e0009)` raised from
`load_xclbin`/`pyxrt.hw_context`, at `EngineSession.__init__`. No dispatch occurred, so **the fix is still
untested on silicon**.

**`xrt-smi` is not a sufficient health gate.** It reported `[003d:00:01.1] : NPU Phoenix` with
`No hardware contexts running on device` immediately before this attempt and again immediately after, while
context creation failed. A future sitting should treat a successful `hw_context` on a known-good container as
the readiness check, not the partition listing.

This is the second time the device has needed recovery in this session; the first cleared with a reboot. NPU
work stopped here rather than retried.

## Sixth attempt, on a healthy machine: still a timeout

After a second reboot the device was re-qualified properly this time. `flagoff_now.ignite` dispatched first as
a readiness check and passed **66/66 layers exact, 8.907 ms** (single cold dispatch), proving the machine
creates contexts and computes correctly.

`probe1_ring2.ignite` — one layer, one chunk, one replay, with each serve pushed for a completion token and
all four awaited before the next tile re-arms — then timed out: `ERT_CMD_STATE_TIMEOUT`. The device stayed
healthy afterwards.

**So bounding the MemTile task queue is not the cause either.** It was a real defect and the fix is right, but
it does not explain the hang. Three separate root causes have now been found, fixed and verified in the built
artifacts, and the ring still does not complete a dispatch.

**What is now excluded, each on evidence rather than argument:**

- Starvation of unserved layers - every layer is served, and the eight-layer probe confirmed it.
- Wrong arming values - reset pulses, cleared `USE_NEXT_BD`, negated acquires and biased repeat counts were all
  read back out of `insts.bin`.
- Colliding descriptor ids - the ring is one window, ids 0/1/2/24/25, none shared.
- An unbounded MemTile task queue - now one tile in flight per column, tokens and syncs verified in the stream
  (TCT 384 to 1,408 exactly as predicted).
- A sick machine, a bad build environment, or lasting damage from the earlier wedges - the flag-off container
  from this same worktree passes 66/66 on the same boot, minutes before the ring probe fails.
- Cross-column interference - every ring register write is scoped by `column=col`.
- Corrupted token accounting - `totals` counts only `w`/`a`/`o` items; `R` and `S` reach no channel.

**What remains, and needs a tool this session did not use:** the hang is inside the ring's runtime handshake
itself, on hardware, with no offline model that reproduces it. The emulator replays column programs and models
no DMA, no lock and no channel, so it cannot see this class of fault by construction. The next step is not
another dispatch-and-guess cycle; it is observation - an AIE trace (`xdp_aie_trace_plugin` ships in the XRT
SDK) or a deliberately instrumented probe that writes a progress marker per stage and is read back after a
timeout, so the exact point of the stall is measured rather than inferred.

Worth suspecting first, when that instrumentation exists: whether the core-side program and the MemTile serve
actually agree. `_core_fn_ring` takes activations through raw `a_cons`/`a_prod` locks against a single buffer,
where the split-ObjectFifo path the engine has always used gives each core a depth-2 fifo. Nothing in this
session tested that handshake in isolation, and it is the one part of the ring that differs from proven code
without having been verified in an artifact.

## Instrumented: the ring computes its first tile exactly, then never starts a second

`verify_engine_container.py` raises on a timeout and compares nothing, which is why six dispatches said only
"it hung". But `read_tensor` is a BO sync plus a read and does not care whether `dispatch()` raised, so the
workspace after a failed dispatch records how far the work got. The probe used here — scratch, uncommitted, and
the one instrument in this note worth rebuilding as a tool — reads each output row before the dispatch and
again after it raises, classifying the row **correct** (equals the reference), **untouched** (still the
pre-dispatch bytes) or **wrong**. Validated on the flag-off container first: 320/320 and 160/160 rows correct.

On `probe1_ring2.ignite` the dispatch timed out after 7.0 s and left: rows 0-19 "wrong", rows 20-319
untouched, layer 1 untouched throughout. Of the 102,400 bytes in those 20 rows, 96,000 were zero and 6,400
carried plausible values - and 6,400 is exactly one 20x20 output tile across layer 0's 16 channels.

Analysed offline from the dumped arrays, the written region is rows 0-19, cols 0-19, 6,400 bytes, and it is
**EXACT**: 0 of 6,400 bytes differ from the reference, all 6,400 non-zero, matching in place. The row-level
"wrong" was an artifact of classifying a whole row - the other fifteen tiles across that band were simply
never written.

**So the ring works.** One shim fetch lands in the MemTile, the serves deliver each core its slice, the cores
compute, and the drain writes a byte-exact tile to DDR. What fails is proceeding to the second tile.

**What differs between tile 1 and tile 2** is exactly the `armed[col]` guard: tile 2 skips the channel reset
and skips every descriptor maskwrite, doing only the lock writes and the pushes. Two fields in a descriptor
are stateful and consumed by execution - `Valid_BD` (word 7 bit 31), which a completed task clears, and
`Iteration_Current` (word 6, bits 28:23), which advances as the BD runs. Neither is in any mask the emitter
writes, so tile 2 pushes a descriptor the hardware no longer considers valid.

The fix is to state the whole descriptor again on every tile rather than only when the chunk count changes,
restoring `Valid_BD` and zeroing `Iteration_Current`. That is safe now for a reason it was not before: the
serves of the previous tile are awaited, so nothing is in flight to disturb.

## What a rebuild must show before any dispatch

Two reset maskwrites on each of `0xA0600`, `0xA0630`, `0xA0638`, `0xA0640`, `0xA0648` per column; a word-1
maskwrite clearing bit 19 on each of the five ring descriptors; iteration maskwrites on the same five; word-7
lock writes whose acquire field is negative; and pushes whose repeat count is one less than the executions
intended. The ring's descriptor ids must appear in the init CDO with the ring's own lengths, and no id may be
shared with another fifo.

## Restoring the consumed descriptor fields changed nothing

`probe1_ring3.ignite` re-arms every descriptor per tile: `Valid_BD` restored on all five (the arrival's word 7
is written with mask `0x80000000` only, so its compiled acquire -1 / release +1 survives), `Iteration_Current`
zeroed by widening the iteration mask from `0x007E0000` to `0x1FFE0000`, and the whole re-arm moved from once
per layer to once per tile - maskwrites 1,504 to 6,312, WRITE and TCT unchanged at 2,560 and 1,408.

Its dispatch is **bit-for-bit identical** to the run before it: one tile at rows 0-19 cols 0-19, the same 6,400
non-zero bytes, the same value histogram, the same 7.0 s timeout. Neither consumed field was the blocker.

## The syncs are not broken - they are the only thing holding correctness

Instrumenting `probe1_ring.ignite`, built before the sync was added, discriminates cleanly:

| container | serve pushes | rows written | data |
|---|---|---|---|
| `probe1_ring` | no token, nothing awaited | **all 320** | 1,304,875 of 1,638,400 bytes wrong |
| `probe1_ring2/3` | token + `npu.sync` per channel | **20** (one tile) | that tile **byte-exact** |

Unbounded, the ring runs ahead of the cores: descriptors are rewritten under live tasks and the four-deep
MemTile queues overrun, so every tile is touched and almost nothing is right. Bounded to one tile, what it
produces is exactly right. Both still time out.

**So the stall is the wait itself.** Tile 1's fill, serve, core compute and drain all complete - its output is
in DDR and correct - and the emitter then blocks on four `aiex.npu.sync` ops for MemTile completion tokens. If
a MemTile MM2S channel does not issue a token such a sync can consume, it waits forever for work that has
already finished, which is precisely what both instrumented runs show.

**The way out does not need a MemTile token.** A tile's drain is a shim task, and shim tasks already carry
completion tokens the emitter awaits routinely through `retire_channel`. Ordering the next re-arm behind the
previous tile's drain is the reverse edge proposed at the outset, built on token machinery this engine has
used since it was written. The open question is whether a drain can be awaited per tile: a token is attached
only to every `retire_batch`-th task of a channel, so the ring would need its drains tokened individually.

## Root cause: the completion token has no route home

The MemTile buffer descriptor runs to completion and lands its bytes - which is why tile 1 is byte-exact - and
`issue_token` sets bit 31 so it emits a task-completion token. The token then has nowhere to go, so the sync
waits forever for work that has already finished. Two things a design needs for a MemTile TCT to arrive, and
this one has neither:

1. `controller_id = #aie.packet_info<pkt_type = ..., pkt_id = ...>` on the MemTile's `aie.tile` op. Without it
   the lowering silently skips programming the TCT controller-ID field - `AIEDmaToNpu.cpp:164-181`, which
   emits that maskwrite only `if the tile carries a controller_id attribute`.
2. An `aie.packet_flow` from `<memtile, "TileControl" : 0>` to `<shim, "South" : 0>`, the token's route back.

A design compiled the ordinary way has neither, because aiecc runs the column-control overlay with
`route-shim-to-tct` left at its default `"shim-only"`, and `AIEGenerateColumnControlOverlay.cpp:326` filters
every non-shim tile out: `if (clRouteShimCTRLToTCT == "shim-only" && !tOp.isShimNOCorPLTile()) continue;`.

**This is not a shim-only mechanism.** Two lit tests execute MemTile token waits on npu1/Phoenix silicon -
`test/npu-xrt/memtile_dmas/writebd_tokens` syncs on row 1 after writing the memtile START_QUEUE at `0xa0604`,
and `test/npu-xrt/memtile_dmas/dma_configure_task_token` awaits a memtile task with `dma_await_task`. The TCT
encoding carries a full 8-bit row (`TxnEncoding.h:158-160`), `AIE_NpuSyncOp` has no verifier, and the op's own
troubleshooting text blames a missing token rather than the tile. Both tests declare exactly the two things
above, with `keep_pkt_header = true, priority_route = true` on the flow.

`dma_await_task` and `npu.sync` are one mechanism with two entry points, both lowering to `NpuSyncOp` ->
`txn_append_sync` -> `TXN_OPC_TCT`, and `DMAAwaitTaskOpPattern` is tile-generic. Only `npu.dma_wait`, which
resolves its tile through a `ShimDMAAllocationOp` symbol, is structurally shim-bound - a limitation of
ObjectFifo symbol resolution, not of tokens.

**Two ways to fix it.** Declare the `controller_id` and the TileControl -> South packet flow in the design, as
the on-silicon tests do; or migrate the ring off raw `push_queue` + `npu.sync` onto
`dma_configure_task`/`dma_start_task`/`dma_await_task` with `issue_token = true`, which is tile-generic by
construction and fails at compile time rather than hanging if a token is ever dropped. Running the overlay with
`route-shim-to-tct=all-tiles` would also generate the flow, but no aiecc or IRON flag exposes that setting.

One caveat to carry: the auto-assigned controller id for a memtile is 26, while both working tests hand-pick
`pkt_id = 1`. The field is 8 bits wide in hardware and the lowering writes five, so 26 is writable, but no test
in the tree exercises a memtile id at or above 16. Pick the id explicitly rather than relying on the default.

## The reverse edge: wait for a drain, which is a token this engine already awaits

Neither fix above was taken. Both declare something new about how the engine builds its tiles; the third
option needs nothing declared, because the ordering the MemTile token was carrying is available from the shim
side already. A tile's drain completes only after the cores emitted its output objects, and the cores could
only do that by consuming every slice the serves delivered. So the drain is proof the serves are done - proof
that travels a route the engine has used since it was written.

The serves are now pushed asking for no token and nothing waits on them. Before a tile re-arms, the column's
outstanding drains are awaited instead. That also restores the bound the waits had been providing on the
MemTile's four-deep task queue: one arrival and one serve per channel are pushed per tile, and the next tile
cannot push until this one's output has reached DDR.

An `R` item carries a flag saying whether that wait may happen, set for every tile of a replayed layer and for
the first pass of a group otherwise. **It is not set between passes of one group**, and that gap is deliberate:
a group has a single drain, issued with its first pass, which cannot complete until the last pass has been
served, so waiting on it there would deadlock the dispatch outright. Passes are therefore unordered against
each other - two layers of yolov8s (`/model.7/conv/Conv`, `/model.19/conv/Conv`, 32 chunks each) and no layer
of yolov8n - and once the serve descriptors were made lock-free, below, they lost their last coupling too,
because a serve no longer waits on `arrived` for the pass before it. Neither layer has been dispatched with
the ring. Each drain issued while a tile is armed carries its own completion
token rather than sharing one with a later drain, so each can be awaited individually.

**Offline, all green.** Per-layer exactness against `graph_reference.run_direct`, seeding the workspace and
emulating one layer back: RING EXACT 66/66 on yolov8n and yolov8s, with the per-group schedule as a control at
66/66 on both. `tests/test_graph_engine_offline` 21 tests OK, `tests/test_conv_engine` 4 OK. Traffic is
unchanged from the tables above, so this adds ordering and not tasks.

**The built stream says the waits are gone**, which is the check worth making before spending a dispatch on a
stale artifact. `probe1_ring4.ignite` - one layer, one chunk, one replay - against the flag-off container:

| | TCT waits | distinct TCT words | WRITE at row 1 | MASKWRITE at row 1 |
|---|---|---|---|---|
| `probe1_ring4` (ring, 1 layer) | 512 | 2 | 1,792 | 4,904 |
| the probe that hung (ring + syncs) | 1,408 | — | 1,792 | 6,312 |
| `yolov8n_flagoff` (66 layers) | 1,671 | 2 | 0 | 0 |

The two distinct wait words of the ring stream, `0x00010100` and `0x01010100`, are **the same two the flag-off
stream uses**, and a flag-off stream waits on shim channels only. So every wait the ring stream makes is drawn
from the shim vocabulary and none is a MemTile wait. The 512 breaks down as the 384 a ring stream made before
the syncs were added, plus 128 for the drains that now carry a token each rather than every second one. Row 1
accounting is exact too: 1,792 writes = 1,280 queue pushes + 512 lock writes, and a flag-off stream touches
row 1 not at all. A transaction address is `(col << 25) | (row << 20) | offset`, so a register write's tile row
comes from its address; the op header's row field is zero for all of them and reading it there proves nothing.

The first attempt to dispatch it never ran: the device refused a hardware context, `0xc01e0009`, from
`pyxrt.hw_context` - on the flag-off container, as the readiness check, before the ring probe ran at all, with
`xrt-smi` having reported `[003d:00:01.1] : NPU Phoenix` and no hardware contexts minutes earlier. Second time
this session the partition listing said healthy immediately before a context failed, third recovery needed.
A reboot cleared it, as the previous two did.

## On silicon: the ring completes a dispatch, and it is the replay that stalls

Same machine, in the sitting that followed that reboot and ran from 2026-09-15 into 2026-09-16.

Readiness first, the real check rather than the partition listing: `yolov8n_flagoff.ignite` **66/66 layers
exact**, first dispatch 9.206 ms, mean 7.399 ms over 20 (min 7.201, max 7.774). The machine creates contexts
and computes correctly, so what follows is the ring's.

| container | layers | result |
|---|---|---|
| `probe1_ring4` | 1 | **completed in 4.600 ms, 320/320 rows of layer 0 correct** |
| `probe8_ring4` | 8 | L0-L5 correct (320 + 160x5 rows); **stalls at L6** `/model.3/conv/Conv` |
| `probe8_serves1` | 8 | replay forced off: **completed, all eight layers correct** |

The first of those is the ring's first completed dispatch after seven timeouts. So the reverse edge works: the
drain is an adequate substitute for the token that has no route, and the single-serve path is correct on
silicon across multi-chunk tiles and across layer boundaries.

**L6 is the first layer with `serves > 1`**, and forcing every layer onto the no-replay path makes the same
eight layers pass. That is a single-variable experiment, so the fault is the replay and nothing else: the
barrier, the re-arm, the drain tokens and the whole no-replay path are exonerated by it.

**What the stalled layer left behind.** Not corruption - absence. Of L6's two output groups, group 1 is
untouched entirely and group 0 holds 9,600 correct bytes with the remaining 16,000 still zero. Mapped to
output objects (4 blocks x 5 rows x 20 columns, 3,200 B each), exactly three objects landed - core 0 at column
tiles 1 and 2, core 1 at column tile 2 - and **every object that landed is byte-exact**. No object is
partially written. Cores 2 and 3 emitted nothing. Two runs stalled after a different number of objects
(44,800 and 41,600 bytes differing) with the same geometry, so how far it gets varies.

**Theories this disproves**, each checked against the dumped arrays rather than argued: the groups are not
swapped, no group holds another's reference, the groups do not hold identical bytes, and nothing equals the
reference rolled by 1-3 channels or rows. Nor is it the iteration counter wrapping inside one task - a
descriptor asked to wrap mid-task would misaddress bytes, and what is missing here is whole objects that were
never produced.

**Replay is the entire benefit, so this cannot be shipped without it.** With replay forced off the ring moves
*more* than the per-group schedule, because fetching once per group through the MemTile moves exactly what the
per-group schedule moved and adds tasks:

| model | ring, no replay | ring off |
|---|---|---|
| yolov8n | 131,920,640 B / 5,424 tasks | 127,487,744 B / 2,972 tasks |
| yolov8s | 353,422,848 B / 10,659 tasks | 349,103,616 B / 7,143 tasks |

Activation bytes with replay off are identical to the flag-off figures. And replay is most of the graph: 37 of
yolov8n's 66 layers have `serves > 1` (histogram 1:29, 2:24, 3:6, 4:7) and 47 of yolov8s's (1:19, 2:17, 3:2,
4:21, 8:7). So there is no partial result to measure and no benchmark to publish.

## The lock protocol, three attempts and the ceiling that stops the third

**One: the serve acquires the window's whole count** (as compiled originally). That makes `arrived` a mutex. A
serve holds it for the length of its transfer, a transfer ends only when its core has taken the bytes, so a
core that runs ahead and fills the output join blocks with its serve still holding the lock, and the other
three channels can never acquire. Measured: the first replayed layer emitted two objects from core 0 - the
join's depth is two - one from core 1, none from cores 2 and 3, then deadlocked.

**Two: the serve holds no lock.** The dispatch completes and every layer is wrong, `/model.0/conv/Conv`
included, which is one chunk and one serve and had been byte-exact twice before. A shim fill task's completion
token says the shim pushed its bytes into the stream, not that this tile's S2MM wrote them into the ring, so
nothing off the tile can stand in for `arrived`: the serves read slots as they were being written. That also
settles a question worth keeping - the ordering has to live on the tile.

**Three: the arrival hands out one token per core per replay** - release count `ROWS * serves`, written per
layer - and each serve takes one per slice and returns none. "Returns none" is a release of zero, not an
omitted release, because a descriptor that touches a lock must carry both ops (`buffer descriptor with a lock
must have both use_lock(acquire) and use_lock(release)`).

This one works, up to a point that is worth stating precisely. An eight-layer container **completed, all eight
layers byte-exact**, including `/model.3/conv/Conv` (four chunks, two replays) and `/model.4/cv1/conv/Conv`
(one chunk, two replays). The full 66-layer container computes **layers 0-12 byte-exact** and then stops dead:
layer 13 `/model.5/conv/Conv` and everything after it untouched, nothing wrong, nothing partial.

**Why it stops is arithmetic.** Consumption matches production over a whole tile, but nothing forces them to
interleave, and the fill is pushed first: if it runs ahead, `arrived` peaks at `slots * ROWS * serves`.

| layer | slots | serves | release per execution | peak `arrived` |
|---|---|---|---|---|
| L6 `/model.3/conv/Conv` | 4 | 2 | 8 | 32 |
| L12 `/model.4/cv2/conv/Conv` | 3 | 2 | 8 | 24 |
| **L13 `/model.5/conv/Conv`** | **8** | **4** | **16** | **128** |
| L20 `/model.7/conv/Conv` | 16 | 2 | 8 | 128 |

A MemTile lock value holds **63** - `getMaxLockValue` returns `0x3F` in `AIETargetModel.h`, confirmed in the
source rather than inferred. Layer 13 is the first of seven yolov8n layers to exceed it, and yolov8s exceeds it
from its own layer 6. This is the ceiling `_activation_ring`'s docstring warned about all along, "one shared
counter would overflow the 63 a lock register holds", and this scheme walked into it.

**It is a structural limit, not a tuning problem.** A 16-slot tile needs `16 * 4 = 64` tokens at a *single*
replay, so no batching of replays, no cap on `serves` and no re-arm schedule rescues a token-per-core-per-replay
counter for the largest tiles.

**And on yolov8s the scheme is not marginally short but short by an order of magnitude.** Its plans, read off a
build rather than derived: `/model.5/conv/Conv` is 16 chunks with 8 serves and peaks at `16 * 4 * 8 = 512`,
`/model.6/cv2/conv/Conv` at 256, `/model.6/cv1/conv/Conv` and `/model.12/cv1/conv/Conv` at 128. Against 63.
Whatever replaces this counter has to be about eight times cheaper in tokens, not a little cheaper, and that
rules out trimming `serves` or the window as a way out.

**What the probes could reach, so this is not concluded from them again:** the eight-layer container's highest
peak is 32, comfortably inside the ceiling. An eight-layer pass therefore says nothing about whether the scheme
scales, and the full container is the only test that does.

**Where to resume:** how a MemTile ring signals per-slot arrival to four consumers, within a 63-value lock and
the one acquire and one release a buffer descriptor can carry.

## Two mechanisms ruled out, so they are not retried

**`ensure` awaiting a drain or weight run before its `S`.** Plausible, and wrong. Holding those tasks so
`ensure` could not retire them left the emitted `insts.bin` **bit-identical** - sha256 `01f6cc395e4f974563e08ea9`
over 1,371,652 bytes, before and after. A hold changes which channel `ensure` retires and so where a TCT wait
lands, so an identical stream proves `ensure` never retired such a task during a yolov8n emit. The change was
also a regression - yolov8s could no longer emit, raising `channel o queue is full of held tasks` where a
column owns eight groups against a four-deep queue - and was reverted.

**The iteration counter wrapping inside one task.** Refuted by the object map: a descriptor asked to wrap
mid-task would misaddress bytes, and what was missing were whole objects that had never been produced.

## The fix that fits the locks, and what the descriptors leave of it

**One lock per slot is the shape that fits.** A slot's lock only ever reaches `ROWS * serves` - 32 at the worst
layer of either model - and, unlike a single counter, it does not grow with the window at all. Against the
plans: the peak falls from 128 to 16 on yolov8n and from 512 to 32 on yolov8s, both far inside 63.

**The descriptors are what it costs.** A descriptor's lock id is a fixed field, so one lock per slot means one
descriptor per slot on each of the five chains - the arrival, and a serve per core. The MemTile's allocation,
read out of `input_with_addresses.mlir` of the build that ran on silicon rather than assumed:

| user | channel | descriptor ids |
|---|---|---|
| ring arrival | S2MM 0 | 0 |
| ring serves | MM2S 0, 1, 2, 3 | 1, 24, 2, 25 |
| output join, to the shim | MM2S 4 | 3-10 |
| output join, from the cores | S2MM 1, 2, 3, 4 | 26, 27 / 11, 12 / 28, 29 / 13, 14 |

21 of 48 ids are taken, and `isBdChannelAccessible` splits the remainder by channel parity: an even channel
reaches only ids below 24, an odd channel only 24 and above - nine free below, eighteen above. Five chains of
`slots` descriptors into 34 free ids gives **`slots <= 6`**, and exactly one arrangement reaches it: the
arrival moved to an odd S2MM channel (5), the serves split two even (MM2S 0, 2) and two odd (1, 3). Leaving
the arrival on an even channel gives 4. The MemTile's SRAM would hold 16 slots and its 64 locks are ample;
descriptors are the only scarce resource here.

**So the question is not what fraction of the replay saving a 6-slot window preserves, but whether a 6-slot
ring beats no ring at all.** Those differ, because the ring *replaces* the split ObjectFifo rather than sitting
beside it: a layer too wide for the window does not return to the per-group schedule, it stays in the ring at
`serves = 1` and pays the ring's task overhead for none of its saving.

Measured over the scheduler at every window size (derived ms = total bytes at 26.8 GB/s plus four instruction
ops per task at 145 ns; a comparator, not a measurement):

| model | ring | layers replayed | tasks | total DDR bytes | derived ms |
|---|---|---|---|---|---|
| yolov8n | 0 | 0 | 2,972 | 127,487,744 | 6.481 |
| yolov8n | **6** | 33 | 4,592 | 106,320,640 | **6.631 (+0.150)** |
| yolov8n | 8 | 36 | 4,512 | 102,224,640 | 6.431 (-0.049) |
| yolov8n | 16 | 37 | 4,496 | 100,586,240 | 6.361 (-0.120) |
| yolov8s | 0 | 0 | 7,143 | 349,103,616 | 17.169 |
| yolov8s | **6** | 31 | 8,343 | 280,821,248 | **15.317 (-1.852)** |
| yolov8s | 8 | 40 | 7,615 | 256,245,248 | 13.978 (-3.191) |
| yolov8s | 16 | 47 | 6,927 | 215,285,248 | 12.051 (-5.119) |

**At the window the descriptors allow, the ring helps yolov8s and hurts yolov8n.** yolov8s moves 68 MB less for
1,200 more tasks; yolov8n moves 21 MB less for 1,620 more. Break-even per task is 2.12 us for yolov8s and
0.49 us for yolov8n, against the 0.58 us that four ops at 145 ns cost - so yolov8s wins with a 3.7x margin and
yolov8n falls just the wrong side of it. (The 2.5 us per DMA task in `docs/BENCHMARKS.md` is itself derived,
and from the 18.3 ms schedule that issued far more tasks; the 145 ns-per-op model is the one since fitted to a
7.87 ms frame.)

**yolov8n needs roughly 260 fewer tasks to break even, and where they went is already known:** the ring's
`serves == 1` path loses the per-group schedule's drain merging and merged weight runs, which is what raises
its count by 1,620 to begin with. That recovery needs no xclbin rebuild, and it belongs *before* the lock
protocol rather than after - it decides whether a 6-slot ring is worth compiling for one model or for both.

**But the recovery collides with the reverse edge, which is worth recording before anyone tries it.**
`schedule_layer_ring` issues one drain per (tile, group) - `run_drain(ws, layer, g, [(y, x0)])`, a single quad -
where `schedule_layer_coarse` gathers up to 16 vertically adjacent quads at one tile column and drains the run
once. Merging them back is where the tasks are. A merged drain, though, spans several tiles, and the drain *is*
the ring's barrier: `ring_barrier` awaits every outstanding `"o"` of a column before `_arm_ring` rewrites the
descriptors they share. A drain covering tiles N..N+k cannot complete until tile N+k has been served, so tile
N+1's re-arm would wait on work its own `S` has not issued yet - precisely the deadlock the schedule already
refuses between passes of one group. The two things that make the ring work, the replay and the reverse edge,
are in tension with the thing that would pay for it.

Merging drains *within* a tile does not help either: a tile's groups differ by `group * OUT_BLOCKS` in the
plane, and `run_drain` has already spent its outermost dimension on the quad repeat, so there is no dimension
left to fold groups into and no fifth one to be had.

**The other lever is descriptors rather than tasks, and it is a dead end - measured, not argued.** The output
join's chain on MM2S 4 is eight descriptors because its fifo is depth 2; at depth 1 the join would hold 8 ids
instead of 14, which lifts the window to 7. A 7-slot ring admits not one extra layer in either model: 33
replayed on yolov8n and 31 on yolov8s, the same as at 6, the same bytes to the byte, and 8 more tasks for the
trouble. Chunk counts run 1, 2, 3, 4, 5, 6, 8, 12, 16, so no layer has exactly 7 and the next one only arrives
at 8. Nor is 8 reachable by rearranging channels: three chains of 8 is 24 descriptors, past the free space in
either half under every assignment of the join's channels, depth 1 included.

| model | ring 6 | ring 7 | ring 8 |
|---|---|---|---|
| yolov8n | +0.150 ms | +0.154 ms | -0.049 ms |
| yolov8s | -1.852 ms | -1.856 ms | -3.191 ms |

So the buildable window is 6, yolov8n's break-even at 8 is out of reach, and giving up the output path's
double buffering would buy nothing at all.

**Which leaves one resolution needing no new mechanism: the ring is already a per-container compile flag.**
`activation_ring` is an argument to `compile_graph_container`, so a per-slot-lock design at 6 slots can be
compiled for the models it helps and left off for those it does not - yolov8s gains its 1.85 ms, yolov8n keeps
the per-group schedule, and nothing regresses. What that costs is the one-program rule: a ring build and a
flag-off build are different xclbins, and whether the engine ships two is a decision above a session's.

## Arming is first order, and counting it reverses every comparison above

Every traffic figure in this note counts DMA tasks and DDR bytes and **none of them counts the register writes
that arm the ring**, because `engine_stream_report` skips the `R` and `S` items by construction: they move no
bytes and issue no task. A flag-off schedule has no arms at all, so arming is overhead the ring must earn back
out of the bytes it saves. It does not earn it back.

Counted rather than modelled, out of the `insts.bin` of containers already built:

| container | insts.bin | instruction ops | derived ms | vs flag-off |
|---|---|---|---|---|
| flag-off | 430,180 B | 12,258 | 1.777 | |
| ring, 16 slots | 1,457,248 B | 48,578 | 7.044 | **+5.266** |
| ring, as committed | 1,371,652 B | 45,521 | 6.601 | **+4.823** |

The counting method checks out against the schedule: flag-off issues 2,972 tasks at `OPS_PER_TASK_ISSUE` of 4,
which is 11,888 ops against 12,258 counted. The ring's 4,496 tasks account for 17,984 of its 45,521, leaving
about 27,500 ops - close to 4 ms - in arming alone. That is against roughly 1.0 ms of transport the 16-slot
window saves on yolov8n.

Three containers in `build/` are the same stream - `yolov8n_flagoff`, `yolov8n_ring8` and `yolov8n_baseline`
all hash `1638da1c474f5b6f`. **`yolov8n_ring8` is not a ring build despite its name** and is evidence of
nothing; a container's manifest does not record `activation_ring`, so the digest is the only way to tell.

With arming included at the 19 register writes `_arm_ring` emits per tile, plus a queue push per core per
serve, the whole comparison inverts:

| model | ring 0 | ring 4 | ring 6 | ring 8 | ring 16 |
|---|---|---|---|---|---|
| yolov8n, arms per frame | 0 | 1,184 | 1,104 | 1,035 | 1,019 |
| yolov8n, total derived ms | 6.481 | +4.296 | **+3.832** | +3.402 | +3.278 |
| yolov8s, arms per frame | 0 | 2,383 | 1,853 | 1,447 | 1,125 |
| yolov8s, total derived ms | 17.169 | +7.367 | **+4.328** | +1.635 | -1.367 |

yolov8n is worse at every window. yolov8s is worse at every window but 16, and that single win is thin enough
to distrust: on yolov8n the model undercounts the measured arming by about 17%, which would take yolov8s's
16-slot case to roughly a wash.

**So the two constraints close on each other.** The window that would pay for itself is 16 - and 16 is exactly
what the 63-value lock forbids under a shared counter, and what the descriptors cannot give per-slot locks. The
window the descriptors do allow is 6, and at 6 arming costs more than the replay saves: 4.3 ms more on yolov8s.
Per-slot locks make that worse rather than better, turning a flat 19-write arm into about 12 writes per chunk -
9.2 to 11.8 ms of arming on yolov8s against 1.85 ms of saving.

**The blocker was never the lock protocol.** It is that the ring arms 1,104 times a frame on yolov8n and 1,853
on yolov8s, and every arm rewrites descriptors that the hardware consumed by running them. Until an arm costs
close to nothing - or a tile is armed once and reused across many layers instead of once per tile - no window
makes this ring pay, and the lock protocol is not worth compiling.

## Costing the arm-once ring

**The shape.** Let the descriptors form a cycle and never clear `Use_Next_BD`. The task then never completes,
so `Valid_BD` is never cleared and `Iteration_Current` never has to be rewound - the two fields that force a
rewrite today - and no queue push is needed at all, because the channel free-runs the way an objectFIFO's does.
The locks alone sequence it.

**The lock shape that makes it balance,** two per slot:

    fill  bd i : Acquire(space[i], N)    Release(arrived[i], N)      N = ROWS * serves
    serve (r,i): Acquire(arrived[i], 1)  Release(space[i], 1)

Every slot returns to its starting value at the end of a tile, so nothing needs rewriting between tiles at all.
Both locks peak at `N`, at most 32 and inside the 63 ceiling, and twelve locks for a six-slot window sit
comfortably in the MemTile's 64. What remains is per layer: the fill descriptors' lock values when `serves`
changes - one maskwrite each, since the acquire value (bits 8-14) and the release value (bits 24-30) share
word 7 - and the chain links when `chunks` changes.

| model | design | config ops | arm ms | total ms | vs flag-off |
|---|---|---|---|---|---|
| yolov8n | flag-off | 0 | 0.000 | 6.481 | |
| yolov8n | per-tile arming (today) | 25,392 | 3.682 | 10.312 | +3.832 |
| yolov8n | **arm-once** | 4,926 | 0.714 | 7.345 | **+0.864** |
| yolov8s | flag-off | 0 | 0.000 | 17.169 | |
| yolov8s | per-tile arming (today) | 42,619 | 6.180 | 21.497 | +4.328 |
| yolov8s | **arm-once** | 7,224 | 1.047 | 16.365 | **-0.804** |

A 5.1 ms swing on yolov8s and 3.0 ms on yolov8n: enough to turn yolov8s from a loss into a win, and to leave
yolov8n 0.86 ms short. The floor, if configuration were free, is +0.150 and -1.852 - the figures from before
arming was counted.

**The remainder is in tasks, and removing the re-arm is what unlocks it.** The ring issues 1,620 more tasks
than flag-off on yolov8n and 1,200 more on yolov8s - 0.940 and 0.696 ms - because its `serves == 1` path loses
drain merging. Merging was blocked by the drain barrier, and the barrier exists only to order re-arms. With no
re-arm there is no barrier and no reason not to merge, which would take yolov8n to about **-0.08 ms** and
yolov8s to about **-1.50 ms**.

**One precondition fails, and it is what to solve first.** A layer's shape is not constant across its tiles: on
yolov8s every one of the 66 ring layers has a column that arms two different widths, and on yolov8n 25 do -
shapes like `(6 slots)` alternating with `(2 slots)`. The cause is the multi-pass path, where a 32-chunk layer
served in passes of six leaves a remainder of two. That is why configuring "on change" costs exactly what
configuring "every layer" costs: the shape alternates rather than settling. It also means a chain would have to
be relinked while the layer is in flight, and a cycle can only be relinked when its channel is quiesced - a
layer barrier is a safe place for that, mid-layer is not, and nothing today proves the channel idle there.

**Which leaves three candidates, none of them built or costed further:** pad every pass to the full window so a
layer has exactly one shape, paying fills on the padding; or split a layer at its shape change and quiesce
there; or keep per-tile arming for the minority of layers that change shape and arm-once for the rest, which on
yolov8n is 25 layers of 66 and on yolov8s all of them - so that third option helps yolov8n and not yolov8s.

## Costing the padding, and the window that makes it unnecessary

**Padding costs more than the design it would enable.** A short pass cannot simply be left short: the cycle
walks all `capacity` slots, so every one must be filled and served each turn or `space[i]` is never released
and the ring stalls; and the core's trip count comes from its weight packet header rather than from how many
packets arrive, so a padding slot needs a real fill *and* a NOP weight packet, or the accumulator is wrong.

| model | padding slots | (tile, group) units | activation bytes | NOP weight bytes | derived |
|---|---|---|---|---|---|
| yolov8n | 3,479 | 779 | 89,062,400 | 32,953,088 | **6.571 ms** |
| yolov8s | 3,146 | 785 | 80,537,600 | 29,798,912 | **5.942 ms** |

That is against the 3.0 ms arm-once saves on yolov8n and the 5.1 ms it swings on yolov8s, and it does not count
core time for the NOP packets. The reason sits in the layer list: the worst cases are the *narrowest* layers -
a one- or two-chunk layer padded out to six wastes four or five slots, times hundreds of (tile, group) units -
and most layers are narrow. The cost scales with `window - chunks`, which is largest exactly where the layers
are smallest. Padding is not worth costing further.

**A window that divides the layer costs nothing and fixes the same problem completely.** The chain length is
configurable per layer; only variation *inside* a layer hurts. Choosing the largest divisor of the chunk count
that fits the window - 8 to 4, 16 to 4, 32 to 4, 9 to 3 - leaves every layer one width and pads nothing. Seven
layers narrow on yolov8n, seventeen on yolov8s.

And the variation is entirely in the width, which is the half that fixes for free:

| model | layer+column pairs arming | single shape | width varies | replay varies | both |
|---|---|---|---|---|---|
| yolov8n | 255 | 230 | 25 | **0** | 0 |
| yolov8s | 257 | 191 | 66 | **0** | 0 |

`serves` never varies within a layer and column on either model, so the fill descriptors' lock values are a
per-layer constant. After a divisor window **no layer+column pair needs a mid-layer rewrite**, and the
precondition that blocked arm-once is met.

**Measured: the divisor window costs 12 tasks on yolov8n and 96 on yolov8s, and not one byte.** Narrower passes
do merge slightly less well, and all of the cost lands in activation fills - 1,762 to 1,774, and 3,997 to
4,093 - while weight tasks, drain tasks and every byte total are unchanged to the byte. In time that is
+0.007 ms and +0.056 ms, against the 3.0 and 5.1 ms arm-once is worth: 0.2% and 1.1% of the benefit.

The measurement ran the real scheduler with `ring_plan` wrapped, rather than reimplementing `pass_fills`'
merging and measuring the reimplementation, and it carried its own guard: a wrapper that failed to take effect
would report a task delta of zero and read as good news, so width variation had to fall from 25 of 255 pairs
and 66 of 257 to zero. It did, on both models.

**Which closes the chain.** Arm-once with a divisor window, against the per-group schedule:

| model | flag-off | ring today | arm-once + divisor window | with drain merging too |
|---|---|---|---|---|
| yolov8n | 6.481 ms | +3.832 | +0.871 | **-0.069** |
| yolov8s | 17.169 ms | +4.328 | **-0.748** | **-1.444** |

So the design is worth roughly 1.4 ms on yolov8s and about break-even on yolov8n, and every step is now costed
rather than assumed. What remains is building it: cycle the descriptors instead of clearing `Use_Next_BD`, two
locks per slot, a per-layer window equal to the largest divisor of the chunk count, configuration moved from
the tile to the layer, and the drain barrier removed along with the re-arm it existed to order. None of that is
written, and all of these milliseconds are derived - the case for building rests on a comparator, so the first
build should be measured against flag-off on silicon before anything is claimed.

## Built, dispatched, and hung: a free-running cycle has no quiescent point

It was built (`bd5579b`, `bfe6480`), and on the instruction stream it does exactly what it was costed to do.
Ops counted from each container's `insts.bin`:

| container | insts.bin | ops | above its own flag-off |
|---|---|---|---|
| yolov8n flag-off | 430,180 | 12,258 | |
| yolov8n ring, per-tile arming | 1,371,652 | 45,521 | +33,263 |
| yolov8n ring, **arm-once** | 793,060 | 23,529 | **+11,271** |
| yolov8s flag-off | 1,019,140 | 28,791 | |
| yolov8s ring, per-tile arming (16 slots) | 1,797,780 | 58,388 | +29,597 |
| yolov8s ring, **arm-once** | 1,371,508 | 40,079 | **+11,288** |

The descriptors landed precisely where the parity arithmetic said they would: the arrival cycling 24-29 on
S2MM 5, the serves 0-5, 30-35, 6-11 and 36-41 on MM2S 0-3, every chain wrapping back to its own head, and the
output join keeping MM2S 4 with 12-23 and 42-45. The even half is exactly full, as predicted.

**Then the eight-layer probe timed out.** `ERT_CMD_STATE_TIMEOUT`, not `0xc01e0009`: the device granted a
context and the dispatch simply never finished. The device was healthy on both sides of it - the known-good
flag-off container verified 66/66 exact immediately before (7.366 ms mean over 20 dispatches) and immediately
after (7.314 ms) - so the hang belongs to the design, not to the machine.

**Why, and it is structural rather than a slip.** A descriptor acquires its lock *before* it transfers. The
arrival descriptor for the next slot has therefore already taken that slot's `space` and is sitting waiting for
stream bytes that will only arrive with the next layer's fills - so the chain is mid-flight at exactly the
moment a layer boundary looks idle. Retiring every shim task proves nothing about the tile side.
`_configure_ring` writes a `space[i]` the DMA has already consumed, crediting it twice, and relinks `next_bd`
on a chain the hardware is walking.

**The rule, stated generally so it is not rediscovered: a MemTile chain is quiescent only when something proves
its last transfer was consumed, and the only such proof this engine has ever had is a completed drain.** That
is what the reverse edge was, and it is what made arming per tile safe. Arm-once deletes the barrier, so it
deletes the only evidence of quiescence there is; the two cannot be had together. The obvious repairs all
collapse: a reset clears the run state and nothing but a queue push restarts it, and a pushed cycle is a task
that never completes so the queue never advances - which is per-tile arming again; one shape for the whole
model needs a fixed `ROWS * serves` and `serves` varies by layer; and reconfiguring only on change was measured
to cost what reconfiguring every layer costs, besides not touching the race.

**The first measured numbers on this branch, and they bear on everything above.** Flag-off on Desktop 2, one
sitting, 20 dispatches each: 7.366 ms and 7.314 ms mean, 66/66 layers exact both times. The derived comparator
for that same schedule is 6.481 ms, so **the model this investigation has used throughout runs about 13%
optimistic**. Every ring verdict derived at 26.8 GB/s and 145 ns per op therefore sits further negative than
written - yolov8s's -0.912 ms included - and nothing here should be quoted as a latency.

## Tested and refuted: the barrier was never missing, and crediting the parked head does not fix it

Two corrections to the section above, both from reading the emitter rather than reasoning about it.

**The drain barrier was never missing.** `run_column_programs` ends by retiring every channel of every column
until each queue is empty - the loop that raises `layer barrier: a channel's newest task carries no token` - and
it is called once per layer. So every layer boundary already awaits every outstanding drain, which is precisely
the proof that the cores consumed every slice the serves delivered. The quiescence the arm-once design was said
to lack is already there on the serve side, at layer granularity, for free. A per-layer barrier was therefore
not the fix, because it was never absent.

**What was left was the fill side, and that is not the whole story either.** A free-running chain always has a
current descriptor. If a descriptor takes its acquire when it becomes current rather than when stream bytes
arrive, then the head of the arrival cycle has already drawn its slot's `space` down and sits parked, and
writing the full count back to every slot hands the DMA tokens it is still holding - the arrival then runs a lap
ahead of the serves reading that slot. The window being a divisor of the tile makes every pass full width, so
the chain returns to its head after every tile and the parked slot is always slot 0: a deterministic correction,
not a guess. Crediting slot 0 zero instead of `ROWS * serves` was built - `insts.bin` sha `14b6eca6...` against
the hung build's `5ba90b00...`, so the stream really did change - and **it timed out in exactly the same way**.

So at least one of these holds, and nothing dispatched here distinguishes them: a descriptor acquires only when
data arrives; or a channel prefetches deeper than one descriptor; or relinking `next_bd` on a live cycle is
fatal by itself. A third variant was not dispatched - two timeouts on one design shape is the point to stop and
ask, not to keep feeding silicon guesses.

**The device is not the variable.** The known-good flag-off container verified 66/66 exact four times across
this sitting - 7.366, 7.314, 7.321 and 7.333 ms mean over 20 dispatches each - including immediately after both
timeouts.

### The lock semantics, settled from the sources, and a diagnostic nobody here has used

A descriptor acquires **before** it transfers, and this is no longer an inference from a hang. AMD's own
programming guide says it in so many words (`mlir-aie/programming_guide/section-2/section-2g/README.md`):
"Each BD says which buffer is being moved and how it synchronizes - *locks acquired before the transfer starts
and released after it completes*. BDs in a chain link to a `next` BD, forming a loop that keeps streaming as
long as the lock protocol permits." The S2MM status register corroborates it structurally: it carries two
*different* stall states, `Stalled_Lock_Acq` and `Stalled_Stream_Starvation`, which a channel that acquired
only on data arrival would not need.

The refinement that matters: whether the parked head is *holding* a token is decided by whether one was there.
Acquire succeeded, so the lock is already decremented and the channel parks on the stream
(`Stalled_Stream_Starvation`); or it blocked, so the lock is untouched and the channel parks on the acquire
(`Stalled_Lock_Acq`). For a producer-side lock at its replenished resting value - which is exactly `space` in
this ring - the first case holds and **the head has already drawn its slot down**. So the correction above was
the architecturally correct one and the ring hangs anyway, which moves the cause elsewhere rather than
vindicating it.

Two constants confirmed rather than assumed, and one trap: MemTile lock fields really are **word 7**
(`DMA_BD0_7` at `0x0001D01C`, and the `aie-rt` driver reads `BdWord[7]` for all five lock fields), so the
emitter's `RING_BD_LOCK_WORD` is right - but a **core** tile keeps them in word **5** (`DMA_BD0_5` at
`0x0001D014`), so this constant is correct only as long as the ring stays on the MemTile. And
`Task_Queue_Size` is **not** the chain's prefetch depth: it counts outstanding start-queue pushes, a different
mechanism from `Use_Next_BD` chaining, and a free-running cycle pushes nothing at all - so that counter says
nothing about this design. How deep a channel prefetches descriptors is **not** documented anywhere on this
machine; the status register exposing a single `Cur_BD` and a single `Stalled_Lock_Acq` bit is suggestive of
one at a time, but that is an argument from absence and is recorded as one.

**Register readback cannot be built - but the probe bisect turned out to be the instrument.** Read this
section as closing one route, not the goal: what actually localized the fault was building containers of 1, 2
and 8 layers and reading their *graded* output (bytes wrong, and whether the first or a repeat dispatch fails),
which needs no register access at all. Do not re-litigate the read path below; it is closed.

The idea was to read
`DMA_S2MM_Status_N` after a hang: it carries `Cur_BD`, bit 2 `Stalled_Lock_Acq`, bit 4
`Stalled_Stream_Starvation` and bits 1:0 as `00=IDLE, 01=STARTING, 10=RUNNING`. **There is no supported way to
read it from the host on Windows/XDNA.** The transaction format has no read opcode (`XAIE_IO_CUSTOM_OP_READ_REGS`
= 130 is defined but never executed and has no emitter); the AIEX dialect has no `npu.read32`, `npu.maskpoll`
or `npu.poll`; and **pyxrt does not export `read_aie_reg`** - zero occurrences in `pyxrt.pyd`, though XRT
declares it in `xrt_aie.h`. `XAIE_IO_MASKPOLL` (opcode 4) does exist, shape-identical to MASKWRITE, but it
returns no value, its encoding carries **no timeout field**, and what the XDNA firmware does with a poll that
never matches is undocumented - a wedge risk, not an instrument.

Two corrections worth keeping anyway, because a mis-mapped access corrupts a tile silently. **`0x1DF00` is the
CORE tile's status register, not the MemTile's.** The MemTile's are `DMA_S2MM_Status_N` at **`0xA0660 + 4N`**
(so channel 5 is `0xA0674`) and `DMA_MM2S_Status_N` at `0xA0680 + 4N`; the channel *control* registers use a
different stride, `0xA0600 + 8i` and `0xA0630 + 8i`. A MemTile has 6 channels each way, so S2MM 5 is the last
one. `Cur_BD` is **6 bits** on a MemTile (`29:24`, 48 descriptors) against 4 on a core tile, so it must be
masked with `0x3F`; and bit 4 is `Stalled_Stream_Starvation` only on S2MM - on MM2S it is
`Stalled_Stream_Backpressure`. The engine's `(col << 25) | (row << 20) | offset` is confirmed correct. MemTile
lock values are readable at `0xC0000 + 0x10*N`, if a read path ever exists.

## The parked head is real: layer 0 goes from corrupt to byte-exact

The instrument turned out to be the probe. A **one-layer** container is one `_configure_ring` and no
reconfiguration at all, and it does not hang - it completes in about 1 ms. What it does instead is compute
layer 0 *nearly* right:

| one-layer container | slot 0 credited | layer 0 |
|---|---|---|
| `probe1_armonce` (`8b33c5ec...`) | `ROWS * serves` | **MISMATCH 45,459 / 1,638,400 bytes** |
| `probe1_headparked` (`7283824...`) | `0` | **EXACT** |

One word of the instruction stream differs between them. At width 1 the over-credit is exactly one extra
fill's worth, so the arrival may run a single lap ahead of the serves and overwrite the slot while it is being
read - which is what 2.8% of bytes sporadically wrong looks like. Crediting the parked head nothing makes it
byte-exact. **So the free-running cycle is sound, the lock protocol is sound, and `RING_HEAD_PARKED` is
confirmed on silicon rather than argued from the architecture.**

That also explains why the eight-layer head-parked probe read as a flat failure: it carried both a correctness
bug and a hang at once. The hang is now bracketed between one layer (correct) and eight (hangs), and the layer
shapes split it - L0-L5 are all `serves 1` with only the *width* changing (1, 2, 1, 1, 2, 2), while L6 is the
first layer to change `serves`.

## The hang is not inside a frame: it is a frame that does not leave the ring re-runnable

A **two-layer** container - L0 at width 1, then L1 at width 2, the first reconfiguration there is - returns
**both layers byte-exact**, so relinking a live cycle's `next_bd` is not fatal and that candidate is closed.
It then times out on a **repeat** dispatch: the traceback is `verify_engine_container.py:104`, the timing loop,
not `:90`, the checked dispatch. The eight-layer container fails at `:90` instead - within the first frame. So
the failure point moves earlier as layers are added, which is the signature of residue that accumulates rather
than of a deadlock.

The mechanism is in `_configure_ring`: it restores locks `for i in range(width)`. A narrow layer relinks its
cycle over the first `width` slots and never touches the rest, so a wider layer before it leaves `space` credit
in slots the new cycle never consumes, and `arrived` is balanced only by serves that no longer run. Nothing in
a frame notices; the next frame starts from it. That predicts exactly what was measured - one layer at a fixed
width survives twenty dispatches, two layers survive one, eight do not survive the first - and the obvious fix
is to restore both locks of **every** slot of the compiled window, at 2 * slots writes a layer.

**That fix was built, and it changes nothing.** With every slot's `space` and `arrived` restored on every layer
- `insts.bin` grew 138,640 -> 140,656 B at two layers and 297,040 -> 304,912 B at eight, so the extra writes
really are in the stream - the two-layer container is still byte-exact on both layers of its first dispatch and
still times out at `verify_engine_container.py:104` on a repeat, at the same failure point. The
residue-in-untouched-slots explanation is refuted, and **what leaves the ring un-re-runnable across a frame
boundary is still unidentified**. Offline is unaffected: 21/21 tests pass with the change.

The device is not the variable, said once for the whole sitting: the known-good flag-off container verified
66/66 exact **six** times between these experiments - 7.366, 7.314, 7.321, 7.333, 7.227 and 7.293 ms mean over
20 dispatches each - including immediately after every timeout.

## The head is parked holding on the first frame and blocked on every later one

Filling in the last cell of the two-by-two settles it. Every container below is two layers, widths 1 then 2,
built from the same source but for the slot-0 credit and whether every slot is restored:

| slot 0 credit | all slots restored | first dispatch | repeat dispatches |
|---|---|---|---|
| `0` | no | L0, L1 **byte-exact** | **times out** at `verify:104` |
| `0` | yes | L0, L1 **byte-exact** | **times out** at `verify:104` |
| `ROWS * serves` | yes | L0 **MISMATCH 51,554**, L1 **MISMATCH 77,485** | **survives 20**, mean 1.305 ms |

So the two failures are not one bug with a single right answer: **crediting the head nothing is correct for the
first frame and fatal for the second, and crediting it fully is correct for every frame after the first and
wrong for the first.** The reason is that a descriptor's acquire either succeeds or blocks depending on whether
a token was there when it became current. Frame 1 starts from the configuration CDO's `aie.dma_start`, where
`space` is at its compiled initial value and the acquire is satisfied - the head is parked *holding*, so the
count it holds must not be handed back. Every later frame starts with the head parked *blocked*, holding
nothing, and zeroing its slot leaves it blocked forever. One absolute value written into a lock whose
held-or-blocked state cannot be observed from the host cannot be right for both.

Two things follow. A fix has to make the head's state at the start of a frame **deterministic** rather than
inferred - and `aiex.dma_channel_reset` exists in the dialect for exactly this shape of problem; mlir-aie's
`npu-xrt/local_reset` test describes its target as "a DMA channel stalled on a lock acquire with a BD queued
behind the lock", which is precisely the later-frame state here. And the ring's byte-exactness and its
re-runnability have now been demonstrated separately, on the same two layers, by containers differing in one
written word - what has never been demonstrated is both at once.

**Measured, and it is a width change rather than a second frame.** The one-layer container at a fixed width 1
(`probe1_headparked`, credit 0) returns layer 0 **EXACT and survives all twenty repeat dispatches**, mean
1.003 ms, no timeout. So with the width held constant, crediting the head nothing is correct *and*
re-runnable - the lock protocol closes on itself frame after frame, and the parked head is holding at the end
of a frame exactly as it is at `dma_start`.

Then the full picture, all with credit 0:

| container | widths | first dispatch | repeats |
|---|---|---|---|
| one layer | 1 | **EXACT** | **survives 20** |
| two layers | 1 then 2 | **both EXACT** | **times out** |

Note which transition is actually new at the boundary. *Inside* a frame the width only ever widens, 1 -> 2, and
both layers come back exact, so widening a live cycle is sound. The frame boundary is the only place the width
**narrows**, 2 -> 1, when the next frame's first layer reconfigures - and that is the one transition never
exercised within a frame. Narrowing is the suspect, not reconfiguration in general.

The cheap way to test that needs no new code looked like `--activation-ring 1`, which makes `_pass_width`
return 1 for every layer so a multi-layer container holds one width throughout and across frames. **That test
was run and it is confounded - do not read it as evidence about width changes.** An eight-layer window-1
container times out on its *first* dispatch (`verify:90`), before a single layer is compared. But forcing the
window to 1 changes a second thing at the same time: every layer whose chunk count exceeds 1 becomes
**multi-pass** - `passes 2` on three of the eight layers and `passes 4` on another - where every layer of the
window-6 builds ran `passes 1`. Two variables moved, so the result separates nothing. It does say that
something in multi-pass or in the later layers breaks the first frame outright, which is a different fault from
the frame-boundary one and is not yet localized.

A properly controlled version has to hold the width fixed **without** introducing passes - a container built
only from layers whose chunk count already equals the window, so the width is constant and every layer still
runs a single pass.

## Controlled and confirmed: a constant width survives, a width change does not

Selecting layers by shape gives that control. The selection cannot be "take these layers off the ring":
`ring_plan` returns a plan for every layer that moves packets, because the ring replaces the split ObjectFifo,
and a layer left on the per-group schedule would fetch activations nothing serves - a hang by construction,
indistinguishable from the fault. So whole layers are dropped from the schedule instead. The graph is
sequential, so a dropped predecessor leaves the next layer reading a region the frame never wrote: those layers
cannot match the reference, and for them the only valid signal is hang versus no-hang.

Six layers at **width 2, serves 1, single pass** - L1, L4, L5, L9, L11 and L26 - so that every one of the six
`_configure_ring` calls writes exactly the same values, within a frame and across frames:

| container | layers | widths | repeats |
|---|---|---|---|
| two-layer prefix | 2 | 1 then 2 | **times out** at `verify:104` |
| shape-selected | 6 | 2 throughout | **survives 20**, mean 1.271 ms |
| shape-selected | 6 | 1 throughout | **survives 20**, mean 1.752 ms, and **L0 EXACT** |

Both carry several layers and several reconfigurations per frame, so "more than one layer in a frame" is
exonerated. **The width change is the differentiator**, and with it held constant the ring runs frame after
frame at six layers just as it did at one. The width-1 selection begins at L0, the one selected layer whose
predecessor is also built, and that layer comes back byte-exact - so at a constant width the ring is exact and
re-runnable together, which is what no container had managed before.

## The fix: reset the channels, then push, so the head's state is known rather than guessed

The fault was never which value to write - it was that the right value depended on something unobservable. A
cyclic chain always has a current descriptor, and whether that descriptor is *holding* its acquire or *blocked*
on it cannot be read back from the host. `dma_channel_reset` removes the question: a reset drains the channel's
start queue and clears its run state, so afterwards nothing is current and no lock is held.

`_configure_ring` now does, per column and per layer, in this order - the order per-tile arming already proved
on silicon:

1. reset each of the ring's five channels (assert bit 1 of the channel control register, then deassert);
2. relink the cycles to this layer's width;
3. restore both locks of **every** slot - `space` full, `arrived` empty;
4. **push last**, one task per channel at the head descriptor.

A reset freezes the channel's bound lock counters, which is why the locks are restored after it rather than
before, and a cyclic chain never completes, so the single push is perpetual and its repeat count never comes
into play - exactly what the configuration CDO's `aie.dma_start` does at load.

**The reset inverts the head credit, which is the counterintuitive part.** Without it the head had already
taken its acquire before the lock write, so slot 0 had to be credited nothing. With it the head acquires *from*
the value written here, so crediting nothing strands it on an empty lock for ever - the same hang by the
opposite route. `RING_HEAD_PARKED` is therefore off; the finding it records still stands, and is precisely why
the state had to be made deterministic instead of guessed.

Measured on Device 0, on the two containers that previously failed:

| container | before | after |
|---|---|---|
| 2 layers, widths 1 -> 2 | both layers exact, **timed out on a repeat** (`verify:104`) | **both exact, survives 20**, mean 1.299 ms |
| 8 layers, widths 1/2/4 and a `serves` change at L6 | **timed out inside the first frame** (`verify:90`) | **all 8 exact, survives 20**, mean 2.992 ms |

## The full model runs, and on yolov8n it is slower than the schedule it replaces

**66/66 layers byte-exact, PASS** - the first completed full-model ring dispatch of this investigation. L13
`/model.5/conv/Conv`, which stalled every earlier attempt, is exact. The overflow that caused that stall is
structurally gone: per-slot locks bound the counter to `ROWS * serves`, and the built shapes peak at `serves 4`,
so 16 against the 63 a MemTile lock holds, where the shared counter reached 128. The container exercises
everything at once - widths 1 to 6, multi-pass layers at 8 and 16 chunks, `replicas` up to 4.

And it is a loss on this model, measured rather than derived:

| yolov8n container | dispatch mean over 20 | first |
|---|---|---|
| flag-off, immediately before | 7.308 ms | 8.892 ms |
| ring, six slots, with the reset | **10.052 ms** (min 9.889, max 10.454) | 11.556 ms |
| flag-off, immediately after | **7.392 ms** (min 7.184, max 7.758) | 8.823 ms |

The flag-off runs bracket the ring in one sitting, so the comparison is not against a remembered number: eight
flag-off verifies across the session land between 7.227 and 7.392 ms, every one of them 66/66 exact.

About **2.7 ms worse**, where the derived comparator said +0.871 ms. Two reasons, both known: the cost model
runs about 13% optimistic (its flag-off figure is 6.481 ms against 7.3 measured), and it never counted the
reset. The reset costs ten register writes, five pushes and `2 * slots` lock writes per layer per column, which
is why `insts.bin` goes 297,040 -> 317,712 B at eight layers and 951,580 B over the full model.

None of that makes the ring wrong - it makes yolov8n the wrong model for it. The design was never predicted to
win here: the costing put yolov8n at +0.871 ms and yolov8s at -0.748 ms, because yolov8s is where a fetched
tile is replayed enough to amortise. `activation_ring` is a per-container compile flag, so the question is
whether yolov8s now runs, and what it measures.

## It runs on yolov8s too, and it loses there as well - which settles the design

yolov8s is the model this was built for, and its container is **66/66 byte-exact** as well. Every number below
is a dispatch mean over 20, measured in one sitting, each container verifying 66/66 exact:

| model | flag-off | ring, six slots, with the reset | ring costs |
|---|---|---|---|
| yolov8n | **7.392 ms** | **10.052 ms** | **+2.66 ms** (+36%) |
| yolov8s | **16.828 ms** | **20.228 ms** | **+3.40 ms** (+20%) |

The derived costing said yolov8s would **win** by 0.748 ms. It loses by 3.40, so the comparator was wrong by
about 4.1 ms on the case it was most confident about, and in the direction that mattered. Two contributions are
known and were already written down - the model runs about 13% optimistic, and it never counted this reset -
but the larger lesson is the one this file has been circling since the first costing: **a DDR saving derived
from a schedule is not a latency measurement.** The ring's -38.3% derived traffic on yolov8s is real as
traffic; it does not convert into time at this task count.

So the ring is finished and it does not pay. What it leaves behind is worth more than the design: the protocol
is proven (a free-running cyclic MemTile chain, per-slot locks bounded by `ROWS * serves`, reset and re-push at
every layer), the parked-head semantics are documented and confirmed on silicon, and a whole class of derived
verdicts in this file now has a measured correction factor.

If anyone returns to it, the two levers not pulled are drain merging (worth a derived 0.940 ms on yolov8n and
0.696 on yolov8s, unblocked once the drain barrier went) and resetting only when the shape actually changes:
the first layer of a frame must reset unconditionally, since its predecessor in execution is the previous
frame's last layer, but every later layer could reset only when its shape differs from the one before it in
program order. Neither closes a 2.7-3.4 ms deficit.

The eight-layer container is the stronger correctness result: it changes the pass width *and* the lock values,
so both kinds of reconfiguration now survive. It costs stream: `insts.bin` grows 138,640 -> 143,856 B at two layers and
297,040 -> 317,712 B at eight, for ten reset writes, five pushes and `2 * slots` lock writes per layer per
column. Offline unaffected: 21/21 offline tests and 4/4 conv-engine tests pass.

The tool that does the selecting is `tools/ring_shape_probe.py`, and two things in it are worth keeping rather
than rediscovering. Layers are dropped from the **schedule**, never from the ring, for the reason above. And it
refuses to write a container unless the filter actually dropped layers: a filter that silently matched
everything would emit an ordinary ring build whose clean run would be read as a result about constant width,
which is the same trap that `yolov8n_ring8.ignite` set earlier in this file.

A hazard this narrows without closing, recorded so it is not rediscovered: `ensure` retires a channel once it
holds `queue_depth` tasks, and retiring means awaiting. A tile whose column owns five or more groups pushes
that many drains before its `S`, and such a drain cannot complete until that `S` runs. Over the schedules,
yolov8s has 196 ring episodes with four or more drains before their `S` (worst eight) and yolov8n has 28
(worst four, one below the depth). The barrier frees each tile's drains at the next re-arm, so far fewer tasks
are ever live, but the ordering itself is untouched deliberately: it needs `hold` semantics, and a helper that
changed those deadlocked the build once already. One variable per dispatch.

**Later, and measured:** this hazard does not fire on yolov8n. Holding those tasks left the emitted stream
bit-identical, which means `ensure` never retires one of them during that emit - see "Two mechanisms ruled
out" below. It remains a real shape for a schedule that queues five or more drains before an `S`, which is
yolov8s only, and yolov8s hits the four-deep queue first.
