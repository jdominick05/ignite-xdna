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
