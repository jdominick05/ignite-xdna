# The MemTile activation ring on silicon — what the build artifacts actually say

Desktop 2 (DESKTOP-CBL5NUA), 2026-09-15. Offline only: every fact here comes from reading build artifacts of
the probe containers and from mlir-aie / aie-rt sources. No dispatch established any of it.

## Method

`tools/disasm_txn.py` prints only five op categories and leaves most decoded ops unprinted, so its silence is
not evidence. Instead: scan every aligned 32-bit word of a binary and report those that look like a MemTile
register address — column in bits 25+, row bit `1 << 20` set. Scripts in this directory:
`scan_memtile_words.py` (BD and lock windows), `scan_memtile_channels.py` (the channel page),
`decode_cdo_bds.py` (a descriptor's eight configured words), `dump_channel_writes.py` (channel registers with
their values), `check_ring_items.py` (served vs unserved fills per layer), `ring_coverage.py` (how each layer
would be served), `gate_ring_exact.py` (per-layer byte-exactness against the direct reference).

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
merged weight runs. Scaled by the measured cost of a task that cancels most of yolov8n's byte saving. Recovering
that merging for the serves-1 path is the next schedule change.

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

## What a rebuild must show before any dispatch

Two reset maskwrites on each of `0xA0600`, `0xA0630`, `0xA0638`, `0xA0640`, `0xA0648` per column; a word-1
maskwrite clearing bit 19 on each of the five ring descriptors; iteration maskwrites on the same five; word-7
lock writes whose acquire field is negative; and pushes whose repeat count is one less than the executions
intended. The ring's descriptor ids must appear in the init CDO with the ring's own lengths, and no id may be
shared with another fifo.
