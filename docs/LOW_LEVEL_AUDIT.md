# Low-level audit — driver, AIE2 microcode, CDO synthesis, 4-D AGU descriptors, container, native ingress

Date: 2026-09-13, Desktop 2 (`DESKTOP-CBL5NUA`, Ryzen 7 8700G, Phoenix NPU Device 0 `[003d:00:01.1]`).

Scope (exclusively): `src/ignite_xdna/runtime/driver.py`, `kernels/aie2/dfl/dfl_decode.cc`,
`kernels/aie2/fused_conv_epilogue/`, `kernels/aie2/sppf/maxpool5x5.cc`,
`src/ignite_xdna/compiler/scheduler.py`, `src/ignite_xdna/compiler/memtile_agu.py`,
`src/ignite_xdna/compiler/serializer.py`, `src/ignite_xdna/pipelines/preprocess_simd.c`,
`src/ignite_xdna/c_api/ignite.cpp`. Findings in adjacent files are listed but not changed.

Evidence tags follow [`docs/SILICON.md`](SILICON.md): **MEASURED** on this machine,
**SPEC** from the AIE2 target model, **DERIVED** by arithmetic from a named artifact,
**NOT DETERMINED** where nothing here settles it. The regression gate is
[`tests/test_low_level_invariants.py`](../tests/test_low_level_invariants.py) (51 checks, no
device), run beside the existing suites; the silicon checks are in §11 and their witness is
[`results/aie/lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log`](../results/aie/lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log).

## Read this first — the two behaviour changes

1. **`ignite-compile` now refuses the current 63-layer flatten.**
   `emit_unified_monolithic_transaction_bundle` flattens every stage into one resident
   parameter set per core and places layer *k* at core-tile offset `0x400 + k·0x1000`.
   Core data memory is 64 KB, so 16 sets fit and set 16 already starts outside it (§1.1).
   The shipped `build/yolov8n.ignite` (63 layers in 9 stages) was built this way; rebuilding
   it is refused with a message naming the stages until parameters are staged per stage.
   The shipped container still loads and runs (§11).
2. **Per-stage init/exec streams gain 8 lock WRITEs each.** The Lock 4/5 initialisation
   branch in `emit_multi_layer_transaction_bundle` compared a 20-bit register offset against
   an absolute address and could never fire (§1.2). Now that it does, a recompiled stage
   exec is 357 ops instead of 349 (+8 × 24 bytes) and a recompiled chained stream would be
   3,353 ops instead of 3,281. No BD in the shipped template acquires lock 4 or 5, so the
   writes are inert for that template; re-run the end-to-end benchmark before quoting a
   number from a recompiled container.

## Summary

| # | Finding | Where | Status |
|---|---|---|---|
| 1.1 | Flattened parameter windows leave core data memory; shipped init writes 2,256 blocks onto memory-module, program-memory and core-module addresses | `scheduler.py:540` | FIXED (refuses), silicon-checked (§11) |
| 1.2 | Lock 4/5 init branch dead: `reg == 0x1C0020` after `& 0xFFFFF` | `scheduler.py:683,751` | FIXED (byte delta above) |
| 1.3 | `import ignite_xdna` fails outside the repo root / when a `tools` package shadows `tools/` | `scheduler.py:132` | FIXED (own parser) |
| 1.4 | No framing/address-window validation of emitted streams | `scheduler.py:221` | ADDED, wired into all emitters |
| 1.5 | Stripped TCTs leave no inter-stage wait; lock WRITEs are not barriers; exec = template ×9 | `scheduler.py:844–946` | DOCUMENTED, stream unchanged |
| 2.1 | Zero step encodes as a one-word step; "in-flight 2× upsample" BD reads a shifted window | `memtile_agu.py:380,604` | FIXED (rejected; 4-pass plan, exact test) |
| 2.2 | Plan defaults allowed a bank to spill into the next; unaligned buffer bases accepted | `memtile_agu.py:47,75` | HARDENED |
| 2.3 | Channel/BD ownership documented as 0..3/4..5; the template shows even/odd | `memtile_agu.py:64` | FIXED (docstring + validator) |
| 3.1 | DFL reciprocal 32768/32784 wraps through `int16_t`; expectation sign flips | `dfl_decode.cc:135` | FIXED, bit-exact model + Peano compile |
| 3.2 | Score egress BD reads from record offset 0, not 16 | `dfl_stage.py:212` (adjacent) | DOCUMENTED, not changed |
| 4 | Fused conv epilogue: acc32 bound, SRS rounding, lock handshake | `conv2d_fused.cc` | VERIFIED, no change |
| 5 | SPPF 5×5 max-pool: border fills, bounds, register reuse | `maxpool5x5.cc` | VERIFIED, no change |
| 6.1 | Manifest re-layout not iterated to a fixed point; a digit-growth case overruns the pad | `serializer.py:200` | FIXED, stress + constructed case |
| 6.2 | Reader trusted header/directory offsets; no per-blob check | `serializer.py:329,369` | HARDENED |
| 7 | `x_tab[1024]` is guarded, not overrun; missing stride check; no AVX2 code exists | `preprocess_simd.c:80,128` | HARDENED; OpenCV parity MEASURED exact |
| 8.1 | Lost wake-up in `stop_worker` (flag stored outside the waiters' mutexes) | `ignite.cpp:277` | FIXED, exercised by every native exit (§11) |
| 8.2 | `exec_bytes` never parsed: the non-monolithic path ran nothing | `ignite.cpp:1190,1252` | FIXED (fail closed) |
| 8.3 | No CRC / size / blob-range check before programming the device | `ignite.cpp:1076` | FIXED, refusal MEASURED (§11) |
| 8.4 | Ticket high-water mark could regress; timings/thresholds data races; dispatch states unchecked | `ignite.cpp:1331,277` | FIXED |
| 8.5 | 8,192 of the 1,228,800 preprocessed bytes reach the device; non-fused decode never reads `bo_out` | `ignite.cpp:1168,762` | DOCUMENTED |
| 9 | Harness had no release path; no `group_id` masking is needed | `driver.py:59` | HARDENED (`close()`); masking declined with reason |

## 1. Transaction emitter — `src/ignite_xdna/compiler/scheduler.py`

### 1.1 Parameter windows leave core data memory (FIXED, fail closed)

The single-layer template's core program reads shift-cut, bias and weights at
`0x37C`, `0x380` and `0x400` of core data memory. `MemTileMultiPassScheduler.schedule`
assigns layer *k* the window `0x37C + k·0x1000 … 0x400 + k·0x1000 + 2304`
(`param_l1_offset`). Core data memory is `0x00000..0x0FFFF` (SPEC: mlir-aie target model;
corroborated by the template, which resets and enables cores through `Core_Control` at
`0x32000`, programs core BDs at `0x1D000` and core locks at `0x1F000`). The last byte of
window *k* is `0xD00 + k·0x1000`, so windows 0–15 fit and window 16 starts at `0x1037C`
(DERIVED). `max_resident_l1_layers()` returns 16 from those constants.

`MultiStageSchedulePlan.to_schedule_plan` (`scheduler.py:339`) concatenates every stage's
layers before the unified init is emitted. The shipped `build/yolov8n.ignite`
(manifest: 9 stages, Stem 7 + P3 7 + P4 7 + P5 6 + Neck_FPN 8 + Neck_PAN 10 + Detect 6+6+6
= 63 layers; header CRC `0x2e06f112`, 5,315,840 bytes, 20 blobs) therefore carries an
`init_monolithic.bin` of 3,381 ops = 349 template ops + 16 cores × 63 layers × 3 BLOCKWRITEs
+ 8 DDR patches (DERIVED; every count reproduced by `parse_transaction_stream`). Of the
3,024 parameter BLOCKWRITEs, 768 land in data memory (windows 0–15), 1,344 on
memory-module addresses `0x10000..0x1CFFF` (windows 16–43), 192 on program memory
`0x20000..0x23FFF` (windows 32–35) and 720 on core-module register space `≥ 0x30000`
(windows 48–62) (DERIVED; `validate_transaction_stream` rejects the blob with exactly
2,256 violations). What the firmware does with those writes is settled only as far as §11
goes: the stream dispatches to completion and the layer-0 egress is byte-identical to the
egress after a 7-layer init.

Fix: `validate_l1_parameter_layout` (`scheduler.py:540`) computes every window from the
packed sizes, requires them disjoint and inside data memory, and is called by
`emit_multi_layer_transaction_bundle` and — with the stage breakdown in the message — by
`emit_unified_monolithic_transaction_bundle` (`scheduler.py:974`). Stage-wise emission of
≤ 16 layers is unaffected.

### 1.2 Dead lock-initialisation branch (FIXED)

Both stream builders compute `reg = addr & 0xFFFFF` and compared it with `0x1C0020`,
which has bit 20 set and cannot survive the mask. MemTile lock *n* is at `0xC0000 +
0x10·n` (SPEC; the template writes locks 0–3 at `0xC0000..0xC0030` and lock 2 gets the
value 4). `memtile_lock_reg()` / `memtile_lock_write()` now build both the match and the
absolute addresses; the prologue/barrier bytes are unchanged (the absolute form
`(col<<25)|(1<<20)|0x1C0020` equals `(col<<25)|(1<<20)|0xC0020`), and
`test_chained_stream_is_byte_identical_to_legacy_builder` pins that. `tools/disasm_txn.py`'s
audit uses the same dead constant in its "MEMTILE LOCK INJECTIONS" section (adjacent, not
changed).

### 1.3 Packaging (FIXED)

`from tools.disasm_txn import disassemble_transaction` made `import ignite_xdna` depend on
the repo root being on `sys.path` and on no other `tools` package being importable; in
`resnet_env17` a regular `tools` package shadows the repo's namespace directory, so the
installed copy failed to import at all (MEASURED). `parse_transaction_stream`
(`scheduler.py:132`) decodes the same fields (`test_parser_matches_disasm_field_by_field`),
is bounded by the header's op count instead of the buffer length — the tool decodes the
chained stream's 32-byte zero pad as a WRITE and raises — and reports truncation and unknown
opcodes as `ValueError`.

### 1.4 Stream validation (ADDED)

`validate_transaction_stream` (`scheduler.py:221`) checks the header size against the
buffer, 4-byte op alignment, known opcodes, in-buffer op sizes, an all-zero tail shorter
than 64 bytes, at least one TCT (a terminal one for the emitter's own streams), tiles
inside five columns × six rows, DDR patches on shim BD registers, and BLOCKWRITE payloads on
data memory or BD registers only. Every emitter validates before writing. On the real
artifacts (MEASURED): the template exec (349 ops) and init (909 ops), the shipped
`exec_monolithic.bin` (3,281 ops, 32-byte pad, one TCT) and every DFL/SPPF `insts.bin`
under `build/` pass; the shipped `init_monolithic.bin` is rejected as in §1.1. One SPPF
trace build ends with event-register writes after its final TCT, which is why the terminal
requirement is opt-in.

### 1.5 Documented, stream unchanged

- **Each stage's exec is the single-layer template.** All nine shipped stage exec blobs are
  byte-identical to `build/layer_conv0_exec.bin` (MEASURED), and
  `chain_stage_transaction_streams` on those nine blobs reproduces the shipped
  `exec_monolithic.bin` byte for byte (MEASURED). The template's DMA program references
  MemTile BDs 0/1/2/24/25/26 and locks 0–3 only; no BD reads or writes `L2_BANK_0`
  (`0x40000`) or `L2_BANK_1` (`0x60000`), and no op selects which parameter window a core
  uses. The ping/pong bank assignments in `PassDescriptor` are therefore bookkeeping that
  the emitted stream does not realise.
- **TCT stripping removes the only inter-stage completion wait** (`scheduler.py:924`). The
  format's opcodes are WRITE, BLOCKWRITE, MASKWRITE, DDR_PATCH and TCT; there is no
  acquire. The "barrier" and prologue ops are lock-value WRITEs (`scheduler.py:879,892`)
  that set values from the instruction stream without waiting for a DMA that holds the
  lock. A chained frame is one TCT at the very end.
- **Prologue/barrier set Lock 5 = 1 while init and per-stage exec set Lock 5 = 0**
  (`LOCK_L2_PONG`, "initial val = 0: idle"). Inert for the shipped template (no BD uses
  lock 5); it is an inconsistency to resolve when a design does.
- **Lock zeroing is skipped only for value 0** (`scheduler.py:742`): core locks with a
  non-zero initial value are re-armed every frame, locks whose initial value is 0 are reset
  by the init stream only. A frame that times out can leave one non-zero.
- **The 64-byte zero pad sits inside the declared size** (`scheduler.py:942`). The shipped
  stream carries 32 such bytes and completes (§11); the validator accepts a pad shorter
  than 64 bytes and rejects any other tail.
- **`fuse_dfl`** routes the standalone DFL transaction (`build/dfl_decode/*/insts.bin`,
  52 ops, 12 DDR patches on kernel arguments 0/1/2) into a runner that binds two BOs and a
  runtime that expects scores at `bo_out + 134,400` (`ignite.cpp:1172`). Argument 2 is
  unbound and argument 0 is a host read the fused design no longer has. NOT DETERMINED on
  silicon: the DFL transport itself has not completed a multi-core dispatch
  (`results/aie/dfl_decode_phoenix_transport_checkpoint_20260913T0950Z.log`).
- **`tests/test_multi_layer_scheduler.py:198,255`** assert that the silicon egress of the
  "3-layer" and "4-layer" runs equals the *layer-0* exact reference; N-layer execution on
  silicon is not evidenced by those tests, consistent with the exec stream above.

## 2. MemTile AGU — `src/ignite_xdna/compiler/memtile_agu.py`

### 2.1 Zero steps and the 2× upsample (FIXED)

The module encodes step fields minus one (SPEC, its own docstring: aie-rt
`_XAieMl_MemTileDmaWriteBd`). `synthesize` accepted `step=0` and emitted field 0, which the
hardware reads as a one-word step. `upsample_2x_nearest_bd` relied on `steps=(4, 0,
channels, 0)` with wraps of 2 to duplicate each pixel and row; what it programmed was a
window shifted by one word per repeat (DERIVED from the encoding). A word cannot be
replicated by the MemTile AGU: no dimension, and not the iteration step either, can have a
stride of zero.

`synthesize` now rejects a zero step on any dimension of size > 1 and a zero iteration step
with a count > 1 (`memtile_agu.py:380,385`). `plan_upsample_2x` (`memtile_agu.py:604`)
lowers 2× nearest-neighbour as four passes: four identical linear MM2S reads of the source
and four S2MM scatters into output phase `(row parity, column parity)` with strides of two
pixels (D1) and two output rows (D2). `MemTileBD.iter_word_offsets` (`memtile_agu.py:170`)
enumerates the words a descriptor moves; `test_upsample_2x_is_exact_nearest_neighbour`
streams a random 4×6×8 map through the descriptors and matches
`np.repeat(np.repeat(x, 2, 0), 2, 1)` exactly, and every output word of the 20×20×64 and
40×40×32 plans is written exactly once. `tests/test_compiler_fusion.py::test_07` asserted
the impossible encoding and was rewritten to the exact-coverage form.

### 2.2 Bank bounds and buffer alignment (HARDENED)

`plan_lateral_concat` and `plan_detect_head_egress` defaulted to whole-MemTile bounds, so a
P4 head (1,600 px × 144 ch = 230,400 B) or a 1,600-px lateral concat at `0x40000` ran
through `0x50000` toward Bank 1 without an error. Bounds now default to the 64 KB bank
containing the base (`bank_bounds`, `memtile_agu.py:47`); explicit bounds keep the old
behaviour available. Buffer *bases* handed to the planners must be 64-byte aligned
(`_cacheline_aligned`, `memtile_agu.py:75`). Alignment is not enforced on every DMA
address on purpose: the BD address field is a 32-bit word address (SPEC), and a C2f slice
at `+32` bytes or a detect-head class branch at `+64` is legal and required; a 64-byte rule
on all addresses would reject correct descriptors. Interleaved concat chunks have
overlapping address spans and disjoint word sets, so `assert_disjoint_destinations`
(`memtile_agu.py:895`) checks by word, not by span; lateral concat, detect-head egress and
channel slicing are verified exact that way.

### 2.3 Channel/BD ownership (FIXED)

`queue_word` documented "channels 0..3 use BD 0..23 and channels 4..5 use BD 24..47". The
template pushes S2MM 0 → BD 0, S2MM 1 → BD 24, S2MM 3 → BD 25, MM2S 0 → BD 3 and
MM2S 1 → BD 26 (DERIVED from its start-queue writes at `0xA0604..0xA063C`): even channels
own BD 0..23, odd channels own BD 24..47. `validate_channel_bd` (`memtile_agu.py:64`)
encodes that rule and `queue_word(bd_id, channel=…)` applies it.

## 3. DFL decode — `kernels/aie2/dfl/dfl_decode.cc`

### 3.1 Reciprocal wraps through `int16_t` (FIXED)

`dfl_expectation` normalises with `reciprocal_fixed(sum >> 4, 26)` and broadcasts
`static_cast<int16_t>(reciprocal)`. The maximum bin always contributes exactly 32767
(z = 0, residual 0), so `sum ≥ 32767` and `sum >> 4 ≥ 2047`; `2^26 // 2047 = 32784` and
`2^26 // 2048 = 32768` exceed `int16_t` and wrap to −32752 / −32768 (DERIVED). The
maximum bin's probability then becomes ≈ −1.0 and the expectation is −k instead of k for
maximum bin k, i.e. a box coordinate error of 2·k·stride, up to 960 px on the P5 scale.
Reachable from INT8 Q4 input whenever the other fifteen bins are ≥ 167 Q4 units (10.4)
below the maximum (sum = 32767) or the second bin is 11–14 halvings down (sum
32768..32783); `test_second_overflow_denominator_reachable` finds such an input by search.

Fix: clamp the reciprocal to `kQ15One` before the cast (`dfl_decode.cc:135`), which costs
1/32768 of scale; the same clamp guards `sigmoid_q15` (`dfl_decode.cc:175`), where
`exp_q15[0] ≥ 10` for every INT8 magnitude keeps `2^30 / (32768 + exp)` below the limit
today (DERIVED, checked exhaustively over −128..127).

Verification: a bit-exact Python model of the kernel arithmetic — round-to-nearest-even
Q15 multiplies, saturating adds, the restoring division, and the `int16_t` wrap — shows the
sign flip before the clamp for every lone-maximum lane and tracks the float softmax
expectation to < 0.06 after it on 3,000 random anchors; the clamp changes no non-overflow
input. Peano (`clang version 22.0.0`, `aie2`, `-O3`) compiles the fixed kernel to the same
vector profile as the baseline — 14 `vmul`, 14 `vadd`, 13 SRS — in 1,024 instructions
against 1,030, 6,552-byte objects both (MEASURED, no device). The oracle in
`tests/test_dfl_kernel.py::_fixed_expectation` never modelled the cast and now differs from
the kernel by 1/32768 in the wrapped case; add the clamp there (adjacent, not changed).

### 3.2 Verified without change

- The exponent argument is `max − logit ≥ 0` for every INT8 pair; "+16.0" is not
  representable in Q4 INT8 (maximum 7.9375). The polynomial's partial sums stay below
  saturation for residuals in `(−ln 2, 0]`.
- All-negative logits: the maximum lane still yields 32767, so the sum is never 0.
- 8,400 anchors = 16 cores × 35 chunks × 15 anchors, no tail; a chunk that straddles
  6,400 or 8,000 computes geometry per anchor. Vector loads read 64-byte-aligned locals.
- Adjacent: `dfl_stage.py:212` programs the score egress BD at `offset = 0` with
  `sizes = [525, 320]`, i.e. bytes 0..319 of each 336-byte record — the four box floats and
  76 class scores — where the record layout `[x1, y1, x2, y2, score_0..79]` puts the scores
  at byte 16. NOT DETERMINED on silicon (transport unqualified); the one-token fix is
  `offset = 16`.

## 4. Fused conv epilogue — `kernels/aie2/fused_conv_epilogue/` (VERIFIED)

- Accumulator: `|bias| + 9·Cin·127·128 + 128·16 < 2^31` holds for Cin ≤ 14,600 (DERIVED);
  the residual enters as `Acc(v, 4)` (×16) before any INT8 clamp, matching the README's
  scale contract.
- SRS: `aie::rounding_mode::conv_even` (round to nearest, ties to even) and
  `saturation_mode::saturate` (asymmetric, [−128, 127]) are set on entry; the intermediate
  `to_vector<int16_t>(0)` clamp at ±128.0 Q8 is beyond the ±8.0 range where SiLU saturates
  the INT8 output, as the README states. The offline oracle checks every Q8 input within
  one LSB of float32 SiLU (MEASURED: `--offline --cin 32` pass, max error 1 LSB).
- Locks: the core acquires output-free (selector 48) once, C++ releases output-ready
  (49) after each pair of 32-byte stores behind `sched_barrier`, four BDs consume one credit
  each and only the last returns the free credit; streamed input acquires 51 per panel and
  releases 50 after the panel's reads. Credits balance; no cycle.

## 5. SPPF max-pool — `kernels/aie2/sppf/maxpool5x5.cc` (VERIFIED)

`shuffle_down_fill(prev, cur, 32/48)` and `shuffle_down_fill(cur, next, 16/32)` yield
pixels p−2, p−1, p+1, p+2 with `−128` filled at both row ends; loads stay inside the
6,400-byte input (`y·320 + 320` maximum) and the vertical pass reads rows y−2..y+2 with
`−128` beyond rows 0 and 19. Five live 512-bit vectors plus the fill constant. The
offline oracle (direct 25-tap integer max, five fixtures) passes (MEASURED).

## 6. Container — `src/ignite_xdna/compiler/serializer.py`

### 6.1 Layout fixed point (FIXED)

The blob directory lives inside the manifest and every blob offset depends on the
manifest's padded size, which depends on the digits of those offsets. `write` re-laid out
once; when the shifted offsets gained a digit the final manifest could exceed its pad, the
pad went negative and was skipped, and the manifest overran the first blob (DERIVED;
`test_layout_fixed_point_covers_digit_growth` constructs such a case by search and
round-trips it). `_plan_layout` (`serializer.py:200`) iterates with a monotonically growing
pad until stable and asserts a non-negative pad.

### 6.2 Reader validation (HARDENED)

At load the reader now requires `total_file_size == len(buffer)`, the manifest range inside
the buffer, unique 64-byte-aligned blob entries inside the file (`serializer.py:329`), and
format version 1 with a known arch id; `verify_blob` (`serializer.py:369`) checks a blob
against its directory CRC. The body CRC deliberately keeps its version-1 definition (offset
64 to EOF); the header's own fields are validated structurally instead. `add_blob` rejects
duplicate and empty names. On the shipped container: checksum and all 20 blob CRCs verify
(MEASURED).

## 7. Ingress preprocessor — `src/ignite_xdna/pipelines/preprocess_simd.c`

- `x_tab[1024]` was guarded (`nw > 1024` returned −2) and `nw ≤ dst_w`, so source widths
  above 1024 px never overran it; the guard limited the *target* width. A heap table now
  serves targets wider than 1024 px (`preprocess_simd.c:128`, verified at 1280×1280).
- `src_stride < src_w·3` read each row's tail from the next row; it is rejected with −3
  (`preprocess_simd.c:80`).
- There are no AVX2 intrinsics in this file (scalar loops under OpenMP), so the aligned /
  unaligned-load question does not arise here; the AVX2 code is in `ignite.cpp`.
- The header's "bit-exact with OpenCV INTER_LINEAR" claim was measured, not assumed: a
  640×480 identity resize and a 1920×1080 → 640×360 letterbox give 0 mismatching bytes out
  of 921,600 and 691,200 against `cv2.resize` (MEASURED, `test_report_opencv_bilinear_agreement`).
- The tracked `preprocess_simd.dll` was rebuilt from the edited source with the recipe in
  `preprocess.py::_compile_simd_dll` (`cl.exe /O2 /fp:fast /openmp /LD`). Adjacent: that
  helper runs `cl.exe` inside the source directory and writes `/Fe:<name>` there, so it only
  succeeds for the in-tree DLL path and leaves `.obj/.exp/.lib` beside the source.

## 8. Native runtime — `src/ignite_xdna/c_api/ignite.cpp`

- **8.1 Lost wake-up (FIXED).** `stop_worker` stored `worker_stop` without holding the
  queue/result mutexes before notifying; a worker between its predicate check and its wait
  missed the notification and `ignite_free` hung. The flag is now stored under all three
  mutexes (`ignite.cpp:277`).
- **8.2 `exec_bytes` (FIXED).** `StageResource.ninstr_exec` was never read from the
  manifest, so `bo_exec` was never created and the non-monolithic path dispatched nothing
  while reporting success. It is parsed (`ignite.cpp:1190`), a manifest count larger than
  its blob is refused, and a container with neither a monolithic nor any stage exec stream
  fails to load (`ignite.cpp:1252`).
- **8.3 Container checks (FIXED).** Header size, `total_file_size` against the mapping,
  body CRC-32 (`ignite.cpp:1076`) and every blob range/alignment are checked before any
  `bo.write` reads the mapping.
- **8.4 Concurrency (FIXED).** `last_completed_ticket` is a monotonic high-water mark;
  `ignite_wait` and the slot gate key on the ticket's own ring entry
  (`ignite.cpp:1331`), so out-of-order completion by several producers cannot free an
  in-flight slot or deliver the wrong entry. `last_timings` is read under `result_mutex`,
  `conf_thres`/`iou_thres` are atomics, init and warm-up dispatch states are checked and a
  per-frame run that does not complete is counted and reported instead of yielding stale
  output silently.
- **8.5 Observations (unchanged).** `in_bytes` is 8,192 for 16 cores (`ignite.cpp:1168`)
  while the preprocessed CHW frame is 3 × 640 × 640 = 1,228,800 bytes; `ignite_run_async`
  copies the first 8,192 bytes to `bo_in`. The Python session marshals input the same way
  (`session.py:627–637`). In the non-fused path `decode_detections_for_slot` returns without
  reading `bo_out` when no reference heads are loaded and otherwise decodes
  `ref_p3_box … ref_p5_cls` (`ignite.cpp:762`) — the `bus_heads.bin` tensors loaded at
  `ignite_load`, which do not depend on the frame. `bo_out` is read only by the
  `fused_dfl` path (`ignite.cpp:652`).
- **`LockFreeRingBuffer`** is not in `ignite.cpp`; it is in `tools/ignite_run_native.cpp`
  (reviewed, unchanged): a single-producer/single-consumer ring with `push` loading `tail`
  acquire and storing `head` release, `pop` symmetric, and a relaxed advisory `size()`. The
  ordering is correct for one producer and one consumer.

## 9. XRT harness — `src/ignite_xdna/runtime/driver.py`

- `kernel.group_id(arg)` is the memory-bank index connected to a kernel argument and is
  passed through unmasked. No context-slot bits are encoded in it; masking it with `0x0F`
  would have no basis and could break the zero-copy bank binding Windows requires
  ([`docs/DECISIONS.md`](DECISIONS.md), "KDMA is unsupported on Windows"). NTSTATUS
  `0xc01e0009` is the sixth-context failure at `hw_context` creation
  (`results/aie/windows_context_switch_bench.log`), unrelated to BO flags.
- Column confinement is a property of the loaded xclbin's partition (`4x4`), not of the
  driver wrapper, which programs no column masks.
- MemTile locks cannot be cleared by the host outside a transaction stream; the
  prologue of each chained frame rewrites locks 2, 4, 5, 6 and 7 (§1.5), which is the
  only lock sanitisation there is. What the harness lacked was a release path:
  `close()` (`driver.py:59`) drops kernel, hardware context, xclbin and device in that
  order and the class is a context manager; no finalizer is installed. Exercised on
  silicon in §11, where two harnesses were opened and closed in one process.

## 10. Verification performed

Run: `pytest tests/test_low_level_invariants.py` (51 passed) and
`tests/test_compiler_fusion.py` (12 passed, with `models/yolov8n_cut_xint8.onnx` present)
in `resnet_env17`; `python -m compileall -q src/ignite_xdna benchmarks tools tests`
(exit 0); `tests/test_dfl_kernel.py --offline`, `tests/test_sppf_kernel.py --offline` and
`tests/test_fused_epilogue_kernel.py --offline --cin 32` (all pass); Peano compile of the
DFL kernel; MSVC 2022 build of `libignite_xdna` and `ignite-run` (no warnings); DLL rebuild.

Not run: the DFL kernel on silicon (its transport is unqualified); any recompiled
container (refused by §1.1); the multi-layer scheduler's silicon tests.

## 11. Silicon checks (MEASURED, Device 0, no other hardware context running)

Witness: [`results/aie/lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log`](../results/aie/lowlevel_audit_silicon_checks_phoenix_20260913T1655Z.log).

1. **Out-of-window init versus in-window init.** With the shipped `im2col_4d_16core.xclbin`
   and one random 8,192-byte input, the 7-layer `stage_stem_init.bin` (289,088 B) and the
   63-layer `init_monolithic.bin` (2,514,752 B) were each dispatched once
   (`ERT_CMD_STATE_COMPLETED`, 1.93 ms and 7.51 ms) followed by three `exec` dispatches
   (all completed). The 4,096-byte egress is byte-identical between the two runs (same
   SHA-256, 2,012 non-zero bytes). The 2,256 out-of-window writes therefore change nothing
   observable in this layer-0 output under this template; whether the firmware drops them
   or they land harmlessly is NOT DETERMINED. The advisory comparison with
   `run_n_layer_fixed_point_reference([stem conv 0])` reads 5.5 % bit agreement — that
   reference assumes 8 input channels and the stem conv has 3, so it is not an oracle for
   this layer; the repo's silicon parity tests use synthetic 8-channel models.
2. **Rebuilt native runtime.** `ignite-run.exe` built from this tree loads the shipped
   container (CRC verified at load), decodes 5 objects on `assets/bus.jpg` and exits
   cleanly: single frame NPU 0.564 ms, glass-to-glass 1.827 ms; 500 asynchronous frames
   after 20 warm-up at 2,176.68 FPS aggregate, glass-to-glass mean 1.322 ms / P95 1.396 /
   P99 1.514, NPU 0.447 ms, ingress 0.381 ms, DFL+NMS 0.108 ms, zero incomplete dispatches.
   A copy with one flipped byte (`Container CRC32 mismatch: header 0x2e06f112, body
   0x39a591bf`) and a copy truncated by 64 bytes (`declares 5315840 bytes but the file
   holds 5315776`) are refused at load before the device is touched.

## 12. Follow-ups (not done here)

- Stage parameters per stage, or give the core program a window selector, before the
  63-layer container can be rebuilt; then re-measure the end-to-end benchmark.
- Decide the inter-stage synchronisation: keep a TCT per stage, or make the MemTile DMA
  chains lock-driven end to end, so that the "barrier" writes stop racing in-flight DMAs.
- `dfl_stage.py`: score egress `offset = 16`; add the reciprocal clamp to the test oracle.
- `tools/disasm_txn.py`: bound the walk by the op count and fix the `0x1C0020` audit constant.
- `preprocess.py::_compile_simd_dll`: build in a scratch directory.
- Feed the whole preprocessed frame (or document that 8,192 bytes is the intended ABI of
  the single-layer template) and read `bo_out` in the non-fused decode path.
