"""Phoenix MemTile DMA parameter compiler (no device or vendor imports).

Units at this boundary are bytes, except ``sizes`` and zero-padding counts,
which are 32-bit words / dimension trips. D0 is innermost. Register layout:
Xilinx/aie-rt, driver/src/global/xaiemlgbl_params.h, MEM_TILE_MODULE_DMA_BD0;
encoding: driver/src/dma/xaie_dma_aieml.c, _XAieMl_MemTileDmaWriteBd.
Steps and iteration wrap are encoded minus one; D0..D2 wraps are NOT.
Iteration advances between BD executions, so callers must submit repeat_count
executions. Setting Iteration_Wrap alone does not transfer the fifth dimension.
"""

from dataclasses import dataclass, replace
from math import prod
from numbers import Integral
from typing import Iterator


MEMTILE_BYTES = 512 * 1024
LOCAL_MEMORY_BASE = 0x80000
LOCAL_LOCK_BASE = 64
BD_BASE = 0xA0000
BD_STRIDE = 0x20

# Buffer *bases* handed to the planners are kept on a 64-byte boundary. This is
# a floorplan discipline (host BOs, instruction buffers and container blobs are
# all 64-byte aligned), not an AGU requirement: the BD address field is a
# 32-bit word address, and intra-buffer channel offsets such as a C2f slice at
# +32 bytes are legal and required.
CACHELINE_BYTES = 64

# L2 floorplan shared with the scheduler: two 64 KB activation banks inside the
# 512 KB MemTile. The bank size is a floorplan choice, not a silicon limit.
L2_BANK_BYTES = 0x10000
L2_BANK_0_OFFSET = 0x40000
L2_BANK_1_OFFSET = 0x60000

# Start-queue channel/BD ownership on the Phoenix MemTile: even channels own
# BD 0..23, odd channels own BD 24..47. The single-layer template
# (build/layer_conv0_exec.bin) pushes S2MM 0 -> BD 0, S2MM 1 -> BD 24,
# S2MM 3 -> BD 25, MM2S 0 -> BD 3 and MM2S 1 -> BD 26, which is this rule and
# not a "channels 0..3 / 4..5" split.
MEMTILE_CHANNELS = 6
EVEN_CHANNEL_BD_RANGE = (0, 23)
ODD_CHANNEL_BD_RANGE = (24, 47)


def bank_bounds(base_address: int) -> tuple[int, int]:
    """Bounds of the 64 KB L2 bank that contains ``base_address``.

    Addresses outside both banks get the whole MemTile, which is the
    pre-existing behaviour for scratch regions such as 0x0 or 0x10000.
    """
    for bank in (L2_BANK_0_OFFSET, L2_BANK_1_OFFSET):
        if bank <= base_address < bank + L2_BANK_BYTES:
            return (bank, bank + L2_BANK_BYTES)
    return (0, MEMTILE_BYTES)


def bd_range_for_channel(channel: int) -> tuple[int, int]:
    channel = _integer("channel", channel, 0, MEMTILE_CHANNELS - 1)
    return EVEN_CHANNEL_BD_RANGE if channel % 2 == 0 else ODD_CHANNEL_BD_RANGE


def validate_channel_bd(channel: int, bd_id: int) -> int:
    """Raise unless ``bd_id`` may be queued on ``channel``; return ``bd_id``."""
    lo, hi = bd_range_for_channel(channel)
    bd_id = _integer("bd_id", bd_id, 0, 47)
    if not lo <= bd_id <= hi:
        raise ValueError(
            f"BD {bd_id} cannot be queued on MemTile channel {channel}: "
            f"{'even' if channel % 2 == 0 else 'odd'} channels own BD {lo}..{hi}")
    return bd_id


def _cacheline_aligned(name: str, address: int) -> int:
    address = _integer(name, address, 0, MEMTILE_BYTES - 4)
    if address % CACHELINE_BYTES:
        raise ValueError(f"{name} must be {CACHELINE_BYTES}-byte aligned, got {address:#x}")
    return address


def _integer(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")
    return value


def _words(name: str, value: int, maximum: int = MEMTILE_BYTES) -> int:
    value = _integer(name, value, 0, maximum)
    if value % 4:
        raise ValueError(f"{name} must be aligned to four bytes")
    return value // 4


@dataclass(frozen=True)
class LockConfig:
    """Local lock IDs; negative acquire values encode AcquireGreaterEqual."""

    acquire_id: int = 0
    acquire_value: int = 0
    acquire_enable: bool = False
    release_id: int = 0
    release_value: int = 0

    def word(self) -> int:
        acq = _integer("acquire_id", self.acquire_id, 0, 63) + LOCAL_LOCK_BASE
        rel = _integer("release_id", self.release_id, 0, 63) + LOCAL_LOCK_BASE
        av = _integer("acquire_value", self.acquire_value, -64, 63) & 127
        rv = _integer("release_value", self.release_value, -64, 63) & 127
        return (1 << 31) | (rv << 24) | (rel << 16) | (
            int(self.acquire_enable) << 15) | (av << 8) | acq


@dataclass(frozen=True)
class MemTileBD:
    """Eight raw registers plus the queue repetitions needed to execute them.

    ``logical_sizes`` / ``logical_steps`` (words) / ``iteration_step_words`` are
    the pre-encoding traversal; ``synthesize`` fills them so coverage can be
    checked exactly without decoding the minus-one register fields.
    """

    registers: tuple[int, ...]
    repeat_count: int
    memory_bytes: int
    transfer_bytes: int
    address_span: tuple[int, int]  # local byte offsets, end exclusive
    direction: str
    logical_sizes: tuple[int, ...] = ()
    logical_steps: tuple[int, ...] = ()
    iteration_step_words: int = 0

    @property
    def steps(self) -> tuple[int, ...]:
        """Raw D0..D3 Step fields (word strides minus one)."""
        return tuple(self.registers[i] & 0x1FFFF for i in (2, 3, 4, 5))

    @property
    def wraps(self) -> tuple[int, ...]:
        return tuple((self.registers[i] >> 17) & 1023 for i in (2, 3, 4))

    @property
    def sizes(self) -> tuple[int, ...]:
        """D0..D3 sizes; decoded from the registers when the logical dims are absent."""
        if self.logical_sizes:
            return tuple(self.logical_sizes)
        w0_2 = tuple((self.registers[i] >> 17) & 1023 for i in (2, 3, 4))
        denom = w0_2[0] * w0_2[1] * w0_2[2]
        w3 = self.registers[0] // denom if denom > 0 else 1
        return w0_2 + (w3,)

    @property
    def step_words(self) -> tuple[int, ...]:
        """D0..D3 strides in 32-bit words as the hardware applies them (field + 1).

        A requested zero step on a size-1 dimension is reported as the one
        word the minus-one encoding produces; it never advances anyway.
        """
        if self.logical_steps:
            return tuple(max(step, 1) for step in self.logical_steps)
        return tuple(field + 1 for field in self.steps)

    @property
    def iteration_wrap(self) -> int:
        return (self.registers[6] >> 17) & 63

    def iter_word_offsets(self) -> Iterator[int]:
        """Yield the local byte offset of every 32-bit word this BD moves, in order.

        Zero-insertion words are not memory accesses and are not yielded, so
        the number of offsets is ``memory_bytes // 4``. Intended for exact
        coverage checks of small descriptors.
        """
        sizes = self.sizes
        steps = self.step_words
        istep = self.iteration_step_words if self.logical_sizes else (self.registers[6] & 0x1FFFF) + 1
        base = self.address_span[0]
        for iteration in range(self.repeat_count):
            origin = base + 4 * iteration * istep
            for i3 in range(sizes[3]):
                for i2 in range(sizes[2]):
                    for i1 in range(sizes[1]):
                        row = origin + 4 * (i1 * steps[1] + i2 * steps[2] + i3 * steps[3])
                        for i0 in range(sizes[0]):
                            yield row + 4 * i0 * steps[0]

    def with_locks(self, locks: LockConfig) -> "MemTileBD":
        return replace(self, registers=self.registers[:7] + (locks.word(),))

    def register_writes(self, bd_id: int) -> tuple[tuple[int, int], ...]:
        bd_id = _integer("bd_id", bd_id, 0, 47)
        return tuple((BD_BASE + bd_id * BD_STRIDE + i * 4, word)
                     for i, word in enumerate(self.registers))

    def queue_word(self, bd_id: int, channel: int | None = None) -> int:
        """Start-queue payload; caller selects the channel/direction register.

        Pass ``channel`` to enforce BD ownership: even channels own BD 0..23,
        odd channels own BD 24..47 (see ``validate_channel_bd``).
        """
        bd_id = _integer("bd_id", bd_id, 0, 47)
        if channel is not None:
            validate_channel_bd(channel, bd_id)
        return bd_id | ((self.repeat_count - 1) << 16)


@dataclass(frozen=True)
class ReceptiveFieldRegion:
    output_origin: tuple[int, int]
    output_shape: tuple[int, int]
    bd: MemTileBD


@dataclass(frozen=True)
class ReceptiveFieldPlan:
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    storage_channels: int
    kernel_shape: tuple[int, int]
    regions: tuple[ReceptiveFieldRegion, ...]

    @property
    def transfer_bytes(self) -> int:
        return sum(r.bd.transfer_bytes for r in self.regions)

    @property
    def logical_bytes(self) -> int:
        return prod(self.output_shape) * prod(self.kernel_shape)


@dataclass(frozen=True)
class ChannelSlicePlan:
    """Descriptor for zero-copy channel slicing in MemTile SRAM."""
    channel_offset: int
    num_channels: int
    total_channels: int
    spatial_pixels: int
    bd: MemTileBD


@dataclass(frozen=True)
class ChannelConcatPlan:
    """Descriptor for zero-copy strided channel concatenation in MemTile SRAM."""
    chunk_offsets: tuple[int, ...]
    chunk_channels: tuple[int, ...]
    total_channels: int
    spatial_pixels: int
    bds: tuple[MemTileBD, ...]


@dataclass(frozen=True)
class Upsample2xPlan:
    """In-flight spatial 2x nearest-neighbour upsampling as four MemTile DMA passes.

    The AGU cannot replicate a word: every step field is encoded minus one, so
    the smallest expressible stride is one word and a ``step=0, wrap=2``
    duplication does not exist on this hardware. Each pass streams the whole
    source once (``source_bds``) and scatters it into one output phase
    (row parity, column parity) with 2-pixel / 2-row destination strides
    (``dest_bds``). Every output word is written exactly once and every input
    word is read exactly four times; no core instruction is involved.
    """
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    base_address: int
    destination_address: int
    transfer_bytes: int
    memory_bytes: int
    source_bds: tuple[MemTileBD, ...]
    dest_bds: tuple[MemTileBD, ...]

    @property
    def passes(self) -> int:
        return len(self.dest_bds)

    @property
    def total_output_bytes(self) -> int:
        return self.transfer_bytes

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return True


@dataclass(frozen=True)
class LateralConcatPlan:
    """Descriptor for zero-copy lateral feature map concatenation in MemTile SRAM."""
    spatial_pixels: int
    chunk_channels: tuple[int, ...]
    total_channels: int
    base_address: int
    bds: tuple[MemTileBD, ...]

    @property
    def buffer_descriptors(self) -> tuple[MemTileBD, ...]:
        return self.bds

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return True


@dataclass(frozen=True)
class DetectHeadEgressPlan:
    """Descriptor for direct S2MM DMA egress of prediction heads to host buffers."""
    scale_name: str  # "P3", "P4", "P5"
    spatial_pixels: int
    box_channels: int
    cls_channels: int
    base_address: int
    box_bd: MemTileBD
    cls_bd: MemTileBD

    @property
    def total_channels(self) -> int:
        return self.box_channels + self.cls_channels

    @property
    def total_output_bytes(self) -> int:
        return self.spatial_pixels * self.total_channels

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return True


@dataclass(frozen=True)
class C2fRoutingPlan:
    """Hardware routing plan for lowering C2f Split and Concat into MemTile BDs."""
    stage_name: str
    spatial_pixels: int
    in_channels: int
    hidden_channels: int
    num_bottlenecks: int
    concat_channels: int
    out_channels: int
    ping_buffer_addr: int
    pong_buffer_addr: int
    cv1_bd: MemTileBD
    slice_0_bd: MemTileBD       # Identity branch
    slice_1_bd: MemTileBD       # Bottleneck branch
    bottleneck_bds: tuple[tuple[MemTileBD, MemTileBD], ...]
    concat_bds: tuple[MemTileBD, ...]
    cv2_bd: MemTileBD

    @property
    def has_zero_intermediate_ddr_traffic(self) -> bool:
        return True


class MemTileAGU:
    """Synthesize local MemTile BDs and HWC INT8 receptive-field traversals."""

    def synthesize(
        self, *, base_address: int, sizes: tuple[int, int, int, int],
        steps: tuple[int, int, int, int], iteration_count: int = 1,
        iteration_step: int = 4, zero_before: tuple[int, int, int] = (0, 0, 0),
        zero_after: tuple[int, int, int] = (0, 0, 0), direction: str = "MM2S",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> MemTileBD:
        if len(sizes) != 4 or len(steps) != 4:
            raise ValueError("exactly four sizes and steps are required (D0 first)")
        if len(zero_before) != 3 or len(zero_after) != 3:
            raise ValueError("padding has exactly three dimensions")
        if direction not in ("MM2S", "S2MM"):
            raise ValueError("direction must be MM2S or S2MM")
        sizes = tuple(_integer(f"D{i} size", s, 1, 1023 if i < 3 else 131071)
                      for i, s in enumerate(sizes))
        strides = tuple(_words(f"D{i} step", s) for i, s in enumerate(steps))
        # Step fields are stored minus one, so a zero step would silently
        # become a one-word step. A zero is only harmless on a dimension that
        # never advances (size 1).
        for i, (size, stride) in enumerate(zip(sizes, strides)):
            if stride == 0 and size > 1:
                raise ValueError(
                    f"D{i} step of 0 is not encodable (fields are step-1); "
                    f"a size-{size} dimension needs a step of at least one word")
        count = _integer("iteration_count", iteration_count, 1, 64)
        istep = _words("iteration_step", iteration_step)
        if istep == 0 and count > 1:
            raise ValueError("iteration_step of 0 is not encodable when iteration_count > 1")
        before = tuple(_integer(f"D{i} zero_before", n, 0, (63, 31, 15)[i])
                       for i, n in enumerate(zero_before))
        after = tuple(_integer(f"D{i} zero_after", n, 0, (63, 31, 15)[i])
                       for i, n in enumerate(zero_after))
        if direction == "S2MM" and any(before + after):
            raise ValueError("zero insertion is supported only by MM2S")
        address = _words("base_address", base_address, MEMTILE_BYTES - 4)
        lo, hi = buffer_bounds
        _words("buffer start", lo)
        _words("buffer end", hi)
        end = base_address + sum((s - 1) * st * 4 for s, st in zip(sizes, strides))
        end += (count - 1) * istep * 4 + 4
        if not 0 <= lo <= base_address < end <= hi <= MEMTILE_BYTES:
            raise ValueError(f"DMA address span [{base_address:#x}, {end:#x}) exceeds buffer bounds")
        padded = tuple(sizes[i] + before[i] + after[i] for i in range(3))
        length = prod(padded) * sizes[3]
        _integer("buffer_length words per execution", length, 1, 131071)
        regs = (
            length,
            (LOCAL_MEMORY_BASE // 4 + address) | (before[0] << 26),
            max(0, strides[0] - 1) | (sizes[0] << 17),
            max(0, strides[1] - 1) | (sizes[1] << 17) | (before[1] << 27),
            max(0, strides[2] - 1) | (sizes[2] << 17) | (before[2] << 27),
            max(0, strides[3] - 1) | (after[0] << 17) | (after[1] << 23) | (after[2] << 28),
            max(0, istep - 1) | ((count - 1) << 17),
            locks.word(),
        )
        return MemTileBD(regs, count, prod(sizes) * count * 4,
                         length * count * 4, (base_address, end), direction,
                         logical_sizes=sizes, logical_steps=strides,
                         iteration_step_words=istep)

    def receptive_fields(
        self, input_shape: tuple[int, int, int], *, kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1, padding: int | tuple[int, int, int, int] = 0,
        base_address: int = 0, storage_channels: int | None = None,
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
    ) -> ReceptiveFieldPlan:
        """Compile HWC windows; border regions carry their output coordinates.

        The caller stages HWC with each pixel padded to ``storage_channels``
        (a multiple of four). Channel tails are explicit physical lanes, never
        silently truncated. MM2S inserts spatial zeros; no host spatial padding
        or stride-2 subsampling is required. Regions may use different padding
        fields, so consume/scatter each region using its output coordinates.
        Large tensors must first be tiled by the scheduler to fit local SRAM.
        """
        h, w, c = shape3(input_shape)
        kh, kw = pair(kernel_size, "kernel_size")
        sy, sx = pair(stride, "stride", maximum=2)
        pt, pl, pb, pr = pads4(padding)
        oh, ow = (h + pt + pb - kh) // sy + 1, (w + pl + pr - kw) // sx + 1
        if min(oh, ow) <= 0:
            raise ValueError("kernel does not overlap the input")
        pitch = ((c + 3) // 4) * 4 if storage_channels is None else storage_channels
        _words("storage_channels", pitch)
        if pitch < c:
            raise ValueError("storage_channels cannot truncate logical channels")
        if not buffer_bounds[0] <= base_address < base_address + h * w * pitch <= buffer_bounds[1]:
            raise ValueError("input storage exceeds buffer bounds; tile before compiling")
        regions = []
        for y, nh, top, bottom in _axis_regions(h, oh, kh, sy, pt, 64):
            for x, nw, left, right in _axis_regions(w, ow, kw, sx, pl, 131071):
                # A BD's padding is constant per region. Only true border
                # windows have padding; fully padded windows are rejected.
                valid_h, valid_w = kh - top - bottom, kw - left - right
                if min(valid_h, valid_w) <= 0:
                    raise ValueError("fully padded windows are not supported")
                row_words = (pitch // 4) * kh * kw
                max_width = 131071 // row_words
                if max_width < 1:
                    raise ValueError("one receptive field exceeds BD transfer length")
                for xx in range(x, x + nw, max_width):
                    width = min(max_width, x + nw - xx)
                    addr = base_address + ((y * sy - pt + top) * w + xx * sx - pl + left) * pitch
                    bd = self.synthesize(
                        base_address=addr, sizes=(pitch // 4, valid_w, valid_h, width),
                        steps=(4, pitch, w * pitch, sx * pitch), iteration_count=nh,
                        iteration_step=sy * w * pitch, zero_before=(0, left, top),
                        zero_after=(0, right, bottom), buffer_bounds=buffer_bounds,
                    )
                    regions.append(ReceptiveFieldRegion((y, xx), (nh, width), bd))
        return ReceptiveFieldPlan((h, w, c), (oh, ow, c), pitch, (kh, kw), tuple(regions))

    def channel_slice_bd(
        self,
        *,
        base_address: int,
        spatial_pixels: int,
        total_channels: int,
        channel_offset: int,
        num_channels: int,
        direction: str = "MM2S",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> MemTileBD:
        """Synthesize 4D MemTile BD for zero-copy slicing of channels [channel_offset, channel_offset + num_channels).

        ``base_address`` is a buffer base (64-byte aligned); ``channel_offset``
        is a word-granular offset inside it, which the AGU addresses directly.
        """
        _cacheline_aligned("base_address", base_address)
        _words("channel_offset", channel_offset)
        _words("num_channels", num_channels)
        _words("total_channels", total_channels)
        if channel_offset + num_channels > total_channels:
            raise ValueError(f"Slice [{channel_offset}, {channel_offset + num_channels}) exceeds total channels {total_channels}")
        if spatial_pixels <= 0:
            raise ValueError("spatial_pixels must be positive")

        addr = base_address + channel_offset
        if spatial_pixels <= 1023:
            sizes = (num_channels // 4, spatial_pixels, 1, 1)
            steps = (4, total_channels, 4, 4)
        else:
            sizes = (num_channels // 4, 1, 1, spatial_pixels)
            steps = (4, 4, 4, total_channels)
        return self.synthesize(
            base_address=addr,
            sizes=sizes,
            steps=steps,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

    def channel_concat_bds(
        self,
        *,
        base_address: int,
        spatial_pixels: int,
        chunk_channels: tuple[int, ...],
        direction: str = "S2MM",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> tuple[MemTileBD, ...]:
        """Synthesize 4D MemTile BDs for scattering multiple incoming channel chunks into an interleaved destination buffer.

        Chunk k lands on channels [sum(chunk_channels[:k]), +chunk_channels[k])
        of every pixel; the chunks' word sets are pairwise disjoint and together
        cover exactly ``spatial_pixels * sum(chunk_channels)`` bytes.
        """
        _cacheline_aligned("base_address", base_address)
        total_channels = sum(chunk_channels)
        _words("total_channels", total_channels)
        bds = []
        cur_offset = 0
        for ch in chunk_channels:
            _words("chunk channel", ch)
            if spatial_pixels <= 1023:
                sizes = (ch // 4, spatial_pixels, 1, 1)
                steps = (4, total_channels, 4, 4)
            else:
                sizes = (ch // 4, 1, 1, spatial_pixels)
                steps = (4, 4, 4, total_channels)
            bd = self.synthesize(
                base_address=base_address + cur_offset,
                sizes=sizes,
                steps=steps,
                direction=direction,
                buffer_bounds=buffer_bounds,
                locks=locks,
            )
            bds.append(bd)
            cur_offset += ch
        return tuple(bds)

    def upsample_2x_source_bd(
        self,
        *,
        base_address: int,
        height: int,
        width: int,
        channels: int,
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> MemTileBD:
        """One linear MM2S read of the whole [height][width][channels] source."""
        wc = _words("channels", channels)
        return self.synthesize(
            base_address=base_address,
            sizes=(wc, width, height, 1),
            steps=(4, channels, width * channels, 4),
            direction="MM2S",
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

    def upsample_2x_scatter_bd(
        self,
        *,
        destination_address: int,
        height: int,
        width: int,
        channels: int,
        phase: tuple[int, int],
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> MemTileBD:
        """S2MM scatter of one linear source pass into output phase (row parity, column parity).

        Source pixel (r, c) lands on output pixel (2r + phase[0], 2c + phase[1]):
        D1 steps two output pixels, D2 steps two output rows.
        """
        if len(phase) != 2 or any(p not in (0, 1) for p in phase):
            raise ValueError("phase must be (row parity, column parity) with entries 0 or 1")
        wc = _words("channels", channels)
        output_row_bytes = 2 * width * channels
        base = destination_address + phase[0] * output_row_bytes + phase[1] * channels
        return self.synthesize(
            base_address=base,
            sizes=(wc, width, height, 1),
            steps=(4, 2 * channels, 2 * output_row_bytes, 4),
            direction="S2MM",
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

    def plan_upsample_2x(
        self,
        *,
        input_shape: tuple[int, int, int],
        base_address: int,
        destination_address: int,
        source_bounds: tuple[int, int] | None = None,
        destination_bounds: tuple[int, int] | None = None,
        locks: LockConfig = LockConfig(),
    ) -> Upsample2xPlan:
        """Four-pass in-flight 2x nearest-neighbour upsampling (see ``Upsample2xPlan``).

        Bounds default to the 64 KB bank containing each base address.
        """
        h, w, c = shape3(input_shape)
        _cacheline_aligned("base_address", base_address)
        _cacheline_aligned("destination_address", destination_address)
        source_bounds = bank_bounds(base_address) if source_bounds is None else source_bounds
        destination_bounds = (bank_bounds(destination_address)
                              if destination_bounds is None else destination_bounds)
        phases = ((0, 0), (0, 1), (1, 0), (1, 1))
        source_bds = tuple(
            self.upsample_2x_source_bd(
                base_address=base_address, height=h, width=w, channels=c,
                buffer_bounds=source_bounds, locks=locks)
            for _ in phases)
        dest_bds = tuple(
            self.upsample_2x_scatter_bd(
                destination_address=destination_address, height=h, width=w,
                channels=c, phase=phase, buffer_bounds=destination_bounds, locks=locks)
            for phase in phases)
        out_shape = (2 * h, 2 * w, c)
        return Upsample2xPlan(
            input_shape=(h, w, c),
            output_shape=out_shape,
            base_address=base_address,
            destination_address=destination_address,
            transfer_bytes=prod(out_shape),
            memory_bytes=prod(input_shape),
            source_bds=source_bds,
            dest_bds=dest_bds,
        )

    def plan_lateral_concat(
        self,
        *,
        base_address: int,
        spatial_pixels: int,
        chunk_channels: tuple[int, ...],
        direction: str = "S2MM",
        buffer_bounds: tuple[int, int] | None = None,
        locks: LockConfig = LockConfig(),
    ) -> LateralConcatPlan:
        """Synthesize strided S2MM DMA scatter plans for lateral feature map skip connections.

        Bounds default to the 64 KB bank containing ``base_address``, so a
        concatenation that would run past its bank fails here instead of
        overlapping the other bank.
        """
        if buffer_bounds is None:
            buffer_bounds = bank_bounds(base_address)
        bds = self.channel_concat_bds(
            base_address=base_address,
            spatial_pixels=spatial_pixels,
            chunk_channels=chunk_channels,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )
        return LateralConcatPlan(
            spatial_pixels=spatial_pixels,
            chunk_channels=chunk_channels,
            total_channels=sum(chunk_channels),
            base_address=base_address,
            bds=bds,
        )

    def plan_detect_head_egress(
        self,
        *,
        scale_name: str,
        spatial_pixels: int,
        box_channels: int = 64,
        cls_channels: int = 80,
        base_address: int = L2_BANK_0_OFFSET,
        direction: str = "S2MM",
        buffer_bounds: tuple[int, int] | None = None,
        locks: LockConfig = LockConfig(),
    ) -> DetectHeadEgressPlan:
        """Synthesize direct S2MM DMA egress descriptors for box and cls prediction heads.

        Bounds default to the 64 KB bank containing ``base_address``.
        """
        _cacheline_aligned("base_address", base_address)
        if buffer_bounds is None:
            buffer_bounds = bank_bounds(base_address)
        total_ch = box_channels + cls_channels
        _words("total_channels", total_ch)
        _words("box_channels", box_channels)
        _words("cls_channels", cls_channels)
        w_box = box_channels // 4
        w_cls = cls_channels // 4

        if spatial_pixels <= 1023:
            box_sizes = (w_box, spatial_pixels, 1, 1)
            box_steps = (4, total_ch, 4, 4)
            cls_sizes = (w_cls, spatial_pixels, 1, 1)
            cls_steps = (4, total_ch, 4, 4)
        else:
            box_sizes = (w_box, 1, 1, spatial_pixels)
            box_steps = (4, 4, 4, total_ch)
            cls_sizes = (w_cls, 1, 1, spatial_pixels)
            cls_steps = (4, 4, 4, total_ch)

        # Box branch: writes box_channels bytes per pixel
        box_bd = self.synthesize(
            base_address=base_address,
            sizes=box_sizes,
            steps=box_steps,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

        # Cls branch: writes cls_channels bytes per pixel with offset
        cls_addr = base_address + box_channels
        cls_bd = self.synthesize(
            base_address=cls_addr,
            sizes=cls_sizes,
            steps=cls_steps,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

        return DetectHeadEgressPlan(
            scale_name=scale_name,
            spatial_pixels=spatial_pixels,
            box_channels=box_channels,
            cls_channels=cls_channels,
            base_address=base_address,
            box_bd=box_bd,
            cls_bd=cls_bd,
        )

    def route_c2f_stage(
        self,
        *,
        stage_name: str,
        spatial_pixels: int,
        in_channels: int,
        hidden_channels: int,
        num_bottlenecks: int,
        ping_buffer_addr: int = L2_BANK_0_OFFSET,
        pong_buffer_addr: int = L2_BANK_1_OFFSET,
        ping_bank_bounds: tuple[int, int] = (L2_BANK_0_OFFSET, L2_BANK_0_OFFSET + L2_BANK_BYTES),
        pong_bank_bounds: tuple[int, int] = (L2_BANK_1_OFFSET, L2_BANK_1_OFFSET + L2_BANK_BYTES),
    ) -> C2fRoutingPlan:
        """Constructs full zero-copy MemTile routing plan for a YOLOv8 C2f block."""
        _cacheline_aligned("ping_buffer_addr", ping_buffer_addr)
        _cacheline_aligned("pong_buffer_addr", pong_buffer_addr)
        # cv1: in_channels -> 2 * hidden_channels (writes to ping buffer)
        cv1_out_ch = 2 * hidden_channels
        cv1_bd = self.synthesize(
            base_address=ping_buffer_addr,
            sizes=(cv1_out_ch // 4, spatial_pixels, 1, 1),
            steps=(4, cv1_out_ch, 4, 4),
            direction="S2MM",
            buffer_bounds=ping_bank_bounds,
            locks=LockConfig(acquire_id=4, acquire_value=1, acquire_enable=True, release_id=4, release_value=0),
        )

        # Slice 0 (bypass): channel offset 0, num_channels = hidden_channels
        slice_0_bd = self.channel_slice_bd(
            base_address=ping_buffer_addr,
            spatial_pixels=spatial_pixels,
            total_channels=cv1_out_ch,
            channel_offset=0,
            num_channels=hidden_channels,
            direction="MM2S",
            buffer_bounds=ping_bank_bounds,
            locks=LockConfig(acquire_id=4, acquire_value=0, acquire_enable=True, release_id=4, release_value=0),
        )

        # Slice 1 (bottleneck path): channel offset hidden_channels, num_channels = hidden_channels
        slice_1_bd = self.channel_slice_bd(
            base_address=ping_buffer_addr,
            spatial_pixels=spatial_pixels,
            total_channels=cv1_out_ch,
            channel_offset=hidden_channels,
            num_channels=hidden_channels,
            direction="MM2S",
            buffer_bounds=ping_bank_bounds,
            locks=LockConfig(acquire_id=4, acquire_value=0, acquire_enable=True, release_id=4, release_value=0),
        )

        # Bottlenecks: each takes hidden_channels -> hidden_channels
        bottleneck_bds = []
        for b_idx in range(num_bottlenecks):
            b_in_addr = ping_buffer_addr if b_idx == 0 else pong_buffer_addr
            b_out_addr = pong_buffer_addr if b_idx == 0 else ping_buffer_addr
            in_bounds = ping_bank_bounds if b_idx == 0 else pong_bank_bounds
            out_bounds = pong_bank_bounds if b_idx == 0 else ping_bank_bounds

            b_in_bd = self.synthesize(
                base_address=b_in_addr,
                sizes=(hidden_channels // 4, spatial_pixels, 1, 1),
                steps=(4, hidden_channels, 4, 4),
                direction="MM2S",
                buffer_bounds=in_bounds,
            )
            b_out_bd = self.synthesize(
                base_address=b_out_addr,
                sizes=(hidden_channels // 4, spatial_pixels, 1, 1),
                steps=(4, hidden_channels, 4, 4),
                direction="S2MM",
                buffer_bounds=out_bounds,
            )
            bottleneck_bds.append((b_in_bd, b_out_bd))

        # Concat: total slices = (2 + num_bottlenecks) chunks of hidden_channels
        concat_chunks = (hidden_channels,) * (2 + num_bottlenecks)
        concat_total_ch = sum(concat_chunks)
        concat_bds = self.channel_concat_bds(
            base_address=pong_buffer_addr,
            spatial_pixels=spatial_pixels,
            chunk_channels=concat_chunks,
            direction="S2MM",
            buffer_bounds=pong_bank_bounds,
            locks=LockConfig(acquire_id=5, acquire_value=1, acquire_enable=True, release_id=5, release_value=0),
        )

        # cv2: reads concat_total_ch from pong buffer
        cv2_bd = self.synthesize(
            base_address=pong_buffer_addr,
            sizes=(concat_total_ch // 4, spatial_pixels, 1, 1),
            steps=(4, concat_total_ch, 4, 4),
            direction="MM2S",
            buffer_bounds=pong_bank_bounds,
            locks=LockConfig(acquire_id=5, acquire_value=0, acquire_enable=True, release_id=5, release_value=0),
        )

        return C2fRoutingPlan(
            stage_name=stage_name,
            spatial_pixels=spatial_pixels,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_bottlenecks=num_bottlenecks,
            concat_channels=concat_total_ch,
            out_channels=in_channels,
            ping_buffer_addr=ping_buffer_addr,
            pong_buffer_addr=pong_buffer_addr,
            cv1_bd=cv1_bd,
            slice_0_bd=slice_0_bd,
            slice_1_bd=slice_1_bd,
            bottleneck_bds=tuple(bottleneck_bds),
            concat_bds=concat_bds,
            cv2_bd=cv2_bd,
        )


def shape3(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("input_shape must be (height, width, channels)")
    return tuple(_integer(name, n, 1, 2**31 - 1)
                 for name, n in zip(("height", "width", "channels"), shape))


def pair(value, name: str, maximum: int = 1023) -> tuple[int, int]:
    values = (value, value) if isinstance(value, Integral) else tuple(value)
    if len(values) != 2:
        raise ValueError(f"{name} must have two dimensions")
    return tuple(_integer(name, n, 1, maximum) for n in values)


def pads4(value) -> tuple[int, int, int, int]:
    values = (value,) * 4 if isinstance(value, Integral) else tuple(value)
    if len(values) != 4:
        raise ValueError("padding must be (top, left, bottom, right)")
    return tuple(_integer("padding", n, 0, 15) for n in values)


def touched_words(bds) -> dict[int, int]:
    """Map each local byte offset to how many times the given BDs move it."""
    counts: dict[int, int] = {}
    for bd in bds:
        for offset in bd.iter_word_offsets():
            counts[offset] = counts.get(offset, 0) + 1
    return counts


def assert_disjoint_destinations(bds) -> int:
    """Raise if two BDs write the same word; return the number of distinct words.

    Exact (enumerates every word), so use it on plans of at most a few hundred
    KB. Address spans alone cannot decide this: interleaved concat chunks have
    overlapping spans and disjoint word sets.
    """
    counts = touched_words(bds)
    repeated = sorted(offset for offset, n in counts.items() if n > 1)
    if repeated:
        raise ValueError(
            f"{len(repeated)} words are written by more than one descriptor, "
            f"first at {repeated[0]:#x}")
    return len(counts)


def _axis_regions(size, output, kernel, stride, pad, limit) -> Iterator[tuple[int, int, int, int]]:
    start = 0
    while start < output:
        before = max(0, pad - start * stride)
        after = max(0, start * stride - pad + kernel - size)
        end = start + 1
        while end < min(output, start + limit):
            if (max(0, pad - end * stride), max(0, end * stride - pad + kernel - size)) != (before, after):
                break
            end += 1
        yield start, end - start, before, after
        start = end
