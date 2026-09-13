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
    """Eight raw registers plus the queue repetitions needed to execute them."""

    registers: tuple[int, ...]
    repeat_count: int
    memory_bytes: int
    transfer_bytes: int
    address_span: tuple[int, int]  # local byte offsets, end exclusive
    direction: str

    @property
    def steps(self) -> tuple[int, ...]:
        """Raw D0..D3 Step fields (word strides minus one)."""
        return tuple(self.registers[i] & 0x1FFFF for i in (2, 3, 4, 5))

    @property
    def wraps(self) -> tuple[int, ...]:
        return tuple((self.registers[i] >> 17) & 1023 for i in (2, 3, 4))

    @property
    def sizes(self) -> tuple[int, ...]:
        """Raw D0..D3 size/wrap fields."""
        w0_2 = tuple((self.registers[i] >> 17) & 1023 for i in (2, 3, 4))
        denom = w0_2[0] * w0_2[1] * w0_2[2]
        w3 = self.registers[0] // denom if denom > 0 else 1
        return w0_2 + (w3,)

    @property
    def iteration_wrap(self) -> int:
        return (self.registers[6] >> 17) & 63

    def with_locks(self, locks: LockConfig) -> "MemTileBD":
        return replace(self, registers=self.registers[:7] + (locks.word(),))

    def register_writes(self, bd_id: int) -> tuple[tuple[int, int], ...]:
        bd_id = _integer("bd_id", bd_id, 0, 47)
        return tuple((BD_BASE + bd_id * BD_STRIDE + i * 4, word)
                     for i, word in enumerate(self.registers))

    def queue_word(self, bd_id: int) -> int:
        """Start-queue payload; caller selects channel/direction register.

        Channel/BD compatibility must also be respected: channels 0..3 use
        BD 0..23 and channels 4..5 use BD 24..47 on Phoenix.
        """
        return _integer("bd_id", bd_id, 0, 47) | ((self.repeat_count - 1) << 16)


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
    """Descriptor for in-flight spatial 2x Nearest-Neighbor upsampling directly in MemTile DMA."""
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    base_address: int
    transfer_bytes: int
    memory_bytes: int
    bd: MemTileBD

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
        if any(s < 0 for s in strides):
            raise ValueError("steps must be non-negative")
        count = _integer("iteration_count", iteration_count, 1, 64)
        istep = _words("iteration_step", iteration_step)
        if istep < 0:
            raise ValueError("iteration_step must be non-negative")
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
            (max(0, strides[0] - 1) if strides[0] > 0 else 0) | (sizes[0] << 17),
            (max(0, strides[1] - 1) if strides[1] > 0 else 0) | (sizes[1] << 17) | (before[1] << 27),
            (max(0, strides[2] - 1) if strides[2] > 0 else 0) | (sizes[2] << 17) | (before[2] << 27),
            (max(0, strides[3] - 1) if strides[3] > 0 else 0) | (after[0] << 17) | (after[1] << 23) | (after[2] << 28),
            (max(0, istep - 1) if istep > 0 else 0) | ((count - 1) << 17),
            locks.word(),
        )
        return MemTileBD(regs, count, prod(sizes) * count * 4,
                         length * count * 4, (base_address, end), direction)

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
        """Synthesize 4D MemTile BD for zero-copy slicing of channels [channel_offset, channel_offset + num_channels)."""
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
        """Synthesize 4D MemTile BDs for scattering multiple incoming channel chunks into an interleaved destination buffer."""
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

    def upsample_2x_nearest_bd(
        self,
        *,
        base_address: int,
        height: int,
        width: int,
        channels: int,
        direction: str = "MM2S",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> MemTileBD:
        """
        Synthesize MemTile DMA BD for in-flight spatial 2x Nearest-Neighbor upsampling.
        Duplicates pixels horizontally (step=0, wrap=2) and vertically across rows
        without executing ALU instructions.
        """
        _words("channels", channels)
        wc = channels // 4
        sizes = (wc, 2, width, 2)
        steps = (4, 0, channels, 0)
        iteration_step = width * channels
        iteration_count = height
        return self.synthesize(
            base_address=base_address,
            sizes=sizes,
            steps=steps,
            iteration_count=iteration_count,
            iteration_step=iteration_step,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )

    def plan_upsample_2x(
        self,
        *,
        input_shape: tuple[int, int, int],
        base_address: int,
        direction: str = "MM2S",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> Upsample2xPlan:
        """Creates an Upsample2xPlan for 2x spatial nearest-neighbor upsampling."""
        h, w, c = shape3(input_shape)
        bd = self.upsample_2x_nearest_bd(
            base_address=base_address,
            height=h,
            width=w,
            channels=c,
            direction=direction,
            buffer_bounds=buffer_bounds,
            locks=locks,
        )
        out_shape = (2 * h, 2 * w, c)
        return Upsample2xPlan(
            input_shape=(h, w, c),
            output_shape=out_shape,
            base_address=base_address,
            transfer_bytes=prod(out_shape),
            memory_bytes=prod(input_shape),
            bd=bd,
        )

    def plan_lateral_concat(
        self,
        *,
        base_address: int,
        spatial_pixels: int,
        chunk_channels: tuple[int, ...],
        direction: str = "S2MM",
        buffer_bounds: tuple[int, int] = (0, MEMTILE_BYTES),
        locks: LockConfig = LockConfig(),
    ) -> LateralConcatPlan:
        """Synthesize strided S2MM DMA scatter plans for lateral feature map skip connections."""
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

    def route_c2f_stage(
        self,
        *,
        stage_name: str,
        spatial_pixels: int,
        in_channels: int,
        hidden_channels: int,
        num_bottlenecks: int,
        ping_buffer_addr: int = 0x40000,
        pong_buffer_addr: int = 0x60000,
        ping_bank_bounds: tuple[int, int] = (0x40000, 0x50000),
        pong_bank_bounds: tuple[int, int] = (0x60000, 0x70000),
    ) -> C2fRoutingPlan:
        """Constructs full zero-copy MemTile routing plan for a YOLOv8 C2f block."""
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
