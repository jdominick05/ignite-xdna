"""Explicit 16-core SPPF DMA graph, with one frame in flight.

Each column owns 64 channels. Its four cores own sixteen channels each.
MemTile ping/pong store four independent [20,20,16] channel shards.
The egress channel snapshots each stage before publishing it to the next pool.
Only that next pool's completed input DMA returns all four write credits.
Thus ping can be reused for Pool3 without racing the Pool1 snapshot or read.
MemTile S2MM scatters [Input, Pool1, Pool2, Pool3] into a local concatenation buffer.
DDR is column-packed: input [4,4,20,20,16], output [4,20,20,4,64].
No CPU concatenation and no intermediate DDR reads or writes occur.
"""
from pathlib import Path

INPUT_BYTES = 20 * 20 * 256
OUTPUT_BYTES = INPUT_BYTES * 4


def module(trace=False, repeats=1):
    lines = ['module {', '  aie.device(npu1) {']
    def emit(s):
        lines.append('    ' + s)
    def lock(name, tile, number, value):
        emit(f'%{name} = aie.lock(%{tile}, {number}) {{init = {value} : i32}}')
    def buf(name, tile, size, address):
        emit(f'%{name} = aie.buffer(%{tile}) {{sym_name = "{name}", address = {address} : i32}} : memref<{size}xi8>')
    def dma(tile, kind, channels):
        emit(f'aie.{kind}(%{tile}) {{')
        for n in (1, 4):
            emit(f'  %c{n} = arith.constant {n} : i32')
        for i, (direction, channel, bds) in enumerate(channels):
            if i:
                emit(f'^ch{i}:')
            nxt = f'^ch{i+1}' if i + 1 < len(channels) else '^end'
            emit(f'  aie.dma_start({direction}, {channel}, ^bd{i}_0, {nxt})')
            for j, (buffer, size, offset, length, dims, acquire, release) in enumerate(bds):
                emit(f'^bd{i}_{j}:')
                emit(f'  aie.use_lock(%{acquire[0]}, AcquireGreaterEqual, %c{acquire[1]})')
                emit(f'  aie.dma_bd(%{buffer} : memref<{size}xi8> offset = {offset} len = {length}{dims})')
                emit(f'  aie.use_lock(%{release[0]}, Release, %c{release[1]})')
                emit(f'  aie.next_bd ^bd{i}_{(j+1) % len(bds)}')
        emit('^end:')
        emit('  aie.end')
        emit('}')

    emit('func.func private @maxpool5x5(memref<6400xi8>, memref<6400xi8>, memref<6400xi8>) -> ()')
    for c in range(4):
        for r in range(6):
            emit(f'%t{c}_{r} = aie.tile({c}, {r})')
        if trace:
            emit(f'aie.trace @dma_trace{c}(%t{c}_0) {{')
            emit('  aie.trace.packet type=shimtile')
            for event in ('DMA_MM2S_0_START_TASK', 'DMA_S2MM_0_FINISHED_TASK', 'USER_EVENT_0'):
                emit(f'  aie.trace.event<"{event}">')
            emit('  aie.trace.start broadcast=15')
            emit('  aie.trace.stop broadcast=14')
            emit('}')
        mt = f't{c}_1'
        for name, addr in (('input', 0), ('ping', 0x40000), ('pong', 0x60000)):
            buf(f'{name}{c}', mt, 25600, addr)
        buf(f'concat{c}', mt, 102400, 0x10000)
        for i, (name, value) in enumerate((('ifree', 4), ('iready', 0), ('icompute', 0),
                  ('pfree', 4), ('pready', 0), ('pcompute', 0),
                  ('qfree', 4), ('qready', 0), ('qcompute', 0), ('p3free', 0),
                  ('catfree', 4), ('catready', 0))):
            lock(f'{name}{c}', mt, i, value)
        emit(f'aie.flow(%t{c}_0, DMA : 0, %{mt}, DMA : 0)')
        # MemTile local loopback is legal only for matching DMA channel indices.
        emit(f'aie.switchbox(%{mt}) {{ aie.connect<DMA : 5, DMA : 5> }}')
        emit(f'aie.flow(%{mt}, DMA : 4, %t{c}_0, DMA : 0)')
        emit(f'aie.shim_dma_allocation @in{c}(%t{c}_0, MM2S, 0)')
        emit(f'aie.shim_dma_allocation @out{c}(%t{c}_0, S2MM, 0)')
        channels = [('S2MM', 0, [(f'input{c}', 25600, 0, 25600, '', (f'ifree{c}', 4), (f'iready{c}', 1))])]
        for r in range(4):
            ct = f't{c}_{r+2}'
            emit(f'aie.flow(%{mt}, DMA : {r}, %{ct}, DMA : 0)')
            emit(f'aie.flow(%{ct}, DMA : 0, %{mt}, DMA : {r+1})')
            bds = [(f'{b}{c}', 25600, r * 6400, 6400, '',
                    (f'{free}{c}', 1), (f'{p}ready{c}', 1))
                   for b, p, free in (('ping', 'p', 'pfree'), ('pong', 'q', 'qfree'),
                                      ('ping', 'p', 'p3free'))]
            channels.append(('S2MM', r+1, bds))
            for name, size, addr in (('ci', 6400, 1024), ('tmp', 6400, 32768), ('co', 6400, 49152)):
                buf(f'{name}{c}_{r}', ct, size, addr)
            for i, (name, value) in enumerate((('cif', 1), ('cir', 0), ('cof', 1), ('cor', 0))):
                lock(f'{name}{c}_{r}', ct, i, value)
            dma(ct, 'mem', [
                ('S2MM', 0, [(f'ci{c}_{r}', 6400, 0, 6400, '', (f'cif{c}_{r}', 1), (f'cir{c}_{r}', 1))]),
                ('MM2S', 0, [(f'co{c}_{r}', 6400, 0, 6400, '', (f'cor{c}_{r}', 1), (f'cof{c}_{r}', 1))])])
            emit(f'aie.core(%{ct}) {{')
            emit('  %one = arith.constant 1 : i32')
            emit('  cf.br ^run')
            emit('^run:')
            emit(f'  aie.use_lock(%cir{c}_{r}, AcquireGreaterEqual, %one)')
            emit(f'  aie.use_lock(%cof{c}_{r}, AcquireGreaterEqual, %one)')
            emit(f'  func.call @maxpool5x5(%ci{c}_{r}, %co{c}_{r}, %tmp{c}_{r}) : (memref<6400xi8>, memref<6400xi8>, memref<6400xi8>) -> ()')
            emit(f'  aie.use_lock(%cif{c}_{r}, Release, %one)')
            emit(f'  aie.use_lock(%cor{c}_{r}, Release, %one)')
            emit('  cf.br ^run')
            emit('} {link_files = ["maxpool5x5.o"]}')
        for r in range(4):
            channels.append(('MM2S', r, [
                (f'{b}{c}', 25600, r * 6400, 6400, '', (f'{p}compute{c}', 1), (f'{free}{c}', 1))
                for b, p, free in (('input', 'i', 'ifree'),
                                  ('ping', 'p', 'p3free'), ('pong', 'q', 'qfree'))]))
        channels.append(('MM2S', 5, [
            (f'{b}{c}', 25600, 0, 25600, ' sizes = [400, 4, 16] strides = [16, 6400, 1]',
             (f'{p}ready{c}', n), (f'{rel}{c}', m))
            for b, p, n, rel, m in (('input', 'i', 1, 'icompute', 4),
               ('ping', 'p', 4, 'pcompute', 4), ('pong', 'q', 4, 'qcompute', 4),
               ('ping', 'p', 4, 'pfree', 4))]))
        channels.append(('S2MM', 5, [
            (f'concat{c}', 102400, stage * 64, 25600,
             ' sizes = [400, 64] strides = [256, 1]', (f'catfree{c}', 1), (f'catready{c}', 1))
            for stage in range(4)]))
        channels.append(('MM2S', 4, [(f'concat{c}', 102400, 0, 102400, '',
                                    (f'catready{c}', 4), (f'catfree{c}', 4))]))
        dma(mt, 'memtile_dma', channels)
    emit(f'aie.runtime_sequence(%x: memref<{INPUT_BYTES}xi8>, %y: memref<{OUTPUT_BYTES}xi8>) {{')
    if trace:
        emit('  aie.trace.host_config {buffer_size = 65536 : i32}')
        for c in range(4):
            emit(f'  aie.trace.start_config @dma_trace{c}')
    # The scoped frame body gives every static task a unique SSA name on unrolling.
    frame_start = len(lines)
    for c in range(4):
        emit(f'  %in{c} = aiex.dma_configure_task_for @in{c} {{')
        emit(f'    aie.dma_bd(%x : memref<{INPUT_BYTES}xi8> offset = {c*25600} len = 25600)')
        emit('    aie.end')
        emit('  }')
        emit(f'  %out{c} = aiex.dma_configure_task_for @out{c} {{')
        emit(f'    aie.dma_bd(%y : memref<{OUTPUT_BYTES}xi8> offset = {c*102400} len = 102400)')
        emit('    aie.end')
        emit('  } {issue_token = true}')
        emit(f'  aiex.dma_start_task(%out{c})')
        emit(f'  aiex.dma_start_task(%in{c})')
    for c in range(4):
        emit(f'  aiex.dma_await_task(%out{c})')
        emit(f'  aiex.dma_free_task(%in{c})')
        emit(f'  aiex.dma_free_task(%out{c})')
    if repeats > 1:
        import re
        frame = lines[frame_start:]
        for repeat in range(1, repeats):
            lines.extend(re.sub(r'%(in|out)([0-3])\b', rf'%\1\2_rep{repeat}', line) for line in frame)
    if trace:
        # Flush full trace packets after the measured output-completion events.
        from aie.utils.trace.events import ShimTileEvent
        emit('  %event_reg = arith.constant 213000 : i32')
        emit(f'  %event_value = arith.constant {int(ShimTileEvent.USER_EVENT_0)} : i32')
        for _ in range(64):
            for c in range(4):
                emit(f'  aiex.npu.write32(%event_reg, %event_value) {{column = {c} : i32, row = 0 : i32}} : i32, i32')
    emit('}')
    lines.extend(['  }', '}'])
    return '\n'.join(lines) + '\n'


def compile_design(directory: Path, trace=False, repeats=1):
    from aie.utils.compile.utils import compile_cxx_core_function, compile_mlir_module
    directory.mkdir(parents=True, exist_ok=True)
    work = directory / 'design.prj'
    work.mkdir(exist_ok=True)
    print('PEANO_TARGET aie2-none-unknown-elf -O3', flush=True)
    compile_cxx_core_function(str(Path(__file__).with_name('maxpool5x5.cc')),
                              'aie2', str(work / 'maxpool5x5.o'), compile_args=['-O3'])
    ir = module(trace, repeats)
    (directory / 'sppf.mlir').write_text(ir, encoding='utf-8')
    compile_mlir_module(ir, insts_path=directory / 'insts.bin',
                        xclbin_path=directory / 'design.xclbin', work_dir=work)
