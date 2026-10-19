"""Generate an export-only arena stream with aggregate command guard checking."""
# Copyright (C) 2026 The ignite-xdna contributors
# SPDX-License-Identifier: AGPL-3.0-or-later


def harness(function, packet, output, blocks, state_words, selected, input_depth=1):
    if output < 16 or not state_words or not selected:
        raise ValueError('Compact output requires state and at least one export')
    if input_depth not in (1,2):raise ValueError('Use one or two input buffers')
    ni, no = blocks*packet, (len(selected)+1)*output
    return f'''module {{
  aie.device(npu1) {{
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    %state = aie.buffer(%t) {{sym_name = "carry_state"}} : memref<{state_words}xf32>
    %scratch = aie.buffer(%t) {{sym_name = "result_scratch"}} : memref<{output}xi32>
    func.func private @{function}(memref<{packet}xi32>, memref<{output}xi32>, memref<{state_words}xf32>)
    aie.objectfifo @inF(%s, {{%t}}, {input_depth} : i32) : !aie.objectfifo<memref<{packet}xi32>>
    aie.objectfifo @outF(%t, {{%s}}, 1 : i32) : !aie.objectfifo<memref<{output}xi32>>
    %core = aie.core(%t) {{
      %zero = arith.constant 0 : index
      %one = arith.constant 1 : index
      %guards = arith.constant 16 : index
      %outputs = arith.constant {output} : index
      %marker_index = arith.constant 11 : index
      %forever = arith.constant 4294967295 : index
      %true = arith.constant true
      %z32 = arith.constant 0 : i32
      %o32 = arith.constant 1 : i32
      %two32 = arith.constant 2 : i32
      %unused = scf.for %it = %zero to %forever step %one iter_args(%ok = %true) -> (i1) {{
          %is = aie.objectfifo.acquire @inF(Consume, 1) : !aie.objectfifosubview<memref<{packet}xi32>>
          %a = aie.objectfifo.subview.access %is[0] : !aie.objectfifosubview<memref<{packet}xi32>> -> memref<{packet}xi32>
          %poison = memref.load %a[%zero] : memref<{packet}xi32>
          %marker = memref.load %a[%marker_index] : memref<{packet}xi32>
          %export_bit = arith.andi %marker, %o32 : i32
          %exporting = arith.cmpi ne, %export_bit, %z32 : i32
          %last_bit = arith.andi %marker, %two32 : i32
          %last = arith.cmpi ne, %last_bit, %z32 : i32
          func.call @{function}(%a, %scratch, %state) : (memref<{packet}xi32>, memref<{output}xi32>, memref<{state_words}xf32>) -> ()
          %checked = scf.for %j = %zero to %guards step %one iter_args(%valid = %ok) -> (i1) {{
            %guard = memref.load %scratch[%j] : memref<{output}xi32>
            %same = arith.cmpi eq, %guard, %poison : i32
            %next = arith.andi %valid, %same : i1
            scf.yield %next : i1
          }}
          scf.if %exporting {{
            %os = aie.objectfifo.acquire @outF(Produce, 1) : !aie.objectfifosubview<memref<{output}xi32>>
            %o = aie.objectfifo.subview.access %os[0] : !aie.objectfifosubview<memref<{output}xi32>> -> memref<{output}xi32>
            memref.copy %scratch, %o : memref<{output}xi32> to memref<{output}xi32>
            aie.objectfifo.release @outF(Produce, 1)
          }}
          aie.objectfifo.release @inF(Consume, 1)
        scf.if %last {{
        %status = aie.objectfifo.acquire @outF(Produce, 1) : !aie.objectfifosubview<memref<{output}xi32>>
        %result = aie.objectfifo.subview.access %status[0] : !aie.objectfifosubview<memref<{output}xi32>> -> memref<{output}xi32>
        scf.for %j = %zero to %outputs step %one {{
          memref.store %poison, %result[%j] : memref<{output}xi32>
        }}
        %bad = arith.xori %poison, %o32 : i32
        %summary = arith.select %checked, %poison, %bad : i32
        memref.store %summary, %result[%zero] : memref<{output}xi32>
        aie.objectfifo.release @outF(Produce, 1)
        }}
        %next_ok = arith.select %last, %true, %checked : i1
        scf.yield %next_ok : i1
      }}
      aie.end
    }} {{link_with = "kernel.o", stack_size = 4096 : i32}}
    aie.runtime_sequence(%a: memref<{ni}xi32>, %o: memref<{no}xi32>) {{
      aiex.npu.dma_memcpy_nd(%a[0,0,0,0][1,1,{ni//16},16][0,0,16,1]) {{id = 0 : i64,metadata = @inF}} : memref<{ni}xi32>
      aiex.npu.dma_memcpy_nd(%o[0,0,0,0][1,1,{no//16},16][0,0,16,1]) {{id = 1 : i64,metadata = @outF,issue_token = true}} : memref<{no}xi32>
      aiex.npu.dma_wait {{symbol = @outF}}
    }}
  }}
}}
'''


def validate_markers(meta, packets):
    """Reject a changed output schedule before DMA could wait for missing frames."""
    import numpy as np
    expected=np.zeros(meta['blocks'],np.uint32)
    expected[meta['output_packets']]=1
    expected[-1]|=2
    if not np.array_equal(packets[:,11],expected):
        raise ValueError('Compact output routing differs from admitted schedule')
