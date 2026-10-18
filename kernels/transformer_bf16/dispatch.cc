// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Single-core packet opcode dispatcher. This shares an image, not a fused graph:
// operations still receive their inputs and return outputs through host BOs.
#include <aie_api/aie.hpp>
#include <stdint.h>
namespace projection {
#include "gemv_precise.cc"
}
namespace norm {
#include "rmsnorm.cc"
}
namespace rotary {
#include "rope.cc"
}
namespace softmax {
#define BF16_EXP_ATTRIBUTES __attribute__((noinline))
#include "softmax.cc"
#undef BF16_EXP_ATTRIBUTES
}
namespace activation {
#include "geglu.cc"
}
namespace residual {
#include "residual.cc"
}
namespace selection {
#include "topk.cc"
}
extern "C" void draft_transformer(const int32_t *packet,int32_t *output,float *state) {
    for(int i=0;i<1056;i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    switch(packet[15]) {
        case 1:projection::draft_projection_precise(packet,output,state);break;
        case 2:norm::draft_norm(packet,output);break;
        case 3:rotary::draft_rope(packet,output);break;
        case 4:softmax::draft_softmax(packet,output,state);break;
        case 5:activation::draft_geglu(packet,output);break;
        case 6:residual::draft_residual(packet,output);break;
        case 7:selection::draft_topk(packet,output);break;
        default:output[0]=0;break; // Host guard rejects unsupported opcodes.
    }
}
