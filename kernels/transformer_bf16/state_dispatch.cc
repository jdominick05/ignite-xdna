// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// A command stream over a 32 KiB BF16 activation arena. Packet weights remain
// immutable in DDR; intermediate activations stay on the compute tile.
#include "dispatch.cc"
#ifndef BF16_ARENA_UPLOAD_MAX
#define BF16_ARENA_UPLOAD_MAX 8192
#endif
static inline bool slot_ok(int offset,int count) {
    return offset>=0 && count>=0 && offset<=16384-count;
}
static void copy_bits(uint16_t *to,const uint16_t *from,int count) {
    int i=0;
#ifdef BF16_VECTOR_COPY
    // Only use aligned complete vectors. Scalar fallback/tails preserve exact
    // allocation bounds for arbitrary public arena offsets and short copies.
    if(((reinterpret_cast<uintptr_t>(to)|reinterpret_cast<uintptr_t>(from))&63)==0)
        for(;i+32<=count;i+=32)aie::store_v(to+i,aie::load_v<32>(from+i));
#endif
    for(;i<count;++i)to[i]=from[i];
}
extern "C" void draft_state_transformer(int32_t *packet,int32_t *output,float *state) {
    auto *arena=reinterpret_cast<uint16_t *>(state+64);
    auto *input=reinterpret_cast<uint16_t *>(packet+16);
    auto *result=reinterpret_cast<uint16_t *>(output+16);
    const int opcode=packet[15],flags=packet[4];
    const int source=packet[5],second=packet[6],destination=packet[7];
    const int nsource=packet[8],nsecond=packet[9],nout=packet[10];
    for(int i=0;i<1056;i+=16)
        aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    bool valid=slot_ok(source,nsource) && slot_ok(second,nsecond) && slot_ok(destination,nout);
    if(opcode==9) {
        valid=valid && slot_ok(destination,nsource) && nsource<=BF16_ARENA_UPLOAD_MAX;
        if(valid)copy_bits(arena+destination,input,nsource);
    } else if(opcode==10) {
        valid=valid && slot_ok(source,nout) && nout<=2048;
        if(valid)copy_bits(result,arena+source,nout);
    } else {
        const int limit=(opcode==1 || opcode==8)?256:opcode==4?512:opcode==7?4096:2048;
        valid=valid && (flags&~7)==0 && nsource<=limit && nout<=2048 && nsecond<=2048;
        if((flags&2)!=0 && opcode!=5 && opcode!=6 && opcode!=11)valid=false;
#ifdef BF16_SMALL_PACKETS
        if(opcode==11) {
            const int table=packet[12];
            valid=valid && flags==7 && nsource==2048 && nsecond==2048 && nout==2048 && slot_ok(table,4096);
            valid=valid && !(destination<table+4096 && table<destination+nout);
        }
#endif
        if(valid) {
            if(flags&1) {
                // Projection/softmax padding must be +0 rather than stale arena
                // bits, including for lengths not divisible by the packet size.
#ifdef BF16_VECTOR_COPY
                for(int i=0;i<limit;i+=32)aie::store_v(input+i,aie::zeros<uint16,32>());
#else
                for(int i=0;i<limit;++i)input[i]=0;
#endif
                copy_bits(input,arena+source,nsource);
            }
            if(flags&2)copy_bits(reinterpret_cast<uint16_t *>(packet+1040),arena+second,nsecond);
#ifdef BF16_SMALL_PACKETS
            if(opcode==11)activation::draft_geglu_resident(packet,output,arena+packet[12]);
            else draft_transformer(packet,output,state);
#else
            draft_transformer(packet,output,state);
#endif
            if(flags&4)copy_bits(arena+destination,result,nout);
        }
    }
    if(!valid)output[0]=0;
}
