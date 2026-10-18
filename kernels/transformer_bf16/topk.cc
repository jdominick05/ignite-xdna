// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
#include <stdint.h>
static inline int ordered(uint16_t bits) {
    return bits & 0x8000 ? 0x8000-int(bits&0x7fff) : 0x8000+int(bits);
}
extern "C" void draft_topk(const int32_t *packet,int32_t *output) {
    const int count=packet[1],k=packet[2];
    const auto *scores=reinterpret_cast<const uint16_t *>(packet+16);
    int indices[32];
    for(int i=0;i<32;++i)indices[i]=-1;
    for(int i=0;i<count;++i) {
        const int value=ordered(scores[i]);
        int slot=k;
        for(int j=0;j<k;++j) {
            if(indices[j]<0 || value>ordered(scores[indices[j]])) {slot=j;break;}
        }
        if(slot<k) {
            for(int j=k-1;j>slot;--j)indices[j]=indices[j-1];
            indices[slot]=i;
        }
    }
    for(int i=0;i<64;++i)output[i]=packet[0];
    for(int i=0;i<k;++i)output[16+i]=indices[i];
}
