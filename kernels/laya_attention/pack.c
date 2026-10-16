/* Copyright (C) 2026 The ignite-xdna contributors
 * SPDX-License-Identifier: AGPL-3.0-or-later */
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#ifndef ATTENTION_PACK_FAST
#define ATTENTION_PACK_FAST 0
#endif

/* Inputs are already rounded bf16 values in contiguous float32 arrays.
 * Preserve their bits; this routine performs layout conversion only. */
__declspec(dllexport) void pack_attention(const float *q,const float *k,const float *v,
                                         uint32_t *out,int valid,int local,uint32_t poison) {
    if(!ATTENTION_PACK_FAST) memset(out,0,798720*4);
    for(int h=0;h<16;++h) {
        for(int t=0;t<32;++t) {
            uint32_t *header=out+(h*32+t)*528;
            if(ATTENTION_PACK_FAST) memset(header,0,64);
            header[0]=t*16; header[1]=valid; header[2]=poison; header[3]=local;
            uint16_t *dst=(uint16_t *)(header+16);
            for(int a=0;a<4;++a) for(int b=0;b<8;++b)
                for(int r=0;r<4;++r) {
                    int query=t*16+a*4+r;
                    int uniform=valid==0 || (local && query-64>=valid);
                    __m128i packed=_mm_setzero_si128();
                    if(!uniform) {
                        __m256i x=_mm256_srli_epi32(_mm256_loadu_si256((const __m256i *)(q+h*32768+query*64+b*8)),16);
                        packed=_mm_packus_epi32(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));
                    }
                    _mm_storeu_si128((__m128i *)dst,packed); dst+=8;
                }
        }
        for(int t=0;t<16;++t) {
            uint16_t *dst=(uint16_t *)(out+270336+(h*16+t)*2064);
            for(int a=0;a<8;++a) for(int b=0;b<8;++b)
                for(int r=0;r<8;r+=4) {
                    const float *base=k+h*32768+(t*32+b*4)*64+a*8+r;
                    __m128 x0=_mm_loadu_ps(base),x1=_mm_loadu_ps(base+64);
                    __m128 x2=_mm_loadu_ps(base+128),x3=_mm_loadu_ps(base+192);
                    _MM_TRANSPOSE4_PS(x0,x1,x2,x3);
                    __m128i p01=_mm_packus_epi32(_mm_srli_epi32(_mm_castps_si128(x0),16),_mm_srli_epi32(_mm_castps_si128(x1),16));
                    __m128i p23=_mm_packus_epi32(_mm_srli_epi32(_mm_castps_si128(x2),16),_mm_srli_epi32(_mm_castps_si128(x3),16));
                    _mm_storeu_si128((__m128i *)dst,p01); _mm_storeu_si128((__m128i *)(dst+8),p23); dst+=16;
                }
            for(int a=0;a<4;++a) for(int b=0;b<16;++b)
                for(int r=0;r<8;++r) {
                    __m128i x=_mm_srli_epi32(_mm_loadu_si128((const __m128i *)(v+h*32768+(t*32+a*8+r)*64+b*4)),16);
                    _mm_storel_epi64((__m128i *)dst,_mm_packus_epi32(x,_mm_setzero_si128())); dst+=4;
                }
            for(int i=0;i<32;++i) *dst++=(uint16_t)i;
        }
    }
}
