// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Experimental compensated inverse-square-root correction. The original
// rmsnorm.cc remains the control; this kernel requires separate qualification.
#include <aie_api/aie.hpp>
#include <stdint.h>
using F = aie::vector<float,16>;
static inline F full(float x) {return aie::broadcast<float,16>(x);}
static inline F mul(F a,F b) {return aie::mul(a,b).to_vector<float>();}
#ifdef BF16_EXACT_PRODUCT
#include "bf16_product.hpp"
#endif
#ifndef BF16_ERROR_ATTRIBUTES
#define BF16_ERROR_ATTRIBUTES inline
#endif
static BF16_ERROR_ATTRIBUTES F product_error(F a,F b,F product) {
    // Split into at most twelve significant bits per high part. Form the
    // product residual explicitly; do not assume FP32 msc is IEEE fused.
    F ah=aie::upshift(aie::downshift(a.cast_to<uint32>(),12),12).cast_to<float>();
    F bh=aie::upshift(aie::downshift(b.cast_to<uint32>(),12),12).cast_to<float>();
    F al=aie::sub(a,ah),bl=aie::sub(b,bh);
    F error=aie::sub(mul(ah,bh),product);
    error=aie::add(error,mul(ah,bl));
    error=aie::add(error,mul(al,bh));
    return aie::add(error,mul(al,bl));
}
#ifndef BF16_HELPER_INLINE
#define BF16_HELPER_INLINE inline
#endif
static BF16_HELPER_INLINE F load_bf16(const bfloat16 *p) {
    auto bits=aie::load_v<16>(p).cast_to<uint16>().unpack();
    return aie::upshift(bits,16).cast_to<float>();
}
static inline void store_bf16(bfloat16 *p,F x) {
    auto bits=x.cast_to<uint32>();
    auto rounded=aie::add(bits,aie::add(aie::broadcast<uint32,16>(0x7fff),
        aie::bit_and(aie::downshift(bits,16),aie::broadcast<uint32,16>(1))));
    aie::store_v(p,aie::filter_even(aie::downshift(rounded,16).cast_to<uint16>()).cast_to<bfloat16>());
}
extern "C" void draft_norm_refined(const int32_t *packet,int32_t *output) {
    const int width=packet[1],rows=packet[2];
    const auto *x=reinterpret_cast<const bfloat16 *>(packet+16);
    const auto *weight=reinterpret_cast<const bfloat16 *>(packet+1040);
    auto *y=reinterpret_cast<bfloat16 *>(output+16);
    for(int i=0;i<1056;i+=16)aie::store_v(output+i,aie::broadcast<int32,16>(packet[0]));
    const float inverse_width=reinterpret_cast<const float *>(packet)[3];
    for(int row=0;row<rows;++row) {
        F squares=full(0);
        for(int i=0;i<width;i+=16) {
            F value=load_bf16(x+row*width+i);
            squares=aie::add(squares,mul(value,value));
        }
        F variance=aie::add(mul(full(aie::reduce_add(squares)),full(inverse_width)),full(1e-6f));
        F inverse=aie::sub(aie::broadcast<uint32,16>(0x5f375a86),
                          aie::downshift(variance.cast_to<uint32>(),1)).cast_to<float>();
        for(int j=0;j<4;++j)
            inverse=mul(inverse,aie::sub(full(1.5f),mul(mul(full(.5f),variance),mul(inverse,inverse))));
#ifdef BF16_NORM_TRACE
        F initial_inverse=inverse;
#endif
        F product=mul(variance,inverse);
        F second_product=mul(inverse,product);
        F residual=aie::sub(full(1),second_product);
        residual=aie::sub(residual,product_error(inverse,product,second_product));
        residual=aie::sub(residual,mul(inverse,product_error(variance,inverse,product)));
        inverse=aie::add(inverse,mul(mul(full(.5f),inverse),residual));
#ifdef BF16_NORM_TRACE
        // Dedicated diagnostic entry only: export four raw FP32 states per
        // row in the usual prefix guard area. Normal kernel builds omit this.
        auto *trace=reinterpret_cast<float *>(output)+4*row;
        trace[0]=variance.get(0);trace[1]=initial_inverse.get(0);
        trace[2]=inverse.get(0);trace[3]=residual.get(0);
#endif
        for(int i=0;i<width;i+=16) {
            F value=load_bf16(x+row*width+i),w=load_bf16(weight+i);
#ifdef BF16_EXACT_PRODUCT
            F first=bf16_operand_product(value,inverse),last=bf16_operand_product(w,first);
#else
            F first=mul(value,inverse),last=mul(first,w);
#endif
            store_bf16(y+row*width+i,last);
#ifdef BF16_NORM_TRACE
            if(i==128) {
                auto *products=reinterpret_cast<float *>(output)+1040+4*row;
                products[0]=value.get(1);products[1]=w.get(1);
                products[2]=first.get(1);products[3]=last.get(1);
            }
#endif
        }
    }
}
