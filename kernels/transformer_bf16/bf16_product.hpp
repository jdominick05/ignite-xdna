// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// RN-even FP32 product when the first operand is exactly BF16. The 8x24-bit
// significand product fits uint32. Exceptional/subnormal cases use the control.
#pragma once
#ifndef BF16_PRODUCT_ATTRIBUTES
#define BF16_PRODUCT_ATTRIBUTES inline
#endif
static BF16_PRODUCT_ATTRIBUTES F bf16_operand_product(F a,F b) {
    using U=aie::vector<uint32,16>;
    auto constant=[](uint32 x){return aie::broadcast<uint32,16>(x);};
    U ab=a.cast_to<uint32>(),bb=b.cast_to<uint32>();
    U ea=aie::bit_and(aie::downshift(ab,23),constant(255));
    U eb=aie::bit_and(aie::downshift(bb,23),constant(255));
    U ma=aie::bit_or(aie::bit_and(aie::downshift(ab,16),constant(127)),constant(128));
    U mb=aie::bit_or(aie::bit_and(bb,constant(0x7fffff)),constant(0x800000));
    U product=aie::mul(ma,mb).to_vector<uint32>();
    U low=aie::downshift(aie::add(product,aie::add(constant(63),
        aie::bit_and(aie::downshift(product,7),constant(1)))),7);
    U high=aie::downshift(aie::add(product,aie::add(constant(127),
        aie::bit_and(aie::downshift(product,8),constant(1)))),8);
    auto upper=aie::gt(product,constant(0x7fffffff));
    U mantissa=aie::select(low,high,upper);
    U exponent=aie::sub(aie::add(ea,eb),constant(128));
    exponent=aie::add(exponent,aie::select(constant(0),constant(1),upper));
    U encoded=aie::add(aie::upshift(exponent,23),mantissa);
    // An unrounded exponent of zero can still round to the smallest normal
    // value. Handle that boundary with one extra significand shift rather
    // than delegating it to the control's underflow behavior.
    auto boundary=aie::eq(exponent,constant(0xffffffff));
    U sub_high=aie::downshift(aie::add(product,aie::add(constant(255),
        aie::bit_and(aie::downshift(product,9),constant(1)))),9);
    encoded=aie::select(encoded,aie::select(high,sub_high,upper),boundary);
    encoded=aie::bit_or(encoded,aie::bit_and(aie::bit_xor(ab,bb),constant(0x80000000)));
    auto special=aie::eq(ea,constant(0))|aie::eq(eb,constant(0))|
                 aie::eq(ea,constant(255))|aie::eq(eb,constant(255))|
                 (aie::gt(exponent,constant(253)) & ~boundary);
    return aie::select(encoded.cast_to<float>(),mul(a,b),special);
}
