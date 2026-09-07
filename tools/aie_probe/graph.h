#pragma once
#include <adf.h>
#include "kernels.h"

using namespace adf;

class simpleGraph : public adf::graph {
public:
    input_plio in;
    output_plio out;
    kernel k;

    simpleGraph() {
        k = kernel::create(passthrough);
        in = input_plio::create("datain", plio_32_bits, "data/input.txt");
        out = output_plio::create("dataout", plio_32_bits, "data/output.txt");
        connect<window<32>>(in.out[0], k.in[0]);
        connect<window<32>>(k.out[0], out.in[0]);
        source(k) = "kernels.cc";
        runtime<ratio>(k) = 0.5;
    }
};
