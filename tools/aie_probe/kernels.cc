#include "kernels.h"

void passthrough(input_window_int8* in, output_window_int8* out) {
    for (unsigned i = 0; i < 32; i++) {
        int8 v = window_readincr(in);
        window_writeincr(out, v);
    }
}
