// Smallest possible ADF graph: one int8 passthrough kernel, two ports.
// Exists to answer one question -- can `aiecompiler` (shipped in
// RyzenAI\1.7.1, package `vaie-overlay`, never a console command: invoke via
// `from vaie_overlay import cli; sys.argv=[...]; cli.aiecompiler()`) be driven
// from this install at all -- not to demonstrate a working kernel. See
// RESEARCH.md's "Custom C++ XRT / hand-written AIE kernels" for the result
// and results/aie/aiecompiler_*.log for every invocation tried.
//
// Reproduce (ryzen-ai-1.7.1 env, not resnet_env17):
//   aiecompiler --include=<vaie_cpplus's installed include dir> \
//     --target=x86sim graph.cc [--part=... | --platform=...]
// Stops at "AIE architecture could not be auto derived" without a valid
// --part/--platform -- none was found anywhere in this SDK as of this commit.

#include "graph.h"

simpleGraph mygraph;

#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
    mygraph.init();
    mygraph.run(1);
    mygraph.end();
    return 0;
}
#endif
