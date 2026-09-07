// Smallest possible ADF graph: one int8 passthrough kernel, two ports.
// Exists to answer one question -- can `aiecompiler` (shipped in
// RyzenAI\1.7.1, package `vaie-overlay`, never a console command: invoke via
// `from vaie_overlay import cli; sys.argv=[...]; cli.aiecompiler()`) be driven
// from this install at all -- not to demonstrate a working kernel. See
// RESEARCH.md's "Custom C++ XRT / hand-written AIE kernels" for the result
// and results/aie/aiecompiler_*.log for every invocation tried.
//
// Reproduce (ryzen-ai-1.7.1 env, not resnet_env17; needs a host C++ compiler
// -- Visual Studio 2022 Build Tools, C++ workload -- see
// results/aie/aiecompiler_hostlib_fixed.log for why):
//   aiecompiler --include=<vaie_cpplus's installed include dir> \
//     --include=<MSVC's include dir> \
//     --include=<Windows SDK's ucrt include dir> \
//     --include=<Windows SDK's shared include dir> \
//     --include=<Windows SDK's um include dir> \
//     --target=hw graph.cc --part=xc10AIE24x5-die-1LP-e-S-es1
// Derives the device (logs "Reading logical device aie2_5x4_device" --
// Phoenix, confirmed via results/aie/notes_aiecompiler_part_db.log) and
// parses this graph cleanly, then stops at "Could not find ... \lib\win64.o\
// physical_device.dll" -- that file does not exist anywhere in this SDK or
// on this machine. See results/aie/aiecompiler_hostlib_fixed.log.

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
