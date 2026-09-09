// dispatch_runner.cpp -- a lightweight C++ XRT host for the AIE dispatch floor.
//
// WHY THIS EXISTS
// ---------------
// results/aie/dispatch_runlist_npu.log reached 36.3 us per dispatch from Python by
// driving raw pyxrt around IRON, and results/aie/dispatch_iron_batch_npu.log then showed
// IRON's own host path adds a near-constant ~500 us per call that is FLAT IN BATCH SIZE
// -- ABI validation, buffer preparation and instruction-buffer setup, not dispatch.
// Both numbers were measured through pybind11. Neither can say how much of the remaining
// per-call cost is the Python binding and how much is the driver, and that distinction
// decides whether a deployable runner can reach the device floor.
//
// This host removes Python entirely. It loads the SAME compiled design the Python
// harness resolved out of ~/.npu/cache/ -- same final.xclbin, same insts.bin, same
// argument layout -- and drives it three ways:
//
//   single      one dispatch at a time, kernel(...) + wait().  Mirrors pyxrt's ~140 us
//               single-dispatch path.  This is the number that says whether that floor
//               was pybind overhead or the driver.
//   rebuild     a fresh xrt::runlist built and torn down every iteration.  This is the
//               apples-to-apples arm: it is exactly what measure_runlist.py does, so the
//               C++/Python difference at the same batch size is the binding's cost.
//   persistent  the runlist is built ONCE and then execute()/wait() in a loop.  The
//               Python harness never tried this -- it noted reuse was documented and
//               rebuilt anyway.  xrt_kernel.h only says execute() "throws if runlist is
//               already executing", and add() only requires the list not be executing,
//               so re-executing a waited list is the documented path.  This is the
//               "cached handles + persistent kernel loop" arm.
//
// WHAT IS AND IS NOT MEASURED
// ---------------------------
// The timed region is submit-to-completion only: execute() + wait(), or kernel() +
// wait().  One-time setup (device open, xclbin register, context, kernel, buffer
// allocation, instruction upload) is timed SEPARATELY and reported, because the whole
// point of a cached-handle runner is that this cost is paid once and not per call.
// Buffer sync is outside the timed region in every arm, matching the Python harness.
//
// CORRECTNESS GATE, NOT OPTIONAL
// ------------------------------
// A run inside a runlist cannot be polled -- xrt_kernel.h says run::state() "is not
// guaranteed to reflect the actual run object state and cannot be called for run objects
// that are part of a runlist".  runlist::wait() is the only completion signal, so
// VERIFYING THE OUTPUT BUFFERS IS THE ONLY CORRECTNESS GATE.  Every batch verifies all
// of its output buffers against the input before its timing is kept, and the table
// prints that column.  Buffers are never shared between queued runs: concurrent buffer
// access within a runlist is undefined and its symptom is wrong data, not an error.
//
// USAGE
//   scripts\build_dispatch_runner.bat            (builds kernels\dispatch_floor\dispatch_runner.exe)
//   dispatch_runner.exe --xclbin <path> --insts <path> [flags]
//   dispatch_runner.exe --cache-newest           (resolve the design out of ~/.npu/cache/)
//
// The design is NOT compiled here -- run kernels/dispatch_floor/measure_runlist.py once
// to populate the IRON cache, then point this at what it resolved.  Nothing is written
// into any of the repo's *cachekey/ compile-cache directories; this reads ~/.npu/cache/
// and writes nothing.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_kernel.h"
#include "xrt/experimental/xrt_xclbin.h"

namespace fs = std::filesystem;
using clk = std::chrono::steady_clock;

// The transaction-opcode the mlir-aie runtime sequence is dispatched under, and the
// argument layout of the generated kernel.  Both are fixed by IRON's host path and are
// mirrored from kernels/dispatch_floor/measure_runlist.py rather than rederived:
//   arg0 opcode(3)  arg1 instruction bo  arg2 instruction word count
//   arg3 A(in)      arg4 B(unused in)    arg5 C(out)
static constexpr uint32_t kOpcode = 3;
static constexpr int kArgOpcode = 0, kArgInstBo = 1, kArgInstCount = 2;
static constexpr int kArgA = 3, kArgB = 4, kArgC = 5;

struct Stats {
    double mean_us = 0.0, min_us = 0.0, median_us = 0.0;
};

static Stats summarize(std::vector<double> v, int batch) {
    Stats s;
    if (v.empty()) return s;
    // Per-dispatch: a batch of `batch` runs completes in one wall interval.
    for (auto &x : v) x /= batch;
    std::sort(v.begin(), v.end());
    s.min_us = v.front();
    s.median_us = v[v.size() / 2];
    s.mean_us = std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
    return s;
}

// --------------------------------------------------------------------------- //
// Cache resolution -- same rule as measure_runlist.py                          //
// --------------------------------------------------------------------------- //
// Require an entry holding BOTH final.xclbin and insts.bin.  Taking the newest
// final.xclbin anywhere in the cache picks up a different design with a different
// argument layout after any other design is compiled, which is a wrong answer rather
// than an error -- the Python harness was fixed for exactly this.
static bool resolve_from_cache(std::string &xclbin_out, std::string &insts_out,
                               std::string &entry_name) {
    const char *home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    if (!home) return false;
    fs::path cache = fs::path(home) / ".npu" / "cache";
    if (!fs::exists(cache)) return false;

    fs::file_time_type best{};
    fs::path best_dir;
    for (const auto &e : fs::directory_iterator(cache)) {
        if (!e.is_directory()) continue;
        fs::path x = e.path() / "final.xclbin";
        fs::path i = e.path() / "insts.bin";
        if (!fs::exists(x) || !fs::exists(i)) continue;
        auto t = fs::last_write_time(x);
        if (best_dir.empty() || t > best) { best = t; best_dir = e.path(); }
    }
    if (best_dir.empty()) return false;
    xclbin_out = (best_dir / "final.xclbin").string();
    insts_out = (best_dir / "insts.bin").string();
    entry_name = best_dir.filename().string();
    return true;
}

static std::vector<uint32_t> read_insts(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open instruction file: " + path);
    auto bytes = static_cast<size_t>(f.tellg());
    if (bytes % 4 != 0) throw std::runtime_error("instruction file is not a whole number of words");
    f.seekg(0);
    std::vector<uint32_t> v(bytes / 4);
    f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

// One run's three buffers.  Every queued run owns its own set: sharing them inside a
// runlist is undefined behaviour whose symptom is wrong data.
struct BufSet {
    xrt::bo a, b, c;
};

static BufSet make_bufset(xrt::device &dev, xrt::kernel &k, int n) {
    const size_t bytes = static_cast<size_t>(n) * sizeof(int32_t);
    BufSet s{
        xrt::bo(dev, bytes, xrt::bo::flags::host_only, k.group_id(kArgA)),
        xrt::bo(dev, bytes, xrt::bo::flags::host_only, k.group_id(kArgB)),
        xrt::bo(dev, bytes, xrt::bo::flags::host_only, k.group_id(kArgC)),
    };
    return s;
}

// arange(1, n+1) in, the same expected back out -- the passthrough design's contract.
static void fill_input(BufSet &s, int n) {
    auto *p = s.a.map<int32_t *>();
    for (int i = 0; i < n; ++i) p[i] = i + 1;
    s.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    auto *q = s.c.map<int32_t *>();
    std::memset(q, 0, static_cast<size_t>(n) * sizeof(int32_t));
    s.c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
}

static bool verify_output(BufSet &s, int n) {
    s.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    auto *p = s.c.map<int32_t *>();
    for (int i = 0; i < n; ++i)
        if (p[i] != i + 1) return false;
    return true;
}

static void set_args(xrt::run &r, xrt::bo &insts_bo, uint32_t n_words, BufSet &s) {
    r.set_arg(kArgOpcode, kOpcode);
    r.set_arg(kArgInstBo, insts_bo);
    r.set_arg(kArgInstCount, n_words);
    r.set_arg(kArgA, s.a);
    r.set_arg(kArgB, s.b);
    r.set_arg(kArgC, s.c);
}

int main(int argc, char **argv) {
    std::string xclbin_path, insts_path, entry_name = "(given on the command line)";
    std::string modes = "single,rebuild,persistent";
    std::string batches = "1,2,4,8,16,32,64";
    int payload = 4096, iters = 100, warmup = 5;
    bool use_cache = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--xclbin") xclbin_path = next();
        else if (a == "--insts") insts_path = next();
        else if (a == "--cache-newest") use_cache = true;
        else if (a == "--payload") payload = std::stoi(next());
        else if (a == "--iters") iters = std::stoi(next());
        else if (a == "--warmup") warmup = std::stoi(next());
        else if (a == "--batch-sizes") batches = next();
        else if (a == "--modes") modes = next();
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "dispatch_runner -- C++ XRT host for the AIE dispatch floor\n\n"
                "  --xclbin PATH      final.xclbin of a compiled IRON design\n"
                "  --insts PATH       insts.bin for the same design\n"
                "  --cache-newest     resolve both out of ~/.npu/cache/ (newest entry\n"
                "                     holding BOTH files), the same rule measure_runlist.py uses\n"
                "  --payload N        int32 element count per buffer (default 4096)\n"
                "  --iters N          timed iterations per point (default 100)\n"
                "  --warmup N         untimed warmup dispatches (default 5)\n"
                "  --batch-sizes L    comma-separated (default 1,2,4,8,16,32,64)\n"
                "  --modes L          any of single,rebuild,persistent (default all three)\n";
            return 0;
        } else {
            std::cerr << "unknown flag: " << a << "\n";
            return 2;
        }
    }

    if (use_cache && xclbin_path.empty()) {
        if (!resolve_from_cache(xclbin_path, insts_path, entry_name)) {
            std::cerr << "ERROR: no ~/.npu/cache/ entry holding both final.xclbin and insts.bin.\n"
                         "       Run kernels/dispatch_floor/measure_runlist.py once to populate it.\n";
            return 1;
        }
    }
    if (xclbin_path.empty() || insts_path.empty()) {
        std::cerr << "ERROR: need --xclbin and --insts, or --cache-newest. Try --help.\n";
        return 2;
    }

    auto want = [&](const char *m) { return modes.find(m) != std::string::npos; };
    std::vector<int> batch_list;
    { std::string t; std::istringstream ss(batches);
      while (std::getline(ss, t, ',')) if (!t.empty()) batch_list.push_back(std::stoi(t)); }

    try {
        std::cout << "== dispatch_runner (C++ XRT host)\n";
        std::cout << "   xclbin: " << xclbin_path << "\n";
        std::cout << "   insts:  " << insts_path << "\n";
        std::cout << "   cache entry: " << entry_name << "\n";
        std::cout << "   payload: " << payload << " int32 (" << payload * 4 << " B) per buffer\n";
        std::cout << "   iters: " << iters << "   warmup: " << warmup << "\n\n";

        // ---- one-time setup, timed as a whole: this is what a cached-handle runner
        // ---- pays once and never again.  IRON pays its ~500 us equivalent PER CALL.
        auto t_setup0 = clk::now();
        xrt::device device(0);
        xrt::xclbin xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());

        auto kernels = xclbin.get_kernels();
        if (kernels.empty()) throw std::runtime_error("no kernels in xclbin");
        std::string kname = kernels[0].get_name();
        xrt::kernel kernel(context, kname);

        auto insts = read_insts(insts_path);
        const auto n_words = static_cast<uint32_t>(insts.size());
        xrt::bo insts_bo(device, insts.size() * 4, xrt::bo::flags::cacheable, kernel.group_id(kArgInstBo));
        std::memcpy(insts_bo.map<uint32_t *>(), insts.data(), insts.size() * 4);
        insts_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto setup_us = std::chrono::duration<double, std::micro>(clk::now() - t_setup0).count();

        std::cout << "   kernel: " << kname << "   instruction words: " << n_words << "\n";
        std::cout << "   ONE-TIME SETUP (device+xclbin+context+kernel+instruction upload): "
                  << std::fixed << std::setprecision(1) << setup_us << " us\n"
                  << "   (paid once; every per-dispatch figure below excludes it)\n\n";

        std::cout << std::left << std::setw(12) << "mode" << std::right
                  << std::setw(7) << "batch" << std::setw(12) << "mean us"
                  << std::setw(12) << "min us" << std::setw(12) << "median us"
                  << std::setw(11) << "verified" << "\n";
        std::cout << std::string(66, '-') << "\n";

        auto row = [&](const char *mode, int batch, const Stats &s, bool ok) {
            std::cout << std::left << std::setw(12) << mode << std::right
                      << std::setw(7) << batch
                      << std::setw(12) << std::fixed << std::setprecision(2) << s.mean_us
                      << std::setw(12) << s.min_us
                      << std::setw(12) << s.median_us
                      << std::setw(11) << (ok ? "yes" : "NO") << "\n";
            std::cout.flush();
        };

        // ---------------- single: one dispatch at a time ----------------
        if (want("single")) {
            BufSet s = make_bufset(device, kernel, payload);
            fill_input(s, payload);
            for (int i = 0; i < warmup; ++i) {
                auto r = kernel(kOpcode, insts_bo, n_words, s.a, s.b, s.c);
                r.wait();
            }
            std::vector<double> t;
            t.reserve(iters);
            for (int i = 0; i < iters; ++i) {
                auto t0 = clk::now();
                auto r = kernel(kOpcode, insts_bo, n_words, s.a, s.b, s.c);
                r.wait();
                t.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
            }
            bool ok = verify_output(s, payload);
            row("single", 1, summarize(t, 1), ok);
        }

        // ---------------- rebuild: fresh runlist per iteration ----------------
        // Apples-to-apples with measure_runlist.py, which rebuilt the list every time.
        if (want("rebuild")) {
            for (int batch : batch_list) {
                std::vector<BufSet> sets;
                sets.reserve(batch);
                for (int i = 0; i < batch; ++i) {
                    sets.push_back(make_bufset(device, kernel, payload));
                    fill_input(sets.back(), payload);
                }
                for (int i = 0; i < warmup; ++i) {
                    auto r = kernel(kOpcode, insts_bo, n_words, sets[0].a, sets[0].b, sets[0].c);
                    r.wait();
                }
                std::vector<double> t;
                t.reserve(iters);
                for (int i = 0; i < iters; ++i) {
                    // DECLARATION ORDER IS LOAD-BEARING. Destruction runs in reverse,
                    // and the runlist holds references to the runs it was given, which
                    // in turn reference the buffers. runs must outlive rl (and sets
                    // must outlive both) or teardown is a use-after-free -- which is a
                    // segfault, not an exception, and it took one down this session.
                    std::vector<xrt::run> runs;
                    runs.reserve(batch);
                    xrt::runlist rl(context);
                    for (int j = 0; j < batch; ++j) {
                        xrt::run r(kernel);          // built UNSTARTED -- kernel(...) would start it
                        set_args(r, insts_bo, n_words, sets[j]);
                        runs.push_back(r);
                        rl.add(runs.back());
                    }
                    auto t0 = clk::now();
                    rl.execute();
                    rl.wait();
                    t.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
                }
                bool ok = true;
                for (auto &s : sets) ok = ok && verify_output(s, payload);
                row("rebuild", batch, summarize(t, batch), ok);
            }
        }

        // ---------------- persistent: build once, re-execute ----------------
        // The lever the Python harness never tried.  add() forbids mutation only while
        // the list is executing, and execute() only throws if it is already executing,
        // so a waited list can be executed again with its runs and arguments intact.
        if (want("persistent")) {
            for (int batch : batch_list) {
                std::vector<BufSet> sets;
                sets.reserve(batch);
                for (int i = 0; i < batch; ++i) {
                    sets.push_back(make_bufset(device, kernel, payload));
                    fill_input(sets.back(), payload);
                }
                // Same destruction-order rule as the rebuild arm: runs before rl.
                std::vector<xrt::run> runs;
                runs.reserve(batch);
                xrt::runlist rl(context);
                for (int j = 0; j < batch; ++j) {
                    xrt::run r(kernel);
                    set_args(r, insts_bo, n_words, sets[j]);
                    runs.push_back(r);
                    rl.add(runs.back());
                }
                for (int i = 0; i < warmup; ++i) { rl.execute(); rl.wait(); }
                std::vector<double> t;
                t.reserve(iters);
                for (int i = 0; i < iters; ++i) {
                    auto t0 = clk::now();
                    rl.execute();
                    rl.wait();
                    t.push_back(std::chrono::duration<double, std::micro>(clk::now() - t0).count());
                }
                bool ok = true;
                for (auto &s : sets) ok = ok && verify_output(s, payload);
                row("persistent", batch, summarize(t, batch), ok);
                // NOT runs.clear() here: the runlist is still alive and references
                // them. Scope exit tears them down in the correct order.
            }
        }

        std::cout << "\nAll per-dispatch figures are wall time around submit+wait only,\n"
                     "divided by the batch size.  A row is kept only if its output buffers\n"
                     "verified -- runlist gives no per-run status, so that is the only gate.\n";
    } catch (const std::exception &e) {
        std::cerr << "\nERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
