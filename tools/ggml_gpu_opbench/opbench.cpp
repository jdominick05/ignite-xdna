// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Per-op iGPU timings for EmbeddingGemma 2 on the llama.cpp ROCm backend (ROCm0), through public GGML APIs, for the
// iGPU/NPU placement question. Same runtime as the llama-server GPU baseline; the loader follows
// tools/ggml_gpu_embedding/embedding.cpp (exports resolved from ggml-base.dll / ggml.dll, backends loaded from BIN).
//
//   opbench BIN SPEC          SPEC lines (whitespace separated):
//     gemm M K N            y[M,N] = x[M,K] f32 @ W[K,N] bf16 (ggml_mul_mat, f32 out)
//     attn L HD H KV W      local (W > 0: |i-j| <= W) or global (W = 0) attention, q f32, k/v f16, scale 1:
//                           both ggml_flash_attn_ext and the classic mul_mat -> soft_max_ext(mask) -> mul_mat
//     rms L D               ggml_rms_norm over rows
//     copy BYTES            ggml_backend_tensor_set then _get of one f32 tensor (host <-> ROCm0 buffer)
//   Each op is built R = 8 times in one graph on shared inputs (independent outputs), so per-op time = graph time / R
//   amortizes launch cost as a forward graph would; the R = 1 graph's time is reported beside it. Median of 7
//   timed computes after 2 warm-ups; ggml_backend_graph_compute returns after the device finishes.
//   Output: one JSON object per line. No CPU fallback: every node must be supported by ROCm0 and live in its buffer;
//   an op ROCm0 does not support is reported with "ms": -1 and never run.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs = std::filesystem;

#define APIS(X) \
  X(ggml_init) X(ggml_free) X(ggml_nbytes) X(ggml_new_tensor_2d) X(ggml_new_tensor_3d) X(ggml_mul_mat) \
  X(ggml_flash_attn_ext) X(ggml_soft_max_ext) X(ggml_rms_norm) X(ggml_set_input) X(ggml_set_output) \
  X(ggml_new_graph_custom) X(ggml_build_forward_expand) X(ggml_graph_n_nodes) X(ggml_graph_node) \
  X(ggml_backend_load_all_from_path) X(ggml_backend_dev_by_name) X(ggml_backend_dev_init) \
  X(ggml_backend_dev_buffer_type) X(ggml_backend_alloc_ctx_tensors) X(ggml_backend_free) \
  X(ggml_backend_buffer_free) X(ggml_backend_buffer_is_host) X(ggml_backend_supports_op) \
  X(ggml_backend_tensor_set) X(ggml_backend_tensor_get) X(ggml_backend_graph_compute) \
  X(ggml_gallocr_new) X(ggml_gallocr_free) X(ggml_gallocr_alloc_graph)

struct Api {
#define DECLARE(name) decltype(&::name) name = nullptr;
  APIS(DECLARE)
#undef DECLARE
  std::vector<HMODULE> mods;
  template <class T> void bind(T &fn, const char *name) {
    for (auto m : mods)
      if (auto s = GetProcAddress(m, name)) { fn = reinterpret_cast<T>(s); return; }
    throw std::runtime_error(std::string("missing export ") + name);
  }
  explicit Api(const fs::path &bin) {
    if (!SetDllDirectoryW(bin.wstring().c_str())) throw std::runtime_error("SetDllDirectory");
    for (auto n : {"ggml-base.dll", "ggml.dll"}) {
      auto m = LoadLibraryExW((bin / n).wstring().c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
      if (!m) throw std::runtime_error(std::string("cannot load ") + n);
      mods.push_back(m);
    }
#define BIND(name) bind(name, #name);
    APIS(BIND)
#undef BIND
    ggml_backend_load_all_from_path(bin.string().c_str());
  }
};

static uint16_t to_bf16(float f) { uint32_t u; std::memcpy(&u, &f, 4); return uint16_t((u + 0x7fff + ((u >> 16) & 1)) >> 16); }
static uint16_t to_f16(float f) {   // small normal values only (inputs are N(0, 0.5))
  uint32_t u; std::memcpy(&u, &f, 4);
  uint32_t s = (u >> 16) & 0x8000; int e = int((u >> 23) & 0xff) - 127 + 15; uint32_t m = (u >> 13) & 0x3ff;
  if (e <= 0) return uint16_t(s); if (e >= 31) return uint16_t(s | 0x7bff);
  return uint16_t(s | (uint32_t(e) << 10) | m);
}

struct Bench {
  Api &api; ggml_backend_dev_t dev; ggml_backend_t be; std::mt19937 rng{7};
  explicit Bench(Api &a) : api(a) {
    dev = api.ggml_backend_dev_by_name("ROCm0");
    if (!dev) throw std::runtime_error("ROCm0 unavailable; no CPU fallback");
    be = api.ggml_backend_dev_init(dev, nullptr);
    if (!be) throw std::runtime_error("cannot init ROCm0");
  }
  void fill(ggml_tensor *t) {
    std::normal_distribution<float> nd(0.f, 0.5f);
    const size_t n = api.ggml_nbytes(t);
    std::vector<uint8_t> buf(n);
    if (t->type == GGML_TYPE_F32) { auto p = reinterpret_cast<float *>(buf.data()); for (size_t i = 0; i < n / 4; ++i) p[i] = nd(rng); }
    else if (t->type == GGML_TYPE_BF16) { auto p = reinterpret_cast<uint16_t *>(buf.data()); for (size_t i = 0; i < n / 2; ++i) p[i] = to_bf16(nd(rng) * 0.05f); }
    else if (t->type == GGML_TYPE_F16) { auto p = reinterpret_cast<uint16_t *>(buf.data()); for (size_t i = 0; i < n / 2; ++i) p[i] = to_f16(nd(rng)); }
    api.ggml_backend_tensor_set(t, buf.data(), 0, n);
  }
  // build(ctx, R) adds R op copies to the graph and returns their outputs; inputs are created once inside build
  // tensors the builder puts in `raw` are written from `raw_data` (e.g. the attention mask), the rest randomly
  template <class F> std::pair<double, double> time(F build, const std::vector<uint16_t> *raw_data = nullptr) {
    double out[2] = {0, 0};
    int idx = 0;
    for (int R : {8, 1}) {
      ggml_init_params p{}; p.mem_size = 64 * 1024 * 1024; p.no_alloc = true;
      std::unique_ptr<ggml_context, decltype(&::ggml_free)> ctx(api.ggml_init(p), api.ggml_free);
      std::vector<ggml_tensor *> inputs, raw;
      auto outs = build(ctx.get(), R, inputs, raw);
      auto g = api.ggml_new_graph_custom(ctx.get(), 1024, false);
      for (auto o : outs) { api.ggml_set_output(o); api.ggml_build_forward_expand(g, o); }
      for (int i = 0; i < api.ggml_graph_n_nodes(g); ++i)
        if (!api.ggml_backend_supports_op(be, api.ggml_graph_node(g, i))) return {-1.0, -1.0};   // unsupported: reported, never run on CPU
      std::unique_ptr<ggml_backend_buffer, decltype(&::ggml_backend_buffer_free)> inbuf(
          api.ggml_backend_alloc_ctx_tensors(ctx.get(), be), api.ggml_backend_buffer_free);
      std::unique_ptr<ggml_gallocr, decltype(&::ggml_gallocr_free)> ga(api.ggml_gallocr_new(api.ggml_backend_dev_buffer_type(dev)), api.ggml_gallocr_free);
      if (!api.ggml_gallocr_alloc_graph(ga.get(), g)) throw std::runtime_error("graph allocation failed");
      for (auto t : inputs) fill(t);
      for (auto t : raw) {
        if (!raw_data || raw_data->size() * 2 != api.ggml_nbytes(t)) throw std::runtime_error("raw input size");
        api.ggml_backend_tensor_set(t, raw_data->data(), 0, api.ggml_nbytes(t));
      }
      for (int i = 0; i < api.ggml_graph_n_nodes(g); ++i)
        if (api.ggml_backend_buffer_is_host(api.ggml_graph_node(g, i)->buffer)) throw std::runtime_error("node in a host buffer");
      std::vector<double> ts;
      for (int rep = 0; rep < 9; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        if (api.ggml_backend_graph_compute(be, g) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
        auto t1 = std::chrono::steady_clock::now();
        if (rep >= 2) ts.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
      }
      std::sort(ts.begin(), ts.end());
      out[idx++] = ts[ts.size() / 2] / R;
    }
    return {out[0], out[1]};
  }
};

static std::vector<uint16_t> window_mask(int64_t L, int64_t W) {
  std::vector<uint16_t> m(size_t(L * L));
  for (int64_t i = 0; i < L; ++i)
    for (int64_t j = 0; j < L; ++j) m[size_t(i * L + j)] = (W == 0 || std::llabs(i - j) <= W) ? 0 : 0xfc00;   // -inf
  return m;
}

int main(int argc, char **argv) {
  try {
    if (argc != 3) { std::cerr << "usage: opbench BIN SPEC\n"; return 2; }
    Api api(argv[1]);
    Bench b(api);
    std::ifstream spec(argv[2]);
    std::string line;
    while (std::getline(spec, line)) {
      std::istringstream in(line);
      std::string op; if (!(in >> op) || op[0] == '#') continue;
      if (op == "gemm") {
        int64_t M, K, N; in >> M >> K >> N;
        auto r = b.time([&](ggml_context *c, int R, auto &inputs, auto &) {
          auto W = api.ggml_new_tensor_2d(c, GGML_TYPE_BF16, K, N), x = api.ggml_new_tensor_2d(c, GGML_TYPE_F32, K, M);
          api.ggml_set_input(W); api.ggml_set_input(x); inputs = {W, x};
          std::vector<ggml_tensor *> o; for (int i = 0; i < R; ++i) o.push_back(api.ggml_mul_mat(c, W, x)); return o;
        });
        std::cout << "{\"op\": \"gemm\", \"M\": " << M << ", \"K\": " << K << ", \"N\": " << N << ", \"ms\": " << r.first
                  << ", \"ms_single\": " << r.second << ", \"gflops\": " << 2.0 * M * K * N / r.first / 1e6 << "}" << std::endl;
      } else if (op == "attn") {
        int64_t L, HD, H, KV, W; in >> L >> HD >> H >> KV >> W;
        auto mask = window_mask(L, W);
        for (int flash : {1, 0}) {
          auto r = b.time([&](ggml_context *c, int R, auto &inputs, auto &raw) {
            auto q = api.ggml_new_tensor_3d(c, GGML_TYPE_F32, HD, L, H);
            auto k = api.ggml_new_tensor_3d(c, GGML_TYPE_F16, HD, L, KV);
            auto v = flash ? api.ggml_new_tensor_3d(c, GGML_TYPE_F16, HD, L, KV) : api.ggml_new_tensor_3d(c, GGML_TYPE_F16, L, HD, KV);
            auto mask_t = api.ggml_new_tensor_2d(c, GGML_TYPE_F16, L, L);
            for (auto t : {q, k, v, mask_t}) api.ggml_set_input(t);
            inputs = {q, k, v};
            raw = {mask_t};
            std::vector<ggml_tensor *> o;
            for (int i = 0; i < R; ++i) {
              if (flash) o.push_back(api.ggml_flash_attn_ext(c, q, k, v, mask_t, 1.0f, 0.0f, 0.0f));
              else {
                auto kq = api.ggml_mul_mat(c, k, q);                         // [L_kv, L_q, H]
                auto p = api.ggml_soft_max_ext(c, kq, mask_t, 1.0f, 0.0f);
                o.push_back(api.ggml_mul_mat(c, v, p));                      // [HD, L_q, H]
              }
            }
            return o;
          }, &mask);
          std::cout << "{\"op\": \"attn\", \"impl\": \"" << (flash ? "flash" : "classic") << "\", \"L\": " << L << ", \"hd\": " << HD
                    << ", \"H\": " << H << ", \"kv\": " << KV << ", \"window\": " << W << ", \"ms\": " << r.first
                    << ", \"ms_single\": " << r.second << "}" << std::endl;
        }
      } else if (op == "rms") {
        int64_t L, D; in >> L >> D;
        auto r = b.time([&](ggml_context *c, int R, auto &inputs, auto &) {
          auto x = api.ggml_new_tensor_2d(c, GGML_TYPE_F32, D, L); api.ggml_set_input(x); inputs = {x};
          std::vector<ggml_tensor *> o; for (int i = 0; i < R; ++i) o.push_back(api.ggml_rms_norm(c, x, 1e-6f)); return o;
        });
        std::cout << "{\"op\": \"rms\", \"L\": " << L << ", \"D\": " << D << ", \"ms\": " << r.first << ", \"ms_single\": " << r.second << "}" << std::endl;
      } else if (op == "copy") {
        int64_t bytes; in >> bytes;
        ggml_init_params p{}; p.mem_size = 1024 * 1024; p.no_alloc = true;
        std::unique_ptr<ggml_context, decltype(&::ggml_free)> ctx(api.ggml_init(p), api.ggml_free);
        auto t = api.ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, bytes / 4, 1);
        std::unique_ptr<ggml_backend_buffer, decltype(&::ggml_backend_buffer_free)> buf(
            api.ggml_backend_alloc_ctx_tensors(ctx.get(), b.be), api.ggml_backend_buffer_free);
        std::vector<uint8_t> host(size_t(bytes), 1);
        std::vector<double> ts_set, ts_get;
        for (int rep = 0; rep < 12; ++rep) {
          auto t0 = std::chrono::steady_clock::now();
          api.ggml_backend_tensor_set(t, host.data(), 0, size_t(bytes));
          auto t1 = std::chrono::steady_clock::now();
          api.ggml_backend_tensor_get(t, host.data(), 0, size_t(bytes));
          auto t2 = std::chrono::steady_clock::now();
          if (rep >= 2) { ts_set.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
                          ts_get.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count()); }
        }
        std::sort(ts_set.begin(), ts_set.end()); std::sort(ts_get.begin(), ts_get.end());
        double s = ts_set[ts_set.size() / 2], g = ts_get[ts_get.size() / 2];
        std::cout << "{\"op\": \"copy\", \"bytes\": " << bytes << ", \"set_ms\": " << s << ", \"get_ms\": " << g
                  << ", \"set_gbps\": " << bytes / s / 1e6 << ", \"get_gbps\": " << bytes / g / 1e6
                  << ", \"buffer_is_host\": " << (api.ggml_backend_buffer_is_host(buf.get()) ? "true" : "false") << "}" << std::endl;
      } else {
        throw std::runtime_error("unknown spec op " + op);
      }
    }
    api.ggml_backend_free(b.be);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR " << e.what() << std::endl;
    return 1;
  }
}
