// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// GPU-only GGUF embedding gather and scale, using public GGML APIs.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
namespace fs=std::filesystem;
#define EMBEDDING_APIS(X) \
 X(gguf_init_from_file) X(gguf_free) X(gguf_find_tensor) X(gguf_get_data_offset) X(gguf_get_tensor_offset) \
 X(ggml_init) X(ggml_free) X(ggml_get_tensor) X(ggml_dup_tensor) X(ggml_nbytes) \
 X(ggml_new_tensor_1d) X(ggml_new_tensor_2d) X(ggml_get_rows) X(ggml_mul) X(ggml_set_input) X(ggml_set_output) \
 X(ggml_new_graph_custom) X(ggml_build_forward_expand) X(ggml_graph_n_nodes) X(ggml_graph_node) \
 X(ggml_backend_load_all_from_path) X(ggml_backend_dev_by_name) X(ggml_backend_dev_init) \
 X(ggml_backend_dev_buffer_type) X(ggml_backend_alloc_ctx_tensors) X(ggml_backend_free) \
 X(ggml_backend_buffer_free) X(ggml_backend_buffer_is_host) X(ggml_backend_supports_op) \
 X(ggml_backend_tensor_set) X(ggml_backend_tensor_get) X(ggml_backend_graph_compute) \
 X(ggml_gallocr_new) X(ggml_gallocr_free) X(ggml_gallocr_alloc_graph)
struct Runtime {
#define DECLARE(name) decltype(&::name) name=nullptr;
    EMBEDDING_APIS(DECLARE)
#undef DECLARE
    std::vector<HMODULE> modules;
    template<class T> void bind(T &fn,const char *name) {
        for(auto module:modules)if(auto symbol=GetProcAddress(module,name)) {fn=reinterpret_cast<T>(symbol);return;}
        throw std::runtime_error(std::string("Missing runtime export: ")+name);
    }
    explicit Runtime(const fs::path &bin) {
        if(!SetDllDirectoryW(bin.wstring().c_str()))throw std::runtime_error("Cannot set runtime DLL path");
        for(auto name:{"ggml-base.dll","ggml.dll"}) {
            auto module=LoadLibraryExW((bin/name).wstring().c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
            if(!module)throw std::runtime_error(std::string("Cannot load runtime: ")+name);
            modules.push_back(module);
        }
#define BIND(name) bind(name,#name);
        EMBEDDING_APIS(BIND)
#undef BIND
        ggml_backend_load_all_from_path(bin.string().c_str());
    }
    // Backend registry modules are process-lifetime objects. Keep this runtime
    // and its loaded libraries alive across lookups, never rebuild per token.
};
using Context=std::unique_ptr<ggml_context,decltype(&::ggml_free)>;
using Gguf=std::unique_ptr<gguf_context,decltype(&::gguf_free)>;
using Backend=std::unique_ptr<ggml_backend,decltype(&::ggml_backend_free)>;
using Buffer=std::unique_ptr<ggml_backend_buffer,decltype(&::ggml_backend_buffer_free)>;
using Allocator=std::unique_ptr<ggml_gallocr,decltype(&::ggml_gallocr_free)>;
struct GpuEmbedding {
    Runtime &api;
    Backend backend;
    Context metadata;
    Gguf gguf;
    Context weights_context;
    Buffer weights_buffer;
    ggml_backend_dev_t device=nullptr;
    ggml_tensor *weight=nullptr;
    size_t weight_bytes=0;
    explicit GpuEmbedding(Runtime &a,const fs::path &model):api(a),backend(nullptr,a.ggml_backend_free),
        metadata(nullptr,a.ggml_free),gguf(nullptr,a.gguf_free),weights_context(nullptr,a.ggml_free),
        weights_buffer(nullptr,a.ggml_backend_buffer_free) {
        device=api.ggml_backend_dev_by_name("ROCm0");
        if(!device)throw std::runtime_error("ROCm0 unavailable; no CPU fallback");
        backend.reset(api.ggml_backend_dev_init(device,nullptr));
        if(!backend)throw std::runtime_error("Cannot initialize ROCm0");
        ggml_context *meta=nullptr;gguf_init_params gp{};gp.no_alloc=true;gp.ctx=&meta;
        gguf.reset(api.gguf_init_from_file(model.string().c_str(),gp));metadata.reset(meta);
        if(!gguf || !metadata)throw std::runtime_error("Cannot read GGUF metadata");
        const auto id=api.gguf_find_tensor(gguf.get(),"token_embd.weight");
        const auto original=api.ggml_get_tensor(metadata.get(),"token_embd.weight");
        if(id<0 || !original || original->ne[0]<1 || original->ne[0]>8192 ||
           original->ne[1]<1 || original->ne[1]>16777216 || original->ne[2]!=1 || original->ne[3]!=1)
            throw std::runtime_error("Unsupported token embedding dimensions");
        ggml_init_params wp{};wp.mem_size=1024*1024;wp.no_alloc=true;
        weights_context.reset(api.ggml_init(wp));
        if(!weights_context)throw std::runtime_error("Weight context allocation failed");
        weight=api.ggml_dup_tensor(weights_context.get(),original);
        if(!weight)throw std::runtime_error("Cannot clone embedding metadata");
        weight_bytes=api.ggml_nbytes(weight);
        if(!weight_bytes || weight_bytes>size_t(2)*1024*1024*1024)throw std::runtime_error("Embedding byte bound");
        weights_buffer.reset(api.ggml_backend_alloc_ctx_tensors(weights_context.get(),backend.get()));
        if(!weights_buffer || api.ggml_backend_buffer_is_host(weights_buffer.get()))
            throw std::runtime_error("Embedding weights must be in a device buffer");
        const auto offset=api.gguf_get_data_offset(gguf.get())+api.gguf_get_tensor_offset(gguf.get(),id);
        std::ifstream file(model,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error("Cannot open model weights");
        const auto size=file.tellg();
        if(size<0 || offset>uint64_t(size) || weight_bytes>uint64_t(size)-offset)
            throw std::runtime_error("Embedding offset exceeds model file");
        file.seekg(std::streamoff(offset));std::vector<char> chunk(16*1024*1024);
        for(size_t done=0;done<weight_bytes;) {
            const auto bytes=std::min(chunk.size(),weight_bytes-done);
            file.read(chunk.data(),std::streamsize(bytes));
            if(!file)throw std::runtime_error("Embedding weight read failed");
            api.ggml_backend_tensor_set(weight,chunk.data(),done,bytes);done+=bytes;
        }
    }
    std::vector<float> lookup(const std::vector<int32_t> &tokens,float scale) {
        if(tokens.empty() || tokens.size()>256 || !std::isfinite(scale) || scale<=0 || scale>128)
            throw std::runtime_error("Lookup input bound");
        for(auto token:tokens)if(token<0 || token>=weight->ne[1])throw std::runtime_error("Token outside vocabulary");
        ggml_init_params gp{};gp.mem_size=1024*1024;gp.no_alloc=true;
        Context graph_context(api.ggml_init(gp),api.ggml_free);
        if(!graph_context)throw std::runtime_error("Graph metadata allocation failed");
        auto ids=api.ggml_new_tensor_1d(graph_context.get(),GGML_TYPE_I32,int64_t(tokens.size()));
        auto rows=api.ggml_get_rows(graph_context.get(),weight,ids);
        // Match llama.cpp's row-scale MUL exactly. SCALE adds a zero bias,
        // which changes negative zero in quantized embedding rows.
        auto scales=api.ggml_new_tensor_2d(graph_context.get(),GGML_TYPE_F32,1,int64_t(tokens.size()));
        auto scaled=api.ggml_mul(graph_context.get(),rows,scales);
        api.ggml_set_input(ids);api.ggml_set_input(scales);api.ggml_set_output(scaled);
        auto graph=api.ggml_new_graph_custom(graph_context.get(),32,false);
        api.ggml_build_forward_expand(graph,scaled);
        if(api.ggml_graph_n_nodes(graph)!=2)throw std::runtime_error("Unexpected lookup graph");
        for(int i=0;i<api.ggml_graph_n_nodes(graph);++i)
            if(!api.ggml_backend_supports_op(backend.get(),api.ggml_graph_node(graph,i)))
                throw std::runtime_error("GPU does not support lookup node; no CPU fallback");
        Allocator allocator(api.ggml_gallocr_new(api.ggml_backend_dev_buffer_type(device)),api.ggml_gallocr_free);
        if(!allocator || !api.ggml_gallocr_alloc_graph(allocator.get(),graph))throw std::runtime_error("GPU graph allocation failed");
        for(int i=0;i<api.ggml_graph_n_nodes(graph);++i)
            if(api.ggml_backend_buffer_is_host(api.ggml_graph_node(graph,i)->buffer))
                throw std::runtime_error("Lookup graph node has a host buffer");
        api.ggml_backend_tensor_set(ids,tokens.data(),0,tokens.size()*sizeof(int32_t));
        const std::vector<float> scale_values(tokens.size(),scale);
        api.ggml_backend_tensor_set(scales,scale_values.data(),0,scale_values.size()*sizeof(float));
        if(api.ggml_backend_graph_compute(backend.get(),graph)!=GGML_STATUS_SUCCESS)
            throw std::runtime_error("GPU lookup failed");
        std::vector<float> result(tokens.size()*size_t(weight->ne[0]));
        api.ggml_backend_tensor_get(scaled,result.data(),0,result.size()*sizeof(float));
        return result;
    }
};
#ifndef IGNITE_GGML_EMBEDDING_LIBRARY
int main(int argc,char **argv) {
    try {
        if(argc!=6)throw std::runtime_error("Usage: ggml_gpu_embedding BIN MODEL TOKENS_I32 NEW_OUTPUT_DIR SCALE");
        const auto out=fs::absolute(argv[4]);
        if(fs::exists(out))throw std::runtime_error("Refusing to replace embedding output");
        std::ifstream input(argv[3],std::ios::binary|std::ios::ate);
        if(!input)throw std::runtime_error("Token file missing");
        const auto bytes=input.tellg();
        if(bytes<4 || bytes>1024 || bytes%4)throw std::runtime_error("Token file byte bound");
        input.seekg(0);std::vector<int32_t> tokens(size_t(bytes)/4);input.read(reinterpret_cast<char *>(tokens.data()),bytes);
        if(!input)throw std::runtime_error("Token file read failed");
        const std::string arg=argv[5];size_t used=0;const float scale=std::stof(arg,&used);
        if(used!=arg.size())throw std::runtime_error("Invalid scale");
        Runtime runtime(fs::absolute(argv[1]));GpuEmbedding embedding(runtime,fs::absolute(argv[2]));
        auto values=embedding.lookup(tokens,scale);
        // Exercise persistent weights/backend and changing graph shapes: A/B/A
        // one-token requests after the batch, with strict signed-zero checks.
        const auto first=embedding.lookup({tokens.front()},scale);
        const auto last=embedding.lookup({tokens.back()},scale);
        const auto repeat=embedding.lookup({tokens.front()},scale);
        const auto row_bytes=size_t(embedding.weight->ne[0])*sizeof(float);
        if(std::memcmp(first.data(),values.data(),row_bytes) ||
           std::memcmp(last.data(),values.data()+values.size()-last.size(),row_bytes) ||
           std::memcmp(first.data(),repeat.data(),row_bytes))
            throw std::runtime_error("Persistent GPU embedding replay differs from batch");
        fs::create_directories(out);std::ofstream data(out/"embeddings.f32",std::ios::binary);
        data.write(reinterpret_cast<const char *>(values.data()),std::streamsize(values.size()*sizeof(float)));data.close();
        if(!data)throw std::runtime_error("Embedding output write failed");
        std::ofstream metadata(out/"lookup.json");
        metadata<<"{\"tokens\":"<<tokens.size()<<",\"hidden\":"<<embedding.weight->ne[0]
                <<",\"vocabulary\":"<<embedding.weight->ne[1]<<",\"weight_type\":"<<int(embedding.weight->type)
                <<",\"weight_bytes\":"<<embedding.weight_bytes<<",\"scale\":"<<std::setprecision(9)<<scale
                <<",\"nodes\":2,\"scale_op\":\"MUL\",\"lookups\":4,\"reuse_checked\":true"
                <<",\"device_buffers\":true,\"backend\":\"ROCm0\",\"complete\":true}\n";
        metadata.close();if(!metadata)throw std::runtime_error("Lookup metadata write failed");
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
#endif
