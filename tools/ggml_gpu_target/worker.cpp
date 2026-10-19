// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Bounded binary-stdio target bridge. Public llama/GGML APIs, GPU arithmetic.
#define IGNITE_GGML_EMBEDDING_LIBRARY
#include "../ggml_gpu_embedding/embedding.cpp"
#include "llama.h"
#include <fcntl.h>
#include <io.h>
#include <map>
#include <chrono>
#include <iomanip>
#include <locale>
#include <sstream>

using TargetClock=std::chrono::steady_clock;
static double elapsed_ms(TargetClock::time_point start) {
    return std::chrono::duration<double,std::milli>(TargetClock::now()-start).count();
}
struct DecodeStats {
    uint64_t calls=0,tokens=0,capture_calls=0,captured_bytes=0,response_bytes=0;
    double process_ms=0,capture_ms=0,logits_ms=0,tensors_pack_ms=0,send_ms=0;
    void json(std::ostream &out) const {
        out<<"{\"calls\":"<<calls<<",\"tokens\":"<<tokens<<",\"capture_calls\":"<<capture_calls
           <<",\"captured_bytes\":"<<captured_bytes<<",\"response_bytes\":"<<response_bytes
           <<",\"process_ms\":"<<process_ms<<",\"capture_ms\":"<<capture_ms
           <<",\"logits_ms\":"<<logits_ms<<",\"tensors_pack_ms\":"<<tensors_pack_ms<<",\"send_ms\":"<<send_ms<<"}";
    }
};

#define TARGET_APIS(X) \
 X(llama_backend_init) X(llama_backend_free) X(llama_model_default_params) \
 X(llama_context_default_params) X(llama_model_load_from_file) X(llama_model_free) \
 X(llama_init_from_model) X(llama_free) X(llama_model_get_vocab) X(llama_vocab_n_tokens) \
 X(llama_batch_ext_init) X(llama_batch_ext_free) X(llama_batch_ext_clear) \
 X(llama_batch_ext_add_token) X(llama_batch_ext_set_pos) X(llama_batch_ext_set_output_logits) \
 X(llama_process) X(llama_synchronize) X(llama_get_logits_ith) X(llama_get_memory) X(llama_memory_seq_rm) X(llama_memory_clear) \
 X(llama_token_to_piece) X(llama_vocab_is_eog)
struct TargetApi {
#define DECLARE(name) decltype(&::name) name=nullptr;
    TARGET_APIS(DECLARE)
#undef DECLARE
    explicit TargetApi(Runtime &runtime,const fs::path &bin) {
        auto module=LoadLibraryExW((bin/"llama.dll").wstring().c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!module)throw std::runtime_error("Cannot load llama.dll");
        runtime.modules.push_back(module);
#define BIND(name) runtime.bind(name,#name);
        TARGET_APIS(BIND)
#undef BIND
    }
};
struct Bytes {
    std::vector<char> data;
    void word(uint32_t x) {const char *p=reinterpret_cast<const char *>(&x);data.insert(data.end(),p,p+4);}
    void raw(const void *p,size_t n) {const char *b=static_cast<const char *>(p);data.insert(data.end(),b,b+n);}
    void text(const std::string &s) {word(uint32_t(s.size()));raw(s.data(),s.size());}
    void send() const {
        const uint32_t n=uint32_t(data.size());std::cout.write(reinterpret_cast<const char *>(&n),4);
        std::cout.write(data.data(),n);std::cout.flush();
        if(!std::cout)throw std::runtime_error("Bridge output pipe closed");
    }
};
struct TensorCopy {
    uint32_t type=0;uint32_t ne[4]{},nb[4]{};std::vector<char> data;
    void append(Bytes &out,const std::string &name) const {
        out.text(name);out.word(type);for(auto x:ne)out.word(x);for(auto x:nb)out.word(x);
        out.word(uint32_t(data.size()));out.raw(data.data(),data.size());
    }
};
struct LiveCapture {
    Runtime &api;std::map<std::string,TensorCopy> tensors;std::string failure;
    double capture_ms=0;uint64_t capture_calls=0,captured_bytes=0;
    explicit LiveCapture(Runtime &a):api(a) {}
    static bool callback(ggml_tensor *tensor,bool ask,void *opaque) {
        auto &self=*static_cast<LiveCapture *>(opaque);const std::string name=tensor->name;
        const bool selected=name=="result_norm" || name=="h_nextn" || name=="Kcur_pos-13" ||
            name=="Vcur_normed-13" || name=="Kcur_pos-14" || name=="Vcur_normed-14";
        if(ask)return selected;
        if(!selected)return true;
        const auto started=TargetClock::now();
        try {
            if(tensor->type!=GGML_TYPE_F32 || !tensor->buffer || self.api.ggml_backend_buffer_is_host(tensor->buffer))
                throw std::runtime_error("Target state must be FP32 in a GPU buffer");
            TensorCopy copy;const auto n=self.api.ggml_nbytes(tensor);
            if(!n || n>16*1024*1024)throw std::runtime_error("Target tensor byte bound");
            for(int i=0;i<4;++i) {
                if(tensor->ne[i]<1 || tensor->ne[i]>65536 || tensor->nb[i]>32*1024*1024)
                    throw std::runtime_error("Target tensor dimension bound");
                copy.ne[i]=uint32_t(tensor->ne[i]);copy.nb[i]=uint32_t(tensor->nb[i]);
            }
            copy.data.resize(n);self.api.ggml_backend_tensor_get(tensor,copy.data.data(),0,n);
            if(!self.tensors.emplace(name,std::move(copy)).second)throw std::runtime_error("Duplicate target tensor in batch");
            self.capture_ms+=elapsed_ms(started);++self.capture_calls;self.captured_bytes+=n;
            return true;
        } catch(const std::exception &e) {self.failure=e.what();return false;}
    }
};
static std::vector<int32_t> read_tokens(const fs::path &path) {
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)throw std::runtime_error("Prompt token file missing");
    const auto n=input.tellg();if(n<4 || n>8192 || n%4)throw std::runtime_error("Prompt token byte bound");
    input.seekg(0);std::vector<int32_t> ids(size_t(n)/4);input.read(reinterpret_cast<char *>(ids.data()),n);
    if(!input)throw std::runtime_error("Prompt token read failed");return ids;
}
int main(int argc,char **argv) {
    _setmode(_fileno(stdin),_O_BINARY);_setmode(_fileno(stdout),_O_BINARY);
    try {
        if(argc!=4 && argc!=5)throw std::runtime_error("Usage: ggml_gpu_target BIN E2B_GGUF PROMPT_I32 [512|2560]");
        uint32_t context=512;
        if(argc==5) {
            const std::string requested=argv[4];
            if(requested=="2560")context=2560;
            else if(requested!="512")throw std::runtime_error("Use context 512 or 2560");
        }
        const size_t prompt_limit=context==512?256:2048;
        const auto bin=fs::absolute(argv[1]);const auto prompt=read_tokens(argv[3]);
        Runtime runtime(bin);TargetApi api(runtime,bin);api.llama_backend_init();
        {
            GpuEmbedding embedding(runtime,fs::absolute(argv[2]));
            if(embedding.weight->ne[0]!=1536 || embedding.weight->ne[1]!=262144)
                throw std::runtime_error("Bridge currently supports Gemma4 E2B only");
            auto buft=runtime.ggml_backend_dev_buffer_type(embedding.device);
            llama_model_tensor_buft_override overrides[]={{"token_embd\\.weight",buft},{"per_layer_token_embd\\.weight",buft},{nullptr,nullptr}};
            auto mp=api.llama_model_default_params();mp.n_gpu_layers=-1;mp.tensor_buft_overrides=overrides;
            std::unique_ptr<llama_model,decltype(api.llama_model_free)> model(api.llama_model_load_from_file(argv[2],mp),api.llama_model_free);
            if(!model)throw std::runtime_error("GPU target model load failed");
            const auto vocab=api.llama_model_get_vocab(model.get());const int vocabulary=api.llama_vocab_n_tokens(vocab);
            auto validate=[&](const std::vector<int32_t> &ids,size_t limit=256) {
                if(ids.empty() || ids.size()>limit)throw std::runtime_error("Batch length outside operation bound");
                for(auto id:ids)if(id<0 || id>=vocabulary)throw std::runtime_error("Token outside vocabulary");
            };validate(prompt,prompt_limit);
            LiveCapture capture(runtime);auto cp=api.llama_context_default_params();
            cp.n_ctx=context;cp.n_batch=uint32_t(prompt_limit);cp.n_ubatch=uint32_t(prompt_limit);cp.n_threads=4;cp.n_threads_batch=4;
            cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
            cp.cb_eval=LiveCapture::callback;cp.cb_eval_user_data=&capture;
            std::unique_ptr<llama_context,decltype(api.llama_free)> ctx(api.llama_init_from_model(model.get(),cp),api.llama_free);
            if(!ctx)throw std::runtime_error("GPU target context failed");
            std::unique_ptr<llama_batch_ext,decltype(api.llama_batch_ext_free)> batch(api.llama_batch_ext_init(ctx.get()),api.llama_batch_ext_free);
            if(!batch)throw std::runtime_error("Target batch allocation failed");
            uint32_t position=0,prefix_floor=uint32_t(prompt.size());
            std::map<uint32_t,DecodeStats> timing;
            auto decode=[&](const std::vector<int32_t> &ids,uint32_t opcode) {
                validate(ids,opcode==1?256:prompt_limit);if(position+ids.size()>context)throw std::runtime_error("Target context bound");
                capture.tensors.clear();capture.failure.clear();api.llama_batch_ext_clear(batch.get());
                capture.capture_ms=0;capture.capture_calls=0;capture.captured_bytes=0;
                for(size_t i=0;i<ids.size();++i) {
                    const auto row=api.llama_batch_ext_add_token(batch.get(),0,ids[i]);llama_pos pos=int(position+i);
                    api.llama_batch_ext_set_pos(batch.get(),row,&pos);
                    api.llama_batch_ext_set_output_logits(batch.get(),row,opcode==1 || i+1==ids.size());
                }
                const auto process_start=TargetClock::now();
                if(api.llama_process(ctx.get(),LLAMA_PROCESS_TYPE_DECODE,batch.get())!=0)
                    throw std::runtime_error("Target decode failed: "+capture.failure);
                api.llama_synchronize(ctx.get());if(!capture.failure.empty())throw std::runtime_error(capture.failure);
                const auto process_ms=elapsed_ms(process_start);
                for(const auto name:{"result_norm","Kcur_pos-13","Vcur_normed-13","Kcur_pos-14","Vcur_normed-14"})
                    if(!capture.tensors.count(name))throw std::runtime_error(std::string("Missing target state: ")+name);
                const auto outputs=opcode==1 ? ids.size() : size_t(1);
                position+=uint32_t(ids.size());Bytes out;out.word(0);out.word(opcode);out.word(position);out.word(uint32_t(outputs));
                const auto logits_start=TargetClock::now();
                for(size_t i=0;i<outputs;++i) {
                    const float *logits=api.llama_get_logits_ith(ctx.get(),opcode==1 ? int(i) : -1);
                    if(!logits)throw std::runtime_error("Missing verification row logits");
                    for(int j=0;j<vocabulary;++j)if(!std::isfinite(logits[j]))throw std::runtime_error("Nonfinite verification logits");
                    out.word(uint32_t(std::max_element(logits,logits+vocabulary)-logits));
                }
                const auto logits_ms=elapsed_ms(logits_start);const auto pack_start=TargetClock::now();
                out.word(uint32_t(capture.tensors.size()));for(const auto &entry:capture.tensors)entry.second.append(out,entry.first);
                const auto pack_ms=elapsed_ms(pack_start);const auto send_start=TargetClock::now();
                out.send();
                auto &stats=timing[opcode];++stats.calls;stats.tokens+=ids.size();
                stats.process_ms+=process_ms;stats.capture_ms+=capture.capture_ms;stats.logits_ms+=logits_ms;
                stats.tensors_pack_ms+=pack_ms;stats.send_ms+=elapsed_ms(send_start);
                stats.capture_calls+=capture.capture_calls;stats.captured_bytes+=capture.captured_bytes;stats.response_bytes+=out.data.size()+4;
            };
            decode(prompt,4);
            for(;;) {
                uint32_t bytes=0;if(!std::cin.read(reinterpret_cast<char *>(&bytes),4))break;
                if(bytes<8 || bytes>8+4*prompt_limit || bytes%4)throw std::runtime_error("Request frame byte bound");
                std::vector<uint32_t> words(bytes/4);std::cin.read(reinterpret_cast<char *>(words.data()),bytes);
                if(!std::cin)throw std::runtime_error("Truncated bridge request");
                const auto op=words[0],n=words[1];if(words.size()!=size_t(n)+2)throw std::runtime_error("Request length mismatch");
                if(op==0 && n==0)break;
                if(op==1) {decode(std::vector<int32_t>(words.begin()+2,words.end()),1);continue;}
                if(op==4 && n==0) {api.llama_memory_clear(api.llama_get_memory(ctx.get()),false);position=0;prefix_floor=uint32_t(prompt.size());decode(prompt,4);continue;}
                if(op==7) {
                    const std::vector<int32_t> ids(words.begin()+2,words.end());validate(ids,prompt_limit);
                    api.llama_memory_clear(api.llama_get_memory(ctx.get()),false);position=0;prefix_floor=n;
                    decode(ids,7);continue;
                }
                Bytes out;out.word(0);out.word(op);
                if(op==8 && n==0) {
                    std::ostringstream report;report.imbue(std::locale::classic());report<<std::setprecision(17)<<"{\"version\":1,\"groups\":{";
                    bool first=true;for(const auto &entry:timing) {
                        if(!first)report<<',';first=false;
                        report<<'"'<<entry.first<<"\":";entry.second.json(report);
                    }
                    report<<"}}";out.word(position);out.word(0);out.word(0);out.text(report.str());out.send();continue;
                }
                if(op==2 && n==1) {
                    const auto keep=words[2];if(keep<prefix_floor || keep>position)throw std::runtime_error("Rollback outside generated suffix");
                    if(!api.llama_memory_seq_rm(api.llama_get_memory(ctx.get()),0,int(keep),-1))throw std::runtime_error("Target rollback refused");
                    position=keep;out.word(position);out.word(0);out.word(0);out.send();continue;
                }
                if(op==3) {
                    const std::vector<int32_t> ids(words.begin()+2,words.end());validate(ids);
                    const auto values=embedding.lookup(ids,std::sqrt(1536.0f));TensorCopy copy;
                    copy.ne[0]=1536;copy.ne[1]=n;copy.ne[2]=copy.ne[3]=1;
                    copy.nb[0]=4;copy.nb[1]=6144;copy.nb[2]=copy.nb[3]=6144*n;
                    copy.data.resize(values.size()*4);std::memcpy(copy.data.data(),values.data(),copy.data.size());
                    out.word(position);out.word(0);out.word(1);copy.append(out,"embedding");out.send();continue;
                }
                if(op==5) {
                    const std::vector<int32_t> ids(words.begin()+2,words.end());validate(ids);
                    std::string pieces;std::vector<uint32_t> eog;
                    for(auto id:ids) {
                        char piece[4096];const auto used=api.llama_token_to_piece(vocab,id,piece,sizeof(piece),0,false);
                        if(used<0 || used>int(sizeof(piece)))throw std::runtime_error("Token piece byte bound");
                        pieces.append(piece,size_t(used));eog.push_back(api.llama_vocab_is_eog(vocab,id)?1u:0u);
                    }
                    out.word(position);out.word(0);out.word(0);out.text(pieces);
                    out.word(uint32_t(eog.size()));for(auto flag:eog)out.word(flag);out.send();continue;
                }
                throw std::runtime_error("Unknown bridge operation");
            }
        }
        api.llama_backend_free();return 0;
    } catch(const std::exception &e) {
        std::cerr<<e.what()<<"\n";Bytes out;out.word(1);out.text(e.what());out.send();return 1;
    }
}
