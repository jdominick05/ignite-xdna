// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Host copies of GPU-produced FP32 KV values; no model arithmetic.
#pragma once

struct PrefixHistory {
    std::vector<int32_t> ids;
    std::map<std::string,TensorCopy> kv;
    size_t capacity;
    explicit PrefixHistory(size_t bound):capacity(bound) {}

    void clear() {ids.clear();kv.clear();}
    static void retain_capture_rows(std::map<std::string,TensorCopy> &captured,size_t count,size_t offset=0) {
        if(!count)throw std::runtime_error("Empty retained capture");
        for(const auto name:{"Kcur_pos-13","Vcur_normed-13","Kcur_pos-14","Vcur_normed-14"}) {
            auto &value=captured.at(name);
            if(offset>value.ne[2] || count>value.ne[2]-offset)throw std::runtime_error("Retained capture exceeds evaluated rows");
            const size_t bytes=offset*value.nb[2];
            if(bytes>value.data.size())throw std::runtime_error("Retained capture offset bound");
            if(bytes)value.data.erase(value.data.begin(),value.data.begin()+std::ptrdiff_t(bytes));
            value.ne[2]=uint32_t(count);
        }
    }
    size_t common(const std::vector<int32_t> &next) const {
        size_t n=0;while(n<ids.size() && n<next.size() && ids[n]==next[n])++n;
        return n;
    }
    void trim(size_t count) {
        if(count>ids.size())throw std::runtime_error("Prefix history trim bound");
        ids.resize(count);
        for(auto &entry:kv) {
            auto &value=entry.second;value.ne[2]=uint32_t(count);
            value.data.resize(count*value.ne[0]*4);value.nb[3]=uint32_t(value.data.size());
        }
    }
    void append(const std::vector<int32_t> &tokens,const std::map<std::string,TensorCopy> &captured) {
        if(tokens.empty() || tokens.size()>capacity-ids.size())throw std::runtime_error("Prefix history capacity");
        const size_t old=ids.size(),total=old+tokens.size();
        for(const auto name:{"Kcur_pos-13","Vcur_normed-13","Kcur_pos-14","Vcur_normed-14"}) {
            const auto found=captured.find(name);
            if(found==captured.end())throw std::runtime_error("Missing prefix KV capture");
            const auto &source=found->second;
            const uint32_t width=std::string(name).find("-13")!=std::string::npos?256u:512u;
            if(source.type || source.ne[0]!=width || source.ne[1]!=1 || source.ne[2]!=tokens.size() || source.ne[3]!=1)
                throw std::runtime_error("Prefix KV capture shape");
            auto &dest=kv[name];
            if(old && (dest.ne[0]!=width || dest.data.size()!=old*width*4))throw std::runtime_error("Prefix history inconsistent");
            dest.ne[0]=width;dest.ne[1]=1;dest.ne[2]=uint32_t(total);dest.ne[3]=1;
            dest.nb[0]=4;dest.nb[1]=dest.nb[2]=width*4;dest.nb[3]=uint32_t(total*width*4);
            dest.data.resize(total*width*4);
            for(size_t row=0;row<tokens.size();++row) {
                const size_t offset=row*source.nb[2];
                const size_t extent=offset+size_t(width-1)*source.nb[0]+4;
                if(extent>source.data.size())throw std::runtime_error("Prefix KV capture extent");
                char *target=dest.data.data()+(old+row)*width*4;
                if(source.nb[0]==4)std::memcpy(target,source.data.data()+offset,width*4);
                else for(size_t x=0;x<width;++x)std::memcpy(target+x*4,source.data.data()+offset+x*source.nb[0],4);
            }
        }
        ids.insert(ids.end(),tokens.begin(),tokens.end());
    }
    std::map<std::string,TensorCopy> complete(const std::map<std::string,TensorCopy> &last) const {
        if(ids.empty() || kv.size()!=4)throw std::runtime_error("Incomplete prefix history");
        auto result=last;
        for(const auto &entry:kv)result[entry.first]=entry.second;
        return result;
    }
};
