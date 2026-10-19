// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Hardware-free checks for complete, strided FP32-bit KV history.
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
struct TensorCopy {
    uint32_t type=0;uint32_t ne[4]{},nb[4]{};std::vector<char> data;
};
#include "prefix_history.hpp"
static void check(bool condition,const char *message) {if(!condition)throw std::runtime_error(message);}
static std::map<std::string,TensorCopy> snapshot(size_t rows,uint32_t seed) {
    std::map<std::string,TensorCopy> result;
    for(const auto name:{"Kcur_pos-13","Vcur_normed-13","Kcur_pos-14","Vcur_normed-14"}) {
        auto &value=result[name];const uint32_t width=std::string(name).find("-13")!=std::string::npos?256u:512u;
        value.ne[0]=width;value.ne[1]=value.ne[3]=1;value.ne[2]=uint32_t(rows);
        value.nb[0]=8;value.nb[1]=value.nb[2]=width*8+16;value.nb[3]=uint32_t(rows*value.nb[2]);
        value.data.resize(rows*value.nb[2],char(0x55));
        for(size_t row=0;row<rows;++row)for(size_t x=0;x<width;++x) {
            const uint32_t bits=uint32_t(seed+row*width+x)^0x7f800000u;
            std::memcpy(value.data.data()+row*value.nb[2]+x*8,&bits,4);
        }
    }
    return result;
}
template<class F>static void rejects(F operation) {
    bool rejected=false;try {operation();}catch(const std::runtime_error &) {rejected=true;}
    check(rejected,"Expected invalid history operation to reject");
}
int main() {
    try {
        PrefixHistory history(4);auto a=snapshot(2,19),b=snapshot(1,91);
        history.append({1,2},a);check(history.common({1,2,3})==2,"Common prefix mismatch");
        check(history.common({1,9})==1 && history.common({9})==0,"Changed prefix mismatch");
        history.trim(1);history.append({3},b);
        for(const auto &entry:history.kv) {
            const auto &value=entry.second;const auto &first=a.at(entry.first);const auto &second=b.at(entry.first);
            check(value.ne[2]==2 && value.nb[0]==4,"Canonical history shape");
            for(size_t x=0;x<value.ne[0];++x) {
                check(std::memcmp(value.data.data()+x*4,first.data.data()+x*8,4)==0,"Reused row changed bits");
                check(std::memcmp(value.data.data()+value.ne[0]*4+x*4,second.data.data()+x*8,4)==0,"Appended row changed bits");
            }
        }
        auto complete=history.complete({{"result_norm",TensorCopy{}}});
        check(complete.size()==5 && complete.count("result_norm"),"Complete state lost hidden output");
        rejects([&]{history.trim(3);});
        rejects([&]{history.append({4,5,6},snapshot(3,0));});
        PrefixHistory missing(4);auto invalid=a;invalid.erase("Kcur_pos-14");
        rejects([&]{missing.append({1,2},invalid);});
        PrefixHistory shape(4);invalid=a;invalid.begin()->second.ne[2]=3;
        rejects([&]{shape.append({1,2},invalid);});
        PrefixHistory extent(4);invalid=a;invalid.begin()->second.data.resize(1);
        rejects([&]{extent.append({1,2},invalid);});
        history.trim(0);rejects([&]{history.complete({});});
        history.append({8},b);check(history.ids==std::vector<int32_t>{8},"Trim-to-zero did not restore history");
        history.clear();check(history.ids.empty() && history.kv.empty(),"Clear retained state");
        auto padded=snapshot(9,7);PrefixHistory::retain_capture_rows(padded,2);
        history.append({1,2},padded);check(history.ids.size()==2,"Padded future rows entered history");
        rejects([&]{PrefixHistory::retain_capture_rows(padded,3);});
        auto leading=snapshot(9,7),original=leading;
        PrefixHistory::retain_capture_rows(leading,2,7);history.clear();history.append({7,8},leading);
        for(const auto &entry:history.kv)check(std::memcmp(entry.second.data.data(),
            original.at(entry.first).data.data()+7*original.at(entry.first).nb[2],4)==0,"Leading padding entered history");
        std::cout<<"PASS: strided KV bits, prefix divergence, trim/append/reset and invalid bounds\n";
        return 0;
    } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
