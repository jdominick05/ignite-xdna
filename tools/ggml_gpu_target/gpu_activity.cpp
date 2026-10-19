// Copyright (C) 2026 The ignite-xdna contributors
// SPDX-License-Identifier: AGPL-3.0-or-later
// Research-only bounded activity on the detected Phoenix Radeon 780M.
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
using Microsoft::WRL::ComPtr;
using Clock=std::chrono::steady_clock;
static void check(HRESULT result) {if(FAILED(result))throw std::runtime_error("D3D11 operation failed");}
static uint32_t iterate(uint32_t value) {
    for(uint32_t i=0;i<32768;++i) {value^=value<<13;value^=value>>17;value^=value<<5;}
    return value;
}
class Activity {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<ID3D11ComputeShader> shader;
    ComPtr<ID3D11Buffer> output,staging,constants;
    ComPtr<ID3D11UnorderedAccessView> view;
    ComPtr<ID3D11Query> done;
    std::thread job;
    std::atomic<bool> stop{false};
    std::string failure;
    uint32_t seed=0,dispatches=0;
    double elapsed_ms=0;
    bool verified=false;
    void finish_gpu() {
        context->End(done.Get());context->Flush();BOOL complete=FALSE;
        const auto deadline=Clock::now()+std::chrono::seconds(2);
        while(true) {
            const auto status=context->GetData(done.Get(),&complete,sizeof(complete),D3D11_ASYNC_GETDATA_DONOTFLUSH);
            check(status);if(status==S_OK && complete)break;
            if(Clock::now()>=deadline)throw std::runtime_error("GPU completion timeout");
            std::this_thread::yield();
        }
    }
    void dispatch() {
        const uint32_t params[4]={32768,++seed,0,0};
        context->UpdateSubresource(constants.Get(),0,nullptr,params,0,0);
        context->Dispatch(1,1,1);finish_gpu();++dispatches;
    }
    void verify() {
        context->CopyResource(staging.Get(),output.Get());finish_gpu();
        D3D11_MAPPED_SUBRESOURCE mapped{};check(context->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped));
        bool equal=true;const auto words=static_cast<const uint32_t *>(mapped.pData);
        for(uint32_t i=0;i<32;++i)equal&=words[i]==iterate(seed^i);
        context->Unmap(staging.Get(),0);
        if(!equal)throw std::runtime_error("GPU activity result differs from integer reference");
        verified=true;
    }
public:
    Activity(const std::filesystem::path &runtime) {
        // HIP 7 public device API: match the graphics API adapter to ROCm device 0.
        // https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.2/reference/hip_runtime_api/modules/initialization_and_version.html
        const auto hip=LoadLibraryExW((std::filesystem::absolute(runtime)/"amdhip64_7.dll").wstring().c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        if(!hip)throw std::runtime_error("Cannot load the pinned HIP runtime");
        // Registry/runtime modules remain loaded until this owned process exits.
        using GetLuid=int (__cdecl *)(char *,unsigned int *,int);
        const auto get_luid=reinterpret_cast<GetLuid>(GetProcAddress(hip,"hipDeviceGetLuid"));
        LUID hip_luid{};unsigned int node_mask=0;static_assert(sizeof(hip_luid)==8);
        if(!get_luid || get_luid(reinterpret_cast<char *>(&hip_luid),&node_mask,0)!=0 || !node_mask)
            throw std::runtime_error("Cannot establish the ROCm device LUID");
        ComPtr<IDXGIFactory1> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        ComPtr<IDXGIAdapter1> selected;DXGI_ADAPTER_DESC1 selected_desc{};
        for(UINT index=0;;++index) {
            ComPtr<IDXGIAdapter1> adapter;const auto status=factory->EnumAdapters1(index,&adapter);
            if(status==DXGI_ERROR_NOT_FOUND)break;check(status);
            DXGI_ADAPTER_DESC1 desc{};check(adapter->GetDesc1(&desc));
            if(desc.VendorId==0x1002 && desc.DeviceId==0x15bf && !(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)
                    && desc.AdapterLuid.LowPart==hip_luid.LowPart && desc.AdapterLuid.HighPart==hip_luid.HighPart) {
                if(selected)throw std::runtime_error("More than one Phoenix graphics adapter");
                selected=adapter;selected_desc=desc;
            }
        }
        if(!selected)throw std::runtime_error("Phoenix Radeon 780M hardware adapter required");
        D3D_FEATURE_LEVEL wanted=D3D_FEATURE_LEVEL_11_0,actual{};
        check(D3D11CreateDevice(selected.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,0,&wanted,1,
                               D3D11_SDK_VERSION,&device,&actual,&context));
        if(actual!=wanted)throw std::runtime_error("D3D11 compute capability required");
        static const char source[]=R"(
            cbuffer Params : register(b0) {uint iterations;uint seed;uint unused0;uint unused1;};
            RWStructuredBuffer<uint> result : register(u0);
            [numthreads(32,1,1)] void main(uint3 id : SV_DispatchThreadID) {
                uint x=seed^id.x;
                [loop] for(uint i=0;i<iterations;++i) {x^=x<<13;x^=x>>17;x^=x<<5;}
                result[id.x]=x;
            })";
        ComPtr<ID3DBlob> code,errors;
        check(D3DCompile(source,sizeof(source)-1,"bounded_activity",nullptr,nullptr,"main","cs_5_0",
                         D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,&code,&errors));
        check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader));
        D3D11_BUFFER_DESC desc{};desc.ByteWidth=128;desc.Usage=D3D11_USAGE_DEFAULT;
        desc.BindFlags=D3D11_BIND_UNORDERED_ACCESS;desc.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;desc.StructureByteStride=4;
        check(device->CreateBuffer(&desc,nullptr,&output));
        D3D11_UNORDERED_ACCESS_VIEW_DESC uv{};uv.Format=DXGI_FORMAT_UNKNOWN;uv.ViewDimension=D3D11_UAV_DIMENSION_BUFFER;uv.Buffer.NumElements=32;
        check(device->CreateUnorderedAccessView(output.Get(),&uv,&view));
        desc={};desc.ByteWidth=128;desc.Usage=D3D11_USAGE_STAGING;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
        check(device->CreateBuffer(&desc,nullptr,&staging));
        desc={};desc.ByteWidth=16;desc.Usage=D3D11_USAGE_DEFAULT;desc.BindFlags=D3D11_BIND_CONSTANT_BUFFER;
        check(device->CreateBuffer(&desc,nullptr,&constants));
        D3D11_QUERY_DESC query{};query.Query=D3D11_QUERY_EVENT;check(device->CreateQuery(&query,&done));
        context->CSSetShader(shader.Get(),nullptr,0);auto uav=view.Get();auto buffer=constants.Get();
        context->CSSetUnorderedAccessViews(0,1,&uav,nullptr);context->CSSetConstantBuffers(0,1,&buffer);
        std::cout<<"{\"ready\":true,\"vendor\":4098,\"device\":5567,\"luid_low\":"<<selected_desc.AdapterLuid.LowPart
                 <<",\"luid_high\":"<<selected_desc.AdapterLuid.HighPart<<",\"hip_luid_matched\":true,\"node_mask\":"<<node_mask
                 <<",\"hardware\":true,\"threads\":32,\"iterations\":32768}\n"<<std::flush;
    }
    ~Activity() {stop=true;if(job.joinable())job.join();}
    void start(unsigned milliseconds) {
        if(milliseconds<1 || milliseconds>100 || job.joinable())throw std::runtime_error("Use one bounded 1..100 ms window");
        stop=false;failure.clear();dispatches=0;verified=false;
        job=std::thread([this,milliseconds] {
            const auto began=Clock::now(),deadline=began+std::chrono::milliseconds(milliseconds);
            try {
                while(!stop && Clock::now()<deadline)dispatch();
                if(dispatches)verify();
            } catch(const std::exception &error) {failure=error.what();}
            elapsed_ms=std::chrono::duration<double,std::milli>(Clock::now()-began).count();
        });
        std::cout<<"{\"started\":true,\"window_ms\":"<<milliseconds<<"}\n"<<std::flush;
    }
    void wait() {
        if(!job.joinable())throw std::runtime_error("No activity window to join");job.join();
        if(!failure.empty())throw std::runtime_error(failure);
        if(!verified || !dispatches)throw std::runtime_error("GPU activity did not complete a checked dispatch");
        std::cout<<"{\"completed\":true,\"gpu_validated\":true,\"dispatches\":"<<dispatches
                 <<",\"elapsed_ms\":"<<std::setprecision(17)<<elapsed_ms<<"}\n"<<std::flush;
    }
    void stop_wait() {stop=true;wait();}
};
int main(int argc,char **argv) {
    try {
        if(argc!=2)throw std::runtime_error("Usage: gpu_activity ROCM_RUNTIME_DIR");
        Activity activity(argv[1]);std::string line;
        while(std::getline(std::cin,line)) {
            if(line=="quit")return 0;
            if(line=="wait") {activity.wait();continue;}
            if(line=="stop") {activity.stop_wait();continue;}
            std::istringstream input(line);std::string op,extra;unsigned milliseconds=0;
            if(line.size()>64 || !(input>>op>>milliseconds) || op!="start" || input>>extra)
                throw std::runtime_error("Use start MS, wait or quit");
            activity.start(milliseconds);
        }
        return 0;
    } catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}
}
