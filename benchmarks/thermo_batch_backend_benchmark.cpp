// SPDX-License-Identifier: GPL-3.0-or-later
// ThermoGPU v2 thermodynamic batch backend benchmark.
#include "thermogpu/thermo_batch_cuda.hpp"
#include "thermomodel/caloric.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef THERMOGPU_HAS_OPENMP
#include <omp.h>
#endif

namespace {
using namespace thermogpu::thermo;
using Clock = std::chrono::steady_clock;
void check(cudaError_t code, const char* operation) {
    if (code != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(code));
}
struct Stream {
    cudaStream_t value{};
    Stream() { check(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), "stream create"); }
    ~Stream() { if (value) cudaStreamDestroy(value); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Event {
    cudaEvent_t value{};
    Event() { check(cudaEventCreate(&value), "event create"); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
template<class T> struct Device {
    T* ptr{};
    explicit Device(std::size_t n) {
        if (n) check(cudaMalloc(reinterpret_cast<void**>(&ptr), n*sizeof(T)), "device allocation");
    }
    ~Device() { if (ptr) cudaFree(ptr); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto n = samples.size();
    return n % 2 ? samples[n/2] : (samples[n/2-1]+samples[n/2])*0.5;
}
BatchInput make_input(std::size_t n, const std::string& workload) {
    BatchInput input;
    input.temperature_K.resize(n);
    input.pressure_Pa.resize(n);
    for (std::size_t i=0; i<n; ++i) {
        const double x = n > 1 ? static_cast<double>(i)/static_cast<double>(n-1) : 0.5;
        if (workload=="uniform") {
            input.temperature_K[i]=320.0;
            input.pressure_Pa[i]=8.0e6;
        } else if (workload=="smooth") {
            input.temperature_K[i]=300.0+35.0*x+4.0*std::sin(6.283185307179586*x);
            input.pressure_Pa[i]=4.0e6+5.0e6*x+2.0e5*std::sin(12.566370614359172*x);
        } else if (workload=="diverse") {
            // Deterministic, inexpensive index scrambling. All states inside PR domain.
            const double a = static_cast<double>((i*104729ULL)%1000003ULL)/1000003.0;
            const double b = static_cast<double>((i*130363ULL+17ULL)%1000033ULL)/1000033.0;
            input.temperature_K[i]=220.0+600.0*a;
            input.pressure_Pa[i]=1.0e5+3.0e7*b;
        } else throw std::invalid_argument("unknown workload");
    }
    return input;
}
void emit(std::ostream& out, const std::string& gpu, const std::string& backend,
          const std::string& workload, std::size_t n, double ms, int repeats,
          int threads, double checksum) {
    const double ns=ms*1e6/static_cast<double>(n);
    out << gpu << ',' << backend << ',' << workload << ',' << n << ','
        << std::setprecision(12) << ms << ',' << ns << ','
        << 1000.0/ns << ',' << repeats << ',' << threads << ',' << checksum << '\n';
}
template<class F> double host_measure(F&& function, int repeats) {
    for (int i=0;i<2;++i) function();
    std::vector<double> samples;
    for (int i=0;i<repeats;++i) {
        auto start=Clock::now();
        function();
        auto stop=Clock::now();
        samples.push_back(std::chrono::duration<double,std::milli>(stop-start).count());
    }
    return median(std::move(samples));
}
double sink=0.0;
} // namespace

int main(int argc, char** argv) {
    try {
        std::string path="thermo_batch_benchmark.csv";
        int repeats=7;
        std::size_t max_batch=1000000;
        for (int i=1;i<argc;++i) {
            const std::string arg=argv[i];
            if ((arg=="--output" || arg=="--repeats" || arg=="--max-batch") && i+1<argc) {
                const std::string value=argv[++i];
                if (arg=="--output") path=value;
                else if (arg=="--repeats") repeats=std::stoi(value);
                else max_batch=static_cast<std::size_t>(std::stoull(value));
            } else throw std::invalid_argument("usage: benchmark [--output file.csv] [--repeats N] [--max-batch N]");
        }
        if (repeats<1 || max_batch<1) throw std::invalid_argument("positive repeats and max-batch required");
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, 0), "device properties");
        check(cudaSetDevice(0), "select device");
        std::string gpu=prop.name;
        std::replace(gpu.begin(), gpu.end(), ',', '_');
        std::ofstream file(path);
        if (!file) throw std::runtime_error("cannot open output CSV");
        file << "device,backend,workload,batch_size,median_ms,ns_per_eval,million_eval_per_sec,repeats,omp_threads,checksum\n";
        CUDAPRParameters config;
        auto caloric=std::make_shared<thermomodel::PolynomialCp>(
            config.caloric_coefficients,config.caloric_scale);
        thermomodel::PengRobinson model(caloric,config.pr);
        int threads=1;
#ifdef THERMOGPU_HAS_OPENMP
        threads=omp_get_max_threads();
#endif
        for (const auto& workload : {"uniform","smooth","diverse"}) {
            for (std::size_t n : {1ULL,10ULL,100ULL,1000ULL,10000ULL,100000ULL,1000000ULL}) {
                if (n>max_batch) continue;
                const auto input=make_input(n,workload);
                auto scalar=[&]{
                    auto result=evaluate_batch_scalar(model,input);
                    sink+=result.Z[n/2];
                };
                const double scalar_ms=host_measure(scalar,repeats);
                emit(file,gpu,"scalar",workload,n,scalar_ms,repeats,1,sink);
#ifdef THERMOGPU_HAS_OPENMP
                auto parallel=[&]{
                    auto result=evaluate_batch_openmp(model,input);
                    sink+=result.Z[n/2];
                };
                emit(file,gpu,"openmp",workload,n,host_measure(parallel,repeats),repeats,threads,sink);
#endif
                auto api=[&]{
                    auto result=evaluate_batch_cuda(config,input);
                    sink+=result.Z[n/2];
                };
                emit(file,gpu,"cuda_api",workload,n,host_measure(api,repeats),repeats,1,sink);
                CUDAThermoWorkspace workspace(config,n);
                auto workspace_call=[&]{
                    workspace.upload(input);
                    workspace.evaluate();
                    auto result=workspace.download();
                    sink+=result.Z[n/2];
                };
                emit(file,gpu,"cuda_workspace",workload,n,host_measure(workspace_call,repeats),repeats,1,sink);
                // G4.6c.1: reuse caller-owned BatchResult storage.
                BatchResult reusable_result;
                auto workspace_reuse_call=[&]{
                    workspace.upload(input);
                    workspace.evaluate();
                    workspace.download_into(reusable_result);
                    sink+=reusable_result.Z[n/2];
                };
                emit(file,gpu,"cuda_workspace_reuse",workload,n,
                     host_measure(workspace_reuse_call,repeats),
                     repeats,1,sink);
                std::vector<CUDADeviceInput> host(n);
                for (std::size_t j=0;j<n;++j)
                    host[j]={input.temperature_K[j],input.pressure_Pa[j]};
                Device<CUDADeviceInput> d_in(n);
                Device<CUDADeviceOutput> d_out(n);
                Stream stream;
                check(cudaMemcpyAsync(d_in.ptr,host.data(),n*sizeof(host[0]),cudaMemcpyHostToDevice,stream.value),"input upload");
                check(cudaStreamSynchronize(stream.value),"input sync");
                Event start,stop;
                for (int j=0;j<2;++j) workspace.evaluate_device(d_in.ptr,d_out.ptr,n,stream.value);
                check(cudaStreamSynchronize(stream.value),"warmup sync");
                std::vector<double> samples;
                for (int j=0;j<repeats;++j) {
                    check(cudaEventRecord(start.value,stream.value),"start event");
                    workspace.evaluate_device(d_in.ptr,d_out.ptr,n,stream.value);
                    check(cudaEventRecord(stop.value,stream.value),"stop event");
                    check(cudaEventSynchronize(stop.value),"event sync");
                    float elapsed=0.0f;
                    check(cudaEventElapsedTime(&elapsed,start.value,stop.value),"event elapsed");
                    samples.push_back(elapsed);
                }
                // Consume one device result to guard against accidentally benchmarking
                // an invalid launch or wholly unobserved result.
                CUDADeviceOutput sample{};
                check(cudaMemcpyAsync(&sample,d_out.ptr+(n/2),sizeof(sample),cudaMemcpyDeviceToHost,stream.value),"sample download");
                check(cudaStreamSynchronize(stream.value),"sample sync");
                if (sample.error!=0 || !std::isfinite(sample.result.state.Z))
                    throw std::runtime_error("invalid CUDA device-resident benchmark result");
                sink+=sample.result.state.Z;
                emit(file,gpu,"cuda_device_kernel",workload,n,median(samples),repeats,1,sink);
                file.flush();
                std::cerr << workload << " n=" << n << " done\n";
            }
        }
        std::cerr << "CSV: " << path << "; checksum: " << sink << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "benchmark error: " << e.what() << '\n';
        return 1;
    }
}
