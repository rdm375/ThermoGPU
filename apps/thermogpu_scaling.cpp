// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/component.hpp"
#include "thermogpu/mixture.hpp"
#ifdef THERMOGPU_HAS_OPENMP
#include <omp.h>
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using clock_type = std::chrono::steady_clock;
volatile double sink = 0.0;
double target_seconds = 0.20;
double calibration_floor = 0.025;
int samples = 9;

std::vector<thermogpu::Component> catalog() {
    using thermogpu::Component;
    return {
        {"methane",190.564,4599200,0.01142,0.01604246},
        {"ethane",305.322,4872200,0.09950,0.03006904},
        {"propane",369.83,4248000,0.15230,0.04409562},
        {"nitrogen",126.192,3395800,0.03720,0.02801340},
        {"carbon_dioxide",304.1282,7377300,0.22394,0.04400950},
    };
}

thermogpu::Mixture make_mixture(std::size_t nc) {
    const auto all=catalog();
    if(nc<1 || nc>all.size()) throw std::invalid_argument("component count must be 1..5");
    thermogpu::Mixture m;
    m.components.assign(all.begin(),all.begin()+static_cast<std::ptrdiff_t>(nc));
    m.mole_fractions.assign(nc,1.0/static_cast<double>(nc));
    m.binary_interactions.assign(nc*nc,0.0);
    return m;
}

thermogpu::MixtureBatch make_batch(std::size_t n,std::size_t nc) {
    thermogpu::MixtureBatch b;
    b.pressure_Pa.resize(n); b.temperature_K.resize(n); b.mole_fractions.resize(n*nc);
    for(std::size_t s=0;s<n;++s) {
        const double u=n>1?static_cast<double>(s)/static_cast<double>(n-1):0.5;
        const double v=static_cast<double>((s*37u)%1009u)/1008.0;
        b.pressure_Pa[s]=1.0e5+u*11.9e6;
        b.temperature_K[s]=240.0+v*180.0;
        double total=0.0;
        for(std::size_t i=0;i<nc;++i) {
            // Positive deterministic weights; composition varies by state and component.
            const double w=0.2+static_cast<double>(((s+1)*(i+3)*101u+17u*i)%997u)/996.0;
            b.mole_fractions[s*nc+i]=w; total+=w;
        }
        for(std::size_t i=0;i<nc;++i) b.mole_fractions[s*nc+i]/=total;
    }
    return b;
}

double checksum(const thermogpu::MixtureBatchResult& r) {
    double x=0.0; for(double v:r.Z)x+=v; for(double v:r.density_kg_per_m3)x+=1e-3*v;
    for(double v:r.ln_phi) x+=1e-2*v;
    return x;
}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
struct Timing{std::size_t calls{};double sec{},mad_pct{},ns_state{},throughput{};};

template<class F> Timing measure(const thermogpu::Mixture& m,const thermogpu::MixtureBatch& b,F&& f){
    auto r=f(m,b); double sum=checksum(r); std::size_t calls=1;
    for(;;){auto a=clock_type::now();for(std::size_t k=0;k<calls;++k)r=f(m,b);auto z=clock_type::now();
        const double e=std::chrono::duration<double>(z-a).count(); if(e>=calibration_floor){
            calls=std::max<std::size_t>(1,static_cast<std::size_t>(std::ceil(std::clamp(calls*target_seconds/e,1.0,1.0e9))));break;} calls*=10;}
    std::vector<double> ts;ts.reserve(samples);for(int q=0;q<samples;++q){auto a=clock_type::now();
        for(std::size_t k=0;k<calls;++k) r=f(m,b);
        auto z=clock_type::now();
        ts.push_back(std::chrono::duration<double>(z-a).count()/calls);}
    sum+=checksum(r);sink=sum;const double med=median(ts);std::vector<double>d;for(double t:ts)d.push_back(std::abs(t-med));const double mad=median(d);
    return {calls,med,100.0*mad/med,med*1e9/b.size(),static_cast<double>(b.size())/med};
}
#ifdef THERMOGPU_HAS_CUDA
Timing measure_resident(const thermogpu::Mixture&m,const thermogpu::MixtureBatch&b){
    thermogpu::CudaBatchWorkspace w(m,b);w.run();auto r=w.download();double sum=checksum(r);std::size_t calls=1;
    for(;;){auto a=clock_type::now();for(std::size_t k=0;k<calls;++k)w.run();auto z=clock_type::now();const double e=std::chrono::duration<double>(z-a).count();
        if(e>=calibration_floor){calls=std::max<std::size_t>(1,static_cast<std::size_t>(std::ceil(std::clamp(calls*target_seconds/e,1.0,1.0e9))));break;}calls*=10;}
    std::vector<double>ts;for(int q=0;q<samples;++q){auto a=clock_type::now();for(std::size_t k=0;k<calls;++k)w.run();auto z=clock_type::now();ts.push_back(std::chrono::duration<double>(z-a).count()/calls);}
    r=w.download();sum+=checksum(r);sink=sum;const double med=median(ts);std::vector<double>d;for(double t:ts)d.push_back(std::abs(t-med));const double mad=median(d);
    return {calls,med,100.0*mad/med,med*1e9/b.size(),static_cast<double>(b.size())/med};
}
#endif

void row(std::size_t nc,std::size_t n,const std::string& backend,int threads,const Timing&t,double scalar,double bestcpu){
    std::cout<<nc<<','<<n<<','<<backend<<','<<threads<<','<<t.calls<<','<<std::setprecision(17)<<t.sec<<','<<t.ns_state<<','<<t.throughput<<','<<t.mad_pct<<','<<scalar/t.sec<<','<<bestcpu/t.sec<<'\n';
}
}

int main(int argc,char**argv){try{
    // Profiling modes deliberately bypass benchmark calibration and reporting.
    // Legacy `--profile NC STATES` remains an alias for the CUDA resident path.
    if(argc>=2 && std::string(argv[1])=="--profile"){
        std::string backend;
        int arg_nc=0, arg_n=0, arg_threads=0;
        if(argc==4){ backend="cuda"; arg_nc=2; arg_n=3; }
        else if(argc==5){ backend=argv[2]; arg_nc=3; arg_n=4; }
        else if(argc==6 && std::string(argv[2])=="omp"){
            backend="omp"; arg_nc=3; arg_n=4; arg_threads=5;
        } else {
            throw std::invalid_argument(
                "profile usage: --profile [cuda|scalar] COMPONENTS STATES or --profile omp COMPONENTS STATES THREADS");
        }

        const auto nc=static_cast<std::size_t>(std::stoull(argv[arg_nc]));
        const auto n=static_cast<std::size_t>(std::stoull(argv[arg_n]));
        if(nc<1 || nc>5) throw std::invalid_argument("profile component count must be 1..5");
        if(!n) throw std::invalid_argument("profile batch size must be positive");
        const auto m=make_mixture(nc);
        const auto b=make_batch(n,nc);

        if(backend=="scalar"){
            auto r=thermogpu::evaluate_mixture_batch_scalar(m,b); // warm-up
            constexpr int profile_calls=25;
            for(int i=0;i<profile_calls;++i) r=thermogpu::evaluate_mixture_batch_scalar(m,b);
            const double sum=checksum(r); sink=sum;
            std::cout<<"profile,backend=scalar,components="<<nc<<",states="<<n
                     <<",calls="<<profile_calls<<",checksum="<<std::setprecision(17)<<sum<<'\n';
            return std::isfinite(sum)?EXIT_SUCCESS:EXIT_FAILURE;
        }

        if(backend=="omp"){
#ifndef THERMOGPU_HAS_OPENMP
            throw std::runtime_error("--profile omp requires THERMOGPU_ENABLE_OPENMP=ON");
#else
            const int threads=std::stoi(argv[arg_threads]);
            if(threads<1) throw std::invalid_argument("profile OpenMP thread count must be positive");
            omp_set_dynamic(0); omp_set_num_threads(threads);
            auto r=thermogpu::evaluate_mixture_batch_openmp(m,b); // warm-up / create team
            constexpr int profile_calls=100;
            for(int i=0;i<profile_calls;++i) r=thermogpu::evaluate_mixture_batch_openmp(m,b);
            const double sum=checksum(r); sink=sum;
            std::cout<<"profile,backend=openmp,components="<<nc<<",states="<<n
                     <<",threads="<<threads<<",calls="<<profile_calls
                     <<",checksum="<<std::setprecision(17)<<sum<<'\n';
            return std::isfinite(sum)?EXIT_SUCCESS:EXIT_FAILURE;
#endif
        }

        if(backend=="cuda"){
#ifndef THERMOGPU_HAS_CUDA
            throw std::runtime_error("--profile cuda requires THERMOGPU_ENABLE_CUDA=ON");
#else
            thermogpu::CudaBatchWorkspace w(m,b);
            w.run(); // warm-up: initialize the CUDA path before fixed profiling launches
            constexpr int profile_launches=10;
            for(int i=0;i<profile_launches;++i) w.run();
            const auto r=w.download();
            const double sum=checksum(r); sink=sum;
            std::cout<<"profile,backend=cuda,components="<<nc<<",states="<<n
                     <<",resident_launches="<<profile_launches
                     <<",checksum="<<std::setprecision(17)<<sum<<'\n';
            return std::isfinite(sum)?EXIT_SUCCESS:EXIT_FAILURE;
#endif
        }

        throw std::invalid_argument("profile backend must be cuda, scalar, or omp");
    }
    std::vector<std::size_t> sizes{100,300,1000,3000,10000,30000,100000,300000,1000000};
    bool crossover_mode=false;
    if(argc==2 && std::string(argv[1])=="--quick"){sizes={100,1000,10000};target_seconds=0.02;calibration_floor=0.005;samples=3;}
    else if(argc==2 && std::string(argv[1])=="--crossover"){crossover_mode=true;}
    else if(argc>1){sizes.clear();for(int i=1;i<argc;++i){auto n=std::stoull(argv[i]);if(!n)throw std::invalid_argument("batch size must be positive");sizes.push_back(n);}}
#ifdef THERMOGPU_HAS_OPENMP
    const int max_threads=omp_get_max_threads();omp_set_dynamic(0);
#else
    const int max_threads=1;
#endif
    std::cerr<<"ThermoGPU M7 component-count scaling; samples="<<samples<<" target_ms="<<target_seconds*1000<<" max_threads="<<max_threads;
    if(crossover_mode) std::cerr<<" mode=crossover";
    std::cerr<<"\n";
    std::cout<<"components,states,backend,threads,calls_per_sample,median_seconds,ns_per_state,states_per_second,mad_percent,speedup_vs_scalar,speedup_vs_best_cpu\n";
    for(std::size_t nc:std::vector<std::size_t>{1,2,3,5}){
        const std::vector<std::size_t> resident_sizes = nc<=2
            ? std::vector<std::size_t>{100,125,150,175,200,225,250,275,300}
            : std::vector<std::size_t>{300,350,400,450,500,550,600,700,800,900,1000};
        const std::vector<std::size_t> e2e_sizes{3000,3500,4000,4500,5000,5500,6000,7000,8000,9000,10000};
        std::vector<std::size_t> run_sizes=sizes;
        if(crossover_mode){
            run_sizes=resident_sizes;
            run_sizes.insert(run_sizes.end(),e2e_sizes.begin(),e2e_sizes.end());
            std::sort(run_sizes.begin(),run_sizes.end());
            run_sizes.erase(std::unique(run_sizes.begin(),run_sizes.end()),run_sizes.end());
        }
        for(std::size_t n:run_sizes){
        const auto m=make_mixture(nc);const auto b=make_batch(n,nc);const auto s=measure(m,b,thermogpu::evaluate_mixture_batch_scalar);
        double best=s.sec;row(nc,n,"scalar",1,s,s.sec,best);
#ifdef THERMOGPU_HAS_OPENMP
        for(int th:std::vector<int>{1,2,4,8}){if(th>max_threads)continue;omp_set_num_threads(th);auto t=measure(m,b,thermogpu::evaluate_mixture_batch_openmp);best=std::min(best,t.sec);row(nc,n,"openmp",th,t,s.sec,best);}
#endif
#ifdef THERMOGPU_HAS_CUDA
        auto r=measure_resident(m,b);row(nc,n,"cuda_resident",0,r,s.sec,best);auto e=measure(m,b,thermogpu::evaluate_mixture_batch_cuda);row(nc,n,"cuda_e2e",0,e,s.sec,best);
#endif
        }
    }
    return std::isfinite(sink)?EXIT_SUCCESS:EXIT_FAILURE;
}catch(const std::exception&e){std::cerr<<"scaling benchmark error: "<<e.what()<<'\n';return EXIT_FAILURE;}}
