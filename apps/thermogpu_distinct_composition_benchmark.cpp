// SPDX-License-Identifier: GPL-3.0-or-later
#include "thermogpu/batch.hpp"
#include "thermogpu/mixture_benchmark.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
using clock_type=std::chrono::steady_clock;
namespace {
thermogpu::MixtureBatch batch(const thermogpu::Mixture&m,std::size_t n){thermogpu::MixtureBatch b;b.pressure_Pa.resize(n);b.temperature_K.resize(n);b.mole_fractions.resize(n*m.size());for(std::size_t s=0;s<n;++s){double u=n>1?double(s)/double(n-1):.5;double v=double((s*37u)%1009u)/1008.;b.pressure_Pa[s]=1e6+u*9e6;b.temperature_K[s]=250+v*100;std::copy(m.mole_fractions.begin(),m.mole_fractions.end(),b.mole_fractions.begin()+s*m.size());}return b;}
double median(std::vector<double>x){std::sort(x.begin(),x.end());return x[x.size()/2];}
template<class W> double timeit(W&w,std::size_t n,int repeats){w.run();std::vector<double> t;for(int q=0;q<repeats;++q){auto a=clock_type::now();w.run();auto z=clock_type::now();t.push_back(std::chrono::duration<double>(z-a).count()*1e9/n);}return median(t);}
}
int main(int argc,char**argv){try{std::size_t n=100000;int repeats=9;if(argc>1)n=std::stoull(argv[1]);if(argc>2)repeats=std::stoi(argv[2]);std::cout<<"components,states,generic_ns_per_state,fixed_ns_per_state,speedup,max_abs_delta_Z\n";for(std::size_t nc:{1u,2u,3u,4u,5u}){auto m=thermogpu::make_distinct_benchmark_mixture(nc);auto b=batch(m,n);thermogpu::CudaBatchWorkspace g(m,b);thermogpu::FixedCompositionCudaWorkspace f(m,b.pressure_Pa,b.temperature_K);double tg=timeit(g,n,repeats),tf=timeit(f,n,repeats);g.run();f.run();auto gz=g.download().Z;auto fz=f.download_z();double d=0;for(std::size_t i=0;i<n;++i)d=std::max(d,std::abs(gz[i]-fz[i]));std::cout<<nc<<','<<n<<','<<std::setprecision(17)<<tg<<','<<tf<<','<<tg/tf<<','<<d<<'\n';}return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
