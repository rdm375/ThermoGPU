// SPDX-License-Identifier: GPL-3.0-or-later
#include "thermogpu/batch.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace thermogpu;
namespace {
void req(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
void close(double a,double b,const std::string&m){double sc=std::max({1.0,std::abs(a),std::abs(b)}); if(std::abs(a-b)>2e-12*sc) throw std::runtime_error(m+": "+std::to_string(a)+" vs "+std::to_string(b));}
Mixture mix(){return {{{"methane",190.56,4.5992e6,.01142,.016043},{"ethane",305.32,4.872e6,.09950,.030070}},{.8,.2},{0,0,0,0}};}
MixtureBatch campaign(std::size_t n){MixtureBatch b; for(std::size_t s=0;s<n;++s){double fp=double((s*37)%997)/996.,ft=double((s*61)%991)/990.,fx=double((s*83)%983)/982.,x=.05+.9*fx;b.pressure_Pa.push_back(1e5+fp*11.9e6);b.temperature_K.push_back(240+ft*180);b.mole_fractions.push_back(x);b.mole_fractions.push_back(1-x);}return b;}
}
int main(){try{constexpr std::size_t n=1000;auto m=mix();auto b=campaign(n);auto a=evaluate_mixture_batch_scalar(m,b);auto g=evaluate_mixture_batch_cuda(m,b);req(g.state_count==n,"state count");for(std::size_t s=0;s<n;++s){close(g.Z[s],a.Z[s],"Z "+std::to_string(s));close(g.density_kg_per_m3[s],a.density_kg_per_m3[s],"density "+std::to_string(s));close(g.cubic_residual[s],a.cubic_residual[s],"residual "+std::to_string(s));req(g.root_classification[s]==a.root_classification[s],"root class "+std::to_string(s));req(g.root_selection_ambiguous[s]==a.root_selection_ambiguous[s],"ambiguity "+std::to_string(s));for(int i=0;i<2;++i)close(g.ln_phi[2*s+i],a.ln_phi[2*s+i],"lnphi "+std::to_string(s));}std::cout<<"M5 CUDA differential smoke test passed: "<<n<<" states\n";return 0;}catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
