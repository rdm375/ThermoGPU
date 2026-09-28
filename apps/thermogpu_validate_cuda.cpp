// SPDX-License-Identifier: GPL-3.0-or-later
#include "thermogpu/batch.hpp"
#include "thermogpu/peng_robinson.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
using namespace thermogpu;
namespace {
struct Metric { std::string name; double max_abs{}; double max_rel{}; std::uint64_t max_ulp{}; std::size_t abs_state{}, rel_state{}, ulp_state{}; std::size_t abs_comp{}, rel_comp{}, ulp_comp{}; };
Mixture mixture(){ return {{{"methane",190.56,4.5992e6,.01142,.016043},{"ethane",305.32,4.872e6,.09950,.030070}},{.8,.2},{0,0,0,0}}; }
void add(MixtureBatch& b,double P,double T,double x){ b.pressure_Pa.push_back(P); b.temperature_K.push_back(T); b.mole_fractions.push_back(x); b.mole_fractions.push_back(1-x); }
MixtureBatch broad(){ MixtureBatch b; constexpr std::size_t n=20000; for(std::size_t s=0;s<n;++s){ double fp=double((s*7919)%19997)/19996., ft=double((s*7877)%19993)/19992., fx=double((s*7841)%19991)/19990.; add(b,1e4+fp*19.99e6,180.+ft*320.,1e-4+(1.-2e-4)*fx); } return b; }
MixtureBatch structured(){
 MixtureBatch b; const auto m=mixture();
 const double ps[]={1e5,5e5,1e6,2e6,3e6,4e6,4.5992e6,4.872e6,6e6,8e6,12e6,20e6};
 const double ts[]={170,180,185,190,190.56,195,210,240,270,300,305.32,310,330,360,420,500};
 const double xs[]={1e-6,.01,.05,.2,.5,.8,.95,.99,1.-1e-6};
 std::size_t rejected=0;
 for(double P:ps)for(double T:ts)for(double x:xs){
   try {
     const Mixture probe{{m.components[0],m.components[1]},{x,1.0-x},m.binary_interactions};
     (void)evaluate_mixture(probe,{P,T});
     add(b,P,T,x);
   } catch(const std::domain_error& e) {
     ++rejected;
     std::cout<<"Structured state rejected by scalar admissibility check:"
              <<" P="<<std::setprecision(17)<<P
              <<" T="<<T<<" z=["<<x<<','<<(1.0-x)<<"]"
              <<" reason=\""<<e.what()<<"\"\n";
   }
 }
 std::cout<<"Structured campaign construction: accepted="<<b.size()<<" rejected-as-nonadmissible="<<rejected<<"\n";
 return b;
}
std::uint64_t ordered(double x){ auto u=std::bit_cast<std::uint64_t>(x); return (u>>63)?~u:(u|(1ULL<<63)); }
std::uint64_t ulps(double a,double b){ if(!std::isfinite(a)||!std::isfinite(b)) return std::numeric_limits<std::uint64_t>::max(); auto x=ordered(a),y=ordered(b); return x>y?x-y:y-x; }
void observe(Metric& m,double a,double g,std::size_t s,std::size_t c=0){ const double ad=std::abs(g-a); const double den=std::max({std::abs(a),std::abs(g),std::numeric_limits<double>::min()}); const double rd=ad/den; const auto ud=ulps(a,g); if(ad>m.max_abs){m.max_abs=ad;m.abs_state=s;m.abs_comp=c;} if(rd>m.max_rel){m.max_rel=rd;m.rel_state=s;m.rel_comp=c;} if(ud>m.max_ulp){m.max_ulp=ud;m.ulp_state=s;m.ulp_comp=c;} }
void state(std::ostream& os,const MixtureBatch& b,std::size_t s,std::size_t c){ os<<"state="<<s<<" P="<<std::setprecision(17)<<b.pressure_Pa[s]<<" T="<<b.temperature_K[s]<<" z=["<<b.mole_fractions[2*s]<<','<<b.mole_fractions[2*s+1]<<"] comp="<<c; }
void compare(const std::string& label,const MixtureBatch& b){ auto m=mixture(); auto cpu=evaluate_mixture_batch_scalar(m,b); auto gpu=evaluate_mixture_batch_cuda(m,b); Metric z{"Z"},rho{"density"},res{"cubic_residual"},phi{"ln_phi"}; std::size_t one=0,three=0,amb=0,class_mismatch=0,amb_mismatch=0; for(std::size_t s=0;s<b.size();++s){ observe(z,cpu.Z[s],gpu.Z[s],s); observe(rho,cpu.density_kg_per_m3[s],gpu.density_kg_per_m3[s],s); observe(res,cpu.cubic_residual[s],gpu.cubic_residual[s],s); for(std::size_t c=0;c<2;++c)observe(phi,cpu.ln_phi[2*s+c],gpu.ln_phi[2*s+c],s,c); one+=cpu.root_classification[s]==RootClassification::one_real; three+=cpu.root_classification[s]==RootClassification::three_real; amb+=cpu.root_selection_ambiguous[s]!=0; class_mismatch+=cpu.root_classification[s]!=gpu.root_classification[s]; amb_mismatch+=cpu.root_selection_ambiguous[s]!=gpu.root_selection_ambiguous[s]; }
 std::cout<<"\nCampaign: "<<label<<" states="<<b.size()<<" one-real="<<one<<" three-real="<<three<<" ambiguous="<<amb<<"\n"; std::cout<<"root-class mismatches="<<class_mismatch<<" ambiguity mismatches="<<amb_mismatch<<"\n";
 for(const Metric* p:{&z,&rho,&res,&phi}){ auto&mtr=*p; std::cout<<std::scientific<<std::setprecision(6)<<std::setw(16)<<mtr.name<<" max_abs="<<mtr.max_abs<<" max_rel="<<mtr.max_rel<<" max_ulp="<<mtr.max_ulp<<"\n  max-rel "; state(std::cout,b,mtr.rel_state,mtr.rel_comp); std::cout<<"\n"; }
 // Acceptance tolerances are intentionally explicit and quantity-specific. Residual is an absolute diagnostic around zero.
 auto fail=[&](const char* q,const Metric& x,double abs_tol,double rel_tol){ if(x.max_abs>abs_tol && x.max_rel>rel_tol) throw std::runtime_error(std::string(label)+" "+q+" exceeds differential tolerance"); };
 fail("Z",z,5e-13,2e-12); fail("density",rho,5e-11,2e-12); fail("ln_phi",phi,5e-13,2e-12); if(res.max_abs>5e-13) throw std::runtime_error(label+" cubic residual absolute difference exceeds tolerance"); if(class_mismatch||amb_mismatch) throw std::runtime_error(label+" root metadata mismatch"); }
}
int main(){ try{ std::cout<<"ThermoGPU M6 CPU/CUDA differential validation\n"; compare("broad deterministic domain",broad()); compare("structured critical/root stress",structured()); std::cout<<"\nM6 differential validation PASSED\n"; return 0; }catch(const std::exception&e){ std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; } }
