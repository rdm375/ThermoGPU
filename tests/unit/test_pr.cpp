// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace thermogpu;

namespace {
bool close(double a,double b,double atol=1e-12,double rtol=1e-11){return std::abs(a-b)<=atol+rtol*std::max(std::abs(a),std::abs(b));}
}

int main(){
 const Component methane{"methane",190.564,4599200.0,0.01142,0.01604246};
 const Component ethane{"ethane",305.322,4872200.0,0.0995,0.03006904};
 const Component propane{"propane",369.83,4248000.0,0.1523,0.04409562};
 const Component nitrogen{"nitrogen",126.192,3395800.0,0.0372,0.0280134};
 const Component co2{"carbon_dioxide",304.1282,7377300.0,0.22394,0.0440095};
 int fail=0; auto check=[&](bool x,const char* m){if(!x){std::cerr<<"FAIL: "<<m<<'\n';++fail;}};
 check(std::abs(alpha(methane,methane.critical_temperature_K)-1.0)<1e-14,"alpha(Tc)=1");
 const auto low=evaluate_pure(methane,{1000.0,300.0});
 check(std::abs(low.Z-1.0)<1e-4,"low pressure Z -> 1");
 check(std::abs(low.cubic_residual)<1e-12,"pure cubic residual");
 check(std::isfinite(low.ln_phi),"finite pure fugacity coefficient");
 const auto normal=evaluate_pure(methane,{5.0e6,300.0});
 check(normal.Z>0.0 && std::isfinite(normal.Z),"positive finite pure Z");
 check(normal.density_kg_per_m3>0.0,"positive pure density");

 // A one-component mixture must collapse exactly to the pure-component equations.
 const Mixture pure_as_mix{{methane},{1.0},{0.0}};
 const auto pm=evaluate_mixture(pure_as_mix,{5.0e6,300.0});
 check(close(pm.A,normal.A),"pure-as-mixture A");
 check(close(pm.B,normal.B),"pure-as-mixture B");
 check(close(pm.Z,normal.Z),"pure-as-mixture Z");
 check(close(pm.density_kg_per_m3,normal.density_kg_per_m3),"pure-as-mixture density");
 check(close(pm.ln_phi[0],normal.ln_phi),"pure-as-mixture fugacity");

 // Binary natural-gas-like mixture, kij=0. These checks exercise all mixture outputs.
 const Mixture binary{{methane,ethane},{0.8,0.2},{0.0,0.0,0.0,0.0}};
 const auto br=evaluate_mixture(binary,{5.0e6,300.0});
 check(br.Z>0.0 && std::isfinite(br.Z),"binary finite Z");
 check(br.density_kg_per_m3>0.0,"binary positive density");
 check(br.ln_phi.size()==2 && std::isfinite(br.ln_phi[0]) && std::isfinite(br.ln_phi[1]),"binary finite fugacities");
 check(std::abs(br.cubic_residual)<1e-12,"binary cubic residual");

 // Regression: cancellation-sensitive one-real-root state found by M6 CPU/CUDA
 // differential validation. Stable Cardano should solve the original cubic
 // to near machine precision without cancellation-driven root error.
 const Mixture m6_regression{{{"methane",190.56,4.5992e6,.01142,.016043},
                              {"ethane",305.32,4.872e6,.09950,.030070}},
                             {4.35330595297648837e-1,5.64669404702351163e-1},
                             {0.0,0.0,0.0,0.0}};
 const auto m6r=evaluate_mixture(m6_regression,{7.68069763952790573e6,2.96542617046818691e2});
 check(std::abs(m6r.cubic_residual)<1e-15,"M6 cancellation-sensitive cubic residual");
 check(std::abs(m6r.Z-5.508474699412217e-1)<5e-15,"M6 cancellation-sensitive stable Z");

 // Full five-component workload with an explicit dense zero-kij matrix.
 const Mixture gas{{methane,ethane,propane,nitrogen,co2},
                   {0.80,0.08,0.04,0.04,0.04},
                   std::vector<double>(25,0.0)};
 const auto gr=evaluate_mixture(gas,{8.0e6,320.0});
 check(gr.ln_phi.size()==5,"five-component fugacity count");
 check(gr.Z>0.0 && gr.density_kg_per_m3>0.0,"five-component physical outputs");
 check(std::abs(gr.cubic_residual)<1e-12,"five-component cubic residual");
 for(double x:gr.ln_phi) check(std::isfinite(x),"five-component finite fugacity");

 // kij must actually affect the attractive mixing rule.
 const Mixture binary_kij{{methane,ethane},{0.8,0.2},{0.0,0.03,0.03,0.0}};
 const auto bkr=evaluate_mixture(binary_kij,{5.0e6,300.0});
 check(!close(bkr.a_m,br.a_m,1e-15,1e-14),"nonzero kij changes a_m");
 check(!close(bkr.Z,br.Z,1e-15,1e-14),"nonzero kij changes Z");

 // Input policy: reject invalid composition rather than silently normalizing it.
 try { (void)evaluate_mixture(Mixture{{methane,ethane},{0.8,0.19},{0,0,0,0}},{1e6,300}); check(false,"reject composition sum != 1"); }
 catch(const std::invalid_argument&) {}
 try { (void)evaluate_mixture(Mixture{{methane,ethane},{0.8,0.2},{0,0.1,0,0}},{1e6,300}); check(false,"reject asymmetric kij"); }
 catch(const std::invalid_argument&) {}

 return fail?1:0;
}
