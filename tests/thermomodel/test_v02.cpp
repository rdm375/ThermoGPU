#include "thermomodel/variable_cp_ideal_gas.hpp"
#include "thermomodel/ideal_gas.hpp"
#include "thermomodel/flash.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace thermomodel;
void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
void near(double a,double b,double tol,const char* s){check(std::abs(a-b)<=tol*std::max({1.0,std::abs(a),std::abs(b)}),s);}
int main(){try {
  auto c=std::make_shared<PolynomialCp>(std::array<double,5>{1000,50,10,2,0.5});
  VariableCpIdealGas g(c);const Model& m=g;
  for(double T:{150.0,298.15,400.0,800.0,2000.0}){
    auto z=m.evaluate_Tp(T,200000);check(bool(z),"state");
    near(z.value.cp,c->cp(T),1e-14,"cp");
    near(z.value.h,z.value.u+z.value.p/z.value.rho,1e-13,"enthalpy identity");
    near(z.value.cp-z.value.cv,287.05,1e-13,"cp-cv");
    auto ph=flash_ph(m,z.value.p,z.value.h);check(bool(ph),"ph flash");near(ph.value.T,T,1e-10,"ph T");
    auto ps=flash_ps(m,z.value.p,z.value.s);check(bool(ps),"ps flash");near(ps.value.T,T,1e-10,"ps T");
    auto d=g.derivatives_Tp(T,z.value.p);check(bool(d),"derivatives");
    const double dt=1e-3,dp=1;
    auto tplus=m.evaluate_Tp(T+dt,z.value.p),tminus=m.evaluate_Tp(T-dt,z.value.p);
    auto pplus=m.evaluate_Tp(T,z.value.p+dp),pminus=m.evaluate_Tp(T,z.value.p-dp);
    check(bool(tplus)&&bool(tminus)&&bool(pplus)&&bool(pminus),"finite difference states");
    near((tplus.value.h-tminus.value.h)/(2*dt),d.value.dh_dT_p,1e-8,"dh/dT");
    near((tplus.value.s-tminus.value.s)/(2*dt),d.value.ds_dT_p,1e-8,"ds/dT");
    near((pplus.value.s-pminus.value.s)/(2*dp),d.value.ds_dp_T,1e-8,"ds/dp");
    near((tplus.value.rho-tminus.value.rho)/(2*dt),d.value.drho_dT_p,1e-8,"drho/dT");
    near((pplus.value.rho-pminus.value.rho)/(2*dp),d.value.drho_dp_T,1e-8,"drho/dp");
  }
  // Polynomial with only constant term must reproduce original backend.
  auto constant=std::make_shared<PolynomialCp>(std::array<double,5>{1004.5,0,0,0,0});
  VariableCpIdealGas vg(constant);IdealGas ig;
  for(double T:{200.0,300.0,700.0}){
    auto a=vg.evaluate_Tp(T,101325),b=ig.evaluate_Tp(T,101325);
    check(bool(a)&&bool(b),"constant comparison");
    near(a.value.h,b.value.h,1e-13,"constant h");near(a.value.s,b.value.s,1e-13,"constant s");
  }
  auto invalid=m.evaluate_Tp(4000,101325);check(!invalid&&invalid.error==ErrorCode::OutOfRange,"domain");
  auto unbracketed=flash_ps(m,101325,1e9);check(!unbracketed&&unbracketed.error==ErrorCode::OutOfRange,"bracket");
  std::cout<<"ThermoModel v0.2 tests passed\n";
  return 0;
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
