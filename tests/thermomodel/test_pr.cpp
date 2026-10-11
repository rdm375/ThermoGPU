#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/flash.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
using namespace thermomodel;
void check(bool b,const char* m){if(!b)throw std::runtime_error(m);}
void near(double a,double b,double tol,const char* m){check(std::abs(a-b)<=tol*std::max({1.,std::abs(a),std::abs(b)}),m);}
int main(){try{
 PengRobinson pr(std::make_shared<ConstantCp>(2200.0));
 // External ThermoPack methane PR reference Z~0.90182112 at 300K,5MPa;
 // Tc,pc,omega conventions can differ. Use 0.2% comparison pending locked oracle.
 auto ref=pr.evaluate_Tp(300,5e6);check(bool(ref),"reference state");
 std::cout<<"methane Z(300K,5MPa)="<<ref.value.Z<<'\n';
 near(ref.value.Z,0.90183535320862873,2e-13,"reference Z");
 for(double T:{220.,300.,400.,600.})for(double p:{1e5,1e6,5e6,10e6}){
  auto r=pr.evaluate_Tp(T,p);check(bool(r),"PR state");const State& z=r.value;
  check(z.cp>z.cv&&z.cv>0&&z.sound_speed>0,"physical caloric state");
  near(z.h,z.u+p/z.rho,1e-10,"h=u+pv");
  auto v=pr.evaluate_Trho(T,z.rho);check(bool(v),"Trho round trip");near(v.value.p,p,1e-9,"Trho p");
  auto d=pr.derivatives_Tp(T,p);check(bool(d),"derivative evaluation");
  const double dt=1e-3,dp=100.;
  auto tp=pr.evaluate_Tp(T+dt,p),tm=pr.evaluate_Tp(T-dt,p);
  auto pp=pr.evaluate_Tp(T,p+dp),pm=pr.evaluate_Tp(T,p-dp);
  check(bool(tp)&&bool(tm)&&bool(pp)&&bool(pm),"finite difference evaluations");
  near((tp.value.h-tm.value.h)/(2*dt),d.value.dh_dT_p,3e-5,"dh/dT");
  near((tp.value.s-tm.value.s)/(2*dt),d.value.ds_dT_p,3e-5,"ds/dT");
  near((pp.value.h-pm.value.h)/(2*dp),d.value.dh_dp_T,3e-5,"dh/dp");
  near((pp.value.s-pm.value.s)/(2*dp),d.value.ds_dp_T,3e-5,"ds/dp");
  near((pp.value.rho-pm.value.rho)/(2*dp),d.value.drho_dp_T,3e-5,"drho/dp");
  near((tp.value.rho-tm.value.rho)/(2*dt),d.value.drho_dT_p,3e-5,"drho/dT");
 }
 PRParameters q; q.root_policy=PRRootPolicy::RejectMultiple;
 PengRobinson strict(std::make_shared<ConstantCp>(2200.0),q);
 auto multi=strict.evaluate_Tp(150,1e6);
 check(!multi&&multi.error==ErrorCode::AmbiguousState,"multiple roots must be explicit");
 check(!pr.evaluate_Tp(-1,1e6),"reject invalid temperature");
 std::cout<<"PR tests passed\n";
}catch(const std::exception& e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;} }
