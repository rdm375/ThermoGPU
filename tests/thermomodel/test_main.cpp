#include "thermomodel/ideal_gas.hpp"
#include "thermomodel/flash.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace thermomodel;
void check(bool ok,const char* name) {if(!ok) throw std::runtime_error(name);}
void near(double a,double b,double tol,const char* name) {check(std::abs(a-b)<=tol*std::max({1.0,std::abs(a),std::abs(b)}),name);}
int main() {
try {
    IdealGas gas;
    const Model& m=gas;
    auto z=m.evaluate_Tp(320.0,250000.0); check(bool(z),"Tp");
    near(z.value.rho,250000.0/(287.05*320.0),1e-14,"density");
    near(z.value.h,z.value.u+z.value.p/z.value.rho,1e-13,"enthalpy identity");
    near(z.value.cp-z.value.cv,287.05,1e-14,"cp-cv");
    near(z.value.Z,1.0,0,"Z");
    auto tr=m.evaluate_Trho(z.value.T,z.value.rho); check(bool(tr),"Trho");
    near(tr.value.p,z.value.p,1e-14,"Trho pressure");
    auto ph=flash_ph(m,z.value.p,z.value.h); check(bool(ph),"ph flash");
    near(ph.value.T,z.value.T,1e-10,"ph roundtrip");
    auto ps=flash_ps(m,z.value.p,z.value.s); check(bool(ps),"ps flash");
    near(ps.value.T,z.value.T,1e-10,"ps roundtrip");
    auto fail=m.evaluate_Tp(-1.0,1e5); check(!fail && fail.error==ErrorCode::InvalidInput,"invalid state");
    auto bad=m.evaluate_Tp(5000.0,1e5); check(!bad && bad.error==ErrorCode::OutOfRange,"out of range");
    auto unbracketed=flash_ph(m,1e5,1e9); check(!unbracketed && unbracketed.error==ErrorCode::OutOfRange,"unbracketed flash");
    auto composition=m.evaluate_Tp(300,1e5,Composition{{1.0}}); check(!composition && composition.error==ErrorCode::Unsupported,"composition contract");
    check(m.capabilities().single_phase_ph_flash,"capabilities");
    std::cout << "All thermomodel v0.1 contract tests passed\n";
    return 0;
} catch(const std::exception& e) {std::cerr << "FAIL: " << e.what() << '\n';return 1;}
}
