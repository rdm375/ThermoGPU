#include "thermomodel/caloric.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace thermomodel {
ConstantCp::ConstantCp(double cp):cp_(cp) { if(!std::isfinite(cp)||cp<=0) throw std::invalid_argument("cp must be positive and finite"); }
double ConstantCp::entropy_thermal_delta(double T,double ref) const noexcept {return cp_*std::log(T/ref);}
PolynomialCp::PolynomialCp(std::array<double,5> a,double scale):a_(a),scale_(scale) {
    if(!std::isfinite(scale)||scale<=0) throw std::invalid_argument("invalid polynomial temperature scale");
    for(double v:a) if(!std::isfinite(v)) throw std::invalid_argument("nonfinite polynomial coefficient");
}
double PolynomialCp::cp(double T) const noexcept {
    const double t=T/scale_;
    return ((((a_[4]*t+a_[3])*t+a_[2])*t+a_[1])*t+a_[0]);
}
double PolynomialCp::enthalpy_delta(double T,double ref) const noexcept {
    const double t=T/scale_, r=ref/scale_;
    double tpow=t,rpow=r,sum=0;
    for(int k=0;k<5;++k) {sum+=a_[k]*(tpow-rpow)/(k+1);tpow*=t;rpow*=r;}
    return scale_*sum;
}
double PolynomialCp::entropy_thermal_delta(double T,double ref) const noexcept {
    const double t=T/scale_,r=ref/scale_;
    double sum=a_[0]*std::log(T/ref),tpow=t,rpow=r;
    for(int k=1;k<5;++k) {sum+=a_[k]*(tpow-rpow)/k;tpow*=t;rpow*=r;}
    return sum;
}
// Conservative positivity check: sample and isolate stationary points via derivative roots
// using interval subdivision; this is a guard, not a certified polynomial positivity proof.
double PolynomialCp::minimum_cp(double Tmin,double Tmax) const noexcept {
    double low=std::min(cp(Tmin),cp(Tmax));
    constexpr int n=1024;
    for(int i=1;i<n;++i) low=std::min(low,cp(Tmin+(Tmax-Tmin)*i/n));
    return low;
}
} // namespace thermomodel
