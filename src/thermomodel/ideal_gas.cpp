#include "thermomodel/ideal_gas.hpp"
#include <cmath>
#include <stdexcept>
namespace thermomodel {
IdealGas::IdealGas(IdealGasParameters p) : params_(p) {
    if (!(std::isfinite(p.R) && std::isfinite(p.cp) && std::isfinite(p.Tref) && std::isfinite(p.pref)
       && p.R>0 && p.cp>p.R && p.Tref>0 && p.pref>0 && p.Tmin>0 && p.Tmax>p.Tmin
       && p.pmin>0 && p.pmax>p.pmin)) throw std::invalid_argument("invalid ideal-gas parameters");
}
Metadata IdealGas::metadata() const {
    return {"calorically-perfect ideal gas", "0.1.0", "h=0 and s=0 at (Tref,pref)", params_.Tmin,params_.Tmax,params_.pmin,params_.pmax};
}
Result<State> IdealGas::evaluate_Tp(double T, double p, const Composition& x) const {
    if (!x.mole_fractions.empty()) return Result<State>::failure(ErrorCode::Unsupported,"ideal-gas instance has fixed composition; supply empty composition");
    if (!(std::isfinite(T) && std::isfinite(p) && T>0 && p>0)) return Result<State>::failure(ErrorCode::InvalidInput,"T and p must be finite and positive");
    if (T<params_.Tmin || T>params_.Tmax || p<params_.pmin || p>params_.pmax) return Result<State>::failure(ErrorCode::OutOfRange,"state outside declared ideal-gas domain");
    const double cv=params_.cp-params_.R, gamma=params_.cp/cv;
    State z;
    z.T=T; z.p=p; z.rho=p/(params_.R*T);
    z.h=params_.cp*(T-params_.Tref);
    z.s=params_.cp*std::log(T/params_.Tref)-params_.R*std::log(p/params_.pref);
    z.u=cv*(T-params_.Tref)-params_.R*params_.Tref; // ensures h=u+p/rho for chosen h reference
    z.cp=params_.cp; z.cv=cv; z.Z=1;
    z.sound_speed=std::sqrt(gamma*params_.R*T);
    z.phase=Phase::Vapor;
    return Result<State>::success(z);
}
Result<State> IdealGas::evaluate_Trho(double T, double rho, const Composition& x) const {
    if (!(std::isfinite(T) && std::isfinite(rho) && T>0 && rho>0)) return Result<State>::failure(ErrorCode::InvalidInput,"T and rho must be finite and positive");
    const double p=rho*params_.R*T;
    if (!std::isfinite(p)) return Result<State>::failure(ErrorCode::OutOfRange,"pressure overflow");
    return evaluate_Tp(T,p,x);
}
Result<Derivatives> IdealGas::derivatives_Tp(double T,double p,const Composition& x) const {
    auto z=evaluate_Tp(T,p,x);
    if(!z) return Result<Derivatives>::failure(z.error,z.message);
    return Result<Derivatives>::success({1/(params_.R*T),-p/(params_.R*T*T),0,params_.cp,-params_.R/p,params_.cp/T});
}
} // namespace thermomodel
