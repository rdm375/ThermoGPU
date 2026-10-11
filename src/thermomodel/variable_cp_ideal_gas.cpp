#include "thermomodel/variable_cp_ideal_gas.hpp"
#include <cmath>
#include <stdexcept>
namespace thermomodel {
VariableCpIdealGas::VariableCpIdealGas(std::shared_ptr<const CaloricModel> c,VariableCpParameters p):caloric_(std::move(c)),params_(p) {
    if(!caloric_||!std::isfinite(p.R)||p.R<=0||!std::isfinite(p.Tref)||p.Tref<=0||!std::isfinite(p.pref)||p.pref<=0||
       !std::isfinite(p.Tmin)||!std::isfinite(p.Tmax)||p.Tmin<=0||p.Tmax<=p.Tmin||p.Tref<p.Tmin||p.Tref>p.Tmax||
       !std::isfinite(p.pmin)||!std::isfinite(p.pmax)||p.pmin<=0||p.pmax<=p.pmin||p.pref<p.pmin||p.pref>p.pmax)
        throw std::invalid_argument("invalid variable-cp ideal-gas parameters");
    // Check positivity at high resolution; runtime checks cover all evaluated states.
    if(!(caloric_->minimum_cp(p.Tmin,p.Tmax)>p.R)) throw std::invalid_argument("cp must exceed R throughout declared domain");
}
Metadata VariableCpIdealGas::metadata() const {return {"thermally-perfect ideal gas","0.2.0","h=s=0 at (Tref,pref)",params_.Tmin,params_.Tmax,params_.pmin,params_.pmax};}
Result<State> VariableCpIdealGas::evaluate_Tp(double T,double p,const Composition& x) const {
    if(!x.mole_fractions.empty()) return Result<State>::failure(ErrorCode::Unsupported,"fixed composition backend");
    if(!std::isfinite(T)||!std::isfinite(p)||T<=0||p<=0) return Result<State>::failure(ErrorCode::InvalidInput,"T,p must be positive finite");
    if(T<params_.Tmin||T>params_.Tmax||p<params_.pmin||p>params_.pmax) return Result<State>::failure(ErrorCode::OutOfRange,"outside model domain");
    const double cp=caloric_->cp(T),cv=cp-params_.R;
    if(!std::isfinite(cp)||cv<=0) return Result<State>::failure(ErrorCode::Unsupported,"invalid heat capacity at state");
    State z;z.T=T;z.p=p;z.rho=p/(params_.R*T);z.cp=cp;z.cv=cv;z.Z=1;
    z.h=caloric_->enthalpy_delta(T,params_.Tref);
    z.s=caloric_->entropy_thermal_delta(T,params_.Tref)-params_.R*std::log(p/params_.pref);
    z.u=z.h-params_.R*T;
    z.sound_speed=std::sqrt(cp/cv*params_.R*T);z.phase=Phase::Vapor;
    if(!std::isfinite(z.h)||!std::isfinite(z.s)||!std::isfinite(z.u)||!std::isfinite(z.sound_speed)) return Result<State>::failure(ErrorCode::Unsupported,"nonfinite state");
    return Result<State>::success(z);
}
Result<State> VariableCpIdealGas::evaluate_Trho(double T,double rho,const Composition& x) const {
    if(!std::isfinite(T)||!std::isfinite(rho)||T<=0||rho<=0) return Result<State>::failure(ErrorCode::InvalidInput,"T,rho must be positive finite");
    const double p=rho*params_.R*T;
    if(!std::isfinite(p)) return Result<State>::failure(ErrorCode::OutOfRange,"pressure overflow");
    return evaluate_Tp(T,p,x);
}
Result<Derivatives> VariableCpIdealGas::derivatives_Tp(double T,double p,const Composition& x) const {
    auto state=evaluate_Tp(T,p,x);
    if(!state) return Result<Derivatives>::failure(state.error,state.message);
    const double R=params_.R;
    return Result<Derivatives>::success({1/(R*T),-p/(R*T*T),0,state.value.cp,-R/p,state.value.cp/T});
}
} // namespace thermomodel
