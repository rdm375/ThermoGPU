#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/detail/pr_coefficients_kernel.hpp"
#include "thermomodel/detail/pr_tp_root_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"
#include "thermomodel/detail/pr_residual_kernel.hpp"
#include "thermomodel/detail/pr_state_kernel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace thermomodel {
namespace {
constexpr double sqrt2 = 1.4142135623730950488;
}

PengRobinson::PengRobinson(std::shared_ptr<const CaloricModel> c,PRParameters q):caloric_(std::move(c)),params_(q) {
    if(!caloric_||!(std::isfinite(q.molar_mass)&&q.molar_mass>0&&std::isfinite(q.Tc)&&q.Tc>0&&
        std::isfinite(q.pc)&&q.pc>0&&std::isfinite(q.acentric)&&
        std::isfinite(q.Tref)&&q.Tref>0&&std::isfinite(q.pref)&&q.pref>0&&
        std::isfinite(q.Tmin)&&q.Tmin>0&&std::isfinite(q.Tmax)&&q.Tmax>q.Tmin&&
        std::isfinite(q.pmin)&&q.pmin>0&&std::isfinite(q.pmax)&&q.pmax>q.pmin&&
        q.Tref>=q.Tmin&&q.Tref<=q.Tmax&&q.pref>=q.pmin&&q.pref<=q.pmax))
        throw std::invalid_argument("invalid PR parameters");
    const auto coefficients = detail::make_pr_coefficients(
        q.molar_mass, q.Tc, q.pc, q.acentric);
    R_ = coefficients.R;
    a0_ = coefficients.a0;
    b_ = coefficients.b;
    kappa_ = coefficients.kappa;
    params_.Tmin = std::max(
        q.Tmin, caloric_->minimum_temperature()
    );
    params_.Tmax = std::min(
        q.Tmax, caloric_->maximum_temperature()
    );

    if (!(params_.Tmax > params_.Tmin &&
          q.Tref >= params_.Tmin &&
          q.Tref <= params_.Tmax))
        throw std::invalid_argument(
            "PR reference temperature or domain outside caloric range"
        );

    if(!(caloric_->minimum_cp(params_.Tmin,params_.Tmax)>R_)) throw std::invalid_argument("ideal cp must exceed R");
}
Metadata PengRobinson::metadata() const {return {"Peng-Robinson pure fluid","0.3.0-alpha","h,s ideal=0 at (Tref,pref); residuals relative to ideal gas at same T,p",params_.Tmin,params_.Tmax,params_.pmin,params_.pmax};}
Result<PengRobinson::Local>
PengRobinson::from_volume(double T,double v,double p) const {
    using Ret = Result<Local>;

    const detail::PRCoefficientParameters coefficients{
        R_, params_.Tc, a0_, b_, kappa_
    };

    const detail::PRStateKernelParameters parameters{
        coefficients, params_.pref
    };

    const detail::PRIdealCaloricState caloric{
        caloric_->cp(T),
        caloric_->enthalpy_delta(T,params_.Tref),
        caloric_->entropy_thermal_delta(T,params_.Tref)
    };

    const auto result =
        detail::pr_evaluate_Tvp(parameters,caloric,T,v,p);

    if (result.status == detail::PRStateStatus::UnstableRoot)
        return Ret::failure(
            ErrorCode::AmbiguousState,
            "unstable or singular PR root"
        );

    if (result.status != detail::PRStateStatus::Success)
        return Ret::failure(
            ErrorCode::Unsupported,
            "nonphysical heat capacities or sound speed"
        );

    State z;
    z.T = result.T;
    z.p = result.p;
    z.rho = result.rho;
    z.Z = result.Z;

    z.u = result.u;
    z.h = result.h;
    z.s = result.s;

    z.cv = result.cv;
    z.cp = result.cp;
    z.sound_speed = result.sound_speed;

    z.phase = T >= params_.Tc
        ? Phase::Supercritical
        : (v > 3*b_ ? Phase::Vapor : Phase::Liquid);

    Derivatives d{
        result.drho_dp_T,
        result.drho_dT_p,
        result.dh_dp_T,
        result.dh_dT_p,
        result.ds_dp_T,
        result.ds_dT_p
    };

    return Ret::success({z,d});
}
Result<State> PengRobinson::evaluate_Tp(double T,double p,const Composition& x) const {
    using Ret=Result<State>;
    if(!x.mole_fractions.empty()) return Ret::failure(ErrorCode::Unsupported,"pure-fluid backend has fixed composition");
    if(!(std::isfinite(T)&&T>0&&std::isfinite(p)&&p>0))return Ret::failure(ErrorCode::InvalidInput,"invalid T,p");
    if(T<params_.Tmin||T>params_.Tmax||p<params_.pmin||p>params_.pmax)return Ret::failure(ErrorCode::OutOfRange,"outside PR domain");
    const detail::PRCoefficientParameters coefficients{
        R_, params_.Tc, a0_, b_, kappa_
    };
    const auto policy = params_.root_policy == PRRootPolicy::Liquid
        ? detail::PRTPRootPolicy::Liquid
        : params_.root_policy == PRRootPolicy::RejectMultiple
            ? detail::PRTPRootPolicy::RejectMultiple
            : detail::PRTPRootPolicy::Vapor;
    const auto selected = detail::pr_select_Tp_root(coefficients,T,p,policy);
    if (selected.status == detail::PRTPStatus::NoPhysicalRoot)
        return Ret::failure(ErrorCode::NoConvergence,"no physical PR volume root");
    if (selected.status == detail::PRTPStatus::MultipleRoots)
        return Ret::failure(ErrorCode::AmbiguousState,"multiple PR volume roots; specify root policy");
    if (selected.status != detail::PRTPStatus::Success)
        return Ret::failure(ErrorCode::InvalidInput,"invalid PR root selection");
    auto r=from_volume(T,selected.selected_v,p);
    if(!r)return Ret::failure(r.error,r.message);
    return Ret::success(r.value.state);
}
Result<State> PengRobinson::evaluate_Trho(double T,double rho,const Composition& x) const {
    using Ret=Result<State>;
    if(!x.mole_fractions.empty())return Ret::failure(ErrorCode::Unsupported,"pure-fluid backend has fixed composition");
    if(!(std::isfinite(T)&&T>0&&std::isfinite(rho)&&rho>0))return Ret::failure(ErrorCode::InvalidInput,"invalid T,rho");
    if(T<params_.Tmin||T>params_.Tmax)return Ret::failure(ErrorCode::OutOfRange,"temperature outside PR domain");
    const double v=1/rho;
    if(!(v>b_))return Ret::failure(ErrorCode::OutOfRange,"density exceeds PR covolume limit");
    const detail::PRCoefficientParameters coefficients{
        R_, params_.Tc, a0_, b_, kappa_
    };
    const double p =
        detail::pr_pressure_derivatives(coefficients,T,v).p;
    if(!(std::isfinite(p)&&p>=params_.pmin&&p<=params_.pmax))return Ret::failure(ErrorCode::OutOfRange,"calculated pressure outside domain");
    auto r=from_volume(T,v,p);
    if(!r)return Ret::failure(r.error,r.message);
    return Ret::success(r.value.state);
}
Result<Derivatives> PengRobinson::derivatives_Tp(double T,double p,const Composition& x) const {
    auto z=evaluate_Tp(T,p,x);
    if(!z)return Result<Derivatives>::failure(z.error,z.message);
    const double v=1/z.value.rho;
    auto r=from_volume(T,v,p);
    if(!r)return Result<Derivatives>::failure(r.error,r.message);
    return Result<Derivatives>::success(r.value.deriv);
}
} // namespace thermomodel
