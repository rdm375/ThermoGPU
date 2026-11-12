#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/detail/cubic_kernel.hpp"
#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace thermomodel {
namespace {
constexpr double sqrt2 = 1.4142135623730950488;
constexpr double Ru = detail::pr_Ru;
}

PengRobinsonMixture::PengRobinsonMixture(
    std::vector<PRMixtureComponent> components,
    std::vector<double> kij, double Tref, double pref,
    double Tmin, double Tmax, double pmin, double pmax,
    PRRootPolicy root_policy)
    : components_(std::move(components)), kij_(std::move(kij)),
      Tref_(Tref), pref_(pref), Tmin_(Tmin), Tmax_(Tmax),
      pmin_(pmin), pmax_(pmax), molar_mass_(0.0),
      gas_constant_(0.0), ideal_cp_(0.0), root_policy_(root_policy)
{
    const std::size_t n = components_.size();
    if (n == 0) throw std::invalid_argument("PR mixture requires components");
    if (kij_.empty()) kij_.assign(n*n, 0.0);
    if (kij_.size() != n*n) throw std::invalid_argument("PR mixture kij must be dense n x n");
    if (!(std::isfinite(Tref_) && Tref_ > 0 && std::isfinite(pref_) && pref_ > 0 &&
          std::isfinite(Tmin_) && Tmin_ > 0 && std::isfinite(Tmax_) && Tmax_ > Tmin_ &&
          std::isfinite(pmin_) && pmin_ > 0 && std::isfinite(pmax_) && pmax_ > pmin_ &&
          Tref_ >= Tmin_ && Tref_ <= Tmax_ && pref_ >= pmin_ && pref_ <= pmax_))
        throw std::invalid_argument("invalid PR mixture reference or domain");
    if (root_policy_ != PRRootPolicy::Vapor && root_policy_ != PRRootPolicy::Liquid &&
        root_policy_ != PRRootPolicy::RejectMultiple)
        throw std::invalid_argument("invalid PR mixture root policy");

    double xsum = 0.0;
    double cp_molar_sum = 0.0;
    for (std::size_t i=0; i<n; ++i) {
        const auto& c = components_[i];
        if (c.name.empty() || !(std::isfinite(c.mole_fraction) && c.mole_fraction >= 0 &&
            std::isfinite(c.molar_mass) && c.molar_mass > 0 &&
            std::isfinite(c.Tc) && c.Tc > 0 && std::isfinite(c.pc) && c.pc > 0 &&
            std::isfinite(c.acentric) && std::isfinite(c.ideal_cp) && c.ideal_cp > 0))
            throw std::invalid_argument("invalid PR mixture component");
        xsum += c.mole_fraction;
        molar_mass_ += c.mole_fraction*c.molar_mass;
        cp_molar_sum += c.mole_fraction*c.molar_mass*c.ideal_cp;
        for (std::size_t j=0; j<n; ++j) {
            const double k = kij_[i*n+j];
            if (!std::isfinite(k) || std::abs(k-kij_[j*n+i]) > 1.0e-14)
                throw std::invalid_argument("PR mixture kij must be finite and symmetric");
        }
    }
    if (std::abs(xsum-1.0) > 1.0e-12)
        throw std::invalid_argument("PR mixture mole fractions must sum to one");
    gas_constant_ = Ru/molar_mass_;
    ideal_cp_ = cp_molar_sum/molar_mass_;
    if (!(ideal_cp_ > gas_constant_))
        throw std::invalid_argument("mixture ideal cp must exceed mixture gas constant");
}

PengRobinsonMixture::MixAtT PengRobinsonMixture::parameters_at(double T) const
{
    const std::size_t n = components_.size();
    std::vector<double> ai(n), aip(n), aipp(n), bi(n);
    for (std::size_t i=0; i<n; ++i) {
        const auto& c = components_[i];
        const double q = 1.0 + (0.37464 + 1.54226*c.acentric -
            0.26992*c.acentric*c.acentric)*(1.0-std::sqrt(T/c.Tc));
        const double qp = -(0.37464 + 1.54226*c.acentric -
            0.26992*c.acentric*c.acentric)/(2.0*std::sqrt(c.Tc*T));
        const double qpp = (0.37464 + 1.54226*c.acentric -
            0.26992*c.acentric*c.acentric)/(4.0*std::sqrt(c.Tc)*std::pow(T,1.5));
        const double a0 = detail::pr_omega_a*Ru*Ru*c.Tc*c.Tc/c.pc;
        ai[i] = a0*q*q;
        aip[i] = 2.0*a0*q*qp;
        aipp[i] = 2.0*a0*(qp*qp+q*qpp);
        bi[i] = detail::pr_omega_b*Ru*c.Tc/c.pc;
    }
    MixAtT mix{0.0,0.0,0.0,0.0};
    for (std::size_t i=0; i<n; ++i) {
        mix.b += components_[i].mole_fraction*bi[i];
        for (std::size_t j=0; j<n; ++j) {
            const double xij = components_[i].mole_fraction*components_[j].mole_fraction;
            const double aij = (1.0-kij_[i*n+j])*std::sqrt(ai[i]*ai[j]);
            const double log_deriv = 0.5*(aip[i]/ai[i]+aip[j]/ai[j]);
            const double aij_p = aij*log_deriv;
            const double aij_pp = aij*(0.5*(aipp[i]/ai[i]+aipp[j]/ai[j])-
                0.25*std::pow(aip[i]/ai[i]-aip[j]/ai[j],2));
            mix.a += xij*aij;
            mix.ap += xij*aij_p;
            mix.app += xij*aij_pp;
        }
    }
    return mix;
}

bool PengRobinsonMixture::composition_matches(const Composition& x) const
{
    if (x.mole_fractions.empty()) return true;
    if (x.mole_fractions.size() != components_.size()) return false;
    for (std::size_t i=0; i<components_.size(); ++i)
        if (std::abs(x.mole_fractions[i]-components_[i].mole_fraction) > 1.0e-12) return false;
    return true;
}

Result<PengRobinsonMixture::Local> PengRobinsonMixture::from_volume(
    double T, double v, double p) const
{
    using Ret = Result<Local>;
    const auto mix = parameters_at(T);
    const double D = v*v+2.0*mix.b*v-mix.b*mix.b;
    if (!(std::isfinite(D) && D > 0 && v > mix.b))
        return Ret::failure(ErrorCode::OutOfRange,"volume is below PR covolume");
    const double pv = -Ru*T/((v-mix.b)*(v-mix.b))+
        mix.a*(2.0*v+2.0*mix.b)/(D*D);
    const double pt = Ru/(v-mix.b)-mix.ap/D;
    if (!(std::isfinite(pv) && pv < 0 && std::isfinite(pt)))
        return Ret::failure(ErrorCode::AmbiguousState,"unstable PR mixture root");
    const double logq = std::log((v+(1.0+sqrt2)*mix.b)/(v+(1.0-sqrt2)*mix.b));
    const double ur_molar = (T*mix.ap-mix.a)*logq/(2.0*sqrt2*mix.b);
    const double sres_molar = Ru*std::log(p*v/(Ru*T)*(v-mix.b)/v)+
        mix.ap*logq/(2.0*sqrt2*mix.b);
    const double cv = ideal_cp_-gas_constant_+
        T*mix.app*logq/(2.0*sqrt2*mix.b)/molar_mass_;
    const double cp = cv-T*(pt/molar_mass_)*(pt/molar_mass_)/(pv/molar_mass_);
    const double c2 = -v*v*pv*cp/(molar_mass_*cv);
    if (!(std::isfinite(cv) && cv > 0 && std::isfinite(cp) && cp > cv &&
          std::isfinite(c2) && c2 > 0))
        return Ret::failure(ErrorCode::Unsupported,"nonphysical PR mixture caloric state");
    const double Z = p*v/(Ru*T);
    const double h = ideal_cp_*(T-Tref_)+ur_molar/molar_mass_+
        (Z-1.0)*gas_constant_*T;
    const double s = ideal_cp_*std::log(T/Tref_)-gas_constant_*std::log(p/pref_)+
        sres_molar/molar_mass_;
    State state;
    state.T=T; state.p=p; state.rho=molar_mass_/v; state.Z=Z;
    state.h=h; state.s=s; state.u=h-p/state.rho; state.cp=cp; state.cv=cv;
    state.sound_speed=std::sqrt(c2);
    state.phase = Phase::Vapor;
    Derivatives d{
        -molar_mass_/(v*v*pv),
        molar_mass_*pt/(v*v*pv),
        v/molar_mass_+T*pt/(molar_mass_*pv),
        cp,
        pt/(molar_mass_*pv),
        cp/T
    };
    return Ret::success({state,d});
}

Result<State> PengRobinsonMixture::evaluate_Tp(double T,double p,const Composition& x) const
{
    using Ret=Result<State>;
    if (!composition_matches(x)) return Ret::failure(ErrorCode::Unsupported,"composition differs from fixed PR mixture");
    if (!(std::isfinite(T)&&T>0&&std::isfinite(p)&&p>0)) return Ret::failure(ErrorCode::InvalidInput,"invalid T,p");
    if (T<Tmin_||T>Tmax_||p<pmin_||p>pmax_) return Ret::failure(ErrorCode::OutOfRange,"outside PR mixture domain");
    const auto mix=parameters_at(T);
    const double A=mix.a*p/(Ru*Ru*T*T), B=mix.b*p/(Ru*T);
    const auto roots=detail::solve_monic_cubic(B-1.0,A-3.0*B*B-2.0*B,-A*B+B*B+B*B*B);
    double valid[3]{}; int count=0;
    for(int i=0;i<roots.count;++i) if(std::isfinite(roots.values[i])&&roots.values[i]>B*(1.0+1.0e-12)) valid[count++]=roots.values[i];
    if(count==0)return Ret::failure(ErrorCode::NoConvergence,"no physical PR mixture root");
    if(count>1&&root_policy_==PRRootPolicy::RejectMultiple)return Ret::failure(ErrorCode::AmbiguousState,"multiple PR mixture roots");
    const double Z=root_policy_==PRRootPolicy::Liquid?valid[0]:valid[count-1];
    auto state=from_volume(T,Z*Ru*T/p,p);
    if(!state)return Ret::failure(state.error,state.message);
    return Ret::success(state.value.state);
}

Result<State> PengRobinsonMixture::evaluate_Trho(double T,double rho,const Composition& x) const
{
    using Ret=Result<State>;
    if (!composition_matches(x)) return Ret::failure(ErrorCode::Unsupported,"composition differs from fixed PR mixture");
    if (!(std::isfinite(T)&&T>0&&std::isfinite(rho)&&rho>0)) return Ret::failure(ErrorCode::InvalidInput,"invalid T,rho");
    if (T<Tmin_||T>Tmax_) return Ret::failure(ErrorCode::OutOfRange,"temperature outside PR mixture domain");
    const double v=molar_mass_/rho;
    const auto mix=parameters_at(T);
    if (!(v>mix.b)) return Ret::failure(ErrorCode::OutOfRange,"density exceeds PR mixture covolume");
    const double D=v*v+2.0*mix.b*v-mix.b*mix.b;
    const double p=Ru*T/(v-mix.b)-mix.a/D;
    if (!(std::isfinite(p)&&p>=pmin_&&p<=pmax_)) return Ret::failure(ErrorCode::OutOfRange,"calculated pressure outside PR mixture domain");
    auto state=from_volume(T,v,p);
    if(!state)return Ret::failure(state.error,state.message);
    return Ret::success(state.value.state);
}

Result<Derivatives> PengRobinsonMixture::derivatives_Tp(double T,double p,const Composition& x) const
{
    auto s=evaluate_Tp(T,p,x);
    if(!s)return Result<Derivatives>::failure(s.error,s.message);
    auto local=from_volume(T,molar_mass_/s.value.rho,p);
    if(!local)return Result<Derivatives>::failure(local.error,local.message);
    return Result<Derivatives>::success(local.value.deriv);
}

Metadata PengRobinsonMixture::metadata() const
{
    return {"Peng-Robinson fixed-composition mixture","2.0.0",
        "constant component ideal cp; mixture PR residuals; h,s ideal reference at Tref,pref",
        Tmin_,Tmax_,pmin_,pmax_};
}
} // namespace thermomodel
