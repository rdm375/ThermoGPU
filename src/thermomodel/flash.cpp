#include "thermomodel/flash.hpp"
#include "thermomodel/derivatives.hpp"
#include <algorithm>
#include <cmath>
namespace thermomodel {
namespace {
Result<State> invert(const Model& model,double p,double target,const Composition& x,FlashOptions opt,bool enthalpy) {
    if(!(std::isfinite(p)&&p>0&&std::isfinite(target)&&std::isfinite(opt.Tmin)&&std::isfinite(opt.Tmax)&&
        opt.Tmin>0&&opt.Tmax>opt.Tmin&&std::isfinite(opt.relative_tolerance)&&opt.relative_tolerance>0&&opt.max_iterations>0))
        return Result<State>::failure(ErrorCode::InvalidInput,"invalid flash arguments");
    const auto meta=model.metadata();
    const double left0=std::max(opt.Tmin,meta.Tmin),right0=std::min(opt.Tmax,meta.Tmax);
    if(left0>=right0) return Result<State>::failure(ErrorCode::OutOfRange,"flash temperature bracket outside model domain");
    auto lo=model.evaluate_Tp(left0,p,x),hi=model.evaluate_Tp(right0,p,x);
    if(!lo) return lo;
    if(!hi) return hi;
    auto residual=[&](const State& z){return (enthalpy?z.h:z.s)-target;};
    double fl=residual(lo.value),fr=residual(hi.value);
    if(!std::isfinite(fl)||!std::isfinite(fr)) return Result<State>::failure(ErrorCode::Unsupported,"nonfinite flash residual");
    if(fl==0) return lo;
    if(fr==0) return hi;
    if(std::signbit(fl)==std::signbit(fr)) return Result<State>::failure(ErrorCode::OutOfRange,"flash target not bracketed");
    double left=left0,right=right0,T=left+0.5*(right-left);
    auto* diff=dynamic_cast<const DifferentiableModel*>(&model);
    for(int i=0;i<opt.max_iterations;++i) {
        auto z=model.evaluate_Tp(T,p,x);if(!z) return z;
        const double f=residual(z.value);
        if(!std::isfinite(f)) return Result<State>::failure(ErrorCode::Unsupported,"nonfinite flash residual");
        if(f==0) return z;
        if(std::signbit(f)==std::signbit(fl)) {left=T;fl=f;} else {right=T;fr=f;}
        if(right-left<=opt.relative_tolerance*std::max(1.0,std::abs(T))) {
            const double mid=left+0.5*(right-left);
            return model.evaluate_Tp(mid,p,x);
        }
        double next=left+0.5*(right-left);
        if(diff) {
            auto d=diff->derivatives_Tp(T,p,x);
            if(d) {
                const double slope=enthalpy?d.value.dh_dT_p:d.value.ds_dT_p;
                if(std::isfinite(slope)&&slope!=0) {
                    const double candidate=T-f/slope;
                    if(std::isfinite(candidate)&&candidate>left&&candidate<right) next=candidate;
                }
            }
        }
        T=next;
    }
    return Result<State>::failure(ErrorCode::NoConvergence,"flash iteration limit reached");
}
}
Result<State> flash_ph(const Model& m,double p,double h,const Composition& x,FlashOptions opt){return invert(m,p,h,x,opt,true);}
Result<State> flash_ps(const Model& m,double p,double s,const Composition& x,FlashOptions opt){return invert(m,p,s,x,opt,false);}
} // namespace thermomodel
