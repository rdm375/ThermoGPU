#pragma once

#include "thermomodel/detail/pr_coefficients_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"
#include "thermomodel/detail/pr_residual_kernel.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_PR_STATE_HD __host__ __device__
#else
#define THERMOMODEL_PR_STATE_HD
#endif

namespace thermomodel::detail {

struct PRIdealCaloricState {
    double cp;
    double enthalpy_delta;
    double entropy_thermal_delta;
};

struct PRStateKernelParameters {
    PRCoefficientParameters coefficients;
    double pref;
};

enum class PRStateStatus : int {
    Success = 0,
    UnstableRoot = 1,
    NonphysicalProperties = 2
};

struct PRStateKernelResult {
    double T;
    double p;
    double rho;
    double Z;

    double u;
    double h;
    double s;

    double cv;
    double cp;
    double sound_speed;

    double drho_dp_T;
    double drho_dT_p;
    double dh_dp_T;
    double dh_dT_p;
    double ds_dp_T;
    double ds_dT_p;

    PRStateStatus status;
};

THERMOMODEL_PR_STATE_HD inline PRStateKernelResult
pr_evaluate_Tvp(
    const PRStateKernelParameters& parameters,
    const PRIdealCaloricState& caloric,
    double T,
    double v,
    double p
) {
    PRStateKernelResult result{};
    result.status = PRStateStatus::UnstableRoot;

    const auto& coefficients = parameters.coefficients;
    const double R = coefficients.R;

    const auto pressure_derivatives =
        pr_pressure_derivatives(coefficients, T, v);

    const double pv = pressure_derivatives.pv;
    const double pt = pressure_derivatives.pt;

    if (!(std::isfinite(pv) && pv < 0.0 && std::isfinite(pt)))
        return result;

    const auto residual =
        pr_residual_properties(coefficients, T, v, p);

    const double ur = residual.ur;
    const double Z = p*v/(R*T);
    const double sres = residual.sres;

    const double cv0 = caloric.cp-R;
    const double cv = cv0+residual.cvres;

    const double cp = cv-T*pt*pt/pv;

    const double dv_dT = -pt/pv;
    const double dv_dp = 1/pv;

    const double c2 = -v*v*pv*cp/cv;

    result.status = PRStateStatus::NonphysicalProperties;

    if (!(std::isfinite(cv) && cv > 0.0 &&
          std::isfinite(cp) && cp > cv &&
          std::isfinite(c2) && c2 > 0.0))
        return result;

    result.T = T;
    result.p = p;
    result.rho = 1/v;
    result.Z = Z;

#if defined(__CUDA_ARCH__)
    const double h0 =
        __dadd_rn(caloric.enthalpy_delta, ur);

    const double pv_h = __dmul_rn(p, v);
    const double h1 = __dadd_rn(h0, pv_h);

    const double RT = __dmul_rn(R, T);
    result.h = __dadd_rn(h1, -RT);
#else
    result.h =
        caloric.enthalpy_delta+ur+p*v-R*T;
#endif

    result.s =
        caloric.entropy_thermal_delta
        - R*::log(p/parameters.pref)+sres;

#if defined(__CUDA_ARCH__)
    const double pv_rounded = __dmul_rn(p, v);
    result.u = __dadd_rn(result.h, -pv_rounded);
#else
    result.u = result.h-p*v;
#endif

    result.cp = cp;
    result.cv = cv;
    result.sound_speed = ::sqrt(c2);

    result.drho_dp_T = -dv_dp/(v*v);
    result.drho_dT_p = -dv_dT/(v*v);

    result.dh_dp_T = v-T*dv_dT;
    result.dh_dT_p = cp;

    result.ds_dp_T = -dv_dT;
    result.ds_dT_p = cp/T;

    result.status = PRStateStatus::Success;

    return result;
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_STATE_HD
