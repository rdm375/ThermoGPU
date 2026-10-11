#pragma once

#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_PR_DERIV_HD __host__ __device__
#else
#define THERMOMODEL_PR_DERIV_HD
#endif

namespace thermomodel::detail {

struct PRPressureDerivatives {
    double p;
    double pv;
    double pt;
};

THERMOMODEL_PR_DERIV_HD inline PRPressureDerivatives
pr_pressure_derivatives(
    const PRCoefficientParameters& coefficients,
    double T,
    double v
) {
    const auto attraction = pr_attraction(coefficients, T);

    const double R = coefficients.R;
    const double b = coefficients.b;
    const double a = attraction.a;
    const double ap = attraction.ap;

#if defined(__CUDA_ARCH__)
    // Match the CPU reference expression's separate rounding.
    const double vv = __dmul_rn(v, v);
    const double two_b = __dmul_rn(2.0, b);
    const double two_bv = __dmul_rn(two_b, v);
    const double bb = __dmul_rn(b, b);

    const double D = __dadd_rn(
        __dadd_rn(vv, two_bv),
        -bb
    );
#else
    const double D = v*v + 2*b*v - b*b;
#endif

    const double p =
        R*T/(v-b) - a/D;

    const double pv =
        -R*T/((v-b)*(v-b))
        + a*(2*v+2*b)/(D*D);

    const double pt =
        R/(v-b) - ap/D;

    return {p, pv, pt};
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_DERIV_HD
