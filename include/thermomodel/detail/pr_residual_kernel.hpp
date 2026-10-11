#pragma once

#include "thermomodel/detail/pr_coefficients_kernel.hpp"

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_PR_RESIDUAL_HD __host__ __device__
#else
#define THERMOMODEL_PR_RESIDUAL_HD
#endif

namespace thermomodel::detail {

struct PRResidualProperties {
    double ur;
    double sres;
    double cvres;
};

THERMOMODEL_PR_RESIDUAL_HD inline PRResidualProperties
pr_residual_properties(
    const PRCoefficientParameters& coefficients,
    double T,
    double v,
    double p
) {
    constexpr double sqrt2 = 1.4142135623730950488;

    const double R = coefficients.R;
    const double b = coefficients.b;

    const auto attraction = pr_attraction(coefficients, T);

    const double a = attraction.a;
    const double ap = attraction.ap;
    const double app = attraction.app;

    const double logq =
        ::log((v+(1+sqrt2)*b)/(v+(1-sqrt2)*b));

    const double ur =
        (T*ap-a)*logq/(2*sqrt2*b);

    const double Z = p*v/(R*T);

    const double sres =
        R*::log(Z*(v-b)/v)
        + ap*logq/(2*sqrt2*b);

    const double cvres =
        T*app*logq/(2*sqrt2*b);

    return {ur, sres, cvres};
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_RESIDUAL_HD
