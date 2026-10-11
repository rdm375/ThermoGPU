#pragma once

#include <cmath>

#if defined(__CUDACC__)
#define THERMOMODEL_PR_HD __host__ __device__
#else
#define THERMOMODEL_PR_HD
#endif

namespace thermomodel::detail {

inline constexpr double pr_Ru = 8.31446261815324;
inline constexpr double pr_omega_a = 0.45723552892138218938;
inline constexpr double pr_omega_b = 0.07779607390388845597;

struct PRCoefficientParameters {
    double R;
    double Tc;
    double a0;
    double b;
    double kappa;
};

struct PRAttraction {
    double a;
    double ap;
    double app;
};

THERMOMODEL_PR_HD inline PRCoefficientParameters
make_pr_coefficients(
    double molar_mass,
    double Tc,
    double pc,
    double acentric)
{
    PRCoefficientParameters q{};

    q.R = pr_Ru / molar_mass;
    q.Tc = Tc;

    q.a0 = pr_omega_a * q.R * q.R * Tc * Tc / pc;
    q.b = pr_omega_b * q.R * Tc / pc;

    q.kappa =
        0.37464 + 1.54226 * acentric
        - 0.26992 * acentric * acentric;

    return q;
}

THERMOMODEL_PR_HD inline PRAttraction
pr_attraction(
    const PRCoefficientParameters& q,
    double T)
{
    const double rootT = ::sqrt(T / q.Tc);

    const double f =
        1.0 + q.kappa * (1.0 - rootT);

    const double a = q.a0 * f * f;

    const double fp =
        -q.kappa / (2.0 * q.Tc * rootT);

    const double fpp =
        q.kappa /
        (4.0 * q.Tc * q.Tc * rootT * rootT * rootT);

    const double ap =
        2.0 * q.a0 * f * fp;

    const double app =
        2.0 * q.a0 * (fp * fp + f * fpp);

    return {a, ap, app};
}

} // namespace thermomodel::detail

#undef THERMOMODEL_PR_HD
