#pragma once
#include "thermomodel/detail/cubic_kernel.hpp"
#include "thermomodel/detail/pr_derivatives_kernel.hpp"
#include <cmath>
#if defined(__CUDACC__)
#define THERMOMODEL_PR_ROOT_HD __host__ __device__
#else
#define THERMOMODEL_PR_ROOT_HD
#endif
namespace thermomodel::detail {
enum class PRTPRootPolicy : int { Vapor = 0, Liquid = 1, RejectMultiple = 2 };
enum class PRTPStatus : int { Success = 0, InvalidInput = 1, NoPhysicalRoot = 2, MultipleRoots = 3, UnstableRoot = 4, NonphysicalProperties = 5 };
struct PRSelectedRoot {
    PRTPStatus status = PRTPStatus::InvalidInput;
    double selected_Z = 0.0;
    double selected_v = 0.0;
    int admissible_roots = 0;
};
THERMOMODEL_PR_ROOT_HD inline PRSelectedRoot
pr_select_Tp_root(const PRCoefficientParameters& q, double T, double p,
                  PRTPRootPolicy policy) {
    PRSelectedRoot result{};
    if (!(std::isfinite(T) && T > 0.0 && std::isfinite(p) && p > 0.0 &&
          std::isfinite(q.R) && q.R > 0.0 &&
          std::isfinite(q.Tc) && q.Tc > 0.0 &&
          std::isfinite(q.b) && q.b > 0.0)) return result;
    if (!(policy == PRTPRootPolicy::Vapor || policy == PRTPRootPolicy::Liquid ||
          policy == PRTPRootPolicy::RejectMultiple)) return result;
    const double a = pr_attraction(q, T).a;

    // Preserve the production implementation's arithmetic order.
    const double A = a*p/(q.R*q.R*T*T);
    const double B = q.b*p/(q.R*T);

    const CubicRoots roots = solve_monic_cubic(
        B-1.0,
        A-3.0*B*B-2.0*B,
        -A*B+B*B+B*B*B
    );

    double valid_roots[3]{};
    int valid_count = 0;

    for (int i = 0; i < roots.count; ++i) {
        const double z = roots.values[i];

        if (std::isfinite(z) && z > B*(1.0+1e-12))
            valid_roots[valid_count++] = z;
    }

    result.admissible_roots = valid_count;

    if (valid_count == 0) {
        result.status = PRTPStatus::NoPhysicalRoot;
        return result;
    }

    if (valid_count > 1 &&
        policy == PRTPRootPolicy::RejectMultiple) {
        result.status = PRTPStatus::MultipleRoots;
        return result;
    }

    const double Z =
        policy == PRTPRootPolicy::Liquid
            ? valid_roots[0]
            : valid_roots[valid_count-1];

    double v = Z*q.R*T/p;

    // For three distinct physical roots, locate the two spinodals by
    // bisecting p_v between consecutive roots.  The liquid and vapor
    // branches lie outside these stationary points.
    double branch_lower = q.b;
    double branch_upper = INFINITY;
    bool branch_bounds_valid = true;
    if (valid_count == 3) {
        double stationary[2]{};
        for (int pair = 0; pair < 2; ++pair) {
            double lo = valid_roots[pair]*q.R*T/p;
            double hi = valid_roots[pair+1]*q.R*T/p;
            const double sign_lo = pr_pressure_derivatives(q,T,lo).pv;
            const double sign_hi = pr_pressure_derivatives(q,T,hi).pv;
            if (!(std::isfinite(sign_lo) && std::isfinite(sign_hi) &&
                  sign_lo*sign_hi < 0.0)) {
                branch_bounds_valid = false;
                break;
            }
            for (int j = 0; j < 48; ++j) {
                const double mid = lo + (hi-lo)*0.5;
                const double sign_mid = pr_pressure_derivatives(q,T,mid).pv;
                if (!std::isfinite(sign_mid)) {
                    branch_bounds_valid = false;
                    break;
                }
                if (sign_mid == 0.0) {
                    lo = hi = mid;
                    break;
                }
                if ((sign_mid < 0.0) == (sign_lo < 0.0))
                    lo = mid;
                else
                    hi = mid;
            }
            if (!branch_bounds_valid) break;
            stationary[pair] = lo + (hi-lo)*0.5;
        }
        if (branch_bounds_valid) {
            if (policy == PRTPRootPolicy::Liquid)
                branch_upper = stationary[0];
            else
                branch_lower = stationary[1];
        }
    }

#if !defined(THERMOMODEL_PR_TP_DISABLE_NEWTON)
#ifndef THERMOMODEL_PR_TP_NEWTON_ITERATIONS
#define THERMOMODEL_PR_TP_NEWTON_ITERATIONS 2
#endif
    for (int iteration = 0;
         iteration < THERMOMODEL_PR_TP_NEWTON_ITERATIONS && branch_bounds_valid;
         ++iteration) {
        const auto derivatives =
            pr_pressure_derivatives(q, T, v);

        if (!(std::isfinite(derivatives.p) &&
              std::isfinite(derivatives.pv) &&
              derivatives.pv != 0.0))
            break;

        const double residual = derivatives.p-p;
        const double correction = residual/derivatives.pv;
        const double candidate = v-correction;

        if (!(std::isfinite(candidate) &&
              candidate > branch_lower && candidate < branch_upper))
            break;

        const auto candidate_derivatives =
            pr_pressure_derivatives(q, T, candidate);

        if (!std::isfinite(candidate_derivatives.p))
            break;

        const double candidate_residual =
            candidate_derivatives.p-p;

        // Do not accept a Newton step that worsens the EOS residual.
        // Equal residuals are permitted at the floating-point floor.
        if (!(std::fabs(candidate_residual) <= std::fabs(residual)))
            break;

        v = candidate;

#if defined(THERMOMODEL_PR_TP_GUARDED_NEWTON)
        // Conditionally perform a second Newton correction.
        // The first accepted correction has already updated v.
        if (iteration == 0) {
            // G3.5g: division-free refinement guard.
            const double residual_limit =
                1.0e-12 * std::fmax(1.0, std::fabs(p));

            const double conditioning_measure =
                std::fabs(v * candidate_derivatives.pv);

            const bool needs_refinement =
                !std::isfinite(candidate_residual) ||
                !std::isfinite(conditioning_measure) ||
                conditioning_measure == 0.0 ||
                std::fabs(candidate_residual) > residual_limit ||
                conditioning_measure < 0.1 * std::fabs(p);

            if (!needs_refinement) {
                break;
            }
        }
#endif
    }
#endif

    result.selected_v = v;
    result.selected_Z = p*v/(q.R*T);
    result.status = PRTPStatus::Success;
    return result;
}
} // namespace thermomodel::detail
#undef THERMOMODEL_PR_ROOT_HD
