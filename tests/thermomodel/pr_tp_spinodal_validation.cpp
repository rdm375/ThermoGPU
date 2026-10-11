#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

using namespace thermomodel::detail;

static PRPolynomialStateParameters make_parameters() {
    const auto q = make_pr_coefficients(0.016043, 190.56, 4599200.0, 0.01142);
    constexpr double nasa[5] = {5.14987613, -1.36709788e-2,
        4.91800599e-5, -4.84743026e-8, 1.66693956e-11};
    PolynomialCpKernel cp{};
    cp.scale = 1000.0;
    double power = 1.0;
    for (int k = 0; k < 5; ++k) {
        cp.a[k] = nasa[k]*q.R*power;
        power *= cp.scale;
    }
    return {{q, 101325.0}, cp, 298.15};
}

// Locate stationary points independently of the cubic root solver.
// The logarithmic scan is deliberately outside the (T,p) evaluator.
static std::vector<double> spinodals(const PRCoefficientParameters& q, double T) {
    std::vector<double> out;
    const double vmin = q.b*(1.0+1e-8);
    const double vmax = q.R*T/100.0*10.0;
    constexpr int samples = 4000;
    double previous_v = vmin;
    double previous_f = pr_pressure_derivatives(q,T,previous_v).pv;
    for (int i = 1; i <= samples; ++i) {
        const double v = vmin*std::pow(vmax/vmin, double(i)/samples);
        const double f = pr_pressure_derivatives(q,T,v).pv;
        if (std::isfinite(f) && std::isfinite(previous_f) &&
            ((f < 0.0) != (previous_f < 0.0))) {
            double lo = previous_v, hi = v;
            const bool lo_negative = previous_f < 0.0;
            for (int k = 0; k < 90; ++k) {
                const double mid = lo + (hi-lo)*0.5;
                if (mid == lo || mid == hi) break;
                const double fm = pr_pressure_derivatives(q,T,mid).pv;
                if ((fm < 0.0) == lo_negative) lo = mid;
                else hi = mid;
            }
            out.push_back(lo+(hi-lo)*0.5);
        }
        previous_v = v;
        previous_f = f;
    }
    return out;
}

int main() {
    const auto parameters = make_parameters();
    const auto& q = parameters.pr.coefficients;
    std::size_t checks = 0, failures = 0, successes = 0, multiple = 0;
    double worst_residual = 0.0;
    auto check = [&](bool ok, const char* reason, double T, double p) {
        ++checks;
        if (!ok) {
            ++failures;
            if (failures <= 20)
                std::cerr << std::setprecision(17) << "FAIL " << reason
                          << " T=" << T << " p=" << p << '\n';
        }
    };

    // Temperatures spanning the subcritical region and the critical limit.
    for (double fraction : {0.50, 0.70, 0.85, 0.95, 0.99,
                            0.999, 0.9999, 0.99999, 0.999999,
                            1.0, 1.000001, 1.001, 1.01, 1.10}) {
        const double T = q.Tc*fraction;
        const auto stationary = spinodals(q,T);
        check(fraction >= 0.999999 || stationary.size() == (fraction < 1.0 ? 2u : 0u),
              "spinodal count", T, 0.0);
        if (fraction < 0.999999 && stationary.size() != 2) continue;

        std::vector<double> pressures;
        pressures.push_back(4599200.0*0.1);
        pressures.push_back(4599200.0);
        pressures.push_back(4599200.0*2.0);
        if (stationary.size() == 2) {
            for (double vspin : stationary) {
                const double ps = pr_pressure_derivatives(q,T,vspin).p;
                if (!(ps > 0.0)) continue;
                for (double delta : {-1e-5, -1e-7, -1e-9, -1e-11,
                                      0.0, 1e-11, 1e-9, 1e-7, 1e-5})
                    pressures.push_back(ps*(1.0+delta));
            }
        }
        for (double p : pressures) {
            if (!(p > 0.0)) continue;
            const auto vapor = pr_evaluate_Tp_polynomial(parameters,T,p,PRTPRootPolicy::Vapor);
            const auto liquid = pr_evaluate_Tp_polynomial(parameters,T,p,PRTPRootPolicy::Liquid);
            const auto reject = pr_evaluate_Tp_polynomial(parameters,T,p,PRTPRootPolicy::RejectMultiple);
            check(vapor.admissible_roots == liquid.admissible_roots &&
                  vapor.admissible_roots == reject.admissible_roots,
                  "root count differs by policy", T,p);
            if (vapor.admissible_roots > 1) {
                ++multiple;
                check(reject.status == PRTPStatus::MultipleRoots,
                      "reject-multiple policy", T,p);
            }
            if (vapor.status == PRTPStatus::Success && liquid.status == PRTPStatus::Success) {
                check(liquid.selected_v <= vapor.selected_v,
                      "root ordering",T,p);
            }
            for (auto result : {vapor, liquid, reject}) {
                if (result.status != PRTPStatus::Success) continue;
                ++successes;
                const double v = result.selected_v;
                check(std::isfinite(v) && v > q.b, "physical volume",T,p);
                const auto d = pr_pressure_derivatives(q,T,v);
                const double residual = std::fabs(d.p-p)/std::max({1.0,std::fabs(d.p),std::fabs(p)});
                worst_residual = std::max(worst_residual,residual);
                // Backward error; tight enough to detect poorly resolved roots,
                // but separate from the 1e-12 CPU/GPU *parity* requirement.
                check(std::isfinite(residual) && residual <= 1e-8,
                      "pressure reconstruction",T,p);
                check(std::isfinite(d.pv) && d.pv < 0.0,
                      "mechanical stability",T,p);
                check(std::fabs(result.selected_Z - p*v/(q.R*T)) <=
                      1e-12*std::max(1.0,std::fabs(result.selected_Z)),
                      "compressibility consistency",T,p);
                if (stationary.size() == 2) {
                    const bool outer = v < stationary[0] || v > stationary[1];
                    check(outer,"spinodal branch",T,p);
                }
            }
        }
    }
    std::cout << std::setprecision(17)
              << "G2e.3c independent checks: " << checks << '\n'
              << "G2e.3c successful evaluations: " << successes << '\n'
              << "G2e.3c multiple-root cases: " << multiple << '\n'
              << "G2e.3c worst pressure residual: " << worst_residual << '\n'
              << "G2e.3c failures: " << failures << '\n';
    return failures ? 1 : 0;
}
