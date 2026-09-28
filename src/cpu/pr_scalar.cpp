// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/peng_robinson.hpp"
#include <algorithm>
#include <cmath>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>

namespace thermogpu {
namespace {
constexpr double sqrt2 = 1.41421356237309504880168872420969808;
constexpr double pr_omega_a = 0.4572355289213822;
constexpr double pr_omega_b = 0.0777960739038884;

void validate_state(const State& s) {
    if (!std::isfinite(s.pressure_Pa) || !std::isfinite(s.temperature_K) ||
        s.pressure_Pa <= 0.0 || s.temperature_K <= 0.0) {
        throw std::invalid_argument("P and T must be finite and positive");
    }
}

double select_largest_root(const CubicRoots& roots) {
    if (roots.count == 0) throw std::runtime_error("PR cubic returned no real roots");
    return *std::max_element(roots.values.begin(), roots.values.begin() + roots.count);
}

double pr_log_term(double Z, double B) {
    if (!(Z > B)) throw std::domain_error("selected PR root requires Z > B");
    const double numerator = Z + (1.0 + sqrt2) * B;
    const double denominator = Z + (1.0 - sqrt2) * B;
    if (!(numerator > 0.0 && denominator > 0.0)) {
        throw std::domain_error("invalid PR fugacity logarithm arguments");
    }
    return std::log(numerator / denominator);
}
} // namespace

double Mixture::kij(std::size_t i, std::size_t j) const {
    if (i >= size() || j >= size()) throw std::out_of_range("kij index out of range");
    return binary_interactions[i * size() + j];
}

void validate_mixture(const Mixture& m, double composition_tolerance) {
    const auto n = m.components.size();
    if (n == 0) throw std::invalid_argument("mixture must contain at least one component");
    if (m.mole_fractions.size() != n) throw std::invalid_argument("composition size does not match component count");
    if (m.binary_interactions.size() != n * n) throw std::invalid_argument("binary interaction matrix must be dense n x n");
    if (!(composition_tolerance >= 0.0) || !std::isfinite(composition_tolerance)) throw std::invalid_argument("invalid composition tolerance");

    double sum = 0.0;
    for (double z : m.mole_fractions) {
        if (!std::isfinite(z) || z < 0.0 || z > 1.0) throw std::invalid_argument("mole fractions must be finite and in [0,1]");
        sum += z;
    }
    if (std::abs(sum - 1.0) > composition_tolerance) throw std::invalid_argument("mole fractions must sum to one; no silent normalization is performed");

    constexpr double symmetry_tolerance = 1.0e-14;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            const double kij = m.binary_interactions[i*n+j];
            if (!std::isfinite(kij)) throw std::invalid_argument("binary interaction coefficients must be finite");
            if (std::abs(kij - m.binary_interactions[j*n+i]) > symmetry_tolerance) throw std::invalid_argument("binary interaction matrix must be symmetric");
        }
    }
}

double kappa(double w) { return 0.37464 + 1.54226*w - 0.26992*w*w; }

double alpha(const Component& c, double T) {
    if (!(T > 0.0) || !std::isfinite(T)) throw std::invalid_argument("temperature must be finite and positive");
    const double q = 1.0 + kappa(c.acentric_factor) * (1.0 - std::sqrt(T / c.critical_temperature_K));
    return q*q;
}

double a_parameter(const Component& c) {
    const double x = gas_constant * c.critical_temperature_K;
    return pr_omega_a * x*x / c.critical_pressure_Pa;
}

double b_parameter(const Component& c) {
    return pr_omega_b * gas_constant * c.critical_temperature_K / c.critical_pressure_Pa;
}

double cubic_residual(double Z, double A, double B) {
    return Z*Z*Z - (1.0-B)*Z*Z + (A-3.0*B*B-2.0*B)*Z - (A*B-B*B-B*B*B);
}

CubicRoots solve_z_cubic(double A, double B) {
    const double a = B - 1.0;
    const double b = A - 3.0*B*B - 2.0*B;
    const double c = -(A*B - B*B - B*B*B);
    const double p = b - a*a/3.0;
    const double q = 2.0*a*a*a/27.0 - a*b/3.0 + c;
    const double disc = q*q/4.0 + p*p*p/27.0;
    CubicRoots o;
    constexpr double eps = 1.0e-14;
    if (disc > eps) {
        const double s = std::sqrt(disc);
        const double x1 = -q/2.0 + s;
        const double x2 = -q/2.0 - s;
        const double x = std::abs(x1) >= std::abs(x2) ? x1 : x2;
        const double u = std::cbrt(x);
        // For a depressed cubic, u*v = -p/3. Recovering the smaller
        // Cardano term this way avoids catastrophic cancellation in x1/x2.
        const double v = (u != 0.0) ? -p/(3.0*u)
                                    : std::cbrt(x == x1 ? x2 : x1);
        o.values[0] = u + v - a/3.0;
        o.count = 1;
        o.classification = RootClassification::one_real;
    } else if (std::abs(p) < eps && std::abs(q) < eps) {
        o.values[0] = -a/3.0;
        o.count = 1;
        o.classification = RootClassification::one_real;
    } else {
        const double r = 2.0 * std::sqrt(std::max(0.0, -p/3.0));
        const double arg = std::clamp((3.0*q/(2.0*p))*std::sqrt(-3.0/p), -1.0, 1.0);
        const double th = std::acos(arg)/3.0;
        for (int k=0; k<3; ++k) o.values[k] = r*std::cos(th-2.0*std::numbers::pi*k/3.0)-a/3.0;
        std::sort(o.values.begin(), o.values.end());
        o.count = 3;
        o.classification = RootClassification::three_real;
    }
    return o;
}

PureResult evaluate_pure(const Component& c, const State& s) {
    validate_state(s);
    PureResult r;
    r.a_alpha = a_parameter(c) * alpha(c, s.temperature_K);
    r.b = b_parameter(c);
    r.A = r.a_alpha*s.pressure_Pa/(gas_constant*gas_constant*s.temperature_K*s.temperature_K);
    r.B = r.b*s.pressure_Pa/(gas_constant*s.temperature_K);
    const auto roots = solve_z_cubic(r.A, r.B);
    r.root_classification = roots.classification;
    r.Z = select_largest_root(roots);
    r.cubic_residual = cubic_residual(r.Z, r.A, r.B);
    r.density_kg_per_m3 = s.pressure_Pa*c.molar_mass_kg_per_mol/(r.Z*gas_constant*s.temperature_K);
    const double L = pr_log_term(r.Z, r.B);
    r.ln_phi = r.Z - 1.0 - std::log(r.Z-r.B) - r.A/(2.0*sqrt2*r.B)*L;
    return r;
}

MixtureDiagnostics evaluate_mixture_diagnostics(const Mixture& m, const State& s) {
    validate_state(s);
    validate_mixture(m);
    const std::size_t n = m.size();
    MixtureDiagnostics d;
    d.alpha.resize(n);
    d.a.resize(n);
    d.a_alpha.resize(n);
    d.b.resize(n);
    d.sum_z_aij.assign(n, 0.0);

    for (std::size_t i=0; i<n; ++i) {
        d.alpha[i] = alpha(m.components[i], s.temperature_K);
        d.a[i] = a_parameter(m.components[i]);
        d.a_alpha[i] = d.a[i] * d.alpha[i];
        d.b[i] = b_parameter(m.components[i]);
    }

    std::vector<double> aij(n*n);
    auto& r = d.result;
    for (std::size_t i=0; i<n; ++i) {
        r.b_m += m.mole_fractions[i] * d.b[i];
        r.molar_mass_kg_per_mol += m.mole_fractions[i] * m.components[i].molar_mass_kg_per_mol;
        for (std::size_t j=0; j<n; ++j) {
            const double a_ij = std::sqrt(d.a_alpha[i]*d.a_alpha[j]) * (1.0 - m.kij(i,j));
            aij[i*n+j] = a_ij;
            d.sum_z_aij[i] += m.mole_fractions[j] * a_ij;
            r.a_m += m.mole_fractions[i] * m.mole_fractions[j] * a_ij;
        }
    }

    r.A = r.a_m*s.pressure_Pa/(gas_constant*gas_constant*s.temperature_K*s.temperature_K);
    r.B = r.b_m*s.pressure_Pa/(gas_constant*s.temperature_K);
    d.roots = solve_z_cubic(r.A, r.B);
    r.root_classification = d.roots.classification;
    r.root_selection_ambiguous = d.roots.count > 1;
    r.Z = select_largest_root(d.roots);
    r.cubic_residual = cubic_residual(r.Z, r.A, r.B);
    r.density_kg_per_m3 = s.pressure_Pa*r.molar_mass_kg_per_mol/(r.Z*gas_constant*s.temperature_K);

    const double L = pr_log_term(r.Z, r.B);
    r.ln_phi.resize(n);
    for (std::size_t i=0; i<n; ++i) {
        const double bi_over_bm = d.b[i] / r.b_m;
        const double attraction = (2.0*d.sum_z_aij[i]/r.a_m) - bi_over_bm;
        r.ln_phi[i] = bi_over_bm*(r.Z-1.0) - std::log(r.Z-r.B)
                    - r.A/(2.0*sqrt2*r.B) * attraction * L;
    }
    return d;
}

MixtureResult evaluate_mixture(const Mixture& m, const State& s) {
    return evaluate_mixture_diagnostics(m, s).result;
}

} // namespace thermogpu
