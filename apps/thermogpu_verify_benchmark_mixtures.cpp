// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/mixture_benchmark.hpp"
#include "thermogpu/peng_robinson.hpp"
#include "thermogpu/tdar_oracle.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace thermogpu;

namespace {
struct Difference {
    double value{};
    double T{};
    double P{};
    double lhs{};
    double rhs{};
    void observe(double a, double b, double t, double p) {
        const double d = std::abs(a-b);
        if (d > value) { value=d; T=t; P=p; lhs=a; rhs=b; }
    }
};

Mixture make_distinct(std::size_t n) {
    if (n < 1 || n > 5) throw std::invalid_argument("distinct mixture size must be 1..5");
    static constexpr Component components[] = {
        {"methane",        190.564,  4599200.0, 0.01142, 0.01604246},
        {"ethane",         305.322,  4872200.0, 0.0995,  0.03006904},
        {"propane",        369.83,   4248000.0, 0.1523,  0.04409562},
        {"nitrogen",       126.192,  3395800.0, 0.0372,  0.0280134},
        {"carbon_dioxide", 304.1282, 7377300.0, 0.22394, 0.0440095},
    };
    static constexpr double final_z[] = {0.80,0.08,0.04,0.04,0.04};
    Mixture m;
    m.components.assign(components, components+n);
    double sum=0.0;
    for (std::size_t i=0;i<n;++i) sum += final_z[i];
    for (std::size_t i=0;i<n;++i) m.mole_fractions.push_back(final_z[i]/sum);
    m.binary_interactions.assign(n*n,0.0);
    validate_mixture(m);
    return m;
}

void print_mixture(const std::string& label, const Mixture& m) {
    std::cout << label << ": ";
    for (std::size_t i=0;i<m.size();++i) {
        if (i) std::cout << ", ";
        std::cout << m.components[i].name << '=' << std::setprecision(12) << m.mole_fractions[i];
    }
    std::cout << '\n';
}

void print_difference(const std::string& name, const Difference& d) {
    std::cout << std::scientific << std::setprecision(17)
              << name << " = " << d.value << " at T=" << d.T << " P=" << d.P
              << " lhs=" << d.lhs << " rhs=" << d.rhs << '\n';
}
}

int main() {
    const Mixture c4=make_fixed_benchmark_mixture(4);
    const Mixture c8=make_fixed_benchmark_mixture(8);
    const Mixture c16=make_fixed_benchmark_mixture(16);
    Difference z48,z416,a48,a416,b48,b416,dt48,dt416,dp48,dp416;
    std::size_t states=0;
    constexpr int NT=65, NP=65;
    constexpr double Tmin=250.0,Tmax=350.0,Pmin=1.0e6,Pmax=10.0e6;
    for (int it=0;it<NT;++it) {
        const double T=Tmin+(Tmax-Tmin)*it/(NT-1.0);
        for (int ip=0;ip<NP;++ip) {
            const double P=Pmin+(Pmax-Pmin)*ip/(NP-1.0);
            const State s{P,T};
            const auto r4=evaluate_mixture(c4,s), r8=evaluate_mixture(c8,s), r16=evaluate_mixture(c16,s);
            const auto o4=evaluate_z_oracle(c4,s), o8=evaluate_z_oracle(c8,s), o16=evaluate_z_oracle(c16,s);
            z48.observe(r4.Z,r8.Z,T,P); z416.observe(r4.Z,r16.Z,T,P);
            a48.observe(r4.a_m,r8.a_m,T,P); a416.observe(r4.a_m,r16.a_m,T,P);
            b48.observe(r4.b_m,r8.b_m,T,P); b416.observe(r4.b_m,r16.b_m,T,P);
            dt48.observe(o4.dZ_dT,o8.dZ_dT,T,P); dt416.observe(o4.dZ_dT,o16.dZ_dT,T,P);
            dp48.observe(o4.dZ_dP,o8.dZ_dP,T,P); dp416.observe(o4.dZ_dP,o16.dZ_dP,T,P);
            ++states;
        }
    }
    std::cout << "=== DUPLICATION INVARIANCE ===\n" << "states tested: " << states << '\n';
    print_difference("max |Z4-Z8|",z48); print_difference("max |Z4-Z16|",z416);
    print_difference("max |a_m4-a_m8|",a48); print_difference("max |a_m4-a_m16|",a416);
    print_difference("max |b_m4-b_m8|",b48); print_difference("max |b_m4-b_m16|",b416);
    print_difference("max |dZ/dT4-dZ/dT8|",dt48); print_difference("max |dZ/dT4-dZ/dT16|",dt416);
    print_difference("max |dZ/dP4-dZ/dP8|",dp48); print_difference("max |dZ/dP4-dZ/dP16|",dp416);
    const bool pass=z48.value<1e-12 && z416.value<1e-12 && a48.value<1e-10 && a416.value<1e-10 &&
                    b48.value<1e-14 && b416.value<1e-14 && dt48.value<1e-10 && dt416.value<1e-10 &&
                    dp48.value<1e-15 && dp416.value<1e-15;
    std::cout << "duplication invariance: " << (pass?"PASS":"FAIL") << "\n\n";

    std::cout << "=== PROPOSED DISTINCT MIXTURES ===\n";
    std::vector<Mixture> distinct;
    for (std::size_t n=1;n<=5;++n) { distinct.push_back(make_distinct(n)); print_mixture("C"+std::to_string(n),distinct.back()); }
    std::cout << "\n=== FIXED-C4 VS DISTINCT-C4 ===\n";
    print_mixture("fixed-C4",c4); print_mixture("distinct-C4",distinct[3]);
    std::cout << "\n=== DISTINCT Z(T,P) SPOT CHECK ===\nT_K,P_Pa,Z_C1,Z_C2,Z_C3,Z_C4,Z_C5\n";
    const State checks[]={{1e6,250},{5e6,250},{1e7,250},{1e6,300},{5e6,300},{1e7,300},{1e6,350},{5e6,350},{1e7,350}};
    std::cout << std::setprecision(17);
    for (const auto& s:checks) {
        std::cout << s.temperature_K << ',' << s.pressure_Pa;
        for (const auto& m:distinct) std::cout << ',' << evaluate_mixture(m,s).Z;
        std::cout << '\n';
    }
    return pass?0:1;
}
