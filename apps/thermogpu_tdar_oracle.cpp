// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/tdar_oracle.hpp"
#include "thermogpu/mixture_benchmark.hpp"

#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

using namespace thermogpu;

namespace {
constexpr Component methane{"methane", 190.564, 4599200.0, 0.01142, 0.01604246};

const char* classification_name(RootClassification c) {
    switch (c) {
        case RootClassification::one_real: return "one_real";
        case RootClassification::three_real: return "three_real";
        default: return "ambiguous";
    }
}

bool parse_double(std::string_view text, double& value) {
    const char* first = text.data();
    const char* last = first + text.size();
    const auto result = std::from_chars(first, last, value);
    return result.ec == std::errc{} && result.ptr == last;
}

std::size_t default_components() {
    if (const char* value = std::getenv("THERMOGPU_TDAR_COMPONENTS")) {
        const auto n = std::stoull(value);
        if (n != 1 && n != 2 && n != 4 && n != 8 && n != 16)
            throw std::invalid_argument("THERMOGPU_TDAR_COMPONENTS must be one of 1,2,4,8,16");
        return n;
    }
    return 1;
}

void emit(double T, double P, std::size_t components=0) {
    if (components == 0) components = default_components();
    const auto r = components == 1 ? evaluate_z_oracle(methane, {P, T})
                                   : evaluate_z_oracle(make_fixed_benchmark_mixture(components), {P, T});
    std::cout << std::setprecision(17)
              << T << '\t' << P << '\t' << r.Z << '\t'
              << r.dZ_dT << '\t' << r.dZ_dP << '\t'
              << classification_name(r.root_classification) << '\t'
              << r.cubic_residual << '\n';
}

void usage(std::ostream& out) {
    out << "Usage:\n"
        << "  thermogpu_tdar_oracle T_K P_Pa\n"
        << "  thermogpu_tdar_oracle --stdin [--components N]\n"
        << "  thermogpu_tdar_oracle --components N T_K P_Pa\n\n"
        << "Output columns (TSV): T_K P_Pa Z dZ_dT dZ_dP root_classification cubic_residual\n"
        << "In --stdin mode, read whitespace-separated T_K P_Pa pairs until EOF.\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
            usage(std::cout);
            return 0;
        }
        if ((argc == 2 || argc == 4) && std::string_view(argv[1]) == "--stdin") {
            std::size_t components = default_components();
            if (argc == 4) {
                if (std::string_view(argv[2]) != "--components") { usage(std::cerr); return 2; }
                components = std::stoull(argv[3]);
            }
            double T = 0.0, P = 0.0;
            while (std::cin >> T >> P) emit(T, P, components);
            if (!std::cin.eof()) {
                std::cerr << "invalid input: expected T_K P_Pa pairs\n";
                return 2;
            }
            return 0;
        }
        if (argc == 5 && std::string_view(argv[1]) == "--components") {
            const std::size_t components = std::stoull(argv[2]);
            double T=0.0, P=0.0;
            if (!parse_double(argv[3], T) || !parse_double(argv[4], P)) return 2;
            emit(T, P, components); return 0;
        }
        if (argc == 3) {
            double T = 0.0, P = 0.0;
            if (!parse_double(argv[1], T) || !parse_double(argv[2], P)) {
                std::cerr << "invalid numeric argument\n";
                return 2;
            }
            emit(T, P);
            return 0;
        }
        usage(std::cerr);
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "thermogpu_tdar_oracle: " << e.what() << '\n';
        return 1;
    }
}
