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
#include <vector>

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

std::string default_family() {
    if (const char* value = std::getenv("THERMOGPU_TDAR_MIXTURE")) return value;
    return "fixed";
}

std::size_t default_components() {
    if (const char* value = std::getenv("THERMOGPU_TDAR_COMPONENTS")) return std::stoull(value);
    return 1;
}

Mixture benchmark_mixture(std::string_view family, std::size_t components) {
    if (family == "fixed") return make_fixed_benchmark_mixture(components);
    if (family == "distinct") return make_distinct_benchmark_mixture(components);
    throw std::invalid_argument("mixture family must be 'fixed' or 'distinct'");
}

void emit(double T, double P, std::size_t components=0, std::string family={}) {
    if (components == 0) components = default_components();
    if (family.empty()) family = default_family();
    const auto r = (family == "fixed" && components == 1)
        ? evaluate_z_oracle(methane, {P, T})
        : evaluate_z_oracle(benchmark_mixture(family, components), {P, T});
    std::cout << std::setprecision(17)
              << T << '\t' << P << '\t' << r.Z << '\t'
              << r.dZ_dT << '\t' << r.dZ_dP << '\t'
              << classification_name(r.root_classification) << '\t'
              << r.cubic_residual << '\n';
}

void usage(std::ostream& out) {
    out << "Usage:\n"
        << "  thermogpu_tdar_oracle T_K P_Pa\n"
        << "  thermogpu_tdar_oracle --stdin [--family fixed|distinct] [--components N]\n"
        << "  thermogpu_tdar_oracle [--family fixed|distinct] [--components N] T_K P_Pa\n\n"
        << "Output columns (TSV): T_K P_Pa Z dZ_dT dZ_dP root_classification cubic_residual\n"
        << "In --stdin mode, read whitespace-separated T_K P_Pa pairs until EOF.\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")) {
            usage(std::cout); return 0;
        }
        bool stdin_mode = false;
        std::size_t components = default_components();
        std::string family = default_family();
        std::vector<std::string_view> positional;
        for (int i=1; i<argc; ++i) {
            const std::string_view arg(argv[i]);
            if (arg == "--stdin") stdin_mode = true;
            else if (arg == "--components" && i+1 < argc) components = std::stoull(argv[++i]);
            else if (arg == "--family" && i+1 < argc) family = argv[++i];
            else if (arg.size() >= 2 && arg.substr(0,2) == "--") { usage(std::cerr); return 2; }
            else positional.push_back(arg);
        }
        // Validate the requested family/count even before reading stdin.
        (void)benchmark_mixture(family, components);
        if (stdin_mode) {
            if (!positional.empty()) { usage(std::cerr); return 2; }
            double T=0.0, P=0.0;
            while (std::cin >> T >> P) emit(T, P, components, family);
            if (!std::cin.eof()) { std::cerr << "invalid input: expected T_K P_Pa pairs\n"; return 2; }
            return 0;
        }
        if (positional.size() != 2) { usage(std::cerr); return 2; }
        double T=0.0, P=0.0;
        if (!parse_double(positional[0], T) || !parse_double(positional[1], P)) {
            std::cerr << "invalid numeric argument\n"; return 2;
        }
        emit(T, P, components, family); return 0;
    } catch (const std::exception& e) {
        std::cerr << "thermogpu_tdar_oracle: " << e.what() << '\n';
        return 1;
    }
}
