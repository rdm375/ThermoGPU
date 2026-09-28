// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/component.hpp"
#include "thermogpu/mixture.hpp"

#ifdef THERMOGPU_HAS_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;
volatile double benchmark_sink = 0.0;

thermogpu::Mixture binary_mixture() {
    using thermogpu::Component;
    thermogpu::Mixture m;
    m.components = {
        Component{"methane", 190.56, 4.5992e6, 0.01142, 0.016043},
        Component{"ethane",  305.32, 4.8720e6, 0.09950, 0.030070},
    };
    m.mole_fractions = {0.8, 0.2};
    m.binary_interactions = {0.0, 0.0,
             0.0, 0.0};
    return m;
}

thermogpu::MixtureBatch make_batch(std::size_t n) {
    thermogpu::MixtureBatch b;
    b.pressure_Pa.resize(n);
    b.temperature_K.resize(n);
    b.mole_fractions.resize(2*n);

    // Deterministic, smoothly varying states.  The formulas intentionally avoid
    // random-number generation so every backend sees byte-identical inputs.
    for (std::size_t s = 0; s < n; ++s) {
        const double u = n > 1 ? static_cast<double>(s) / static_cast<double>(n - 1) : 0.5;
        const double v = static_cast<double>((s * 37u) % 1009u) / 1008.0;
        const double x1 = 0.05 + 0.90 * static_cast<double>((s * 101u) % 997u) / 996.0;
        b.pressure_Pa[s] = 1.0e5 + u * 11.9e6;
        b.temperature_K[s] = 240.0 + v * 180.0;
        b.mole_fractions[2*s] = x1;
        b.mole_fractions[2*s + 1] = 1.0 - x1;
    }
    return b;
}

double checksum(const thermogpu::MixtureBatchResult& r) {
    double sum = 0.0;
    for (double x : r.Z) sum += x;
    for (double x : r.density_kg_per_m3) sum += x * 1.0e-3;
    for (double x : r.ln_phi) sum += x * 1.0e-2;
    return sum;
}

struct Timing {
    double seconds_per_call{};
    double eos_per_second{};
    double ns_per_state{};
    double checksum{};
};

template<class Evaluator>
Timing measure(const thermogpu::Mixture& mixture,
               const thermogpu::MixtureBatch& batch,
               Evaluator&& evaluator) {
    // Warm-up initializes the OpenMP runtime and faults in code/data before timing.
    auto result = evaluator(mixture, batch);
    double sum = checksum(result);

    const std::size_t n = batch.size();
    const std::size_t calls_per_sample = std::clamp<std::size_t>(100000 / std::max<std::size_t>(n, 1), 1, 100000);
    constexpr int samples = 5;
    std::vector<double> times;
    times.reserve(samples);

    for (int sample = 0; sample < samples; ++sample) {
        const auto t0 = clock_type::now();
        for (std::size_t k = 0; k < calls_per_sample; ++k)
            result = evaluator(mixture, batch);
        const auto t1 = clock_type::now();
        sum += checksum(result);
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(elapsed / static_cast<double>(calls_per_sample));
    }

    std::sort(times.begin(), times.end());
    const double seconds = times[times.size()/2];
    benchmark_sink = sum;
    return Timing{
        seconds,
        static_cast<double>(n) / seconds,
        seconds * 1.0e9 / static_cast<double>(n),
        sum,
    };
}

void print_row(const std::string& backend, int threads, std::size_t states,
               const Timing& t, double scalar_seconds) {
    const double speedup = scalar_seconds / t.seconds_per_call;
    std::cout << std::left << std::setw(10) << backend
              << std::right << std::setw(8) << threads
              << std::setw(12) << states
              << std::setw(14) << std::fixed << std::setprecision(1) << t.ns_per_state
              << std::setw(16) << std::scientific << std::setprecision(3) << t.eos_per_second
              << std::setw(12) << std::fixed << std::setprecision(3) << speedup
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::size_t> sizes{1, 10, 100, 1000, 10000, 100000, 1000000};
        if (argc > 1) {
            sizes.clear();
            for (int i = 1; i < argc; ++i) {
                const auto value = std::stoull(argv[i]);
                if (value == 0) throw std::invalid_argument("batch size must be positive");
                sizes.push_back(static_cast<std::size_t>(value));
            }
        }

        const auto mixture = binary_mixture();
#ifdef THERMOGPU_HAS_OPENMP
        const int startup_max_threads = omp_get_max_threads();
#endif
        std::cout << "ThermoGPU PR batch benchmark\n"
                  << "Mixture: CH4/C2H6, deterministic varying P/T/z states\n"
                  << "Timing: median of 5 samples after warm-up; validation excluded\n";
#ifdef THERMOGPU_HAS_OPENMP
        std::cout << "OpenMP max threads at startup: " << startup_max_threads << "\n";
#else
        std::cout << "OpenMP: disabled\n";
#endif
        std::cout << '\n'
                  << std::left << std::setw(10) << "Backend"
                  << std::right << std::setw(8) << "Threads"
                  << std::setw(12) << "States"
                  << std::setw(14) << "ns/state"
                  << std::setw(16) << "EOS/s"
                  << std::setw(12) << "Speedup"
                  << '\n';

        for (const std::size_t n : sizes) {
            const auto batch = make_batch(n);
            const Timing scalar = measure(mixture, batch, thermogpu::evaluate_mixture_batch_scalar);
            print_row("scalar", 1, n, scalar, scalar.seconds_per_call);

#ifdef THERMOGPU_HAS_OPENMP
            std::vector<int> thread_counts{1, 2, 4, 8};
            for (int threads : thread_counts) {
                if (threads > startup_max_threads) continue;
                omp_set_dynamic(0);
                omp_set_num_threads(threads);
                const Timing parallel = measure(mixture, batch, thermogpu::evaluate_mixture_batch_openmp);
                print_row("OpenMP", threads, n, parallel, scalar.seconds_per_call);
            }
#endif
            std::cout << '\n';
        }

        // Make the observable sink visible without contaminating the timed region.
        if (!std::isfinite(benchmark_sink)) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "benchmark error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
