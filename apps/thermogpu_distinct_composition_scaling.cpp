// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Richard Myers

#include "thermogpu/batch.hpp"
#include "thermogpu/mixture_benchmark.hpp"

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
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

volatile double sink = 0.0;
double target_seconds = 0.20;
double calibration_floor = 0.025;
int samples = 9;

thermogpu::MixtureBatch make_batch(
    const thermogpu::Mixture& mixture,
    std::size_t n)
{
    thermogpu::MixtureBatch b;
    const std::size_t nc = mixture.size();

    b.pressure_Pa.resize(n);
    b.temperature_K.resize(n);
    b.mole_fractions.resize(n * nc);

    for (std::size_t s = 0; s < n; ++s) {
        const double u =
            n > 1 ? static_cast<double>(s) /
                        static_cast<double>(n - 1)
                  : 0.5;
        const double v =
            static_cast<double>((s * 37u) % 1009u) / 1008.0;

        // Match the frozen distinct-mixture TDAR domain.
        b.pressure_Pa[s] = 1.0e6 + u * 9.0e6;
        b.temperature_K[s] = 250.0 + v * 100.0;

        std::copy(
            mixture.mole_fractions.begin(),
            mixture.mole_fractions.end(),
            b.mole_fractions.begin() + s * nc);
    }

    return b;
}

double checksum(const thermogpu::MixtureBatchResult& r)
{
    double x = 0.0;
    for (double v : r.Z)
        x += v;
    for (double v : r.density_kg_per_m3)
        x += 1.0e-3 * v;
    for (double v : r.ln_phi)
        x += 1.0e-2 * v;
    return x;
}

double checksum_z(const std::vector<double>& z)
{
    double x = 0.0;
    for (double v : z)
        x += v;
    return x;
}

double median(std::vector<double> x)
{
    std::sort(x.begin(), x.end());
    return x[x.size() / 2];
}

struct Timing {
    std::size_t calls{};
    double sec{};
    double mad_pct{};
    double ns_state{};
    double throughput{};
};

template <class F>
Timing measure_cpu(
    const thermogpu::Mixture& mixture,
    const thermogpu::MixtureBatch& batch,
    F&& f)
{
    auto result = f(mixture, batch);
    double sum = checksum(result);

    std::size_t calls = 1;

    for (;;) {
        const auto a = clock_type::now();
        for (std::size_t k = 0; k < calls; ++k)
            result = f(mixture, batch);
        const auto z = clock_type::now();

        const double elapsed =
            std::chrono::duration<double>(z - a).count();

        if (elapsed >= calibration_floor) {
            calls = std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::ceil(std::clamp(
                        calls * target_seconds / elapsed,
                        1.0,
                        1.0e9))));
            break;
        }

        calls *= 10;
    }

    std::vector<double> times;
    times.reserve(samples);

    for (int q = 0; q < samples; ++q) {
        const auto a = clock_type::now();

        for (std::size_t k = 0; k < calls; ++k)
            result = f(mixture, batch);

        const auto z = clock_type::now();

        times.push_back(
            std::chrono::duration<double>(z - a).count() /
            static_cast<double>(calls));
    }

    sum += checksum(result);
    sink = sum;

    const double med = median(times);

    std::vector<double> deviations;
    deviations.reserve(times.size());

    for (double t : times)
        deviations.push_back(std::abs(t - med));

    const double mad = median(deviations);

    return {
        calls,
        med,
        100.0 * mad / med,
        med * 1.0e9 / static_cast<double>(batch.size()),
        static_cast<double>(batch.size()) / med
    };
}

#ifdef THERMOGPU_HAS_CUDA

Timing measure_generic_resident(
    const thermogpu::Mixture& mixture,
    const thermogpu::MixtureBatch& batch)
{
    thermogpu::CudaBatchWorkspace workspace(mixture, batch);

    workspace.run();
    auto result = workspace.download();
    double sum = checksum(result);

    std::size_t calls = 1;

    for (;;) {
        const auto a = clock_type::now();
        for (std::size_t k = 0; k < calls; ++k)
            workspace.run();
        const auto z = clock_type::now();

        const double elapsed =
            std::chrono::duration<double>(z - a).count();

        if (elapsed >= calibration_floor) {
            calls = std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::ceil(std::clamp(
                        calls * target_seconds / elapsed,
                        1.0,
                        1.0e9))));
            break;
        }

        calls *= 10;
    }

    std::vector<double> times;
    times.reserve(samples);

    for (int q = 0; q < samples; ++q) {
        const auto a = clock_type::now();

        for (std::size_t k = 0; k < calls; ++k)
            workspace.run();

        const auto z = clock_type::now();

        times.push_back(
            std::chrono::duration<double>(z - a).count() /
            static_cast<double>(calls));
    }

    result = workspace.download();
    sum += checksum(result);
    sink = sum;

    const double med = median(times);

    std::vector<double> deviations;
    for (double t : times)
        deviations.push_back(std::abs(t - med));

    const double mad = median(deviations);

    return {
        calls,
        med,
        100.0 * mad / med,
        med * 1.0e9 / static_cast<double>(batch.size()),
        static_cast<double>(batch.size()) / med
    };
}

Timing measure_fixed_resident(
    const thermogpu::Mixture& mixture,
    const thermogpu::MixtureBatch& batch)
{
    thermogpu::FixedCompositionCudaWorkspace workspace(
        mixture,
        batch.pressure_Pa,
        batch.temperature_K);

    workspace.run();
    auto result = workspace.download_z();
    double sum = checksum_z(result);

    std::size_t calls = 1;

    for (;;) {
        const auto a = clock_type::now();
        for (std::size_t k = 0; k < calls; ++k)
            workspace.run();
        const auto z = clock_type::now();

        const double elapsed =
            std::chrono::duration<double>(z - a).count();

        if (elapsed >= calibration_floor) {
            calls = std::max<std::size_t>(
                1,
                static_cast<std::size_t>(
                    std::ceil(std::clamp(
                        calls * target_seconds / elapsed,
                        1.0,
                        1.0e9))));
            break;
        }

        calls *= 10;
    }

    std::vector<double> times;
    times.reserve(samples);

    for (int q = 0; q < samples; ++q) {
        const auto a = clock_type::now();

        for (std::size_t k = 0; k < calls; ++k)
            workspace.run();

        const auto z = clock_type::now();

        times.push_back(
            std::chrono::duration<double>(z - a).count() /
            static_cast<double>(calls));
    }

    result = workspace.download_z();
    sum += checksum_z(result);
    sink = sum;

    const double med = median(times);

    std::vector<double> deviations;
    for (double t : times)
        deviations.push_back(std::abs(t - med));

    const double mad = median(deviations);

    return {
        calls,
        med,
        100.0 * mad / med,
        med * 1.0e9 / static_cast<double>(batch.size()),
        static_cast<double>(batch.size()) / med
    };
}

#endif

void emit(
    std::size_t components,
    std::size_t states,
    const std::string& backend,
    int threads,
    const Timing& timing)
{
    std::cout
        << components << ','
        << states << ','
        << backend << ','
        << threads << ','
        << timing.calls << ','
        << std::setprecision(17)
        << timing.sec << ','
        << timing.ns_state << ','
        << timing.throughput << ','
        << timing.mad_pct << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        std::vector<std::size_t> sizes{
            1, 10, 100, 1000, 10000, 100000, 1000000
        };

        if (argc > 1) {
            sizes.clear();

            for (int i = 1; i < argc; ++i) {
                const auto n = std::stoull(argv[i]);
                if (!n)
                    throw std::invalid_argument(
                        "batch size must be positive");
                sizes.push_back(n);
            }
        }

#ifdef THERMOGPU_HAS_OPENMP
        const int max_threads = omp_get_max_threads();
        omp_set_dynamic(0);
#else
        const int max_threads = 1;
#endif

        std::cerr
            << "ThermoGPU distinct fixed-composition scaling; "
            << "samples=" << samples
            << " target_ms=" << target_seconds * 1000.0
            << " max_threads=" << max_threads
            << '\n';

        std::cout
            << "components,states,backend,threads,"
               "calls_per_sample,median_seconds,"
               "ns_per_state,states_per_second,mad_percent\n";

        for (std::size_t nc : {1u, 2u, 3u, 4u, 5u}) {
            const auto mixture =
                thermogpu::make_distinct_benchmark_mixture(nc);

            for (std::size_t n : sizes) {
                const auto batch = make_batch(mixture, n);

                const auto scalar = measure_cpu(
                    mixture,
                    batch,
                    thermogpu::evaluate_mixture_batch_scalar);

                emit(nc, n, "scalar", 1, scalar);

#ifdef THERMOGPU_HAS_OPENMP
                for (int threads : {1, 2, 4, 8}) {
                    if (threads > max_threads)
                        continue;

                    omp_set_num_threads(threads);

                    const auto parallel = measure_cpu(
                        mixture,
                        batch,
                        thermogpu::evaluate_mixture_batch_openmp);

                    emit(
                        nc,
                        n,
                        "openmp",
                        threads,
                        parallel);
                }
#endif

#ifdef THERMOGPU_HAS_CUDA
                const auto generic =
                    measure_generic_resident(mixture, batch);

                emit(
                    nc,
                    n,
                    "cuda_generic_resident",
                    0,
                    generic);

                const auto fixed =
                    measure_fixed_resident(mixture, batch);

                emit(
                    nc,
                    n,
                    "cuda_fixed_resident",
                    0,
                    fixed);
#endif
            }
        }

        return std::isfinite(sink)
            ? EXIT_SUCCESS
            : EXIT_FAILURE;

    } catch (const std::exception& e) {
        std::cerr
            << "distinct-composition scaling benchmark error: "
            << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
