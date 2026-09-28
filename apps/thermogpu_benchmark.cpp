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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

namespace {

using clock_type = std::chrono::steady_clock;
volatile double benchmark_sink = 0.0;
constexpr double target_sample_seconds = 0.200;
constexpr double calibration_floor_seconds = 0.025;
constexpr int sample_count = 9;
constexpr double unstable_mad_percent = 10.0;

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

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Timing {
    std::size_t calls_per_sample{};
    double min_seconds_per_call{};
    double median_seconds_per_call{};
    double max_seconds_per_call{};
    double mad_seconds_per_call{};
    double eos_per_second{};
    double ns_per_state{};
    double relative_mad_percent{};
    double checksum{};
};

template<class Evaluator>
std::size_t calibrate_calls(const thermogpu::Mixture& mixture,
                            const thermogpu::MixtureBatch& batch,
                            Evaluator&& evaluator,
                            thermogpu::MixtureBatchResult& result) {
    std::size_t calls = 1;
    for (;;) {
        const auto t0 = clock_type::now();
        for (std::size_t k = 0; k < calls; ++k)
            result = evaluator(mixture, batch);
        const auto t1 = clock_type::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();

        if (elapsed >= calibration_floor_seconds) {
            const double estimated = static_cast<double>(calls) * target_sample_seconds / elapsed;
            const double bounded = std::clamp(estimated, 1.0,
                static_cast<double>(std::numeric_limits<std::size_t>::max() / 2));
            return std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(bounded)));
        }

        if (calls > std::numeric_limits<std::size_t>::max() / 10)
            return calls;
        calls *= 10;
    }
}

template<class Evaluator>
Timing measure(const thermogpu::Mixture& mixture,
               const thermogpu::MixtureBatch& batch,
               Evaluator&& evaluator) {
    // Warm-up initializes the OpenMP runtime and faults in code/data before calibration.
    auto result = evaluator(mixture, batch);
    double sum = checksum(result);

    const std::size_t n = batch.size();
    const std::size_t calls_per_sample = calibrate_calls(mixture, batch, evaluator, result);
    sum += checksum(result);

    std::vector<double> times;
    times.reserve(sample_count);
    for (int sample = 0; sample < sample_count; ++sample) {
        const auto t0 = clock_type::now();
        for (std::size_t k = 0; k < calls_per_sample; ++k)
            result = evaluator(mixture, batch);
        const auto t1 = clock_type::now();
        sum += checksum(result);
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        times.push_back(elapsed / static_cast<double>(calls_per_sample));
    }

    const double med = median(times);
    const auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());
    std::vector<double> deviations;
    deviations.reserve(times.size());
    for (double t : times) deviations.push_back(std::abs(t - med));
    const double mad = median(deviations);

    benchmark_sink = sum;
    return Timing{
        calls_per_sample,
        *min_it,
        med,
        *max_it,
        mad,
        static_cast<double>(n) / med,
        med * 1.0e9 / static_cast<double>(n),
        med > 0.0 ? 100.0 * mad / med : 0.0,
        sum,
    };
}


#ifdef THERMOGPU_HAS_CUDA
Timing measure_cuda_resident(const thermogpu::Mixture& mixture,
                             const thermogpu::MixtureBatch& batch) {
    thermogpu::CudaBatchWorkspace workspace(mixture, batch);
    workspace.run();
    auto result = workspace.download();
    double sum = checksum(result);
    const std::size_t n = batch.size();

    std::size_t calls = 1;
    for (;;) {
        const auto t0 = clock_type::now();
        for (std::size_t k=0;k<calls;++k) workspace.run();
        const auto t1 = clock_type::now();
        const double elapsed=std::chrono::duration<double>(t1-t0).count();
        if (elapsed >= calibration_floor_seconds) {
            calls=std::max<std::size_t>(1,static_cast<std::size_t>(std::ceil(
                std::clamp(static_cast<double>(calls)*target_sample_seconds/elapsed,1.0,
                           static_cast<double>(std::numeric_limits<std::size_t>::max()/2)))));
            break;
        }
        if(calls>std::numeric_limits<std::size_t>::max()/10) break;
        calls*=10;
    }

    std::vector<double> times; times.reserve(sample_count);
    for(int sample=0;sample<sample_count;++sample){
        const auto t0=clock_type::now();
        for(std::size_t k=0;k<calls;++k) workspace.run();
        const auto t1=clock_type::now();
        times.push_back(std::chrono::duration<double>(t1-t0).count()/static_cast<double>(calls));
    }
    result=workspace.download(); sum+=checksum(result);
    const double med=median(times); const auto [min_it,max_it]=std::minmax_element(times.begin(),times.end());
    std::vector<double> deviations; for(double t:times) deviations.push_back(std::abs(t-med));
    const double mad=median(deviations); benchmark_sink=sum;
    return Timing{calls,*min_it,med,*max_it,mad,static_cast<double>(n)/med,
                  med*1.0e9/static_cast<double>(n),med>0.0?100.0*mad/med:0.0,sum};
}
#endif

struct Observation {
    std::size_t states{};
    double seconds_per_call{};
    double mad_percent{};
};

struct LinearFit {
    double intercept_seconds{};
    double slope_seconds_per_state{};
    double r_squared{};
    double rms_residual_seconds{};
    std::size_t points{};
};

LinearFit fit_linear(const std::vector<Observation>& observations) {
    std::vector<Observation> stable;
    for (const auto& o : observations)
        if (o.mad_percent <= unstable_mad_percent) stable.push_back(o);
    if (stable.size() < 2) throw std::runtime_error("not enough stable points for performance model");

    double mean_x = 0.0, mean_y = 0.0;
    for (const auto& o : stable) {
        mean_x += static_cast<double>(o.states);
        mean_y += o.seconds_per_call;
    }
    mean_x /= static_cast<double>(stable.size());
    mean_y /= static_cast<double>(stable.size());

    double sxx = 0.0, sxy = 0.0;
    for (const auto& o : stable) {
        const double dx = static_cast<double>(o.states) - mean_x;
        sxx += dx * dx;
        sxy += dx * (o.seconds_per_call - mean_y);
    }
    if (sxx == 0.0) throw std::runtime_error("degenerate performance-model grid");

    const double slope = sxy / sxx;
    const double intercept = mean_y - slope * mean_x;
    double ss_res = 0.0, ss_tot = 0.0;
    for (const auto& o : stable) {
        const double predicted = intercept + slope * static_cast<double>(o.states);
        const double residual = o.seconds_per_call - predicted;
        ss_res += residual * residual;
        const double dy = o.seconds_per_call - mean_y;
        ss_tot += dy * dy;
    }
    const double r2 = ss_tot > 0.0 ? 1.0 - ss_res / ss_tot : 1.0;
    const double rms = std::sqrt(ss_res / static_cast<double>(stable.size()));
    return {intercept, slope, r2, rms, stable.size()};
}

double crossover_states(const LinearFit& a, const LinearFit& b) {
    const double denominator = a.slope_seconds_per_state - b.slope_seconds_per_state;
    if (std::abs(denominator) < std::numeric_limits<double>::epsilon())
        return std::numeric_limits<double>::quiet_NaN();
    return (b.intercept_seconds - a.intercept_seconds) / denominator;
}

void print_performance_model(const std::map<int, std::vector<Observation>>& observations) {
    const auto scalar_it = observations.find(0);
    if (scalar_it == observations.end()) return;
    const LinearFit scalar = fit_linear(scalar_it->second);

    std::cout << "Performance model: T(N) = a + b*N, fitted to stable crossover medians\n"
              << "Backend    Threads   Points       a (us)    b (ns/state)       R^2   RMS resid (us)  Crossover vs scalar\n";

    auto print_fit = [&](const std::string& backend, int threads, const LinearFit& fit, double crossover) {
        std::cout << std::left << std::setw(10) << backend
                  << std::right << std::setw(8) << threads
                  << std::setw(9) << fit.points
                  << std::setw(13) << std::fixed << std::setprecision(3) << fit.intercept_seconds * 1.0e6
                  << std::setw(16) << std::setprecision(3) << fit.slope_seconds_per_state * 1.0e9
                  << std::setw(10) << std::setprecision(5) << fit.r_squared
                  << std::setw(17) << std::setprecision(3) << fit.rms_residual_seconds * 1.0e6;
        if (std::isfinite(crossover) && crossover >= 0.0)
            std::cout << std::setw(20) << std::setprecision(2) << crossover;
        else
            std::cout << std::setw(20) << "n/a";
        std::cout << '\n';
    };

    print_fit("scalar", 1, scalar, std::numeric_limits<double>::quiet_NaN());
    for (const auto& [key, rows] : observations) {
        if (key == 0) continue;
        const LinearFit fit = fit_linear(rows);
        print_fit("OpenMP", key, fit, crossover_states(scalar, fit));
    }
    std::cout << "\nModel note: crossover is the intersection of fitted total-call-time lines; "
                 "raw measured bracketing remains authoritative when the linear fit is poor.\n";
}

void print_row(const std::string& backend, int threads, std::size_t states,
               const Timing& t, double scalar_seconds) {
    const double speedup = scalar_seconds / t.median_seconds_per_call;
    const double min_ns = t.min_seconds_per_call * 1.0e9 / static_cast<double>(states);
    const double max_ns = t.max_seconds_per_call * 1.0e9 / static_cast<double>(states);
    std::cout << std::left << std::setw(10) << backend
              << std::right << std::setw(8) << threads
              << std::setw(12) << states
              << std::setw(10) << t.calls_per_sample
              << std::setw(13) << std::fixed << std::setprecision(1) << min_ns
              << std::setw(13) << t.ns_per_state
              << std::setw(13) << max_ns
              << std::setw(10) << std::setprecision(2) << t.relative_mad_percent
              << std::setw(16) << std::scientific << std::setprecision(3) << t.eos_per_second
              << std::setw(12) << std::fixed << std::setprecision(3) << speedup
              << std::setw(8) << (t.relative_mad_percent > unstable_mad_percent ? "UNSTABLE" : "")
              << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::size_t> sizes{1, 10, 100, 1000, 10000, 100000, 1000000};
        if (argc > 1) {
            sizes.clear();
            if (std::string(argv[1]) == "--crossover") {
                if (argc != 2) throw std::invalid_argument("--crossover takes no batch-size arguments");
                sizes = {10, 15, 20, 25, 30, 40, 50, 60, 75, 100};
            } else if (std::string(argv[1]) == "--gpu-crossover") {
                if (argc != 2) throw std::invalid_argument("--gpu-crossover takes no batch-size arguments");
#ifndef THERMOGPU_HAS_CUDA
                throw std::invalid_argument("--gpu-crossover requires THERMOGPU_ENABLE_CUDA=ON");
#else
                sizes = {100, 150, 200, 250, 300, 400, 500, 750, 1000,
                         1500, 2000, 3000, 4000, 5000, 7500, 10000};
#endif
            } else if (std::string(argv[1]) == "--gpu-e2e-crossover") {
                if (argc != 2) throw std::invalid_argument("--gpu-e2e-crossover takes no batch-size arguments");
#ifndef THERMOGPU_HAS_CUDA
                throw std::invalid_argument("--gpu-e2e-crossover requires THERMOGPU_ENABLE_CUDA=ON");
#else
                sizes = {3200, 3400, 3600, 3800, 4000, 4200, 4400, 4600, 4800};
#endif
            } else {
                for (int i = 1; i < argc; ++i) {
                    const auto value = std::stoull(argv[i]);
                    if (value == 0) throw std::invalid_argument("batch size must be positive");
                    sizes.push_back(static_cast<std::size_t>(value));
                }
            }
        }

        const auto mixture = binary_mixture();
#ifdef THERMOGPU_HAS_OPENMP
        const int startup_max_threads = omp_get_max_threads();
        omp_set_dynamic(0);
#endif
        std::cout << "ThermoGPU PR batch benchmark\n"
                  << "Mixture: CH4/C2H6, deterministic varying P/T/z states\n"
                  << "Timing: calibrated to ~" << static_cast<int>(target_sample_seconds * 1000.0)
                  << " ms/sample; median of " << sample_count
                  << " samples after warm-up; validation excluded\n"
                  << "Spread: min/median/max ns/state; MAD% = median absolute deviation / median\n";
#ifdef THERMOGPU_HAS_OPENMP
        std::cout << "OpenMP max threads at startup: " << startup_max_threads << "\n";
#else
        std::cout << "OpenMP: disabled\n";
#endif
#ifdef THERMOGPU_HAS_CUDA
        std::cout << "CUDA: enabled (resident-kernel and end-to-end rows)\n";
#else
        std::cout << "CUDA: disabled\n";
#endif
        std::cout << '\n' 
                  << std::left << std::setw(10) << "Backend"
                  << std::right << std::setw(8) << "Threads"
                  << std::setw(12) << "States"
                  << std::setw(10) << "Calls"
                  << std::setw(13) << "min ns/st"
                  << std::setw(13) << "med ns/st"
                  << std::setw(13) << "max ns/st"
                  << std::setw(10) << "MAD%"
                  << std::setw(16) << "EOS/s"
                  << std::setw(12) << "Speedup"
                  << std::setw(8) << "Status"
                  << '\n';

        std::map<int, std::vector<Observation>> model_observations;
#ifdef THERMOGPU_HAS_CUDA
        struct GpuCrossoverRow {
            std::size_t states{};
            std::string cpu_backend;
            double cpu_seconds{};
            Timing cuda_resident{};
            Timing cuda_e2e{};
        };
        std::vector<GpuCrossoverRow> gpu_crossover_rows;
#endif

        for (const std::size_t n : sizes) {
            const auto batch = make_batch(n);
            const Timing scalar = measure(mixture, batch, thermogpu::evaluate_mixture_batch_scalar);
            print_row("scalar", 1, n, scalar, scalar.median_seconds_per_call);
            model_observations[0].push_back({n, scalar.median_seconds_per_call, scalar.relative_mad_percent});
            double best_cpu_seconds = scalar.median_seconds_per_call;
            std::string best_cpu_backend = "scalar";

#ifdef THERMOGPU_HAS_OPENMP
            const std::vector<int> thread_counts{1, 2, 4, 8};
            for (int threads : thread_counts) {
                if (threads > startup_max_threads) continue;
                omp_set_num_threads(threads);
                const Timing parallel = measure(mixture, batch, thermogpu::evaluate_mixture_batch_openmp);
                print_row("OpenMP", threads, n, parallel, scalar.median_seconds_per_call);
                model_observations[threads].push_back({n, parallel.median_seconds_per_call, parallel.relative_mad_percent});
                if (parallel.relative_mad_percent <= unstable_mad_percent &&
                    parallel.median_seconds_per_call < best_cpu_seconds) {
                    best_cpu_seconds = parallel.median_seconds_per_call;
                    best_cpu_backend = "OMP" + std::to_string(threads);
                }
            }
#endif
#ifdef THERMOGPU_HAS_CUDA
            const Timing cuda_resident = measure_cuda_resident(mixture, batch);
            print_row("CUDA-res", 0, n, cuda_resident, scalar.median_seconds_per_call);
            const Timing cuda_e2e = measure(mixture, batch, thermogpu::evaluate_mixture_batch_cuda);
            print_row("CUDA-e2e", 0, n, cuda_e2e, scalar.median_seconds_per_call);
            if (argc == 2 && (std::string(argv[1]) == "--gpu-crossover" ||
                              std::string(argv[1]) == "--gpu-e2e-crossover"))
                gpu_crossover_rows.push_back({n, best_cpu_backend, best_cpu_seconds, cuda_resident, cuda_e2e});
#endif
            std::cout << '\n';
        }

        if (argc == 2 && std::string(argv[1]) == "--crossover") {
            print_performance_model(model_observations);
        }
#ifdef THERMOGPU_HAS_CUDA
        if (argc == 2 && (std::string(argv[1]) == "--gpu-crossover" ||
                          std::string(argv[1]) == "--gpu-e2e-crossover")) {
            std::cout << "GPU crossover summary: CUDA speedup relative to fastest stable measured CPU backend\n"
                      << std::left << std::setw(10) << "States"
                      << std::setw(12) << "CPU-best"
                      << std::right << std::setw(14) << "CPU ns/st"
                      << std::setw(14) << "CUDA-res"
                      << std::setw(12) << "res/CPU"
                      << std::setw(14) << "CUDA-e2e"
                      << std::setw(12) << "e2e/CPU" << '\n';
            for (const auto& row : gpu_crossover_rows) {
                const double cpu_ns = row.cpu_seconds * 1.0e9 / static_cast<double>(row.states);
                std::cout << std::left << std::setw(10) << row.states
                          << std::setw(12) << row.cpu_backend
                          << std::right << std::setw(14) << std::fixed << std::setprecision(1) << cpu_ns
                          << std::setw(14) << row.cuda_resident.ns_per_state
                          << std::setw(12) << std::setprecision(3)
                          << row.cpu_seconds / row.cuda_resident.median_seconds_per_call
                          << std::setw(14) << std::setprecision(1) << row.cuda_e2e.ns_per_state
                          << std::setw(12) << std::setprecision(3)
                          << row.cpu_seconds / row.cuda_e2e.median_seconds_per_call << '\n';
            }
            std::cout << "\nCrossover note: values > 1 mean CUDA is faster than the measured CPU envelope. "
                         "Bracket crossover from adjacent measured points around 1; no GPU fit is applied.\n";
        }
#endif

        if (!std::isfinite(benchmark_sink)) return EXIT_FAILURE;
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "benchmark error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
