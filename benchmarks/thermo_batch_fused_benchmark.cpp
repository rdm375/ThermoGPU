#include "thermogpu/thermo_batch_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace thermogpu::thermo;

static double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const auto n = values.size();
    return n % 2 ? values[n/2]
                 : 0.5 * (values[n/2-1] + values[n/2]);
}

static BatchInput make_input(
    std::size_t n, const std::string& workload)
{
    BatchInput input;
    input.temperature_K.resize(n);
    input.pressure_Pa.resize(n);

    constexpr double pi = 3.14159265358979323846;

    for (std::size_t i = 0; i < n; ++i) {
        const double x = n > 1
            ? static_cast<double>(i)/static_cast<double>(n-1)
            : 0.5;

        if (workload == "uniform") {
            input.temperature_K[i] = 320.0;
            input.pressure_Pa[i] = 8e6;
        } else if (workload == "smooth") {
            input.temperature_K[i] =
                300.0 + 35.0*x + 4.0*std::sin(2*pi*x);
            input.pressure_Pa[i] =
                4e6 + 5e6*x + 2e5*std::sin(4*pi*x);
        } else if (workload == "invalid") {
            input.temperature_K[i] =
                i % 3 == 0 ? -1.0 : 320.0;
            input.pressure_Pa[i] =
                i % 3 == 1 ? -1.0 : 8e6;
        } else {
            const double a =
                static_cast<double>((i*104729ULL)%1000003ULL)
                / 1000003.0;
            const double b =
                static_cast<double>((i*130363ULL+17ULL)%1000033ULL)
                / 1000033.0;

            input.temperature_K[i] = 220.0 + 600.0*a;
            input.pressure_Pa[i] = 1e5 + 3e7*b;
        }
    }

    return input;
}

int main(int argc, char** argv)
{
    try {


        if (argc == 5 &&
            std::string(argv[1]) == "--storage-workload" &&
            std::string(argv[3]) == "--storage-size") {

            const std::string workload = argv[2];
            const std::size_t n = std::stoull(argv[4]);

            if (workload != "uniform" &&
                workload != "smooth" &&
                workload != "diverse" &&
                workload != "invalid")
                throw std::invalid_argument("Unknown workload");

            if (!n)
                throw std::invalid_argument("Size must be positive");

            CUDAThermoWorkspace ws(CUDAPRParameters{}, n);
            ws.upload(make_input(n, workload));

            const auto warmup = ws.profile_storage();

            if (warmup.mismatches)
                throw std::runtime_error(
                    "Storage warmup parity failure");

            const auto result = ws.profile_storage();

            if (result.mismatches)
                throw std::runtime_error(
                    "Storage parity failure");

            std::cout << std::fixed << std::setprecision(6)
                      << "workload=" << workload
                      << " n=" << n
                      << " full_ms=" << result.full_ms
                      << " padded_ms=" << result.padded_ms
                      << " compact_ms=" << result.compact_ms
                      << " full_bytes=" << result.full_record_bytes
                      << " compact_bytes=" << result.compact_record_bytes
                      << " mismatches=" << result.mismatches
                      << " checksum=" << result.checksum
                      << '\n';

            return 0;
        }

        if (argc == 3 &&
            std::string(argv[1]) == "--storage-csv") {

            const int repeats = std::stoi(argv[2]);

            if (repeats < 1)
                throw std::invalid_argument(
                    "repeats must be positive");

            const std::string path =
                "benchmarks/results-thermo-batch/"
                "7680-storage-control.csv";

            std::ofstream csv(path);

            if (!csv)
                throw std::runtime_error(
                    "Cannot open storage CSV");

            csv << "workload,n,full_ms,padded_ms,compact_ms,"
                   "full_over_padded,padded_over_compact,parity\n";

            csv << std::setprecision(12);

            for (const std::string workload :
                 {"uniform", "smooth", "diverse", "invalid"}) {

                for (std::size_t n :
                     {1000ULL, 10000ULL, 100000ULL, 1000000ULL}) {

                    CUDAThermoWorkspace ws(CUDAPRParameters{}, n);
                    ws.upload(make_input(n, workload));

                    if (ws.profile_storage().mismatches)
                        throw std::runtime_error(
                            "Storage warmup parity failure");

                    std::vector<double> full;
                    std::vector<double> padded;
                    std::vector<double> compact;

                    for (int r = 0; r < repeats; ++r) {
                        const auto p =
                            ws.profile_storage(r % 2 != 0);

                        if (p.mismatches)
                            throw std::runtime_error(
                                "Storage parity failure");

                        full.push_back(p.full_ms);
                        padded.push_back(p.padded_ms);
                        compact.push_back(p.compact_ms);
                    }

                    const double a = median(full);
                    const double b = median(padded);
                    const double c = median(compact);

                    csv << workload << ',' << n << ','
                        << a << ',' << b << ',' << c << ','
                        << a/b << ',' << b/c << ",PASS\n";

                    std::cout << workload
                              << " N=" << n
                              << " full=" << a
                              << " padded=" << b
                              << " compact=" << c
                              << " padded/compact=" << b/c
                              << "x PARITY PASS\n";
                }
            }

            std::cout << "CSV: " << path << '\n';
            return 0;
        }

        if (argc == 5 &&
            std::string(argv[1]) == "--profile-workload" &&
            std::string(argv[3]) == "--profile-size") {

            const std::string workload = argv[2];
            const std::size_t n = std::stoull(argv[4]);

            if (workload != "uniform" &&
                workload != "smooth" &&
                workload != "diverse" &&
                workload != "invalid")
                throw std::invalid_argument("Unknown workload");

            if (n == 0)
                throw std::invalid_argument("Size must be positive");

            CUDAThermoWorkspace ws(CUDAPRParameters{}, n);
            ws.upload(make_input(n, workload));

            // Warm up before the profiled comparison.
            const auto warmup = ws.profile_fused(false);
            if (warmup.mismatches)
                throw std::runtime_error("Warmup parity failure");

            // One comparison: full EOS, packing, fused EOS.
            const auto result = ws.profile_fused(false);

            if (result.mismatches)
                throw std::runtime_error("Profile parity failure");

            std::cout << std::fixed << std::setprecision(6)
                      << "workload=" << workload
                      << " n=" << n
                      << " full_ms=" << result.reference_kernel_ms
                      << " pack_ms=" << result.packing_kernel_ms
                      << " fused_ms=" << result.fused_kernel_ms
                      << " mismatches=" << result.mismatches
                      << " checksum=" << result.checksum
                      << '\n';

            return 0;
        }

        const std::string path =
            argc > 1 ? argv[1] : "fused-output.csv";
        const int repeats =
            argc > 2 ? std::stoi(argv[2]) : 15;

        if (repeats < 1)
            throw std::invalid_argument("repeats must be positive");

        std::ofstream csv(path);
        if (!csv)
            throw std::runtime_error("Cannot open CSV");

        csv << "workload,n,reference_ms,packing_ms,"
               "separate_total_ms,fused_ms,speedup,parity\n";
        csv << std::setprecision(12);

        for (const std::string workload :
             {"uniform", "smooth", "diverse", "invalid"}) {

            for (std::size_t n :
                 {1000ULL, 10000ULL, 100000ULL, 1000000ULL}) {

                CUDAThermoWorkspace ws(CUDAPRParameters{}, n);
                ws.upload(make_input(n, workload));

                auto warmup = ws.profile_fused();
                if (warmup.mismatches)
                    throw std::runtime_error("Warmup parity failure");

                std::vector<double> reference;
                std::vector<double> packing;
                std::vector<double> fused;
                std::vector<double> separate_total;

                for (int r = 0; r < repeats; ++r) {
                    auto p = ws.profile_fused(r % 2 != 0);

                    if (p.mismatches)
                        throw std::runtime_error(
                            "Fused parity failure: " + workload +
                            " n=" + std::to_string(n) +
                            " mismatches=" +
                            std::to_string(p.mismatches));

                    reference.push_back(p.reference_kernel_ms);
                    packing.push_back(p.packing_kernel_ms);
                    fused.push_back(p.fused_kernel_ms);
                    separate_total.push_back(
                        p.reference_kernel_ms + p.packing_kernel_ms);
                }

                const double a = median(reference);
                const double b = median(packing);
                const double c = median(separate_total);
                const double d = median(fused);

                csv << workload << ',' << n << ','
                    << a << ',' << b << ','
                    << c << ',' << d << ','
                    << c/d << ",PASS\n";

                std::cout << std::setw(8) << workload
                          << " N=" << std::setw(8) << n
                          << " reference=" << std::fixed
                          << std::setprecision(4) << a
                          << " pack=" << b
                          << " separate=" << c
                          << " fused=" << d
                          << " speedup=" << c/d
                          << "x PARITY PASS\n";
            }
        }

        std::cout << "CSV: " << path << '\n';
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
