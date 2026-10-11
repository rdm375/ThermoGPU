#pragma once

#include "thermogpu/thermo_batch.hpp"
#include "thermomodel/peng_robinson.hpp"

#include <array>
#include <cstddef>
#include <memory>

namespace thermogpu::thermo {

// Pure-fluid Peng-Robinson with polynomial ideal-gas heat capacity.
//
// cp(T) = sum_{k=0}^4 a[k] * (T / caloric_scale)^k
//
// All thermodynamic properties use mass-specific SI units.
struct CUDAPRParameters {
    thermomodel::PRParameters pr{};

    std::array<double, 5> caloric_coefficients{
        2200.0, 0.0, 0.0, 0.0, 0.0
    };

    double caloric_scale = 1000.0;
};

// Owns persistent CUDA buffers. Not thread-safe; operations are synchronous.
// upload() invalidates the previous result; evaluate() must precede download().
class CUDAThermoWorkspace {
public:
    CUDAThermoWorkspace(const CUDAPRParameters& parameters,
                        std::size_t capacity);
    ~CUDAThermoWorkspace();
    CUDAThermoWorkspace(const CUDAThermoWorkspace&) = delete;
    CUDAThermoWorkspace& operator=(const CUDAThermoWorkspace&) = delete;
    CUDAThermoWorkspace(CUDAThermoWorkspace&&) noexcept;
    CUDAThermoWorkspace& operator=(CUDAThermoWorkspace&&) noexcept;

    void upload(const BatchInput& input);
    void evaluate();
    [[nodiscard]] BatchResult download() const;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

BatchResult evaluate_batch_cuda(
    const CUDAPRParameters& parameters,
    const BatchInput& input
);

} // namespace thermogpu::thermo
