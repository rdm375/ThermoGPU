#pragma once

#include "thermogpu/thermo_batch.hpp"
#include "thermomodel/peng_robinson.hpp"
#include "thermomodel/detail/pr_tp_state_kernel.hpp"

#include <cuda_runtime_api.h>

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

// Device ABI: input and output pointers must refer to CUDA-accessible memory.
// The result contains the complete PR state, including its derivatives.
struct CUDADeviceInput {
    double T;
    double p;
};

struct CUDADeviceOutput {
    thermomodel::detail::PRTPResult result;
    int error;
};

// Diagnostic compact result. Invalid entries contain NaN for Z and rho.
struct CUDACompactOutput {
    double Z;
    double rho;
    int error;
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

    // Enqueue into the caller's stream without synchronizing. Caller owns
    // buffers and must keep them alive until stream completion. count must
    // not exceed capacity(). Inputs and outputs must not overlap.
    // Error codes match the internal CUDA error mapping; invalid states
    // have no meaningful result fields. Call cudaStreamSynchronize before
    // reading results on the host.
    void evaluate_device(const CUDADeviceInput* device_inputs,
                         CUDADeviceOutput* device_outputs,
                         std::size_t count,
                         cudaStream_t stream = nullptr) const;
    // Diagnostic: wall-clock breakdown of the synchronous workspace path.
    // Times include synchronization and host work; kernel_ms is CUDA-event
    // device elapsed time. Not safe for concurrent use of this workspace.
    struct Profile {
        double pack_ms = 0.0;
        double h2d_ms = 0.0;
        double kernel_wall_ms = 0.0;
        double kernel_event_ms = 0.0;
        double d2h_ms = 0.0;
        double conversion_ms = 0.0;
        double result_init_ms = 0.0;
        double error_translation_ms = 0.0;
        double numeric_phase_ms = 0.0;
        BatchResult result;
    };
    [[nodiscard]] Profile profile(const BatchInput& input);

    // Diagnostic only: requires a completed evaluate().
    // Measures transfers of the same EOS results in two layouts.
    struct CompactProfile {
        std::size_t count = 0;
        std::size_t full_record_bytes = 0;
        std::size_t compact_record_bytes = 0;
        double full_d2h_ms = 0.0;
        double pack_kernel_ms = 0.0;
        double compact_d2h_ms = 0.0;
        double full_extract_ms = 0.0;
        double compact_extract_ms = 0.0;
        std::size_t mismatches = 0;
        double checksum = 0.0;
    };

    [[nodiscard]] CompactProfile profile_compact() const;
    struct FusedProfile {
        std::size_t count = 0;
        std::size_t mismatches = 0;
        double reference_kernel_ms = 0.0;
        double packing_kernel_ms = 0.0;
        double fused_kernel_ms = 0.0;
        double checksum = 0.0;
    };

    // Diagnostic: compares two fully GPU-resident evaluation paths.
    // Requires upload(); evaluates both paths using the same input.
    [[nodiscard]] FusedProfile profile_fused(
        bool fused_first = false) const;

    // Diagnostic only: compare specialized computation with
    // compact and full-sized output storage.
    struct StorageProfile {
        std::size_t count = 0;
        std::size_t mismatches = 0;
        std::size_t full_record_bytes = 0;
        std::size_t compact_record_bytes = 0;
        double full_ms = 0.0;
        double padded_ms = 0.0;
        double compact_ms = 0.0;
        double checksum = 0.0;
    };

    [[nodiscard]] StorageProfile profile_storage(
        bool compact_first = false) const;

    [[nodiscard]] BatchResult download() const;

    // Download into caller-owned storage, reusing vector capacity.
    // Preserves NaN, phase, error, and message semantics.
    void download_into(BatchResult& result) const;
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
