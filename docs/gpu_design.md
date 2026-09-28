# GPU design

## M5 baseline

The V1 baseline maps one thermodynamic state to one CUDA thread. Each thread performs the complete Peng--Robinson state calculation: pure-component temperature corrections, mixture rules, cubic solution and largest-root selection, density, and all component fugacity coefficients.

This is intentionally a correctness-first port. M5 does not use shared-memory tiling, component-level cooperative kernels, constant-memory tuning, kernel fusion, or other GPU-specific optimization. Those changes must be measured against this baseline after CPU/GPU differential validation.

Host-side batch and mixture validation is reused before launch. Component definitions and the dense `kij` matrix are invariant across the batch; pressure, temperature, and state-major composition vary by state. Results retain the same `MixtureBatchResult` contract as the scalar and OpenMP backends.

The baseline uses fixed-size thread-local component work arrays and therefore supports at most 32 components per mixture. The host API rejects larger mixtures rather than silently truncating them. Removing or changing this ceiling is a post-baseline design decision.

M5 includes a deterministic 1,000-state CUDA-versus-scalar smoke campaign. M6 remains responsible for the comprehensive CPU/GPU numerical validation campaign and investigation of any floating-point differences.
