# ThermoGPU

ThermoGPU is an experimental scientific-computing project investigating whether GPUs can materially accelerate large batches of engineering equation-of-state calculations without sacrificing numerical quality.

V1.0 targets single-phase multicomponent Peng–Robinson calculations and compares scalar C++, OpenMP, and CUDA backends. The repository is designed to produce a reproducible computational result, not merely a CUDA demonstration.

## Current status

**Current milestone: M5 CUDA characterization complete, pending final acceptance run/tag.** Scalar, OpenMP, and baseline CUDA Peng–Robinson batch backends are implemented and differentially tested.

Implemented now:
- C++20/CMake project
- Peng–Robinson `a`, `b`, alpha and kappa relations
- analytic cubic solution for compressibility factor `Z`
- explicit one-real/three-real root classification
- pure-component density and fugacity coefficient
- multicomponent quadratic attractive mixing rule
- explicit dense symmetric binary-interaction (`kij`) matrix
- composition validation with no silent normalization
- mixture molecular weight, density, and component fugacity coefficients
- pure-as-mixture equivalence, limiting, mixture, `kij`, and invalid-input tests
- initial five-component property table
- optional CUDA batch backend with resident-workspace and end-to-end benchmark paths
- M3 structure-of-arrays batch input: pressure, temperature, and state-major flattened composition
- scalar batch evaluator returning Z, density, residual, root metadata, and state-major component ln(phi)
- batch-vs-single-state differential tests across varying P, T, and composition
- explicit empty-batch behavior and batch layout/composition validation

M4 OpenMP batched execution is complete. M5 adds the CUDA batch backend, CUDA-vs-scalar differential testing, and resident versus end-to-end CUDA characterization. `thermogpu_benchmark` measures scalar, OpenMP, and (when enabled) CUDA throughput without mixing correctness checks into the timed region.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

CUDA remains optional. For the tested CUDA 12.4 environment, use GCC 13 consistently for both C++ and the nvcc host compiler; CUDA 12.4 does not support GCC 15 as a host compiler:

```bash
rm -rf build
cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
    -DTHERMOGPU_ENABLE_OPENMP=ON \
    -DTHERMOGPU_ENABLE_CUDA=ON
```

See `docs/benchmarking.md` for the toolchain diagnosis and M5 CUDA characterization. On the characterized Dell Precision 7710 / i7-6920HQ / Quadro M3000M, resident CUDA crossed the measured CPU envelope between 250–300 states (~275 interpolated), while end-to-end CUDA crossed between 3800–4000 states (~3900; 3936 by local interpolation). These are machine- and workload-specific measurements, not universal dispatch thresholds.

## Mixture API policy

`Mixture` stores components, mole fractions, and an explicit dense row-major `n x n` binary-interaction matrix. Mole fractions must sum to one within the documented validation tolerance. The scalar API does **not** silently normalize compositions. The `kij` matrix must be finite and symmetric.

For the current declared single-phase calculation, the evaluator selects the largest real PR root. If the cubic has three real roots, `root_selection_ambiguous` is set so downstream verification can classify that state rather than silently treating the selection as unambiguous phase determination.

## M3 scalar batch API

`MixtureBatch` is a structure-of-arrays workload representation intended to remain stable as OpenMP and CUDA backends are added. For `S` states and `N` components it stores `pressure_Pa[S]`, `temperature_K[S]`, and a flattened state-major `mole_fractions[S*N]`, indexed as `state*N + component`.

`evaluate_mixture_batch_scalar()` evaluates each state with the validated scalar Peng-Robinson kernel and returns structure-of-arrays outputs for `Z`, density, cubic residual, root classification/ambiguity, and flattened state-major `ln_phi[S*N]`. Each state composition is independently validated and is never silently normalized. An empty batch is valid and returns empty output arrays.

The M3 unit test differentially compares every scalar-batch state against direct `evaluate_mixture()` evaluation. M4 adds `evaluate_mixture_batch_openmp()`, which parallelizes the outer state loop with `#pragma omp parallel for schedule(static)`. `test_pr_batch_openmp` deterministically generates 1,000 binary-mixture states spanning pressure, temperature, and composition and compares every OpenMP result with the scalar-batch reference. Thread count is controlled with the standard `OMP_NUM_THREADS` environment variable.

## M4 CPU benchmark

`thermogpu_benchmark` sweeps deterministic CH4/C2H6 batches through the scalar backend and OpenMP at 1, 2, 4, and 8 threads (when available). It reports nanoseconds per state, EOS evaluations per second, and speedup relative to scalar at the same batch size. The benchmark performs a warm-up, calibrates to approximately 200 ms per sample, and reports the median of nine timed samples; correctness validation is deliberately outside the timed region. See `docs/benchmarking.md` for methodology.

```bash
./build/thermogpu_benchmark
```

Pass explicit batch sizes to shorten or focus a run, for example `./build/thermogpu_benchmark 1000 10000 100000`.

## Scientific policy

The scalar implementation is the readable numerical reference. Optimized CPU and GPU backends will be differentially tested against it. The scalar implementation is validated by invariant tests, frozen cross-implementation regression cases, published-reference checks, and controlled external comparison against ThermoPack 2.2.3. Numerical disagreements are investigated causally rather than hidden by relaxed tolerances.

See `docs/V1_SPEC.md` for the full V1.0 specification.

## License

ThermoGPU is free software licensed under the GNU General Public License, version 3 or (at your option) any later version (`GPL-3.0-or-later`).

See `LICENSE` for the full license text.

## Inspect a scalar EOS calculation

M2 adds `thermogpu_eval`, a small diagnostic executable for inspecting the scalar
Peng-Robinson reference calculation.  It currently evaluates a built-in
five-component natural-gas-like state at 8 MPa and 320 K with zero binary
interaction coefficients.

```bash
./build/thermogpu_eval
```

The report includes the input composition, mixture molecular weight, `a_m`,
`b_m`, dimensionless `A` and `B`, all real cubic roots, the selected `Z`, cubic
residual, density, and component `ln(phi)` / `phi` values.

The printed values are ThermoGPU results.  They are intended to make the
calculation inspectable; the diagnostic output is covered by the M2 validation framework described below.

## Validation

Run all unit and external-reference tests with:

```bash
ctest --test-dir build --output-on-failure
```

To inspect the third-party Peng-Robinson regression directly:

```bash
./build/test_pr_reference
```

External expected values and provenance are documented in `tests/reference/README.md`.

## Frozen cross-implementation validation

The M2 validation layer includes a separate Python Peng-Robinson reference
implementation with literal frozen inputs and committed CSV outputs. It shares no
C++ implementation code and emits intermediate values so numerical disagreements
can be localized. Because it was written from the same mathematical specification,
this is described as a cross-implementation check rather than an independent
mathematical validation. Controlled ThermoPack 2.2.3 pseudo-component cases provide the strongest external third-party check; the published `thermo` regression remains a literature anchor. See `tests/reference/README.md`.

Run all validation with:

```bash
ctest --test-dir build --output-on-failure
```

For the diagnostic oracle test alone:

```bash
ctest --test-dir build -R pr_frozen_oracle -V
```
