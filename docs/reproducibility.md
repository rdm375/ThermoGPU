# Reproducibility

ThermoGPU separates the production C++/CUDA build from the Python environment used for validation and performance analysis.

## Production build

### Requirements

The CPU implementation requires:

- CMake 3.20 or newer
- a C++20 compiler
- OpenMP when `THERMOGPU_ENABLE_OPENMP=ON`

CUDA support additionally requires:

- the NVIDIA CUDA toolkit
- a CUDA-capable NVIDIA GPU
- a host compiler supported by the installed CUDA toolkit

Python is not required to build or use the ThermoGPU C++/CUDA code.

### CPU build

OpenMP is enabled by default and CUDA is disabled by default.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build -j

ctest --test-dir build --output-on-failure
```

To explicitly disable OpenMP:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHERMOGPU_ENABLE_OPENMP=OFF
```

### CUDA build

Enable CUDA explicitly and select the architecture for the target GPU:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTHERMOGPU_ENABLE_OPENMP=ON \
  -DTHERMOGPU_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=<architecture>

cmake --build build -j

ctest --test-dir build --output-on-failure
```

The two M7 systems used different CUDA architectures:

- Quadro M3000M (Maxwell): `52`
- RTX 4090 Laptop GPU (Ada): `89`

ThermoGPU intentionally does not hard-code a CUDA architecture in `CMakeLists.txt`. The architecture should describe the GPU on which the build will run.

If the CUDA toolkit requires a particular host compiler, it can be selected at configure time. For example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DTHERMOGPU_ENABLE_OPENMP=ON \
  -DTHERMOGPU_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89
```

### PTXAS resource reporting

CUDA compiler resource information can be enabled with:

```text
-DTHERMOGPU_PTXAS_VERBOSE=ON
```

This passes `-Xptxas=-v` to CUDA compilation. It is intended for diagnostics and profiling rather than as a requirement for a normal build.

## Python validation and analysis environment

Python tooling is separate from the production build. It is used for reference calculations, independent external validation, and analysis of benchmark data.

Create an isolated environment:

```bash
python3 -m venv .venv-validation
source .venv-validation/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-analysis.txt
```

The non-standard Python dependencies are:

- NumPy
- Matplotlib
- ThermoPack

The roles of the Python tools are intentionally distinct.

### `tools/pr_reference.py`

This is ThermoGPU's Python cross-implementation reference. It is useful for checking formulas and generating frozen reference data, but it is not treated as an independent EOS implementation.

### `tools/thermopack_validation.py`

This uses ThermoPack as an independent external implementation for controlled Peng-Robinson comparisons.

The controlled validation setup and assumptions are documented in `docs/verification.md` and `docs/m6_validation.md`.

### `tools/analyze_m7.py`

This analyzes M7 scaling measurements and produces derived crossover summaries and plots. NumPy and Matplotlib are analysis dependencies; they are not runtime dependencies of ThermoGPU itself.

## Test suite

After configuring and building, run:

```bash
ctest --test-dir build --output-on-failure
```

The exact set of tests depends on the enabled backends. A CUDA-enabled V1 build exercises the scalar, OpenMP, CUDA, external-reference, controlled-ThermoPack, and frozen-oracle validation layers.

A successful test run establishes agreement with the repository's defined test criteria. It does not by itself establish performance portability; performance evidence is recorded separately.

## Performance reproducibility

Performance results are machine-specific. ThermoGPU therefore keeps the numerical implementation and validation requirements separate from performance crossover measurements.

Do not assume that a crossover measured on one CPU/GPU pair is a universal dispatch threshold.

M7 distinguishes two CUDA timing models:

- **resident**: state and result data are already resident on the GPU
- **end-to-end (E2E)**: host/device transfer costs are included

These answer different application-level questions and should not be interchanged.

Benchmark methodology is documented in:

- `docs/benchmarking.md`
- `docs/m7_performance.md`

Raw and derived M7 evidence is retained under:

- `results/m7/precision-7710/`
- `results/m7/precision-7680/`

The cross-machine synthesis is:

- `results/m7/CROSS_MACHINE_ANALYSIS.md`

## Validated M7 environments

The V1 implementation and numerical validation were exercised unchanged on two substantially different systems.

### Dell Precision 7710

- native Linux
- Intel Core i7-6920HQ
- NVIDIA Quadro M3000M
- Maxwell, compute capability 5.2
- CUDA architecture `52`

This machine supplied the primary controlled M7 CPU/GPU crossover and performance characterization.

### Dell Precision 7680

- Windows 11 / WSL2
- Intel Core i9-13950HX
- NVIDIA RTX 4090 Laptop GPU
- Ada, compute capability 8.9
- CUDA architecture `89`

The V1 numerical tests passed unchanged on this system. CUDA resident large-batch performance was reproducible, while precise CPU/GPU E2E crossover characterization was limited by variation in the WSL2-visible CPU baseline and the flattened presentation of the hybrid CPU topology.

The 7680 results are therefore useful evidence of numerical and CUDA portability, but its measured E2E crossover values should not be promoted as universal dispatch thresholds.

## Profiling reproducibility

Profiling tools and hardware-counter capabilities vary by GPU generation and operating environment.

On the Precision 7710, Nsight Systems profiling was available, while Nsight Compute hardware-counter profiling was not supported for the Maxwell GPU used in the study.

On the Precision 7680, Nsight Compute hardware counters were available after GPU performance-counter access was enabled on the Windows host.

Profiling evidence is retained with the corresponding machine results rather than being required for a normal ThermoGPU build.

## Numerical reproducibility

ThermoGPU treats unexplained numerical differences as defects to investigate rather than as reasons to widen tolerances.

Important numerical decisions, including the stable one-real-root Cardano formulation used after the M6 CPU/CUDA differential investigation, are documented in:

- `docs/theory.md`
- `docs/verification.md`
- `docs/m6_validation.md`

The project uses:

```text
R = 8.31446261815324 J mol^-1 K^-1
```

Changes to numerical algorithms, constants, compiler behavior, or backend implementations should therefore be validated causally before changing test tolerances.

## Reproducing historical results

The repository retains benchmark, validation, environment, and profiling evidence from the development milestones. These files document particular machines and toolchains; they are not expected to reproduce identical timings on different hardware.

For a new machine, the appropriate workflow is:

1. configure a Release build for the available backends;
2. build ThermoGPU;
3. run the complete applicable CTest suite;
4. record the machine and toolchain environment;
5. run the benchmark/scaling tools;
6. analyze crossover behavior for that machine rather than importing a threshold from another system;
7. retain enough raw evidence to distinguish measured results from derived summaries.

This preserves the distinction between numerical reproducibility, cross-platform portability, and machine-specific performance.
