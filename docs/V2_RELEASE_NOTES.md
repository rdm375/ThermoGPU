# ThermoGPU 2.0.0 Release Notes

## Overview

ThermoGPU 2.0 extends the original Peng–Robinson equation-of-state
research implementation into an installable thermodynamic evaluation
library with CPU, OpenMP, and CUDA execution.

The release introduces a common thermodynamic model interface,
expanded thermodynamic properties, batch evaluation, and reusable
CUDA execution infrastructure.

ThermoGPU 1.0 remains available as the original validated baseline.

## Major changes

### ThermoModel integration

ThermoGPU 2.0 incorporates the ThermoModel thermodynamic foundation.

The new API provides:

- Peng–Robinson equation-of-state evaluation;
- compressibility factor and mass density;
- enthalpy, entropy, and internal energy;
- isobaric and isochoric heat capacities;
- speed of sound;
- thermodynamic derivatives; and
- phase and error metadata.

The thermodynamic API is exposed through:

```cpp
#include <thermogpu/thermo.hpp>
```

### Batch execution

The new thermodynamic batch interface supports scalar CPU and
optional OpenMP execution.

```cpp
#include <thermogpu/thermo_batch.hpp>
```

Batch results include the expanded thermodynamic properties and
per-state error information.

### CUDA execution

The CUDA backend provides Peng–Robinson thermodynamic batch
evaluation with reusable device workspaces.

The implementation supports:

- persistent device allocations;
- explicit input upload and result download;
- GPU-resident evaluation;
- caller-managed CUDA streams for device-resident execution; and
- profiling interfaces for execution and transfer analysis.

The CUDA interface is available through:

```cpp
#include <thermogpu/thermo_batch_cuda.hpp>
```

CUDA execution requires a CUDA-enabled build and a compatible
NVIDIA GPU and CUDA Toolkit.

### Installable CMake package

ThermoGPU 2.0 introduces installation and exported CMake targets.

External applications can use:

```cmake
find_package(ThermoGPU 2.0 REQUIRED CONFIG)

target_link_libraries(my_application PRIVATE
    thermogpu::thermo_batch
)
```

Available targets include:

- `thermomodel::thermomodel`
- `thermogpu::thermo`
- `thermogpu::thermo_batch`
- `thermogpu::scalar`

CUDA-enabled installations additionally provide:

- `thermogpu::thermo_cuda`
- `thermogpu::cuda`

Installation instructions are provided in the project README.

## Validation

The release-packaging qualification completed successfully
on the Dell Precision 7710.

| Configuration | Regression tests |
|---|---:|
| CPU-only | 12/12 |
| CPU + OpenMP | 14/14 |
| CPU + OpenMP + CUDA | 24/24 |

All three configurations also passed independent CMake consumer
tests using installed packages relocated outside the source tree.

The functional consumer tests evaluated Peng–Robinson states
and verified agreement between scalar and batch results for
compressibility factor and density.

The CUDA configuration used:

- NVIDIA Quadro M3000M (`sm_52`);
- CUDA Toolkit 12.4; and
- GCC 13 as the CUDA host compiler.

The CUDA regression suite exercises CUDA functionality separately
from the external consumer tests.

## Numerical policy

ThermoGPU retains its established numerical-validation approach:

- independent and cross-implementation reference comparisons;
- deterministic regression tests;
- CPU/CUDA differential validation;
- explicit investigation of floating-point discrepancies; and
- no unexplained relaxation of numerical tolerances.

Historical ThermoGPU 1.0 validation and benchmark results remain
documented separately.

## Compatibility and limitations

- The project requires C++20 and CMake 3.20 or newer.
- OpenMP is optional.
- CUDA is optional and requires compatible hardware and tooling.
- CUDA-specific headers are installed only for CUDA-enabled builds.
- The new CUDA thermodynamic batch implementation targets the
  supported Peng–Robinson model and its documented caloric form.
- The new thermodynamic batch interface should not be interpreted
  as a general-purpose multiphase flash solver.
- GPU performance depends on hardware, batch size, data movement,
  and execution mode.
- Published benchmark measurements are specific to their tested
  hardware and workloads.

## Release documentation

- [README](../README.md)
- [ThermoGPU 1.0 final report](V1_REPORT.md)
- [GPU specialization closeout](G4.5c.2c.9-closeout.md)

## License

ThermoGPU is licensed under GPL-3.0-or-later.
