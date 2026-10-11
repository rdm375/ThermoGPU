# G4.6c: CUDA Thermodynamic Batch Profiling and Output Reuse

## Platform

- System: Dell Precision 7710
- GPU: NVIDIA Quadro M3000M (SM 5.2)
- CPU: Intel Core i7-6920HQ
- CUDA toolkit: 12.4
- Host compiler: GCC 13.4
- OpenMP threads: 4

## G4.6c: Profiling

Profiling separates CUDA thermodynamic batch execution into:

- Input packing
- Host-to-device transfer
- GPU kernel execution
- Device-to-host transfer
- Result initialization
- Error translation
- Numerical result conversion

Results: `results-thermo-batch/g46c-sm52/profile.csv`

At one million states, output conversion and result initialization
account for substantial host-side overhead.

## G4.6c.1: Reusable Output Storage

The existing CUDAThermoWorkspace::download_into() API allows
callers to reuse BatchResult storage.

Measured end-to-end performance at one million states:

| Workload | Fresh (ms) | Reused (ms) | Speedup |
|---|---:|---:|---:|
| Uniform | 130.1386 | 80.3532 | 1.620x |
| Smooth | 118.6340 | 74.9792 | 1.582x |
| Diverse | 127.0018 | 80.0220 | 1.587x |

Results: `results-thermo-batch/g46c1-sm52/reuse.csv`

The optimization reduces execution time by approximately 37%
for one-million-state workloads without changing the numerical
algorithm or production CUDA implementation.

## Validation

The CUDA integration test validates fresh versus reused output
storage, repeated downloads, changing batch sizes, and transitions
between successful and failed evaluations.

Results:

- CUDA BatchResult reuse parity: PASS
- CPU/CUDA numerical comparisons: 409632
- Status mismatches: 0
- Message mismatches: 0
- Phase mismatches: 0
- Tolerance violations: 0
- Complete CTest suite: 24/24 PASS

## Conclusions

Reusable output storage substantially reduces host-side overhead
for large CUDA thermodynamic batches.

Future investigations:

- G4.6c.2: Reduce remaining result initialization and conversion costs.
- G4.6b: Establish CPU/GPU crossover points.
- G4.6d: Evaluate repeated GPU-resident execution.
