# Precision 7680 CUDA Profiling

This directory contains Nsight Compute profiling evidence collected for
ThermoGPU `v0.7.0` on a Dell Precision 7680 under WSL2.

The profiling experiment compares the same CUDA Peng-Robinson batch kernel
at identical batch size and launch geometry while changing only mixture
component count.

## System

- Dell Precision 7680
- NVIDIA RTX 4090 Laptop GPU
- CUDA compute capability 8.9
- 76 SMs
- CUDA 13.0
- Nsight Compute 2025.3.1
- WSL2 / Ubuntu 24.04
- ThermoGPU `v0.7.0`

Detailed system and toolchain information is recorded in:

    ../environment.txt

## Experiment

Two CUDA-resident workloads were profiled:

    thermogpu_scaling --profile cuda 1 100000
    thermogpu_scaling --profile cuda 5 100000

Both evaluate 100,000 thermodynamic states using the same
`pr_batch_kernel`.

Raw Nsight Compute output:

    ncu_1c_100k.txt
    ncu_5c_100k.txt

The comparison was chosen to investigate why CUDA kernel time increases
with mixture component count.

## Results

Representative measurements are:

| Metric | 1 component | 5 components |
|---|---:|---:|
| Kernel duration | ~269 us | ~942 us |
| Relative kernel duration | 1.00x | ~3.50x |
| Compute (SM) throughput | ~80.6% | ~80.2-80.7% |
| DRAM throughput | ~1.9-2.0% | ~1.3% |
| Registers per thread | 56 | 56 |
| Theoretical occupancy | 75% | 75% |
| Achieved occupancy | ~55.8% | ~56.1% |
| Waves per SM | 1.14 | 1.14 |

The measurements were highly repeatable across the profiled launches.
The 1-component kernel remained near 269 us and the 5-component kernel
remained near 942 us.

## Interpretation

Increasing the mixture from one to five components increases CUDA kernel
duration by approximately 3.5x.

The increase is not accompanied by a change in the kernel's static register
requirement, theoretical occupancy, launch geometry, or meaningful loss of
achieved occupancy. Compute throughput remains near 80% in both cases.

DRAM throughput is very low in both cases and decreases rather than
increases for the five-component workload. The observed component-count
scaling therefore does not indicate a DRAM-bandwidth bottleneck.

The evidence supports the interpretation that the increased kernel duration
is primarily caused by additional per-state computational work as component
count increases. Peng-Robinson mixture evaluation contains component-dependent
work including a double sum in the mixture attraction calculation and
additional component-dependent fugacity calculations.

In particular, the experiment does not support explanations based on:

- increasing register pressure with component count;
- occupancy collapse;
- changing launch geometry;
- DRAM bandwidth saturation; or
- failure to keep the GPU computational units active.

The kernel is compute-heavy and register-constrained in absolute terms:
Nsight Compute reports 56 registers per thread, 75% theoretical occupancy,
and approximately 56% achieved occupancy. However, those quantities are
essentially unchanged between the one- and five-component experiments and
therefore do not explain the approximately 3.5x component-scaling penalty.

## Launch-tail observation

At 100,000 states the launch consists of 782 blocks of 128 threads and
produces approximately 1.14 waves per SM. Nsight Compute identifies a
partial-wave tail and reports it as a possible optimization opportunity.

This may affect absolute kernel performance at this batch size, but it
does not explain the one-component versus five-component scaling observed
here because both experiments use the same batch size and launch geometry.

Nsight Compute optimization estimates are treated as diagnostic guidance,
not as measured achievable speedups.

## Relation to Precision 7710 results

Earlier profiling on the Precision 7710 showed that the five-component
kernel took approximately 3.4x as long as the one-component kernel, but
hardware performance counters were unavailable on the Maxwell M3000M.

The approximately 3.5x ratio observed here on Ada is consistent with that
earlier timing result. The Ada hardware counters additionally show that the
component-count increase does not cause a memory-bandwidth bottleneck,
register-count increase, or occupancy collapse.

Because the two systems differ in GPU architecture, CUDA toolkit, operating
environment, and other hardware, this is not a controlled hardware-only
comparison. The agreement nevertheless provides useful cross-platform
evidence that the component-scaling behavior is associated with the
Peng-Robinson workload/kernel structure rather than being unique to the
older Maxwell GPU.

## Scope

These profiles are diagnostic measurements for the M7 performance study.
They are not intended to establish universal performance characteristics
or universal CPU/GPU crossover thresholds.

Machine-specific crossover and throughput measurements are documented in
the surrounding Precision 7680 and Precision 7710 M7 result sets.
