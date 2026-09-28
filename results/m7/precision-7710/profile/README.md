# M7 profiling notes

For static CUDA kernel resource statistics, configure with the normal M7 CUDA toolchain plus:

```bash
-DTHERMOGPU_PTXAS_VERBOSE=ON
```

Example:

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13 \
  -DTHERMOGPU_ENABLE_OPENMP=ON \
  -DTHERMOGPU_ENABLE_CUDA=ON \
  -DTHERMOGPU_PTXAS_VERBOSE=ON

cmake --build build --clean-first --verbose \
  > results/m7/precision-7710/profile/build_ptxas.txt 2>&1

grep -Ei 'ptxas info|register|spill|stack frame|cmem|gmem' \
  results/m7/precision-7710/profile/build_ptxas.txt
```

The option is OFF by default and changes only assembler reporting, not the numerical implementation or optimization level.
