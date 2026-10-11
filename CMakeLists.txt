cmake_minimum_required(VERSION 3.20)
project(ThermoGPU VERSION 1.0.0 LANGUAGES CXX)
option(THERMOGPU_ENABLE_OPENMP "Build OpenMP backend" ON)
option(THERMOGPU_ENABLE_CUDA "Build CUDA backend" OFF)
option(THERMOGPU_PTXAS_VERBOSE "Report CUDA ptxas resource usage during compilation" OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# ThermoModel v2.0 thermodynamic foundation.
# The existing ThermoGPU implementation remains available for comparison.
add_library(thermomodel STATIC
  src/thermomodel/ideal_gas.cpp
  src/thermomodel/caloric.cpp
  src/thermomodel/variable_cp_ideal_gas.cpp
  src/thermomodel/flash.cpp
  src/thermomodel/peng_robinson.cpp
)
add_library(thermomodel::thermomodel ALIAS thermomodel)
target_include_directories(thermomodel PUBLIC include)
target_compile_features(thermomodel PUBLIC cxx_std_20)
target_compile_options(thermomodel PRIVATE -Wall -Wextra -Wpedantic)

add_library(thermogpu_scalar
  src/cpu/pr_scalar.cpp
  src/cpu/pr_batch_scalar.cpp
  src/cpu/tdar_oracle.cpp
  src/cpu/mixture_benchmark.cpp
)
target_include_directories(thermogpu_scalar PUBLIC include)
target_compile_options(thermogpu_scalar PRIVATE -Wall -Wextra -Wpedantic)
if(THERMOGPU_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED)
  target_sources(thermogpu_scalar PRIVATE src/cpu/pr_batch_openmp.cpp)
  target_link_libraries(thermogpu_scalar PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(thermogpu_scalar PUBLIC THERMOGPU_HAS_OPENMP=1)
endif()
if(THERMOGPU_ENABLE_CUDA)
  enable_language(CUDA)
  add_library(thermogpu_cuda src/cuda/pr_cuda.cu)
  target_include_directories(thermogpu_cuda PUBLIC include)
  target_link_libraries(thermogpu_cuda PUBLIC thermogpu_scalar)
  target_compile_definitions(thermogpu_cuda PUBLIC THERMOGPU_HAS_CUDA=1)
  set_target_properties(thermogpu_cuda PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  if(THERMOGPU_PTXAS_VERBOSE)
    target_compile_options(thermogpu_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>)
  endif()
endif()
add_executable(thermogpu_eval apps/thermogpu_eval.cpp)
target_link_libraries(thermogpu_eval PRIVATE thermogpu_scalar)
target_compile_options(thermogpu_eval PRIVATE -Wall -Wextra -Wpedantic)

add_executable(thermogpu_tdar_oracle apps/thermogpu_tdar_oracle.cpp)
target_link_libraries(thermogpu_tdar_oracle PRIVATE thermogpu_scalar)
target_compile_options(thermogpu_tdar_oracle PRIVATE -Wall -Wextra -Wpedantic)

add_executable(thermogpu_verify_benchmark_mixtures
    apps/thermogpu_verify_benchmark_mixtures.cpp)
target_link_libraries(thermogpu_verify_benchmark_mixtures PRIVATE thermogpu_scalar)
target_compile_options(thermogpu_verify_benchmark_mixtures
    PRIVATE -Wall -Wextra -Wpedantic)

add_executable(thermogpu_benchmark apps/thermogpu_benchmark.cpp)
target_link_libraries(thermogpu_benchmark PRIVATE thermogpu_scalar)
if(THERMOGPU_ENABLE_CUDA)
  target_link_libraries(thermogpu_benchmark PRIVATE thermogpu_cuda)
endif()
target_compile_options(thermogpu_benchmark PRIVATE -Wall -Wextra -Wpedantic)

add_executable(thermogpu_scaling apps/thermogpu_scaling.cpp)
target_link_libraries(thermogpu_scaling PRIVATE thermogpu_scalar)
if(THERMOGPU_ENABLE_CUDA)
  target_link_libraries(thermogpu_scaling PRIVATE thermogpu_cuda)
endif()
target_compile_options(thermogpu_scaling PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(test_pr tests/unit/test_pr.cpp)
target_link_libraries(test_pr PRIVATE thermogpu_scalar)
add_test(NAME pr_scalar_unit COMMAND test_pr)

add_executable(test_tdar_oracle tests/unit/test_tdar_oracle.cpp)
target_link_libraries(test_tdar_oracle PRIVATE thermogpu_scalar)
target_compile_options(test_tdar_oracle PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pr_tdar_oracle COMMAND test_tdar_oracle)

add_executable(test_pr_batch tests/unit/test_batch.cpp)
target_link_libraries(test_pr_batch PRIVATE thermogpu_scalar)
target_compile_options(test_pr_batch PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pr_scalar_batch COMMAND test_pr_batch)

if(THERMOGPU_ENABLE_CUDA)
  add_executable(thermogpu_fixed_composition_benchmark apps/thermogpu_fixed_composition_benchmark.cpp)
  target_link_libraries(thermogpu_fixed_composition_benchmark PRIVATE thermogpu_cuda)
  target_compile_options(thermogpu_fixed_composition_benchmark PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(thermogpu_distinct_composition_benchmark apps/thermogpu_distinct_composition_benchmark.cpp)
  target_link_libraries(thermogpu_distinct_composition_benchmark PRIVATE thermogpu_cuda)
  target_compile_options(thermogpu_distinct_composition_benchmark PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(thermogpu_distinct_composition_scaling
    apps/thermogpu_distinct_composition_scaling.cpp)
  target_link_libraries(thermogpu_distinct_composition_scaling PRIVATE thermogpu_cuda)
  target_compile_options(thermogpu_distinct_composition_scaling PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(test_pr_batch_cuda tests/unit/test_batch_cuda.cpp)
  target_link_libraries(test_pr_batch_cuda PRIVATE thermogpu_cuda)
  target_compile_options(test_pr_batch_cuda PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME pr_cuda_batch COMMAND test_pr_batch_cuda)

  add_executable(thermogpu_validate_cuda apps/thermogpu_validate_cuda.cpp)
  target_link_libraries(thermogpu_validate_cuda PRIVATE thermogpu_cuda)
  target_compile_options(thermogpu_validate_cuda PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME pr_cuda_differential COMMAND thermogpu_validate_cuda)

  add_executable(thermogpu_diagnose_state apps/thermogpu_diagnose_state.cpp)
  target_link_libraries(thermogpu_diagnose_state PRIVATE thermogpu_cuda)
  target_compile_options(thermogpu_diagnose_state PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(THERMOGPU_ENABLE_OPENMP)
  add_executable(test_pr_batch_openmp tests/unit/test_batch_openmp.cpp)
  target_link_libraries(test_pr_batch_openmp PRIVATE thermogpu_scalar)
  target_compile_options(test_pr_batch_openmp PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME pr_openmp_batch COMMAND test_pr_batch_openmp)
endif()

add_executable(test_pr_reference tests/reference/test_prmix_thermo_docs.cpp)
target_link_libraries(test_pr_reference PRIVATE thermogpu_scalar)
target_compile_options(test_pr_reference PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pr_external_reference COMMAND test_pr_reference)

add_executable(test_pr_thermopack tests/reference/test_thermopack_controlled.cpp)
target_link_libraries(test_pr_thermopack PRIVATE thermogpu_scalar)
target_compile_options(test_pr_thermopack PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pr_thermopack_controlled COMMAND test_pr_thermopack)

add_executable(test_pr_frozen_oracle tests/reference/test_frozen_oracle.cpp)
target_link_libraries(test_pr_frozen_oracle PRIVATE thermogpu_scalar)
target_compile_options(test_pr_frozen_oracle PRIVATE -Wall -Wextra -Wpedantic)
add_test(NAME pr_frozen_oracle COMMAND test_pr_frozen_oracle
  ${CMAKE_SOURCE_DIR}/tests/reference/data)

# Imported ThermoModel regression tests.
add_executable(thermomodel_tests
    tests/thermomodel/test_main.cpp
)
target_link_libraries(thermomodel_tests PRIVATE thermomodel::thermomodel)
add_test(NAME thermomodel_contract COMMAND thermomodel_tests)

add_executable(thermomodel_v02_tests
    tests/thermomodel/test_v02.cpp
)
target_link_libraries(thermomodel_v02_tests PRIVATE thermomodel::thermomodel)
add_test(NAME thermomodel_v02 COMMAND thermomodel_v02_tests)

add_executable(thermomodel_pr_tests
    tests/thermomodel/test_pr.cpp
)
target_link_libraries(thermomodel_pr_tests PRIVATE thermomodel::thermomodel)
add_test(NAME thermomodel_pr COMMAND thermomodel_pr_tests)

# ThermoGPU v1 / ThermoModel numerical comparison.
add_executable(compare_pure_pr
    tests/integration/compare_pure_pr.cpp
)

target_link_libraries(compare_pure_pr PRIVATE
    thermogpu_scalar
    thermomodel::thermomodel
)

target_compile_options(compare_pure_pr PRIVATE
    -Wall -Wextra -Wpedantic
)

add_test(
    NAME thermogpu_v2_pure_pr_equivalence
    COMMAND compare_pure_pr
)

# ============================================================
# ThermoModel CUDA device-kernel parity validation
# ============================================================

if(THERMOGPU_ENABLE_CUDA)

    function(add_thermomodel_cuda_parity_test test_name cuda_standard)

        set(target_name "thermomodel_${test_name}")

        add_executable(${target_name}
            tests/thermomodel/${test_name}.cu
        )

        target_link_libraries(${target_name} PRIVATE
            thermomodel::thermomodel
        )

        set_target_properties(${target_name} PROPERTIES
            CUDA_STANDARD ${cuda_standard}
            CUDA_STANDARD_REQUIRED ON
        )

        add_test(
            NAME ${target_name}
            COMMAND ${target_name}
        )

    endfunction()

    # G2a-G2d: CUDA C++20 kernels.
    add_thermomodel_cuda_parity_test(
        polynomial_cp_cuda_parity 20
    )

    add_thermomodel_cuda_parity_test(
        pr_coefficients_cuda_parity 20
    )

    add_thermomodel_cuda_parity_test(
        cubic_cuda_parity 20
    )

    add_thermomodel_cuda_parity_test(
        pr_derivatives_cuda_parity 20
    )

    add_thermomodel_cuda_parity_test(
        pr_residual_cuda_parity 20
    )

    # G2e: CUDA C++17 kernels.
    add_thermomodel_cuda_parity_test(
        pr_state_cuda_parity 17
    )

    add_thermomodel_cuda_parity_test(
        pr_tp_cuda_parity 17
    )

endif()
