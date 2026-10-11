#!/usr/bin/env python3
"""Compare ThermoModel against independent ThermoPack 2.2.3 data."""

import argparse
import csv
import math
import subprocess
from pathlib import Path

RU = 8.31446261815324
MW = 0.016043
R = RU / MW

CP_IG = 2200.0
TREF = 298.15
PREF = 101325.0

REFERENCE = Path(
    "tests/reference/data/methane_pr_caloric_thermopack.csv"
)

# Tight double-precision comparison tolerances.
# These are initial validation thresholds, not fitted values.
TOLERANCES = {
    "Z": (5e-13, 5e-14),
    "rho_kg_m3": (5e-13, 5e-12),
    "h_res_J_kg": (5e-12, 5e-8),
    "s_res_J_kgK": (5e-12, 5e-9),
    "cp_res_J_kgK": (5e-11, 5e-9),
}


def evaluate(executable, T, p):
    proc = subprocess.run(
        [
            str(executable),
            "state",
            "--model", "methane-pr",
            "--T", str(T),
            "--p", str(p),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    lines = proc.stdout.strip().splitlines()
    if len(lines) != 2:
        raise RuntimeError(
            f"Unexpected CLI output:\n{proc.stdout}"
        )

    return {
        k: float(v)
        for k, v in zip(
            lines[0].split(","),
            lines[1].split(","),
            strict=True,
        )
    }


def residuals(state, T, p):
    hig = CP_IG * (T - TREF)
    sig = (
        CP_IG * math.log(T / TREF)
        - R * math.log(p / PREF)
    )

    return {
        "Z": state["Z"],
        "rho_kg_m3": state["rho_kg_m3"],
        "h_res_J_kg": state["h_J_kg"] - hig,
        "s_res_J_kgK": state["s_J_kgK"] - sig,
        "cp_res_J_kgK": state["cp_J_kgK"] - CP_IG,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("build/thermomodel_cli"),
    )
    args = parser.parse_args()

    executable = args.executable.resolve()
    failures = 0
    comparisons = 0
    max_errors = {name: 0.0 for name in TOLERANCES}

    with REFERENCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    if len(rows) != 16:
        raise RuntimeError(
            f"Expected 16 reference states, got {len(rows)}"
        )

    for row in rows:
        T = float(row["T_K"])
        p = float(row["p_Pa"])

        actual = residuals(
            evaluate(executable, T, p), T, p
        )

        for name, (rtol, atol) in TOLERANCES.items():
            expected = float(row[name])
            observed = actual[name]
            error = abs(observed - expected)
            limit = atol + rtol * abs(expected)

            comparisons += 1
            max_errors[name] = max(
                max_errors[name], error
            )

            if not math.isfinite(observed) or error > limit:
                failures += 1
                print(
                    f"FAIL T={T:g} p={p:g} {name}\n"
                    f"  ThermoModel: {observed:.17g}\n"
                    f"  ThermoPack:  {expected:.17g}\n"
                    f"  Difference:  {error:.8e}\n"
                    f"  Tolerance:   {limit:.8e}"
                )

    print(f"States:      {len(rows)}")
    print(f"Comparisons: {comparisons}")
    print(f"Failures:    {failures}")
    print("Maximum absolute differences:")

    for name, value in max_errors.items():
        print(f"  {name:16s} {value:.8e}")

    print(
        "PASS" if failures == 0 else "FAIL",
        "ThermoPack external caloric reference",
    )

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
