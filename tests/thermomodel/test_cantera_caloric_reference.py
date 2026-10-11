#!/usr/bin/env python3
"""Compare ThermoModel methane ideal-gas caloric properties to Cantera."""
import csv
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "tests/reference/data/methane_nasa7_cantera.csv"
EXE = Path(sys.argv[1]).resolve()

# ThermoModel and Cantera use the same NASA7 coefficients but
# different evaluation paths. These are numerical tolerances,
# not fitted or empirical uncertainty estimates.
TOL = {
    "cp": (2e-11, 1e-7),
    "dh": (2e-10, 1e-6),
    "ds": (2e-10, 1e-8),
}

def close(name, actual, expected):
    rtol, atol = TOL[name]
    if not math.isfinite(actual) or not math.isclose(
        actual, expected, rel_tol=rtol, abs_tol=atol
    ):
        raise AssertionError(
            f"{name}: actual={actual:.17g}, "
            f"expected={expected:.17g}, "
            f"error={actual-expected:.9g}"
        )

def state(T, pressure):
    output = subprocess.check_output([
        str(EXE), "state",
        "--model", "methane-pr-variable-cp",
        "--T", format(T, ".17g"),
        "--p", format(pressure, ".17g"),
    ], text=True)
    return next(csv.DictReader(output.splitlines()))

# Remove PR residual contributions by differencing against a
# constant-cp PR model at the same T and p.
#
# h_var - h_const = h_ig,var - cp_const*(T-Tref)
# s_var - s_const = s_ig,var - cp_const*ln(T/Tref)
# cp_var - cp_const = cp_ig,var - cp_const
#
# Both PR models must have identical EOS parameters.
CP_CONST = 2200.0
TREF = 298.15
PRESSURE = 5e6

rows = list(csv.DictReader(CSV.open(newline="")))
count = 0

for row in rows:
    T = float(row["T_K"])
    variable = state(T, PRESSURE)
    constant = state(T, PRESSURE)

    # The second call must explicitly select the baseline model.
    output = subprocess.check_output([
        str(EXE), "state",
        "--model", "methane-pr",
        "--T", format(T, ".17g"),
        "--p", format(PRESSURE, ".17g"),
    ], text=True)
    constant = next(csv.DictReader(output.splitlines()))

    cp = (
        float(variable["cp_J_kgK"])
        - float(constant["cp_J_kgK"])
        + CP_CONST
    )
    dh = (
        float(variable["h_J_kg"])
        - float(constant["h_J_kg"])
        + CP_CONST*(T-TREF)
    )
    ds = (
        float(variable["s_J_kgK"])
        - float(constant["s_J_kgK"])
        + CP_CONST*math.log(T/TREF)
    )

    close("cp", cp, float(row["cp_ig_J_kgK"]))
    close("dh", dh, float(row["dh_ig_J_kg"]))
    close("ds", ds, float(row["ds_ig_J_kgK"]))
    count += 3

print(f"PASS Cantera ideal-gas caloric reference: {count} comparisons")
