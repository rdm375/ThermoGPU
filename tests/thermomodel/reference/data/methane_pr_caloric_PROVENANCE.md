# Methane Peng–Robinson external caloric reference

## Reference implementation

- ThermoPack version: 2.2.3
- EOS: Peng–Robinson
- Mixing rule: van der Waals
- Alpha model: Classic
- Composition: pure pseudo-component methane
- Phase selection: vapor root

## Component parameters

- Critical temperature: 190.56 K
- Critical pressure: 4,599,200 Pa
- Acentric factor: 0.01142
- Molar mass: 0.016043 kg/mol

## Reference grid

- Temperatures: 220, 300, 400, 600 K
- Pressures: 0.1, 1, 5, 10 MPa
- Total states: 16

## Quantities

- Compressibility factor Z
- Density, kg/m^3
- Residual enthalpy, J/kg
- Residual entropy, J/(kg K)
- Residual isobaric heat capacity, J/(kg K)

Residual quantities are relative to the corresponding
ideal-gas properties at the same temperature and pressure.

## Generation

python tools/thermopack_caloric_reference.py

The generated CSV is frozen for regression testing.
ThermoPack is required to regenerate the CSV but is not
required to run the regression test.

## Validation

ThermoModel was compared against ThermoPack 2.2.3:

- 16 states
- 80 comparisons
- 0 failures

Maximum absolute errors:

- Z: 3.33066907e-16
- Density: 1.42108547e-13 kg/m^3
- Residual enthalpy: 1.45519152e-10 J/kg
- Residual entropy: 3.97903932e-13 J/(kg K)
- Residual cp: 4.09272616e-12 J/(kg K)

This validates the selected PR residual-property states.
It does not constitute experimental validation or
validation of two-phase behavior.
