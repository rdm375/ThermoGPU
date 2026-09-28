#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Richard Myers
"""Reproduce the controlled ThermoPack 2.2.3 external validation cases.

Optional developer audit tool; ThermoPack is intentionally not a ThermoGPU build
or test dependency. Pseudo slots must be declared during the first initialization.
"""
import numpy as np
from thermopack.cubic import cubic


def model(Tc, Pc, acf, Mw, names):
    n=len(Tc)
    eos=cubic(",".join(["PSEUDO"]*n), "PR", mixing="vdW")
    eos.init_pseudo(comps=",".join(names), Tclist=np.array(Tc), Pclist=np.array(Pc),
                    acflist=np.array(acf), Mwlist=np.array(Mw), mixing="vdW", alpha="Classic")
    for i in range(1,n+1):
        for j in range(i+1,n+1): eos.set_kij(i,j,0.0)
    return eos

T=300.0; P=5.0e6
pure=model([190.56],[4.5992e6],[0.01142],[0.016043],["ThermoGPU_CH4"])
z=np.array([1.0])
print("pure methane")
print(f"Z={pure.zfac(T,P,z,pure.VAPPH)[0]:.16e}")
print(f"lnphi={pure.thermo(T,P,z,pure.VAPPH)[0][0]:.16e}")

binary=model([190.56,305.32],[4.5992e6,4.872e6],[0.01142,0.09950],
             [0.016043,0.030070],["ThermoGPU_CH4","ThermoGPU_C2H6"])
z=np.array([0.8,0.2]); lp=binary.thermo(T,P,z,binary.VAPPH)[0]
print("\nbinary CH4/C2H6")
print(f"Z={binary.zfac(T,P,z,binary.VAPPH)[0]:.16e}")
print(f"lnphi[0]={lp[0]:.16e}")
print(f"lnphi[1]={lp[1]:.16e}")
