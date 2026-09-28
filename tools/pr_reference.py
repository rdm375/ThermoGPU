#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Richard Myers
"""Independent, standard-library Peng-Robinson oracle for frozen regression fixtures.

This file intentionally shares no ThermoGPU implementation code.  All inputs are
literal and all intermediate quantities are emitted so disagreements localize.
"""
import csv, math, sys

R = 8.31446261815324
SQRT2 = math.sqrt(2.0)
PR_OMEGA_A = 0.4572355289213822
PR_OMEGA_B = 0.0777960739038884

CASES = {
    "methane_low_pressure": {
        "T": 300.0, "P": 1.0e3,
        "components": [("methane",190.56,4.5992e6,0.01142,0.016043,1.0)],
        "kij": [[0.0]],
    },
    "methane_moderate_pressure": {
        "T": 300.0, "P": 5.0e6,
        "components": [("methane",190.56,4.5992e6,0.01142,0.016043,1.0)],
        "kij": [[0.0]],
    },
    "methane_ethane_binary": {
        "T": 300.0, "P": 5.0e6,
        "components": [
            ("methane",190.56,4.5992e6,0.01142,0.016043,0.80),
            ("ethane",305.32,4.872e6,0.0995,0.030070,0.20),
        ],
        "kij": [[0.0,0.0],[0.0,0.0]],
    },
    "binary_nonzero_kij": {
        "T": 300.0, "P": 8.0e6,
        "components": [
            ("methane",190.56,4.5992e6,0.01142,0.016043,0.70),
            ("carbon_dioxide",304.13,7.3773e6,0.22394,0.0440095,0.30),
        ],
        "kij": [[0.0,0.03],[0.03,0.0]],
    },
    "five_component": {
        "T": 320.0, "P": 8.0e6,
        "components": [
            ("methane",190.56,4.5992e6,0.01142,0.016043,0.80),
            ("ethane",305.32,4.872e6,0.0995,0.030070,0.08),
            ("propane",369.83,4.248e6,0.1523,0.044097,0.04),
            ("nitrogen",126.20,3.3958e6,0.0372,0.0280134,0.04),
            ("carbon_dioxide",304.13,7.3773e6,0.22394,0.0440095,0.04),
        ],
        "kij": [[0.0]*5 for _ in range(5)],
    },
    "high_pressure_five_component": {
        "T": 400.0, "P": 5.0e7,
        "components": [
            ("methane",190.56,4.5992e6,0.01142,0.016043,0.80),
            ("ethane",305.32,4.872e6,0.0995,0.030070,0.08),
            ("propane",369.83,4.248e6,0.1523,0.044097,0.04),
            ("nitrogen",126.20,3.3958e6,0.0372,0.0280134,0.04),
            ("carbon_dioxide",304.13,7.3773e6,0.22394,0.0440095,0.04),
        ],
        "kij": [[0.0]*5 for _ in range(5)],
    },
    "binary_three_root": {
        "T": 115.0, "P": 1.0e6,
        "components": [
            ("nitrogen",126.1,33.94e5,0.04,0.0280134,0.5),
            ("methane",190.6,46.04e5,0.011,0.01604246,0.5),
        ],
        "kij": [[0.0,0.0],[0.0,0.0]],
    },
    "methane_near_critical": {
        "T": 190.0, "P": 4.5e6,
        "components": [("methane",190.56,4.5992e6,0.01142,0.016043,1.0)],
        "kij": [[0.0]],
    },
}

def roots_pr(A, B):
    # Independent Cardano/trigonometric cubic implementation.
    aa = B - 1.0
    bb = A - 3.0*B*B - 2.0*B
    cc = -(A*B - B*B - B**3)
    p = bb - aa*aa/3.0
    q = 2.0*aa**3/27.0 - aa*bb/3.0 + cc
    disc = q*q/4.0 + p**3/27.0
    if disc > 0.0:
        s = math.sqrt(disc)
        cbrt = lambda x: math.copysign(abs(x)**(1.0/3.0), x)
        return [cbrt(-q/2+s)+cbrt(-q/2-s)-aa/3]
    r = 2.0*math.sqrt(max(0.0, -p/3.0))
    arg = max(-1.0, min(1.0, (3*q/(2*p))*math.sqrt(-3/p)))
    th = math.acos(arg)/3.0
    return sorted(r*math.cos(th-2*math.pi*k/3)-aa/3 for k in range(3))

def evaluate(case):
    T, P, comps, kij = case["T"], case["P"], case["components"], case["kij"]
    names=[c[0] for c in comps]; zs=[c[5] for c in comps]
    kappas=[]; alphas=[]; avals=[]; aa=[]; bs=[]
    for _,Tc,Pc,w,_,_ in comps:
        kap=0.37464+1.54226*w-0.26992*w*w
        al=(1+kap*(1-math.sqrt(T/Tc)))**2
        a=PR_OMEGA_A*(R*Tc)**2/Pc
        b=PR_OMEGA_B*R*Tc/Pc
        kappas.append(kap); alphas.append(al); avals.append(a); aa.append(a*al); bs.append(b)
    n=len(comps)
    aij=[[math.sqrt(aa[i]*aa[j])*(1-kij[i][j]) for j in range(n)] for i in range(n)]
    sums=[sum(zs[j]*aij[i][j] for j in range(n)) for i in range(n)]
    am=sum(zs[i]*zs[j]*aij[i][j] for i in range(n) for j in range(n))
    bm=sum(zs[i]*bs[i] for i in range(n))
    mw=sum(zs[i]*comps[i][4] for i in range(n))
    A=am*P/(R*R*T*T); B=bm*P/(R*T)
    roots=roots_pr(A,B); Z=max(roots)
    L=math.log((Z+(1+SQRT2)*B)/(Z+(1-SQRT2)*B))
    lnphi=[]
    for i in range(n):
        br=bs[i]/bm
        attraction=2*sums[i]/am-br
        lnphi.append(br*(Z-1)-math.log(Z-B)-A/(2*SQRT2*B)*attraction*L)
    rho=P*mw/(Z*R*T)
    residual=Z**3-(1-B)*Z**2+(A-3*B*B-2*B)*Z-(A*B-B*B-B**3)
    out={"R":R,"T":T,"P":P,"a_m":am,"b_m":bm,"molar_mass":mw,"A":A,"B":B,
         "root_count":float(len(roots)),"Z":Z,"density":rho,"cubic_residual":residual}
    for i,name in enumerate(names):
        out[f"kappa.{name}"]=kappas[i]; out[f"alpha.{name}"]=alphas[i]
        out[f"a.{name}"]=avals[i]; out[f"a_alpha.{name}"]=aa[i]; out[f"b.{name}"]=bs[i]
        out[f"sum_z_aij.{name}"]=sums[i]; out[f"ln_phi.{name}"]=lnphi[i]
        out[f"phi.{name}"]=math.exp(lnphi[i]); out[f"fugacity.{name}"]=zs[i]*P*math.exp(lnphi[i])
    for i,x in enumerate(roots): out[f"root.{i}"]=x
    return out

def main():
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        raise SystemExit(f"usage: {sys.argv[0]} {{{'|'.join(CASES)}}} output.csv")
    out=evaluate(CASES[sys.argv[1]])
    with open(sys.argv[2], "w", newline="") as f:
        w=csv.writer(f); w.writerow(["quantity","value"])
        for k,v in out.items(): w.writerow([k, format(v, ".17g")])
    print(f"wrote {sys.argv[2]} ({len(out)} quantities)")
if __name__ == "__main__": main()
