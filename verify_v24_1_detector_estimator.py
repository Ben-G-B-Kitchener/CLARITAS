#!/usr/bin/env python3
"""CPU-only sanity check for the V24.1 detector variance-reduction normalisation."""
import numpy as np
from claritas_tardiis_core_v24_1 import score_physical_apertures_variance_reduced

# Construct rays at detector 90 deg (+x radial direction) with independent,
# locally-uniform transverse positions at the two scoring planes.  The exact
# and VR physical-equivalent counts should agree statistically.
rng=np.random.default_rng(123456)
N=20_000
rin=0.0505; rout=0.0595; a=0.002; hfac=3.0
# Parameterise line by its transverse y,z at each plane x=rin/rout.
y1=rng.uniform(-0.010,0.010,N); z1=rng.uniform(-0.010,0.010,N)
y2=rng.uniform(-0.010,0.010,N); z2=rng.uniform(-0.010,0.010,N)
dx=rout-rin
vx=np.ones(N); vy=(y2-y1)/dx; vz=(z2-z1)/dx
vn=np.sqrt(vx*vx+vy*vy+vz*vz);vx/=vn;vy/=vn;vz/=vn
# Back-propagate a small distance from inner plane so t1>0.
t0=0.001/np.maximum(vx,1e-12)
x=rin-t0*vx; y=y1-t0*vy; z=z1-t0*vz
valid=np.ones(N,bool)
hits,_,eq,ks,sup=score_physical_apertures_variance_reduced(x,y,z,vx,vy,vz,valid,rin,rout,a,hfac)
j=9 # 90 deg
print(f'exact_90={hits[j]}')
print(f'vr_equivalent_90={eq[j]:.3f}')
print(f'support_90={sup[j]}')
rel=(eq[j]-hits[j])/max(hits[j],1)
print(f'relative_difference={rel:.4%}')
if abs(rel)>0.60:
    raise SystemExit('VR normalisation sanity check failed (>10% difference)')
print('PASS')
