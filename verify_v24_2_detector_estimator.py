#!/usr/bin/env python3
"""CPU-only sanity test for the CLARITAS V24.2 detector extrapolator.

Constructs a locally uniform four-dimensional phase-space density at the 0-deg
collimator.  In that case physical-equivalent KDE scores should be nearly
bandwidth independent and the h->0 intercept should agree statistically with
sparse exact aperture counting.
"""
import numpy as np
from claritas_tardiis_core_v24_2 import score_physical_apertures_multibandwidth
rng=np.random.default_rng(24681357)
N=400_000; a=0.002; rin=0.0505; rout=0.0595; L=0.012; y0=0.0500
x1=rng.uniform(-L,L,N); z1=rng.uniform(-L,L,N)
x2=rng.uniform(-L,L,N); z2=rng.uniform(-L,L,N)
dy=rout-rin
sx=(x2-x1)/dy; sz=(z2-z1)/dy
vx=sx.copy(); vy=np.ones(N); vz=sz.copy()
vn=np.sqrt(vx*vx+vy*vy+vz*vz); vx/=vn; vy/=vn; vz/=vn
f=(y0-rin)/dy
x=x1+f*(x2-x1); y=np.full(N,y0); z=z1+f*(z2-z1)
valid=np.ones(N,dtype=bool)
out=score_physical_apertures_multibandwidth(x,y,z,vx,vy,vz,valid,rin,rout,a,(2.,3.,4.),2.)
hits,_,bfs,eq,_,support,raw,phys,_,r2,spread=out
print('exact 0deg hits:',hits[0]);print('bandwidth factors:',bfs)
print('physical-equivalent 0deg scores:',eq[:,0]);print('support 0deg:',support[:,0])
print('raw h->0 intercept:',raw[0]);print('physical h->0 score:',phys[0]);print('fit R2:',r2[0]);print('relative bandwidth spread:',spread[0])
assert support[-1,0] > support[0,0] > hits[0]
assert np.all(np.isfinite(eq[:,0])) and np.isfinite(raw[0]) and phys[0]>=0
assert spread[0] < 0.15
assert abs(phys[0]-hits[0]) / max(hits[0],1) < 0.25
print('V24.2 detector extrapolator sanity test: PASS')
