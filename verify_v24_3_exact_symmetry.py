#!/usr/bin/env python3
"""CPU-only checks for the V24.3 exact detector estimator (no CUDA required)."""
import numpy as np
from claritas_tardiis_core_v24_3 import detect_physical_apertures,detect_symmetry_averaged_exact

def main():
 rng=np.random.default_rng(1234);N=400000
 # Synthetic air-exit rays deliberately drawn from an x-symmetric distribution.
 phi=rng.uniform(-np.pi,np.pi,N); r=np.full(N,0.0500003);x=r*np.sin(phi);y=r*np.cos(phi);z=rng.normal(0,0.003,N)
 # Mostly outward directions with random angular perturbations.
 vx=np.sin(phi)+rng.normal(0,0.035,N);vy=np.cos(phi)+rng.normal(0,0.035,N);vz=rng.normal(0,0.025,N)
 q=np.sqrt(vx*vx+vy*vy+vz*vz);vx/=q;vy/=q;vz/=q;valid=np.ones(N,dtype=bool)
 nat,mir,sym,_,_=detect_symmetry_averaged_exact(x,y,z,vx,vy,vz,valid,0.0505,0.0595,0.002)
 # Direct native score for an independently mirrored sample must match the helper.
 mir2,_=detect_physical_apertures(-x,y,z,-vx,vy,vz,valid,0.0505,0.0595,0.002)
 assert np.array_equal(mir,mir2)
 assert np.allclose(sym,0.5*(nat+mir))
 print('native total:',nat.sum());print('mirror total:',mir.sum());print('symmetry-equivalent total:',sym.sum())
 print('max |native normalized - mirror normalized|:',np.max(np.abs(nat/nat.sum()-mir/mir.sum())))
 print('PASS: only hard-aperture hits are used; symmetry score is the exact 1/2 native+mirror average.')
if __name__=='__main__':main()
