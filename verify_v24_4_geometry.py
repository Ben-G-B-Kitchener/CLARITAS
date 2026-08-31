#!/usr/bin/env python3
"""CPU/static verification for CLARITAS V24.4 reconstructed collimator geometry."""
from pathlib import Path
import math,re
import numpy as np
from claritas_tardiis_core_v24_4 import detect_reconstructed_hardware,detect_hardware_symmetry_averaged,detect_legacy_acceptance

HERE=Path(__file__).resolve().parent

def block(text,start,end):
 a=text.index(start);b=text.index(end,a);return text[a:b]

def main():
 # Drawing-derived dimensions.
 rin=0.0505;rout=0.0655;depth=0.008;throat_out=rout-depth;rth=0.002;rcb=0.00435
 assert abs((throat_out-rin)-0.007)<1e-15
 half=math.degrees(math.atan2(rth,rout-rin))
 assert 7.5<half<7.7

 # Central 0-degree radial ray passes the mechanical channel.
 x=np.array([0.0]);y=np.array([0.05001]);z=np.array([0.0]);vx=np.array([0.0]);vy=np.array([1.0]);vz=np.array([0.0]);valid=np.array([True])
 h,idx=detect_reconstructed_hardware(x,y,z,vx,vy,vz,valid,rin,throat_out,rout,rth,rcb)
 assert h[0]==1 and h.sum()==1 and idx[0]==0

 # A ray steep enough to leave the 4-mm throat fails.
 vx2=np.array([0.30]);vy2=np.sqrt(1-vx2*vx2);vz2=np.array([0.0])
 h2,_=detect_reconstructed_hardware(x,y,z,vx2,vy2,vz2,valid,rin,throat_out,rout,rth,rcb)
 assert h2.sum()==0

 # Symmetry score is exactly the half-sum of two hard scorers.
 xx=np.array([0.0,0.001,-0.001]);yy=np.full(3,0.05001);zz=np.zeros(3);vxx=np.zeros(3);vyy=np.ones(3);vzz=np.zeros(3);vv=np.ones(3,dtype=bool)
 n,m,s,_,_=detect_hardware_symmetry_averaged(xx,yy,zz,vxx,vyy,vzz,vv,rin,throat_out,rout,rth,rcb)
 assert np.array_equal(s,0.5*(n+m))

 # Legacy +/-6.5-degree scorer overlaps neighbouring 10-degree channels as old CLARITAS did.
 ang=np.deg2rad(np.array([5.0]));rr=0.05
 lx=rr*np.sin(ang);ly=rr*np.cos(ang)
 lh=detect_legacy_acceptance(lx,ly,np.array([True]),np.array([1]),6.5)
 assert lh[0]==1 and lh[1]==1

 # Static regression: particle Fresnel routine, acrylic annulus, and water-transport loop are unchanged from V24.3.
 old=(HERE/'claritas_tardiis_core_v24_3.py').read_text()
 new=(HERE/'claritas_tardiis_core_v24_4.py').read_text()
 assert block(old,'__device__ int sphere_fresnel_interaction_3d','// Propagate a ray already inside the acrylic annulus') == block(new,'__device__ int sphere_fresnel_interaction_3d','// Propagate a ray already inside the acrylic annulus')
 assert block(old,'__device__ int acrylic_annulus','__device__ void raster_water_segment') == block(new,'__device__ int acrylic_annulus','__device__ void raster_water_segment')
 oldloop=block(old,'    for(int step=0;step<(int)MAX_ITERATIONS;++step){','    goto WRITE_OUT;')
 newloop=block(new,'    for(int step=0;step<(int)MAX_ITERATIONS;++step){','    goto WRITE_OUT;')
 assert oldloop==newloop
 print('PASS V24.4 geometry/static verification')
 print(f'ring radial thickness={(rout-rin)*1e3:.3f} mm; through-bore length={(throat_out-rin)*1e3:.3f} mm; counterbore depth={depth*1e3:.3f} mm')
 print(f'center-to-inner-throat half-angle diagnostic={half:.4f} deg; legacy comparator half-angle=6.5 deg')
 print('particle Fresnel, acrylic-annulus, and water transport blocks match V24.3 byte-for-byte')
if __name__=='__main__':main()
