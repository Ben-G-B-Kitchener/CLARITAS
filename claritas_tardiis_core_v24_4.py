#!/usr/bin/env python3
"""Shared CLARITAS V24.4 / TARDIIS apparatus forward model.

Physics:
- fully 3-D ray state;
- geometric particle encounter cross section pi*r^2 (no Mie);
- full Snell + unpolarised Fresnel physics at spherical particles;
- real TARDIIS cylindrical water/acrylic/air side-wall geometry;
- source and detectors constrained by the physical radial collimator apertures;
- multiple acrylic-wall reflections and TIR are followed explicitly.

V24.4 keeps the V23.2 particle interaction physics and V24 water/acrylic/air
cell optics unchanged, but corrects the TARDIIS sensor-ring collimator geometry
from the supplied 2019 design paper. The ring is reconstructed as a 4-mm-diameter
through-bore followed by an 8.7-mm-diameter, 8-mm-deep counterbore. A legacy
+/-6.5-degree CLARITAS exit-position scorer is retained as a comparison channel
only; it is not the selected physical detector response.

The paper contains a prose/drawing inconsistency (3 mm / 10 mm in section 5.3
versus 4 mm / 8 mm in Fig. 3, Fig. 4 and the machining caption). V24.4 defaults
to the engineering drawing / machining values: 4 mm through and 8 mm deep.
All dimensions are configurable. The default PMMA refractive index is configurable
and is not fitted by the PSD inference program.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import math
import numpy as np

LOESS_DIAMETER_M = np.array([1.729e-6, 1.981e-6, 2.269e-6, 2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6,
    4.472e-6, 5.122e-6, 5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6,
    11.565e-6, 13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6,
    26.111e-6, 29.907e-6, 34.255e-6, 39.234e-6, 44.938e-6, 51.471e-6,
    58.953e-6, 67.523e-6, 77.34e-6, 88.583e-6, 101.46e-6, 116.21e-6,
    133.103e-6, 152.453e-6, 174.616e-6, 200.000e-6, 229.075e-6, 262.376e-6], dtype=np.float64)
LOESS_WEIGHTS = np.array([157, 227, 294, 354, 414, 487, 592, 747, 975, 1291, 1704, 2197, 2736,
    3288, 3822, 4196, 4372, 4391, 4352, 4362, 4508, 4826, 5279, 5758,
    6080, 6106, 5786, 5149, 4342, 3404, 2456, 1662, 1175, 858, 631, 463, 333, 230], dtype=np.float64)
KAOLIN_DIAMETER_M = np.array([0.172e-6, 0.197e-6, 0.226e-6, 0.259e-6, 0.296e-6, 0.339e-6, 0.389e-6,
    0.445e-6, 0.51e-6, 0.584e-6, 0.669e-6, 0.766e-6, 0.877e-6, 1.005e-6,
    1.151e-6, 1.318e-6, 1.51e-6, 1.729e-6, 1.981e-6, 2.269e-6,
    2.599e-6, 2.976e-6, 3.409e-6, 3.905e-6, 4.472e-6, 5.122e-6,
    5.867e-6, 6.72e-6, 7.697e-6, 8.816e-6, 10.097e-6, 11.565e-6,
    13.246e-6, 15.172e-6, 17.377e-6, 19.904e-6, 22.797e-6], dtype=np.float64)
KAOLIN_WEIGHTS = np.array([217, 547, 1112, 2032, 2985, 3492, 3308, 2644, 1893, 1300, 916, 700, 601,
    584, 637, 757, 948, 1208, 1530, 1899, 2309, 2770, 3312, 3973,
    4772, 5681, 6583, 7267, 7478, 7042, 6113, 5057, 3680, 2330, 1287, 631, 284], dtype=np.float64)

DETECTOR_ANGLES_DEG = np.arange(0, 180, 10, dtype=np.float64)

CUDA_SRC = r'''extern "C" {

#define INF_F 3.402823466e+38F
#define PI_F 3.14159265358979323846f
#define EPS_F 2.0e-7f

__device__ unsigned int xorshift32_state(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x; return x;
}

__device__ float rnd_uniform(unsigned int* state) {
    return (float)(xorshift32_state(state) * 2.3283064e-10f);
}

__device__ float dot3(float ax,float ay,float az,float bx,float by,float bz) {
    return ax*bx + ay*by + az*bz;
}

__device__ void normalize3(float* x,float* y,float* z) {
    float q=(*x)*(*x)+(*y)*(*y)+(*z)*(*z);
    if(q<=0.0f){*x=0.0f;*y=1.0f;*z=0.0f;return;}
    float s=rsqrtf(q); *x*=s;*y*=s;*z*=s;
}

__device__ void reflect3(float ix,float iy,float iz,float nx,float ny,float nz,
                         float* ox,float* oy,float* oz) {
    float d=dot3(ix,iy,iz,nx,ny,nz);
    *ox=ix-2.0f*d*nx; *oy=iy-2.0f*d*ny; *oz=iz-2.0f*d*nz;
    normalize3(ox,oy,oz);
}

__device__ int refract3(float ix,float iy,float iz,
                        float nx,float ny,float nz,
                        float n1,float n2,
                        float* ox,float* oy,float* oz) {
    // n points into the incident medium.
    float ci=-dot3(ix,iy,iz,nx,ny,nz);
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float eta=n1/n2;
    float k=1.0f-eta*eta*(1.0f-ci*ci);
    if(k<=0.0f) return 0;
    float ct=sqrtf(k);
    *ox=eta*ix+(eta*ci-ct)*nx;
    *oy=eta*iy+(eta*ci-ct)*ny;
    *oz=eta*iz+(eta*ci-ct)*nz;
    normalize3(ox,oy,oz); return 1;
}

__device__ float fresnel_R(float ci,float n1,float n2,int* tir) {
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float eta=n1/n2;
    float st2=eta*eta*fmaxf(0.0f,1.0f-ci*ci);
    if(st2>=1.0f){*tir=1;return 1.0f;}
    *tir=0;
    float ct=sqrtf(fmaxf(0.0f,1.0f-st2));
    float ds=n1*ci+n2*ct, dp=n1*ct+n2*ci;
    float rs=1.0f,rp=1.0f;
    if(fabsf(ds)>1e-12f){float a=(n1*ci-n2*ct)/ds;rs=a*a;}
    if(fabsf(dp)>1e-12f){float a=(n1*ct-n2*ci)/dp;rp=a*a;}
    return fminf(fmaxf(0.5f*(rs+rp),0.0f),1.0f);
}

__device__ float cylinder_hit(float x,float y,float vx,float vy,float R) {
    float a=vx*vx+vy*vy;
    if(a<1e-18f) return INF_F;
    float b=x*vx+y*vy;
    float c=x*x+y*y-R*R;
    float d=b*b-a*c;
    if(d<0.0f) return INF_F;
    float sd=sqrtf(fmaxf(d,0.0f));
    float t1=(-b-sd)/a, t2=(-b+sd)/a;
    float t=INF_F;
    if(t1>EPS_F) t=t1;
    if(t2>EPS_F && t2<t) t=t2;
    return t;
}

__device__ void radial_normal(float x,float y,float* nx,float* ny) {
    float r=sqrtf(fmaxf(x*x+y*y,1e-30f)); *nx=x/r; *ny=y/r;
}

__device__ void perpendicular_basis(float vx,float vy,float vz,
                                    float* e1x,float* e1y,float* e1z,
                                    float* e2x,float* e2y,float* e2z) {
    float rx,ry,rz;
    if(fabsf(vy)<0.9f){rx=0.0f;ry=1.0f;rz=0.0f;}else{rx=1.0f;ry=0.0f;rz=0.0f;}
    *e1x=ry*vz-rz*vy; *e1y=rz*vx-rx*vz; *e1z=rx*vy-ry*vx;
    normalize3(e1x,e1y,e1z);
    *e2x=vy*(*e1z)-vz*(*e1y); *e2y=vz*(*e1x)-vx*(*e1z); *e2z=vx*(*e1y)-vy*(*e1x);
    normalize3(e2x,e2y,e2z);
}

__device__ int sphere_fresnel_interaction_3d(
    unsigned int* state,float n_medium,float n_particle,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections) {
    if(radius<=0.0f) return 1;
    float rho=sqrtf(fminf(fmaxf(rnd_uniform(state),0.0f),0.99999994f));
    float ci=sqrtf(fmaxf(0.0f,1.0f-rho*rho));
    float phi=rnd_uniform(state)*2.0f*PI_F;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(*vx,*vy,*vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float qx=cosf(phi)*e1x+sinf(phi)*e2x;
    float qy=cosf(phi)*e1y+sinf(phi)*e2y;
    float qz=cosf(phi)*e1z+sinf(phi)*e2z;
    float nx=-ci*(*vx)+rho*qx, ny=-ci*(*vy)+rho*qy, nz=-ci*(*vz)+rho*qz;
    normalize3(&nx,&ny,&nz);
    float cx=*x-radius*nx, cy=*y-radius*ny, cz=*z-radius*nz;
    int tir=0; float R=fresnel_R(ci,n_medium,n_particle,&tir);
    if(tir || rnd_uniform(state)<R){
        float a,b,c; reflect3(*vx,*vy,*vz,nx,ny,nz,&a,&b,&c);
        *vx=a;*vy=b;*vz=c;(*p_reflections)++;(*p_entry_reflections)++;return 1;
    }
    float a,b,c;
    if(!refract3(*vx,*vy,*vz,nx,ny,nz,n_medium,n_particle,&a,&b,&c)) return 0;
    *vx=a;*vy=b;*vz=c;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        float rx=*x-cx,ry=*y-cy,rz=*z-cz;
        float chord=-2.0f*dot3(rx,ry,rz,*vx,*vy,*vz);
        if(chord<=1e-12f) return 0;
        *x+=chord*(*vx);*y+=chord*(*vy);*z+=chord*(*vz);*internal_path+=chord;
        float nox=(*x-cx)/radius,noy=(*y-cy)/radius,noz=(*z-cz)/radius;normalize3(&nox,&noy,&noz);
        float cii=fminf(fmaxf(dot3(*vx,*vy,*vz,nox,noy,noz),0.0f),1.0f);
        int tiri=0; float Ri=fresnel_R(cii,n_particle,n_medium,&tiri);
        if(!tiri && rnd_uniform(state)>=Ri){
            if(refract3(*vx,*vy,*vz,-nox,-noy,-noz,n_particle,n_medium,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;return 1;
            }
        }
        reflect3(*vx,*vy,*vz,nox,noy,noz,&a,&b,&c);*vx=a;*vy=b;*vz=c;
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces) return 0;
    }
    return 0;
}

// Propagate a ray already inside the acrylic annulus until it re-enters water
// or transmits through the outer acrylic surface into air.
// return: 1=water, 2=air, 0=failure.
__device__ int acrylic_annulus(
    unsigned int* state,float n_water,float n_acrylic,float n_air,
    float Rin,float Rout,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* inner_reflections,int* outer_reflections,int* cell_tir) {
    for(int bounce=0;bounce<=max_bounces;++bounce){
        float ti=cylinder_hit(*x,*y,*vx,*vy,Rin);
        float to=cylinder_hit(*x,*y,*vx,*vy,Rout);
        int hit_inner = ti < to;
        float t=hit_inner?ti:to;
        if(t>=INF_F*0.5f) return 0;
        *x+=t*(*vx);*y+=t*(*vy);*z+=t*(*vz);*acrylic_path+=t;
        float nx,ny; radial_normal(*x,*y,&nx,&ny);
        float a,b,c; int tir=0;
        if(hit_inner){
            // acrylic -> water; +radial points into incident acrylic medium.
            float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);
            float R=fresnel_R(ci,n_acrylic,n_water,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,nx,ny,0.0f,n_acrylic,n_water,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*x-=EPS_F*nx;*y-=EPS_F*ny;return 1;
            }
            reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;
            (*inner_reflections)++; if(tir)(*cell_tir)++; *x+=EPS_F*nx;*y+=EPS_F*ny;
        } else {
            // acrylic -> air; -radial points into incident acrylic medium.
            float ci=fminf(fmaxf(dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);
            float R=fresnel_R(ci,n_acrylic,n_air,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,-nx,-ny,0.0f,n_acrylic,n_air,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*x+=EPS_F*nx;*y+=EPS_F*ny;return 2;
            }
            reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;
            (*outer_reflections)++; if(tir)(*cell_tir)++; *x-=EPS_F*nx;*y-=EPS_F*ny;
        }
        if(bounce==max_bounces) return 0;
    }
    return 0;
}

__device__ void raster_water_segment(float x0,float y0,float z0,float vx,float vy,float vz,float dist,
    int vis,float Rin,float zmin,float zmax,float* hxy,float* hzy) {
    if(vis<=0 || dist<=0.0f) return;
    const float spacing=1.0e-5f;
    int n=(int)ceilf(dist/spacing); if(n<1)n=1; if(n>20000)n=20000;
    for(int k=1;k<=n;++k){
        float f=(float)k/(float)n;
        float x=x0+vx*dist*f,y=y0+vy*dist*f,z=z0+vz*dist*f;
        int ix=(int)(((x+Rin)/(2.0f*Rin))*vis);
        int iy=vis-1-(int)(((y+Rin)/(2.0f*Rin))*vis);
        int iz=(int)(((z-zmin)/(zmax-zmin))*vis);
        if(ix>=0&&ix<vis&&iy>=0&&iy<vis) atomicAdd(&hxy[iy*vis+ix],1.0f);
        if(iz>=0&&iz<vis&&iy>=0&&iy<vis) atomicAdd(&hzy[iy*vis+iz],1.0f);
    }
}

__global__ void trace_kernel(
    const float MAX_ITERATIONS,const float MU_GEOM,
    const float N_WATER,const float N_ACRYLIC,const float N_AIR,
    const int MAX_PARTICLE_BOUNCES,const int MAX_CELL_BOUNCES,
    const float RIN,const float ROUT,const float ZMIN,const float ZMAX,
    const float RING_IN_R,const float THROAT_OUT_R,const float RING_OUT_R,const float SOURCE_R,
    const float THROAT_R,const float COUNTERBORE_R,const float BEAM_SIGMA,
    const int VIS_SIZE,
    const double* polar_init,const double* azimuth_init,const int N_rays,
    const double* particle_cdf,const double* particle_n,const double* particle_radius,const int n_particles,
    float* heat_xy,float* heat_zy,
    float* air_x,float* air_y,float* air_z,float* air_vx,float* air_vy,float* air_vz,
    float* water_path,float* acrylic_path,
    int* interaction_count,int* particle_reflections,int* particle_entry_reflections,int* particle_internal_reflections,
    int* cell_inner_reflections,int* cell_outer_reflections,int* cell_tir,
    int* entered_water,int* status,
    unsigned int seed0,unsigned int seed1,const unsigned int ray_offset) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x; if(tid>=N_rays)return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    unsigned int state=seed0+gid*74729u+13u, stateOPT=seed1+gid*104729u+29u;
    int ic=0,pr=0,per=0,pir=0,cir=0,cor=0,ctir=0,entered=0,st=0;
    float wp=0.0f,ap=0.0f;

    // TARDIIS LED module is at 180 degrees. V24.4 reconstructs the sensor-ring
    // bore from the engineering drawing: ring OD 131 mm / ID 101 mm, an
    // 8.7-mm counterbore 8 mm deep, then a 4-mm through-bore to the inner face.
    // The emitting spot itself remains the pre-existing narrow Gaussian source;
    // only the mechanical launch/collimation geometry is changed here.
    float u1=fmaxf(rnd_uniform(&state),1e-12f),u2=rnd_uniform(&state);
    float mag=sqrtf(-2.0f*logf(u1));
    float x=BEAM_SIGMA*mag*cosf(2.0f*PI_F*u2);
    float z=BEAM_SIGMA*mag*sinf(2.0f*PI_F*u2);
    float y=-SOURCE_R;
    float polar=(float)polar_init[tid],az=(float)azimuth_init[tid];
    float sp=sinf(polar),vx=sp*cosf(az),vy=cosf(polar),vz=sp*sinf(az);normalize3(&vx,&vy,&vz);
    if(vy<=1e-12f){st=3;goto WRITE_OUT;}

    // At the outer ring face the source must lie inside the 8.7-mm counterbore.
    if(x*x+z*z>COUNTERBORE_R*COUNTERBORE_R){st=3;goto WRITE_OUT;}

    // The narrow 4-mm throat occupies RING_IN_R .. THROAT_OUT_R. A straight
    // ray is inside the cylindrical throat throughout iff both end-plane
    // transverse radii are inside the throat circle (norm is convex along a line).
    {
        float t=(-THROAT_OUT_R-y)/vy;
        if(t<=0.0f){st=3;goto WRITE_OUT;}
        float xt=x+t*vx, zt=z+t*vz;
        if(xt*xt+zt*zt>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
        t=(-RING_IN_R-y)/vy;
        if(t<=0.0f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        if(x*x+z*z>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
    }

    // Reach outer acrylic cylinder from the air gap and apply air -> acrylic Fresnel/Snell.
    {
        float t=cylinder_hit(x,y,vx,vy,ROUT);
        if(t>=INF_F*0.5f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        float nx,ny;radial_normal(x,y,&nx,&ny);
        float ci=fminf(fmaxf(-dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,N_AIR,N_ACRYLIC,&tir);
        float a,b,c;
        if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,nx,ny,0.0f,N_AIR,N_ACRYLIC,&a,&b,&c)){
            reflect3(vx,vy,vz,nx,ny,0.0f,&a,&b,&c);vx=a;vy=b;vz=c;cor++;if(tir)ctir++;
            x+=EPS_F*nx;y+=EPS_F*ny;st=1;goto AIR_EXIT;
        }
        vx=a;vy=b;vz=c;x-=EPS_F*nx;y-=EPS_F*ny;
        int where=acrylic_annulus(&state,N_WATER,N_ACRYLIC,N_AIR,RIN,ROUT,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&ctir);
        if(where==2){st=1;goto AIR_EXIT;}
        if(where!=1){st=4;goto WRITE_OUT;}
        entered=1;
    }

    for(int step=0;step<(int)MAX_ITERATIONS;++step){
        if(z<=ZMIN || z>=ZMAX){st=2;break;}
        float tside=cylinder_hit(x,y,vx,vy,RIN);
        float tz=INF_F;
        if(vz>1e-14f) tz=(ZMAX-z)/vz;
        else if(vz<-1e-14f) tz=(ZMIN-z)/vz;
        if(tz<=EPS_F) tz=INF_F;
        int axial = tz < tside;
        float tbound=axial?tz:tside;
        if(tbound>=INF_F*0.5f){st=4;break;}
        float freep=INF_F;
        if(MU_GEOM>0.0f){float u=fmaxf(rnd_uniform(&state),1e-12f);freep=-logf(u)/MU_GEOM;}
        int do_particle=freep<tbound;
        float travel=do_particle?freep:tbound;
        float x0=x,y0=y,z0=z;
        raster_water_segment(x0,y0,z0,vx,vy,vz,travel,VIS_SIZE,RIN,ZMIN,ZMAX,heat_xy,heat_zy);
        x+=travel*vx;y+=travel*vy;z+=travel*vz;wp+=travel;
        if(do_particle){
            ic++;
            float u=rnd_uniform(&state);int pidx=n_particles-1;
            for(int j=0;j<n_particles-1;++j){if(u<=(float)particle_cdf[j]){pidx=j;break;}}
            float internal=0.0f;
            if(!sphere_fresnel_interaction_3d(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_radius[pidx],
                MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&internal,&pr,&per,&pir)){st=4;break;}
            continue;
        }
        if(axial){st=2;break;}

        // water -> acrylic at the cylindrical sample-cell wall.
        float nx,ny;radial_normal(x,y,&nx,&ny);
        float ci=fminf(fmaxf(dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,N_WATER,N_ACRYLIC,&tir);float a,b,c;
        if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,-nx,-ny,0.0f,N_WATER,N_ACRYLIC,&a,&b,&c)){
            reflect3(vx,vy,vz,nx,ny,0.0f,&a,&b,&c);vx=a;vy=b;vz=c;cir++;if(tir)ctir++;
            x-=EPS_F*nx;y-=EPS_F*ny;continue;
        }
        vx=a;vy=b;vz=c;x+=EPS_F*nx;y+=EPS_F*ny;
        int where=acrylic_annulus(&state,N_WATER,N_ACRYLIC,N_AIR,RIN,ROUT,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&ctir);
        if(where==1) continue;
        if(where==2){st=1;goto AIR_EXIT;}
        st=4;break;
    }
    goto WRITE_OUT;

AIR_EXIT:
    air_x[tid]=x;air_y[tid]=y;air_z[tid]=z;air_vx[tid]=vx;air_vy[tid]=vy;air_vz[tid]=vz;
WRITE_OUT:
    water_path[tid]=wp;acrylic_path[tid]=ap;
    interaction_count[tid]=ic;particle_reflections[tid]=pr;particle_entry_reflections[tid]=per;particle_internal_reflections[tid]=pir;
    cell_inner_reflections[tid]=cir;cell_outer_reflections[tid]=cor;cell_tir[tid]=ctir;
    entered_water[tid]=entered;status[tid]=st;
}
}
'''


def get_material_psd(material: str):
    m=material.strip().lower()
    if m=='loess': d,w=LOESS_DIAMETER_M.copy(),LOESS_WEIGHTS.copy()
    elif m=='kaolin': d,w=KAOLIN_DIAMETER_M.copy(),KAOLIN_WEIGHTS.copy()
    else: raise ValueError("material must be 'loess' or 'kaolin'")
    return d,w/w.sum()


def build_geometric_transport(diameters_m,weights,concentration_g_per_L,density_kg_per_m3=2600.0,weight_mode='mass_fraction'):
    d=np.asarray(diameters_m,float);w=np.asarray(weights,float)
    if d.ndim!=1 or w.ndim!=1 or d.size!=w.size: raise ValueError('diameters/weights mismatch')
    if np.any(d<=0) or np.any(w<0) or not np.any(w>0): raise ValueError('invalid PSD')
    w=w/w.sum();r=d/2.0
    pm=(4.0/3.0)*np.pi*r**3*float(density_kg_per_m3);c=float(concentration_g_per_L)
    if weight_mode=='mass_fraction': nd=c*w/pm
    elif weight_mode=='number_fraction':
        mm=np.sum(w*pm);nd=(c/mm if mm>0 else 0.0)*w
    else: raise ValueError('weight_mode must be mass_fraction or number_fraction')
    sigma=np.pi*r*r;mu_bin=nd*sigma;mu=float(mu_bin.sum())
    ew=mu_bin/mu if mu>0 else np.zeros_like(mu_bin);cdf=np.cumsum(ew)
    if cdf.size and cdf[-1]>0:cdf/=cdf[-1]
    return {'weights':w,'radii_m':r,'number_density_by_bin':nd,'geometric_cross_section_m2':sigma,
            'mu_geom_by_bin':mu_bin,'mu_geom':mu,'particle_event_weights':ew,'particle_event_cdf':cdf}



def detect_reconstructed_hardware(x,y,z,vx,vy,vz,valid,
                                  ring_inner_radius_m,throat_outer_radius_m,ring_outer_radius_m,
                                  through_bore_radius_m,counterbore_radius_m):
    """Exact hard-wall scorer for the reconstructed TARDIIS sensor-ring bore.

    Engineering-drawing default geometry (all configurable):
      - ring inner radius: 50.5 mm (101-mm ID),
      - ring outer radius: 65.5 mm (131-mm OD),
      - 4-mm-diameter through-bore from 50.5 to 57.5 mm radius,
      - 8.7-mm-diameter counterbore from 57.5 to 65.5 mm radius.

    A ray must pass the two end circles of the narrow through-bore and the outer
    counterbore opening. Because the transverse ray coordinate varies linearly
    with propagation distance and the Euclidean norm is convex, these endpoint
    checks are sufficient for the two cylindrical bore sections.

    This models the *mechanical optical channel*. The paper does not specify a
    photodiode active-area diameter or exact LED die plane, so V24.4 does not
    invent an additional sensor-area aperture.
    """
    p=np.column_stack([x,y,z]).astype(float,copy=False)
    v=np.column_stack([vx,vy,vz]).astype(float,copy=False)
    valid=np.asarray(valid,bool)
    best_t=np.full(p.shape[0],np.inf);best=np.full(p.shape[0],-1,dtype=np.int16)
    rth2=float(through_bore_radius_m)**2
    rcb2=float(counterbore_radius_m)**2
    for j,deg in enumerate(DETECTOR_ANGLES_DEG):
        th=np.deg2rad(deg);u=np.array([np.sin(th),np.cos(th),0.0])
        den=v@u;pdot=p@u
        good=valid & (den>1e-10)
        if not np.any(good): continue
        t0=np.full(p.shape[0],np.nan);t1=np.full(p.shape[0],np.nan);t2=np.full(p.shape[0],np.nan)
        t0[good]=(ring_inner_radius_m-pdot[good])/den[good]
        t1[good]=(throat_outer_radius_m-pdot[good])/den[good]
        t2[good]=(ring_outer_radius_m-pdot[good])/den[good]
        good &= (t0>0)&(t1>t0)&(t2>t1)
        if not np.any(good): continue
        ids=np.flatnonzero(good)
        q0=p[ids]+t0[ids,None]*v[ids]-ring_inner_radius_m*u[None,:]
        q1=p[ids]+t1[ids,None]*v[ids]-throat_outer_radius_m*u[None,:]
        q2=p[ids]+t2[ids,None]*v[ids]-ring_outer_radius_m*u[None,:]
        aperture=(np.einsum('ij,ij->i',q0,q0)<=rth2) & (np.einsum('ij,ij->i',q1,q1)<=rth2) & (np.einsum('ij,ij->i',q2,q2)<=rcb2)
        if not np.any(aperture): continue
        hit_ids=ids[aperture]
        upd=t0[hit_ids]<best_t[hit_ids]
        hit_ids=hit_ids[upd]
        best_t[hit_ids]=t0[hit_ids];best[hit_ids]=j
    hits=np.bincount(best[best>=0],minlength=DETECTOR_ANGLES_DEG.size).astype(np.int64)
    return hits,best


def detect_hardware_symmetry_averaged(x,y,z,vx,vy,vz,valid,
                                      ring_inner_radius_m,throat_outer_radius_m,ring_outer_radius_m,
                                      through_bore_radius_m,counterbore_radius_m):
    """Unbiased x-reflection symmetry average of the same hard hardware scorer."""
    native,native_id=detect_reconstructed_hardware(
        x,y,z,vx,vy,vz,valid,ring_inner_radius_m,throat_outer_radius_m,ring_outer_radius_m,
        through_bore_radius_m,counterbore_radius_m)
    mirror,mirror_id=detect_reconstructed_hardware(
        -np.asarray(x),y,z,-np.asarray(vx),vy,vz,valid,
        ring_inner_radius_m,throat_outer_radius_m,ring_outer_radius_m,
        through_bore_radius_m,counterbore_radius_m)
    score=0.5*(native.astype(np.float64)+mirror.astype(np.float64))
    return native,mirror,score,native_id,mirror_id


def detect_legacy_acceptance(x,y,valid,interaction_count,acceptance_deg=6.5):
    """Original-CLARITAS-style +/-acceptance exit-position channel scorer.

    This intentionally reproduces the old *comparison* semantics rather than the
    V24.4 physical detector: boundary exit-position angle, 0..170 degree detector
    centres, overlapping +/-6.5-degree windows, and no ballistic contribution to
    detector centres >=90 degrees. A ray may therefore contribute to more than
    one legacy channel. It is not a physical hit probability and is never used as
    V24.4 ``normalized_response``.
    """
    x=np.asarray(x,float);y=np.asarray(y,float);valid=np.asarray(valid,bool)
    ic=np.asarray(interaction_count)
    ang=(np.rad2deg(np.arctan2(x,y))+360.0)%360.0
    in_side=valid & (ang>=0.0) & (ang<=180.0)
    hits=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.int64)
    if not np.any(in_side): return hits
    a=ang[in_side]; interacted=ic[in_side]>0
    diffs=np.abs(a[:,None]-DETECTOR_ANGLES_DEG[None,:])
    mask=diffs<=float(acceptance_deg)
    mask[(~interacted)[:,None] & (DETECTOR_ANGLES_DEG[None,:]>=90.0)]=False
    return mask.sum(axis=0).astype(np.int64)


def detect_legacy_symmetry_averaged(x,y,valid,interaction_count,acceptance_deg=6.5):
    native=detect_legacy_acceptance(x,y,valid,interaction_count,acceptance_deg)
    mirror=detect_legacy_acceptance(-np.asarray(x),y,valid,interaction_count,acceptance_deg)
    return native,mirror,0.5*(native.astype(float)+mirror.astype(float))


def _histogram_median(hist):
    hist=np.asarray(hist,dtype=np.int64)
    n=int(hist.sum())
    if n<=0:return float('nan')
    cs=np.cumsum(hist)
    if n%2:
        return float(np.searchsorted(cs,n//2+1))
    a=int(np.searchsorted(cs,n//2)); b=int(np.searchsorted(cs,n//2+1))
    return 0.5*(a+b)


def _jackknife_normalized_se(batch_scores):
    """Batch-jackknife SE for the normalized 18-channel detector response."""
    if len(batch_scores)<3:
        return np.full(DETECTOR_ANGLES_DEG.size,np.nan,dtype=float)
    s=np.asarray(batch_scores,dtype=float)
    total=s.sum(axis=0)
    loo=[]
    for b in range(s.shape[0]):
        q=total-s[b]; den=q.sum()
        if den>0: loo.append(q/den)
    if len(loo)<3:return np.full(DETECTOR_ANGLES_DEG.size,np.nan,dtype=float)
    loo=np.asarray(loo); m=loo.mean(axis=0); B=loo.shape[0]
    return np.sqrt((B-1.0)/B*np.sum((loo-m)**2,axis=0))


@dataclass
class ForwardResult:
    detector_angles_deg: np.ndarray
    raw_hits: np.ndarray                       # compatibility alias: native reconstructed-hardware hits
    normalized_response: np.ndarray            # selected: symmetry-averaged reconstructed hardware
    native_exact_hits: np.ndarray              # compatibility alias: hardware native
    mirror_exact_hits: np.ndarray              # compatibility alias: hardware mirror
    symmetry_exact_scores: np.ndarray          # compatibility alias: hardware symmetry score
    normalized_native_exact_response: np.ndarray
    normalized_mirror_exact_response: np.ndarray
    normalized_symmetry_exact_response: np.ndarray
    normalized_response_jackknife_se: np.ndarray
    hardware_native_hits: np.ndarray
    hardware_mirror_hits: np.ndarray
    hardware_symmetry_scores: np.ndarray
    normalized_hardware_native_response: np.ndarray
    normalized_hardware_mirror_response: np.ndarray
    normalized_hardware_symmetry_response: np.ndarray
    hardware_normalized_jackknife_se: np.ndarray
    legacy_native_channel_counts: np.ndarray
    legacy_mirror_channel_counts: np.ndarray
    legacy_symmetry_channel_scores: np.ndarray
    normalized_legacy_native_response: np.ndarray
    normalized_legacy_mirror_response: np.ndarray
    normalized_legacy_symmetry_response: np.ndarray
    legacy_normalized_jackknife_se: np.ndarray
    legacy_acceptance_deg: float
    valid_exit_count: int
    n_rays: int
    requested_n_rays: int
    stopped_adaptively: bool
    stop_reason: str
    mu_geom_per_m: float
    optical_depth_diameter: float
    mean_interactions: float
    median_interactions: float
    ballistic_fraction: float
    mean_fresnel_reflections: float
    entry_reflection_fraction: float
    internal_reflection_fraction: float
    mean_cell_inner_reflections: float
    mean_cell_outer_reflections: float
    cell_tir_fraction: float
    source_collimator_acceptance_fraction: float
    entered_water_fraction: float
    air_exit_fraction: float
    axial_loss_fraction: float
    detector_detection_fraction: float         # selected symmetry-equivalent score / ray
    native_detector_detection_fraction: float
    mirror_detector_detection_fraction: float
    legacy_channel_score_per_ray: float
    symmetry_variance_reduction: bool
    batch_diagnostics: Optional[list]=None
    ray_data: Optional[Dict[str,np.ndarray]]=None
    heatmap_xy: Optional[np.ndarray]=None
    heatmap_zy: Optional[np.ndarray]=None
    def to_dict(self):
        d={k:v for k,v in self.__dict__.items() if k not in {'ray_data','heatmap_xy','heatmap_zy'}}
        for k,v in list(d.items()):
            if isinstance(v,np.ndarray): d[k]=v.tolist()
            elif isinstance(v,np.generic): d[k]=v.item()
        return d


class TardiisForwardModel:
    def __init__(self,n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,
                 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
                 water_height_m=0.426,sensor_height_above_bottom_m=0.102,
                 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,
                 counterbore_depth_m=0.008,through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,
                 source_launch_radius_m=0.0655,legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1.0e-5,
                 alpha1=1.0,alpha2=100.0,density_kg_per_m3=2600.0,
                 max_iterations=int(1e6),max_internal_bounces=64,max_cell_bounces=64,
                 chunk_size=250000,statistics_batch_rays=250000,
                 symmetry_average_exact=True):
        try: import cupy as cp
        except Exception as exc: raise RuntimeError('CuPy/CUDA is required for CLARITAS V24.4.') from exc
        self.cp=cp;self.n_water=float(n_water);self.n_particle=float(n_particle);self.n_acrylic=float(n_acrylic);self.n_air=float(n_air)
        self.rin=float(tube_inner_radius_m);self.rout=float(tube_outer_radius_m)
        self.water_height=float(water_height_m);self.sensor_height=float(sensor_height_above_bottom_m)
        self.zmin=-self.sensor_height;self.zmax=self.water_height-self.sensor_height
        self.ring_in=float(sensor_ring_inner_radius_m);self.ring_out=float(sensor_ring_outer_radius_m)
        self.counterbore_depth=float(counterbore_depth_m);self.throat_out=self.ring_out-self.counterbore_depth
        self.throat_radius=0.5*float(through_bore_diameter_m);self.counterbore_radius=0.5*float(counterbore_diameter_m)
        self.source_launch_radius=float(source_launch_radius_m)
        self.legacy_acceptance_deg=float(legacy_detector_acceptance_deg);self.beam_sigma=float(source_beam_sigma_m)
        self.alpha1=float(alpha1);self.alpha2=float(alpha2);self.density=float(density_kg_per_m3)
        self.max_iterations=int(max_iterations);self.max_internal_bounces=int(max_internal_bounces);self.max_cell_bounces=int(max_cell_bounces)
        self.chunk_size=int(chunk_size);self.statistics_batch_rays=int(statistics_batch_rays)
        self.symmetry_average_exact=bool(symmetry_average_exact)
        if self.chunk_size<1 or self.statistics_batch_rays<1: raise ValueError('chunk/batch sizes must be positive')
        if not (0<self.rin<self.rout<self.ring_in<self.throat_out<=self.source_launch_radius<=self.ring_out): raise ValueError('TARDIIS radial/source geometry is inconsistent')
        if not (0<self.throat_radius<self.counterbore_radius): raise ValueError('through-bore must be narrower than counterbore')
        if not (0.0<self.legacy_acceptance_deg<90.0): raise ValueError('legacy detector acceptance must be between 0 and 90 degrees')
        if not (self.zmin<0<self.zmax): raise ValueError('sensor plane must lie within water column')
        self.module=cp.RawModule(code=CUDA_SRC,options=('-std=c++11',));self.kernel=self.module.get_function('trace_kernel')

    def simulate(self,material,concentration_g_per_L,weights=None,n_rays=100000,seed=12345,
                 weight_mode='mass_fraction',collect_rays=False,heatmap_size=0,
                 target_detector_score=None,min_rays=None,max_rays=None,
                 stability_l1_tolerance=None,stability_window=2):
        """Run V24 cell/particle transport with V24.4 reconstructed hardware scoring.

        Fixed-statistics mode (default): exactly ``n_rays`` are traced.
        Adaptive mode: set ``target_detector_score``. The run continues in
        statistics batches until the symmetry-equivalent exact detector score
        reaches the requested target (and optional response-stability condition),
        or ``max_rays`` is reached.

        The selected response uses only the reconstructed hard mechanical bore.
        The legacy +/-6.5-degree channel is exported for comparison only.
        No KDE / widened physical detector / extrapolation is used.
        """
        cp=self.cp;d,bw=get_material_psd(material);w=bw if weights is None else np.asarray(weights,float)
        if w.size!=d.size: raise ValueError(f'{material} requires {d.size} PSD weights')
        tr=build_geometric_transport(d,w,concentration_g_per_L,self.density,weight_mode);mu=tr['mu_geom'];cdf=tr['particle_event_cdf'];radii=tr['radii_m']
        nidx=np.full_like(radii,self.n_particle)
        requested=int(n_rays)
        adaptive=target_detector_score is not None
        if adaptive:
            target=float(target_detector_score)
            if target<=0: raise ValueError('target_detector_score must be >0')
            maxN=int(max_rays if max_rays is not None else max(requested,self.statistics_batch_rays))
            minN=int(min_rays if min_rays is not None else min(self.statistics_batch_rays,maxN))
            if minN<0 or maxN<minN: raise ValueError('invalid min_rays/max_rays')
        else:
            target=None;maxN=requested;minN=requested
        if maxN<1: raise ValueError('number of rays must be positive')

        # Separate deterministic source streams make common-random-number comparisons
        # stable for a fixed seed and sampling schedule, without storing all ray outputs.
        ss=np.random.SeedSequence(int(seed)); sp,sa=ss.spawn(2)
        rng_p=np.random.default_rng(sp); rng_a=np.random.default_rng(sa)
        cdfd=cp.asarray(cdf,dtype=cp.float64);nd=cp.asarray(nidx,dtype=cp.float64);rd=cp.asarray(radii,dtype=cp.float64)
        vis=int(heatmap_size);hxy=cp.zeros(max(1,vis*vis),dtype=cp.float32);hzy=cp.zeros(max(1,vis*vis),dtype=cp.float32)
        seed0=np.uint32((int(seed)*1664525+1013904223)&0x7fffffff or 1);seed1=np.uint32((int(seed)*22695477+1)&0x7fffffff or 7)
        threads=256

        native_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.int64)
        mirror_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.int64)
        sym_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.float64)
        legacy_native_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.int64)
        legacy_mirror_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.int64)
        legacy_sym_total=np.zeros(DETECTOR_ANGLES_DEG.size,dtype=np.float64)
        batch_scores=[];legacy_batch_scores=[];batch_rows=[];prev_cum=None;recent_l1=[]
        total_rays=0; valid_air_total=0; entered_total=0; source_accept_total=0; axial_total=0
        sum_ic=0.0; ballistic_total=0; ic_hist=np.zeros(1,dtype=np.int64)
        sum_pr=0.0; rays_entry_pr=0; rays_internal_pr=0;sum_cir=0.0;sum_cor=0.0;rays_ctir=0
        ray_chunks=[] if collect_rays else None
        stop_reason='fixed_n_rays'

        while total_rays<maxN:
            batchN=min(self.statistics_batch_rays,maxN-total_rays)
            polar_batch=rng_p.beta(self.alpha1,self.alpha2,batchN)*(np.pi/2.0)
            az_batch=rng_a.uniform(0.0,2.0*np.pi,batchN)
            batch_native=np.zeros_like(native_total);batch_mirror=np.zeros_like(mirror_total)
            batch_legacy_native=np.zeros_like(legacy_native_total);batch_legacy_mirror=np.zeros_like(legacy_mirror_total)
            batch_start=total_rays

            for local_start in range(0,batchN,self.chunk_size):
                local_end=min(batchN,local_start+self.chunk_size);n=local_end-local_start;global_start=batch_start+local_start
                pd=cp.asarray(polar_batch[local_start:local_end],dtype=cp.float64)
                ad=cp.asarray(az_batch[local_start:local_end],dtype=cp.float64)
                outf=[cp.full(n,cp.nan,dtype=cp.float32) for _ in range(8)]
                outi=[cp.zeros(n,dtype=cp.int32) for _ in range(9)]
                blocks=(n+threads-1)//threads
                self.kernel((blocks,),(threads,),(
                    np.float32(self.max_iterations),np.float32(mu),np.float32(self.n_water),np.float32(self.n_acrylic),np.float32(self.n_air),
                    np.int32(self.max_internal_bounces),np.int32(self.max_cell_bounces),np.float32(self.rin),np.float32(self.rout),np.float32(self.zmin),np.float32(self.zmax),
                    np.float32(self.ring_in),np.float32(self.throat_out),np.float32(self.ring_out),np.float32(self.source_launch_radius),
                    np.float32(self.throat_radius),np.float32(self.counterbore_radius),np.float32(self.beam_sigma),np.int32(vis),
                    pd,ad,np.int32(n),cdfd,nd,rd,np.int32(cdf.size),hxy,hzy,
                    *outf,*outi,seed0,seed1,np.uint32(global_start)))
                cp.cuda.Stream.null.synchronize()
                af=[cp.asnumpy(a) for a in outf]; ai=[cp.asnumpy(a) for a in outi]
                arrays=dict(zip(['air_x','air_y','air_z','air_vx','air_vy','air_vz','water_path','acrylic_path'],af))
                ints=dict(zip(['interaction_count','particle_reflections','particle_entry_reflections','particle_internal_reflections','cell_inner_reflections','cell_outer_reflections','cell_tir','entered_water','status'],ai))
                valid_air=(ints['status']==1)&np.isfinite(arrays['air_x'])&np.isfinite(arrays['air_vx'])
                nat,mir,sym,nid,mid=detect_hardware_symmetry_averaged(
                    arrays['air_x'],arrays['air_y'],arrays['air_z'],arrays['air_vx'],arrays['air_vy'],arrays['air_vz'],valid_air,
                    self.ring_in,self.throat_out,self.ring_out,self.throat_radius,self.counterbore_radius)
                lnat,lmir,lsym=detect_legacy_symmetry_averaged(
                    arrays['air_x'],arrays['air_y'],valid_air,ints['interaction_count'],self.legacy_acceptance_deg)
                batch_native+=nat;batch_mirror+=mir
                batch_legacy_native+=lnat;batch_legacy_mirror+=lmir

                entered=ints['entered_water']>0; ic=ints['interaction_count'][entered]
                valid_air_total+=int(valid_air.sum());entered_total+=int(entered.sum());source_accept_total+=int(np.count_nonzero(ints['status']!=3));axial_total+=int(np.count_nonzero(ints['status']==2))
                if ic.size:
                    sum_ic+=float(ic.sum());ballistic_total+=int(np.count_nonzero(ic==0))
                    bc=np.bincount(ic.astype(np.int64));
                    if bc.size>ic_hist.size: ic_hist=np.pad(ic_hist,(0,bc.size-ic_hist.size))
                    ic_hist[:bc.size]+=bc
                    pr=ints['particle_reflections'][entered];per=ints['particle_entry_reflections'][entered];pir=ints['particle_internal_reflections'][entered]
                    cir=ints['cell_inner_reflections'][entered];cor=ints['cell_outer_reflections'][entered];ct=ints['cell_tir'][entered]
                    sum_pr+=float(pr.sum());rays_entry_pr+=int(np.count_nonzero(per>0));rays_internal_pr+=int(np.count_nonzero(pir>0));sum_cir+=float(cir.sum());sum_cor+=float(cor.sum());rays_ctir+=int(np.count_nonzero(ct>0))
                if collect_rays:
                    ray_chunks.append({**arrays,**ints,'native_detector_index':nid,'mirror_detector_index':mid})
                del pd,ad,outf,outi,af,ai

            total_rays+=batchN
            native_total+=batch_native;mirror_total+=batch_mirror
            legacy_native_total+=batch_legacy_native;legacy_mirror_total+=batch_legacy_mirror
            batch_sym=0.5*(batch_native.astype(float)+batch_mirror.astype(float)) if self.symmetry_average_exact else batch_native.astype(float)
            batch_legacy_sym=0.5*(batch_legacy_native.astype(float)+batch_legacy_mirror.astype(float)) if self.symmetry_average_exact else batch_legacy_native.astype(float)
            sym_total+=batch_sym;legacy_sym_total+=batch_legacy_sym
            batch_scores.append(batch_sym.copy());legacy_batch_scores.append(batch_legacy_sym.copy())
            den=float(sym_total.sum());cum=sym_total/den if den>0 else np.zeros_like(sym_total)
            l1=float(np.sum(np.abs(cum-prev_cum))) if prev_cum is not None else float('nan')
            if np.isfinite(l1): recent_l1.append(l1)
            prev_cum=cum.copy()
            batch_rows.append({'batch':len(batch_rows)+1,'batch_rays':batchN,'cumulative_rays':total_rays,
                               'batch_native_hits':int(batch_native.sum()),'batch_mirror_hits':int(batch_mirror.sum()),
                               'batch_symmetry_score':float(batch_sym.sum()),'cumulative_symmetry_score':float(sym_total.sum()),
                               'batch_legacy_channel_score':float(batch_legacy_sym.sum()),'cumulative_legacy_channel_score':float(legacy_sym_total.sum()),
                               'cumulative_l1_change':l1})

            if adaptive and total_rays>=minN and sym_total.sum()>=target:
                stable=True
                if stability_l1_tolerance is not None:
                    wdw=max(1,int(stability_window));stable=len(recent_l1)>=wdw and all(q<=float(stability_l1_tolerance) for q in recent_l1[-wdw:])
                if stable:
                    stop_reason='target_detector_score_and_stability' if stability_l1_tolerance is not None else 'target_detector_score'
                    break
        if adaptive and total_rays>=maxN and sym_total.sum()<target: stop_reason='max_rays_before_target'
        elif adaptive and total_rays>=maxN and stop_reason.startswith('fixed'): stop_reason='max_rays'

        nsum=float(native_total.sum());msum=float(mirror_total.sum());ssum=float(sym_total.sum())
        nn=native_total.astype(float)/nsum if nsum>0 else np.zeros_like(sym_total)
        nm=mirror_total.astype(float)/msum if msum>0 else np.zeros_like(sym_total)
        ns=sym_total/ssum if ssum>0 else np.zeros_like(sym_total)
        lnsum=float(legacy_native_total.sum());lmsum=float(legacy_mirror_total.sum());lssum=float(legacy_sym_total.sum())
        lnn=legacy_native_total.astype(float)/lnsum if lnsum>0 else np.zeros_like(legacy_sym_total)
        lnm=legacy_mirror_total.astype(float)/lmsum if lmsum>0 else np.zeros_like(legacy_sym_total)
        lns=legacy_sym_total/lssum if lssum>0 else np.zeros_like(legacy_sym_total)
        se=_jackknife_normalized_se(batch_scores);legacy_se=_jackknife_normalized_se(legacy_batch_scores)
        denom_enter=max(entered_total,1)
        ray_data=None
        if collect_rays and ray_chunks:
            keys=ray_chunks[0].keys();ray_data={k:np.concatenate([q[k] for q in ray_chunks]) for k in keys}
            ni=ray_data['native_detector_index']; mi=ray_data['mirror_detector_index']
            ray_data['native_detector_deg']=np.where(ni>=0,DETECTOR_ANGLES_DEG[np.clip(ni,0,len(DETECTOR_ANGLES_DEG)-1)],np.nan)
            ray_data['mirror_detector_deg']=np.where(mi>=0,DETECTOR_ANGLES_DEG[np.clip(mi,0,len(DETECTOR_ANGLES_DEG)-1)],np.nan)
        hmxy=cp.asnumpy(hxy).reshape(vis,vis) if vis>0 else None;hmzy=cp.asnumpy(hzy).reshape(vis,vis) if vis>0 else None
        return ForwardResult(
            detector_angles_deg=DETECTOR_ANGLES_DEG.copy(),raw_hits=native_total.copy(),normalized_response=ns,
            native_exact_hits=native_total.copy(),mirror_exact_hits=mirror_total.copy(),symmetry_exact_scores=sym_total.copy(),
            normalized_native_exact_response=nn,normalized_mirror_exact_response=nm,normalized_symmetry_exact_response=ns,
            normalized_response_jackknife_se=se,
            hardware_native_hits=native_total.copy(),hardware_mirror_hits=mirror_total.copy(),hardware_symmetry_scores=sym_total.copy(),
            normalized_hardware_native_response=nn,normalized_hardware_mirror_response=nm,normalized_hardware_symmetry_response=ns,
            hardware_normalized_jackknife_se=se,
            legacy_native_channel_counts=legacy_native_total.copy(),legacy_mirror_channel_counts=legacy_mirror_total.copy(),legacy_symmetry_channel_scores=legacy_sym_total.copy(),
            normalized_legacy_native_response=lnn,normalized_legacy_mirror_response=lnm,normalized_legacy_symmetry_response=lns,
            legacy_normalized_jackknife_se=legacy_se,legacy_acceptance_deg=self.legacy_acceptance_deg,
            valid_exit_count=valid_air_total,n_rays=total_rays,requested_n_rays=requested,
            stopped_adaptively=bool(adaptive and total_rays<maxN),stop_reason=stop_reason,mu_geom_per_m=mu,optical_depth_diameter=mu*(2*self.rin),
            mean_interactions=sum_ic/denom_enter if entered_total else float('nan'),median_interactions=_histogram_median(ic_hist),
            ballistic_fraction=ballistic_total/denom_enter if entered_total else float('nan'),mean_fresnel_reflections=sum_pr/denom_enter if entered_total else float('nan'),
            entry_reflection_fraction=rays_entry_pr/denom_enter if entered_total else float('nan'),internal_reflection_fraction=rays_internal_pr/denom_enter if entered_total else float('nan'),
            mean_cell_inner_reflections=sum_cir/denom_enter if entered_total else float('nan'),mean_cell_outer_reflections=sum_cor/denom_enter if entered_total else float('nan'),
            cell_tir_fraction=rays_ctir/denom_enter if entered_total else float('nan'),source_collimator_acceptance_fraction=source_accept_total/max(total_rays,1),
            entered_water_fraction=entered_total/max(total_rays,1),air_exit_fraction=valid_air_total/max(total_rays,1),axial_loss_fraction=axial_total/max(total_rays,1),
            detector_detection_fraction=ssum/max(total_rays,1),native_detector_detection_fraction=nsum/max(total_rays,1),mirror_detector_detection_fraction=msum/max(total_rays,1),
            legacy_channel_score_per_ray=lssum/max(total_rays,1),symmetry_variance_reduction=self.symmetry_average_exact,batch_diagnostics=batch_rows,ray_data=ray_data,heatmap_xy=hmxy,heatmap_zy=hmzy)
