#!/usr/bin/env python3
"""Shared CLARITAS V24.1 / TARDIIS apparatus forward model.

V24.1 changes detector *estimation*, not optical physics.  In addition to the
exact physical two-aperture hit test, it provides a variance-reduced 4-D
phase-space kernel estimator of detector irradiance.  The kernel operates only
after the complete physical ray has exited the acrylic cell into air; it does
not alter source rays, particle encounters, Snell/Fresnel decisions, wall
reflections, or ray directions.

Physics:
- fully 3-D ray state;
- geometric particle encounter cross section pi*r^2 (no Mie);
- full Snell + unpolarised Fresnel physics at spherical particles;
- real TARDIIS cylindrical water/acrylic/air side-wall geometry;
- source and detectors constrained by the physical radial collimator apertures;
- exact detector hit counts retained for validation;
- variance-reduced detector scoring from a product Epanechnikov kernel in the
  two transverse aperture planes, normalised to the real 4-mm aperture etendue;
- multiple acrylic-wall reflections and TIR are followed explicitly.

The sensor-ring / cell dimensions are based on the supplied 2019 TARDIIS design
paper. The default PMMA refractive index is configurable and is not fitted by the
PSD inference program.
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
    const float COLL_IN_R,const float COLL_OUT_R,const float APERTURE_R,const float BEAM_SIGMA,
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

    // LED module is at 180 degrees. Start at the outer end of its radial collimator.
    float u1=fmaxf(rnd_uniform(&state),1e-12f),u2=rnd_uniform(&state);
    float mag=sqrtf(-2.0f*logf(u1));
    float x=BEAM_SIGMA*mag*cosf(2.0f*PI_F*u2);
    float z=BEAM_SIGMA*mag*sinf(2.0f*PI_F*u2);
    float y=-COLL_OUT_R;
    float polar=(float)polar_init[tid],az=(float)azimuth_init[tid];
    float sp=sinf(polar),vx=sp*cosf(az),vy=cosf(polar),vz=sp*sinf(az);normalize3(&vx,&vy,&vz);
    if(vy<=1e-12f){st=3;goto WRITE_OUT;}

    // Physical source collimator: require passage through the inner 4-mm aperture.
    {
        float t=(-COLL_IN_R-y)/vy;
        if(t<=0.0f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        if(x*x+z*z>APERTURE_R*APERTURE_R){st=3;goto WRITE_OUT;}
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


def score_physical_apertures_variance_reduced(
    x, y, z, vx, vy, vz, valid,
    inner_plane_radius_m, outer_plane_radius_m, aperture_radius_m,
    bandwidth_factor=3.0,
):
    """Score the 18 physical TARDIIS collimators with exact and VR estimators.

    Exact scoring requires a ray to pass through both circular apertures.

    The variance-reduced score is a 4-D local phase-space density estimate.
    For detector j, q1 and q2 are the two transverse miss vectors where the
    external ray intersects the inner and outer aperture planes.  A product of
    two 2-D Epanechnikov kernels is evaluated using bandwidth h >= a, where a
    is the real aperture radius::

        K = max(0, 1-|q1|^2/h^2) * max(0, 1-|q2|^2/h^2)

    For locally uniform radiance, the product-kernel integral is
    (pi*h^2/2)^2 whereas the real two-aperture acceptance volume is
    (pi*a^2)^2.  Therefore

        physical_equivalent_score = sum(K) * 4 * (a/h)^4

    is on the scale of an expected exact physical hit count.  The estimator is
    consistent as ray count increases and deliberately trades a small local
    smoothing bias for a very large variance reduction.  No ray trajectory or
    optical probability is modified.

    A fixed bandwidth is used for all detector angles and all optimiser
    evaluations; this is important for a smooth common-random-number inverse
    objective.
    """
    p = np.column_stack([x, y, z]).astype(np.float64, copy=False)
    v = np.column_stack([vx, vy, vz]).astype(np.float64, copy=False)
    valid = np.asarray(valid, dtype=bool)
    a = float(aperture_radius_m)
    bf = float(bandwidth_factor)
    if not np.isfinite(bf) or bf < 1.0:
        raise ValueError('detector VR bandwidth_factor must be >= 1.0')
    h = a * bf
    a2 = a*a
    h2 = h*h

    n_det = DETECTOR_ANGLES_DEG.size
    exact_hits = np.zeros(n_det, dtype=np.int64)
    kernel_sum = np.zeros(n_det, dtype=np.float64)
    support_counts = np.zeros(n_det, dtype=np.int64)

    # Exact physical detector assignment remains exclusive (at most one detector
    # per ray), matching V24.  Kernel density scores are estimators evaluated at
    # every detector centre and therefore need not be exclusive.
    best_t = np.full(p.shape[0], np.inf, dtype=np.float64)
    best = np.full(p.shape[0], -1, dtype=np.int16)

    for j, deg in enumerate(DETECTOR_ANGLES_DEG):
        th = np.deg2rad(deg)
        u = np.array([np.sin(th), np.cos(th), 0.0], dtype=np.float64)
        den = v @ u
        pdot = p @ u
        good = valid & (den > 1.0e-10)
        if not np.any(good):
            continue

        t1 = np.full(p.shape[0], np.nan, dtype=np.float64)
        t2 = np.full(p.shape[0], np.nan, dtype=np.float64)
        t1[good] = (float(inner_plane_radius_m) - pdot[good]) / den[good]
        t2[good] = (float(outer_plane_radius_m) - pdot[good]) / den[good]
        good &= (t1 > 0.0) & (t2 > t1)
        ids = np.flatnonzero(good)
        if ids.size == 0:
            continue

        q1 = p[ids] + t1[ids, None]*v[ids] - float(inner_plane_radius_m)*u[None, :]
        q2 = p[ids] + t2[ids, None]*v[ids] - float(outer_plane_radius_m)*u[None, :]
        r1sq = np.einsum('ij,ij->i', q1, q1)
        r2sq = np.einsum('ij,ij->i', q2, q2)

        ex = (r1sq <= a2) & (r2sq <= a2)
        if np.any(ex):
            eid = ids[ex]
            upd = t1[eid] < best_t[eid]
            if np.any(upd):
                chosen = eid[upd]
                best_t[chosen] = t1[chosen]
                best[chosen] = j

        sup = (r1sq < h2) & (r2sq < h2)
        support_counts[j] = int(np.count_nonzero(sup))
        if np.any(sup):
            k1 = 1.0 - r1sq[sup]/h2
            k2 = 1.0 - r2sq[sup]/h2
            kernel_sum[j] = float(np.sum(k1*k2, dtype=np.float64))

    if np.any(best >= 0):
        exact_hits = np.bincount(best[best >= 0], minlength=n_det).astype(np.int64)

    # Product 2-D Epanechnikov normalisation to the physical two-aperture
    # phase-space acceptance.  This common geometry scale preserves absolute
    # comparability with expected exact hit counts as well as angular shape.
    scale = 4.0 * (a/h)**4
    physical_equivalent = kernel_sum * scale
    return exact_hits, best, physical_equivalent, kernel_sum, support_counts


def detect_physical_apertures(x,y,z,vx,vy,vz,valid,inner_plane_radius_m,outer_plane_radius_m,aperture_radius_m):
    """Compatibility wrapper returning the exact V24 binary-aperture result."""
    hits, best, _, _, _ = score_physical_apertures_variance_reduced(
        x,y,z,vx,vy,vz,valid,
        inner_plane_radius_m,outer_plane_radius_m,aperture_radius_m,
        bandwidth_factor=1.0,
    )
    return hits, best


@dataclass
class ForwardResult:
    detector_angles_deg: np.ndarray
    raw_hits: np.ndarray                         # exact physical binary hits
    detector_scores: np.ndarray                  # selected scoring estimator
    normalized_response: np.ndarray              # selected scoring estimator
    normalized_exact_response: np.ndarray
    normalized_variance_reduced_response: np.ndarray
    variance_reduced_physical_equivalent_scores: np.ndarray
    variance_reduced_kernel_sums: np.ndarray
    variance_reduced_support_counts: np.ndarray
    detector_scoring_mode: str
    detector_vr_bandwidth_factor: float
    valid_exit_count: int
    n_rays: int
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
    detector_detection_fraction: float          # exact hits / launched rays
    detector_vr_equivalent_fraction: float       # sum VR eq. scores / launched rays
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
                 collimator_inner_plane_radius_m=0.0505,collimator_outer_plane_radius_m=0.0595,
                 collimator_diameter_m=0.004,source_beam_sigma_m=1.0e-5,
                 alpha1=1.0,alpha2=100.0,density_kg_per_m3=2600.0,
                 max_iterations=int(1e6),max_internal_bounces=64,max_cell_bounces=64,chunk_size=250000,
                 detector_scoring_mode='variance_reduced',detector_vr_bandwidth_factor=3.0):
        try: import cupy as cp
        except Exception as exc: raise RuntimeError('CuPy/CUDA is required for CLARITAS V24.') from exc
        self.cp=cp;self.n_water=float(n_water);self.n_particle=float(n_particle);self.n_acrylic=float(n_acrylic);self.n_air=float(n_air)
        self.rin=float(tube_inner_radius_m);self.rout=float(tube_outer_radius_m)
        self.water_height=float(water_height_m);self.sensor_height=float(sensor_height_above_bottom_m)
        self.zmin=-self.sensor_height;self.zmax=self.water_height-self.sensor_height
        self.coll_in=float(collimator_inner_plane_radius_m);self.coll_out=float(collimator_outer_plane_radius_m)
        self.aperture_radius=0.5*float(collimator_diameter_m);self.beam_sigma=float(source_beam_sigma_m)
        self.alpha1=float(alpha1);self.alpha2=float(alpha2);self.density=float(density_kg_per_m3)
        self.max_iterations=int(max_iterations);self.max_internal_bounces=int(max_internal_bounces);self.max_cell_bounces=int(max_cell_bounces)
        self.chunk_size=int(chunk_size)
        self.detector_scoring_mode=str(detector_scoring_mode).strip().lower()
        if self.detector_scoring_mode not in {'variance_reduced','exact'}:
            raise ValueError("detector_scoring_mode must be 'variance_reduced' or 'exact'")
        self.detector_vr_bandwidth_factor=float(detector_vr_bandwidth_factor)
        if self.detector_vr_bandwidth_factor < 1.0:
            raise ValueError('detector_vr_bandwidth_factor must be >= 1.0')
        if not (0<self.rin<self.rout<self.coll_in<self.coll_out): raise ValueError('TARDIIS radial geometry is inconsistent')
        if not (self.zmin<0<self.zmax): raise ValueError('sensor plane must lie within water column')
        self.module=cp.RawModule(code=CUDA_SRC,options=('-std=c++11',));self.kernel=self.module.get_function('trace_kernel')

    def simulate(self,material,concentration_g_per_L,weights=None,n_rays=100000,seed=12345,weight_mode='mass_fraction',collect_rays=False,heatmap_size=0):
        cp=self.cp;d,bw=get_material_psd(material);w=bw if weights is None else np.asarray(weights,float)
        if w.size!=d.size: raise ValueError(f'{material} requires {d.size} PSD weights')
        tr=build_geometric_transport(d,w,concentration_g_per_L,self.density,weight_mode);mu=tr['mu_geom'];cdf=tr['particle_event_cdf'];radii=tr['radii_m']
        nidx=np.full_like(radii,self.n_particle)
        rng=np.random.default_rng(int(seed));N=int(n_rays)
        polar=rng.beta(self.alpha1,self.alpha2,N)*(np.pi/2.0);az=rng.uniform(0,2*np.pi,N)
        cdfd=cp.asarray(cdf,dtype=cp.float64);nd=cp.asarray(nidx,dtype=cp.float64);rd=cp.asarray(radii,dtype=cp.float64)
        vis=int(heatmap_size);hxy=cp.zeros(max(1,vis*vis),dtype=cp.float32);hzy=cp.zeros(max(1,vis*vis),dtype=cp.float32)
        arrays={k:np.full(N,np.nan,dtype=np.float32) for k in ['air_x','air_y','air_z','air_vx','air_vy','air_vz','water_path','acrylic_path']}
        ints={k:np.zeros(N,dtype=np.int32) for k in ['interaction_count','particle_reflections','particle_entry_reflections','particle_internal_reflections','cell_inner_reflections','cell_outer_reflections','cell_tir','entered_water','status']}
        seed0=np.uint32((int(seed)*1664525+1013904223)&0x7fffffff or 1);seed1=np.uint32((int(seed)*22695477+1)&0x7fffffff or 7)
        threads=256
        for start in range(0,N,self.chunk_size):
            end=min(N,start+self.chunk_size);n=end-start;pd=cp.asarray(polar[start:end],dtype=cp.float64);ad=cp.asarray(az[start:end],dtype=cp.float64)
            outf=[cp.full(n,cp.nan,dtype=cp.float32) for _ in range(8)];outi=[cp.zeros(n,dtype=cp.int32) for _ in range(9)]
            blocks=(n+threads-1)//threads
            self.kernel((blocks,),(threads,),(
                np.float32(self.max_iterations),np.float32(mu),np.float32(self.n_water),np.float32(self.n_acrylic),np.float32(self.n_air),
                np.int32(self.max_internal_bounces),np.int32(self.max_cell_bounces),np.float32(self.rin),np.float32(self.rout),np.float32(self.zmin),np.float32(self.zmax),
                np.float32(self.coll_in),np.float32(self.coll_out),np.float32(self.aperture_radius),np.float32(self.beam_sigma),np.int32(vis),
                pd,ad,np.int32(n),cdfd,nd,rd,np.int32(cdf.size),hxy,hzy,
                *outf,*outi,seed0,seed1,np.uint32(start)))
            cp.cuda.Stream.null.synchronize()
            for key,a in zip(arrays,outf): arrays[key][start:end]=cp.asnumpy(a)
            for key,a in zip(ints,outi): ints[key][start:end]=cp.asnumpy(a)
            del pd,ad,outf,outi
        valid_air=(ints['status']==1)&np.isfinite(arrays['air_x'])&np.isfinite(arrays['air_vx'])
        hits,det_id,vr_eq,vr_kernel,vr_support=score_physical_apertures_variance_reduced(
            arrays['air_x'],arrays['air_y'],arrays['air_z'],arrays['air_vx'],arrays['air_vy'],arrays['air_vz'],
            valid_air,self.coll_in,self.coll_out,self.aperture_radius,self.detector_vr_bandwidth_factor)
        exact_total=int(hits.sum())
        exact_norm=hits.astype(np.float64)/exact_total if exact_total>0 else np.zeros_like(hits,dtype=np.float64)
        vr_total=float(vr_eq.sum())
        vr_norm=vr_eq/vr_total if vr_total>0 else np.zeros_like(vr_eq,dtype=np.float64)
        if self.detector_scoring_mode == 'exact':
            selected_scores=hits.astype(np.float64)
            norm=exact_norm
        else:
            selected_scores=vr_eq.astype(np.float64,copy=False)
            norm=vr_norm
        entered=ints['entered_water']>0;den=max(int(entered.sum()),1);ic=ints['interaction_count'][entered]
        ray_data=None
        if collect_rays:
            ray_data={**arrays,**ints,'detector_index_exact':det_id,'detector_deg_exact':np.where(det_id>=0,DETECTOR_ANGLES_DEG[np.clip(det_id,0,len(DETECTOR_ANGLES_DEG)-1)],np.nan)}
        hmxy=cp.asnumpy(hxy).reshape(vis,vis) if vis>0 else None;hmzy=cp.asnumpy(hzy).reshape(vis,vis) if vis>0 else None
        return ForwardResult(
            detector_angles_deg=DETECTOR_ANGLES_DEG.copy(),raw_hits=hits,
            detector_scores=selected_scores,normalized_response=norm,
            normalized_exact_response=exact_norm,normalized_variance_reduced_response=vr_norm,
            variance_reduced_physical_equivalent_scores=vr_eq,
            variance_reduced_kernel_sums=vr_kernel,variance_reduced_support_counts=vr_support,
            detector_scoring_mode=self.detector_scoring_mode,
            detector_vr_bandwidth_factor=self.detector_vr_bandwidth_factor,
            valid_exit_count=int(valid_air.sum()),n_rays=N,mu_geom_per_m=mu,optical_depth_diameter=mu*(2*self.rin),
            mean_interactions=float(np.mean(ic)) if ic.size else float('nan'),median_interactions=float(np.median(ic)) if ic.size else float('nan'),
            ballistic_fraction=float(np.mean(ic==0)) if ic.size else float('nan'),
            mean_fresnel_reflections=float(np.mean(ints['particle_reflections'][entered])) if ic.size else float('nan'),
            entry_reflection_fraction=float(np.mean(ints['particle_entry_reflections'][entered]>0)) if ic.size else float('nan'),
            internal_reflection_fraction=float(np.mean(ints['particle_internal_reflections'][entered]>0)) if ic.size else float('nan'),
            mean_cell_inner_reflections=float(np.mean(ints['cell_inner_reflections'][entered])) if ic.size else float('nan'),
            mean_cell_outer_reflections=float(np.mean(ints['cell_outer_reflections'][entered])) if ic.size else float('nan'),
            cell_tir_fraction=float(np.mean(ints['cell_tir'][entered]>0)) if ic.size else float('nan'),
            source_collimator_acceptance_fraction=float(np.mean(ints['status']!=3)),entered_water_fraction=float(np.mean(entered)),
            air_exit_fraction=float(np.mean(valid_air)),axial_loss_fraction=float(np.mean(ints['status']==2)),
            detector_detection_fraction=float(exact_total/N),detector_vr_equivalent_fraction=float(vr_total/N),
            ray_data=ray_data,heatmap_xy=hmxy,heatmap_zy=hmzy)
