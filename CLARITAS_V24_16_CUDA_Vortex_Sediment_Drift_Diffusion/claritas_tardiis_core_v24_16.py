#!/usr/bin/env python3
"""Shared CLARITAS V24.16 / TARDIIS CUDA-authoritative vortex sediment-transport diagnostic.

V24.16 retains the DGB-calibrated source/apparatus state and the independently
validated V24.15 localized, volume-preserving Scully free surface.  The new
ray-determining mechanism is a process-based, size-resolved sediment field for
the continuously stirred suspension.

For each PSD bin the model balances two particle drift processes against an
eddy-diffusive representation of vertical/radial stirring and recirculation:

    radial inertial drift:   w_r = tau_p * v_theta(r)^2 / r
    gravitational settling: w_s = (rho_p-rho_w) g d^2 / (18 mu)

with Stokes response time tau_p=rho_p*d^2/(18*mu) and the V24.15 Scully
velocity ansatz v_theta=Omega_c*r/(1+(r/a)^2).  The steady zero-flux
radial/vertical drift-diffusion solution has the separable local factor

    F_i(r,z) = exp[A_i r^2/(a^2+r^2) - B_i (z-z_bottom)],

where A_i=tau_i*Omega_c^2*a^2/(2*D_t) and B_i=w_s,i/D_t.  Every size bin is
volume-normalised over the actual curved cell on CUDA, so its total mass is
preserved exactly up to quadrature precision.  No material-specific or
concentration-specific optical/scattering multiplier is introduced.

Spatially varying event rates are traced with CUDA delta/Woodcock tracking
using a rigorously constructed per-bin majorant.  Candidate interactions are
accepted from the local size-resolved extinction/event field, then the local
particle-size distribution is sampled from that same field.  The historical
uniform-suspension branch remains explicit so it retains the V24.15 random
stream and provides a clean regression control.

Python performs configuration, orchestration, diagnostics aggregation and I/O;
Mie/event/source/free-surface/sediment-field/ray/detector calculations that
change ray outcomes are CUDA-resident.
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

# ============================ V24.5 COMPLEX PARTICLE OPTICS ============================
# Explicit per-bin complex refractive-index profiles at 622 nm.
# These are exploratory hypotheses, NOT measured optical constants for the user's samples.
# They exist to test whether size-dependent n(d)+i*k(d) can materially change the detector response.
#
# Loess profile: smooth effective-mineral mixture that is clay/iron richer in the fine fraction
# and quartz/feldspar richer in the coarse fraction. Endmember choices are documented in the README.
# Every one of the 38 loess PSD bins receives its own n and k value.
DEFAULT_LOESS_N_REAL = np.array([1.607451331, 1.605504695, 1.603586747, 1.601692366, 1.599827047, 1.597981118, 1.596160167, 1.594367874, 1.592599298, 1.590855346, 1.589138183, 1.587447487, 1.585783658, 1.584147841, 1.582538755, 1.580958153, 1.579405657, 1.577883002, 1.576389253, 1.57492664, 1.573494964, 1.572095154, 1.570728147, 1.569395256, 1.568097028, 1.56683506, 1.565610853, 1.564425894, 1.563282045, 1.562181851, 1.561127985, 1.560123766, 1.559173695, 1.558283428, 1.557461192, 1.556719679, 1.556082305, 1.555628472], dtype=np.float64)
DEFAULT_LOESS_K_IMAG = np.array([0.001036314391, 0.00100409545, 0.0009725096047, 0.0009414702151, 0.0009110650451, 0.0008811348103, 0.0008517691111, 0.0008230252467, 0.000794822161, 0.0007671732463, 0.0007401114665, 0.0007136302637, 0.0006877346223, 0.0006624410284, 0.000637728537, 0.000613623132, 0.000590118135, 0.0005672389351, 0.0005449708121, 0.000523346451, 0.0005023623575, 0.0004820319492, 0.0004623687456, 0.0004433917156, 0.0004251090282, 0.0004075438773, 0.0003907181311, 0.0003746536642, 0.0003593777604, 0.0003449270403, 0.0003313401273, 0.0003186645197, 0.0003069633941, 0.0002963152118, 0.0002868310859, 0.0002786768356, 0.0002721434123, 0.0002680974742], dtype=np.float64)

# Kaolin gets an explicit 37-bin conservative clay-dominant profile so the code remains usable
# for both materials; the first intended diagnostic is loess 0.5 g/L.
DEFAULT_KAOLIN_N_REAL = np.array([1.585, 1.584305748, 1.583603199, 1.582905961, 1.58222285, 1.58152895, 1.580825132, 1.580137092, 1.579439632, 1.5787465, 1.57805136, 1.5773587, 1.576666415, 1.575969469, 1.575275553, 1.574582453, 1.573886743, 1.573193902, 1.572497863, 1.571803467, 1.571108815, 1.570415873, 1.569720958, 1.569026043, 1.568332463, 1.567638212, 1.566943504, 1.566249071, 1.565554649, 1.564860252, 1.564166201, 1.563471767, 1.562777499, 1.562083009, 1.561388825, 1.560694246, 1.56], dtype=np.float64)
DEFAULT_KAOLIN_K_IMAG = np.array([0.0008, 0.0007805609537, 0.0007608895687, 0.0007413668971, 0.0007222398082, 0.000702810588, 0.0006831036844, 0.0006638385767, 0.0006443096912, 0.0006249020138, 0.0006054380686, 0.0005860435904, 0.0005666596099, 0.0005471451365, 0.0005277154959, 0.0005083086708, 0.0004888288063, 0.0004694292508, 0.0004499401693, 0.0004304970855, 0.0004110468237, 0.0003916444313, 0.0003721868175, 0.000352729216, 0.0003333089771, 0.0003138699307, 0.0002944181087, 0.0002749739813, 0.0002555301742, 0.0002360870658, 0.000216653628, 0.0001972094688, 0.0001777699824, 0.0001583242393, 0.0001388871099, 0.0001194389002, 0.0001], dtype=np.float64)

DEFAULT_PARTICLE_WAVELENGTH_M = 622e-9


def vortex_surface_geometry(model: str, zmean: float, radius: float,
                            delta_h: float, core_radius: float):
    """Return deterministic geometry diagnostics for the selected free surface.

    ``delta_h`` is wall height minus centre height.  For the localized Scully
    surface, ``core_radius`` is the velocity-core radius ``a``.  The returned
    centre/wall positions use the same z coordinate as ``zmean``.
    """
    model=str(model).strip().lower()
    zmean=float(zmean); R=float(radius); D=float(delta_h); a=float(core_radius)
    if model=='flat' or D<=0.0:
        return dict(center_z=zmean,wall_z=zmean,h_inf=0.0,h_bar=0.0,
                    equivalent_rigid_body_rpm=0.0,equivalent_core_rpm=0.0)
    if model=='parabolic_vortex':
        rpm=math.sqrt(2.0*9.81*D/(R*R))*60.0/(2.0*math.pi)
        return dict(center_z=zmean-0.5*D,wall_z=zmean+0.5*D,h_inf=D,h_bar=0.5*D,
                    equivalent_rigid_body_rpm=rpm,equivalent_core_rpm=float('nan'))
    if model!='localized_scully_vortex':
        raise ValueError(f'unknown free-surface model {model!r}')
    if not (a>0.0 and R>0.0):
        raise ValueError('localized Scully vortex requires positive core and vessel radii')
    a2=a*a; R2=R*R
    h_inf=D*(a2+R2)/R2
    h_bar=h_inf*(1.0-(a2/R2)*math.log1p(R2/a2))
    omega_c=math.sqrt(2.0*9.81*h_inf/a2) if D>0.0 else 0.0
    return dict(center_z=zmean-h_bar,wall_z=zmean-h_bar+D,h_inf=h_inf,h_bar=h_bar,
                equivalent_rigid_body_rpm=float('nan'),
                equivalent_core_rpm=omega_c*60.0/(2.0*math.pi))


def sediment_transport_coefficients_host(diameter_m, zmean, zmin, radius, delta_h, core_radius,
                                         eddy_diffusivity_m2_s, particle_density_kg_m3=2600.0,
                                         water_density_kg_m3=997.0, water_dynamic_viscosity_pa_s=8.9e-4,
                                         gravity_m_s2=9.81, quadrature_points=8192,
                                         transport_model='stokes_drift_diffusion'):
    """Independent host reference for the V24.16 size-resolved transport field.

    Production coefficients are generated on CUDA.  This routine exists for
    verification and documentation.  Each returned ``normalization_scaled`` is
    the volume mean of exp(logF-logF_peak), so ``1/normalization_scaled`` is the
    exact majorant scale used by delta tracking for that bin.
    """
    d=np.asarray(diameter_m,dtype=np.float64)
    tm=str(transport_model).strip().lower()
    if tm=='uniform':
        one=np.ones_like(d);zero=np.zeros_like(d)
        return dict(response_time_s=zero,settling_velocity_m_s=zero,radial_log_coefficient=zero,
                    vertical_inverse_length_m=zero,normalization_scaled=one,max_density_scale=one,
                    omega_rad_s=0.0)
    if tm!='stokes_drift_diffusion':
        raise ValueError(f'unknown sediment transport model {transport_model!r}')
    Dt=float(eddy_diffusivity_m2_s);rho_p=float(particle_density_kg_m3);rho_w=float(water_density_kg_m3)
    mu=float(water_dynamic_viscosity_pa_s);g=float(gravity_m_s2);R=float(radius);a=float(core_radius)
    if not (Dt>0 and rho_p>rho_w>0 and mu>0 and g>0 and R>0 and a>0):
        raise ValueError('invalid V24.16 drift-diffusion physical parameters')
    vg=vortex_surface_geometry('localized_scully_vortex',float(zmean),R,float(delta_h),a)
    omega2=(2.0*g*vg['h_inf']/(a*a)) if delta_h>0 else 0.0
    tau=rho_p*d*d/(18.0*mu)
    ws=(rho_p-rho_w)*g*d*d/(18.0*mu)
    A=tau*omega2*a*a/(2.0*Dt)
    B=ws/Dt
    nquad=max(64,int(quadrature_points));dr=R/nquad
    rr=(np.arange(nquad,dtype=np.float64)+0.5)*dr
    f=rr*rr/(a*a+rr*rr);fR=R*R/(a*a+R*R)
    # actual curved free-surface height above the acrylic/water boundary
    Hinf=vg['h_inf'];hbar=vg['h_bar']
    zsurf=float(zmean)-hbar+Hinf*rr*rr/(a*a+rr*rr)
    H=np.maximum(zsurf-float(zmin),0.0)
    Hmean=float(zmean)-float(zmin)
    norm=np.empty_like(d)
    for i in range(d.size):
        radial=np.exp(A[i]*(f-fR))
        if abs(B[i])<1e-14:
            iz=radial*H
        else:
            iz=radial*(-np.expm1(-B[i]*H))/B[i]
        norm[i]=(2.0/(R*R*Hmean))*np.sum(rr*iz)*dr
    return dict(response_time_s=tau,settling_velocity_m_s=ws,radial_log_coefficient=A,
                vertical_inverse_length_m=B,normalization_scaled=norm,max_density_scale=1.0/norm,
                omega_rad_s=math.sqrt(max(omega2,0.0)))

def get_material_optical_constants(material: str, profile_override=None, n_fallback=1.59):
    m=material.strip().lower()
    if m=='loess':
        d=LOESS_DIAMETER_M
        n=DEFAULT_LOESS_N_REAL.copy(); k=DEFAULT_LOESS_K_IMAG.copy()
    elif m=='kaolin':
        d=KAOLIN_DIAMETER_M
        n=DEFAULT_KAOLIN_N_REAL.copy(); k=DEFAULT_KAOLIN_K_IMAG.copy()
    else:
        raise ValueError("material must be 'loess' or 'kaolin'")
    if profile_override:
        q=profile_override.get(m, profile_override.get(material, None))
        if q is not None:
            n=np.asarray(q.get('n_real', n), dtype=np.float64)
            k=np.asarray(q.get('k_imag', k), dtype=np.float64)
    if n.size != d.size or k.size != d.size:
        raise ValueError(f"{m} complex-optics profile must contain {d.size} n and k values")
    if np.any(~np.isfinite(n)) or np.any(~np.isfinite(k)) or np.any(n<=0) or np.any(k<0):
        raise ValueError(f"{m} complex-optics profile contains invalid n/k values")
    return n,k




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


#define PI_D 3.141592653589793238462643383279502884
#define MIE_MAX_ORDER 4096

struct cpxd { double re; double im; };
__device__ cpxd CD(double re,double im){cpxd z;z.re=re;z.im=im;return z;}
__device__ cpxd cd_add(cpxd a,cpxd b){return CD(a.re+b.re,a.im+b.im);}
__device__ cpxd cd_sub(cpxd a,cpxd b){return CD(a.re-b.re,a.im-b.im);}
__device__ cpxd cd_mul(cpxd a,cpxd b){return CD(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);}
__device__ cpxd cd_scale(cpxd a,double s){return CD(a.re*s,a.im*s);}
__device__ cpxd cd_div(cpxd a,cpxd b){
    double d=b.re*b.re+b.im*b.im;if(d<1e-300)d=1e-300;
    return CD((a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d);
}
__device__ double cd_abs2(cpxd a){return a.re*a.re+a.im*a.im;}
__device__ double cd_re_mul_conj(cpxd a,cpxd b){return a.re*b.re+a.im*b.im;}

__device__ float normal_box_muller(unsigned int* state){
    float u1=fmaxf(rnd_uniform(state),1e-12f),u2=rnd_uniform(state);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*PI_F*u2);
}

__device__ float gamma_shape_sample(unsigned int* state,float shape){
    if(shape<=0.0f) return 0.0f;
    if(fabsf(shape-1.0f)<1e-7f) return -logf(fmaxf(rnd_uniform(state),1e-12f));
    float correction=1.0f;
    float a=shape;
    if(a<1.0f){
        float u=fmaxf(rnd_uniform(state),1e-12f);
        correction=powf(u,1.0f/a);
        a+=1.0f;
    }
    float d=a-0.3333333333333333f;
    float c=rsqrtf(9.0f*d);
    for(int it=0;it<128;++it){
        float x=normal_box_muller(state);
        float v=1.0f+c*x;
        if(v<=0.0f) continue;
        v=v*v*v;
        float u=rnd_uniform(state);
        if(u<1.0f-0.0331f*x*x*x*x) return correction*d*v;
        if(logf(fmaxf(u,1e-12f))<0.5f*x*x+d*(1.0f-v+logf(v))) return correction*d*v;
    }
    return correction*d;
}

__device__ float beta_sample(unsigned int* state,float a,float b){
    if(a<=0.0f||b<=0.0f) return 0.0f;
    // Exact inverse forms cover CLARITAS' default Beta(1,100) source efficiently.
    if(fabsf(a-1.0f)<1e-7f){
        float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
        return 1.0f-powf(1.0f-u,1.0f/b);
    }
    if(fabsf(b-1.0f)<1e-7f){
        float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
        return powf(u,1.0f/a);
    }
    float ga=gamma_shape_sample(state,a),gb=gamma_shape_sample(state,b);
    float s=ga+gb;return s>0.0f?ga/s:0.0f;
}

// V24.10.2 source-law correction.
//
// The historical CLARITAS Beta(a,b) law was a one-dimensional angular
// beam-intensity/profile law in the horizontal plane. Extending that same
// angular radiance profile I(theta) to an axisymmetric 3-D source requires
// sampling probability per polar angle proportional to I(theta)*sin(theta),
// because dOmega = sin(theta) dtheta dphi.
//
// If u=theta/(pi/2) and I(theta) is proportional to BetaPDF(u;a,b), the
// desired u density is BetaPDF(u;a,b)*sin(pi*u/2).
//
// Exact efficient rejection proposal:
//   q(u)=BetaPDF(u;a+1,b)
// with acceptance sin(c*u)/(c*u), c=pi/2. The acceptance is bounded below
// by 2/pi, so this avoids the poor efficiency of direct sin(theta) rejection.
__device__ float beta_radiance_polar_sample(unsigned int* state,float a,float b){
    const float c=0.5f*PI_F;
    for(int it=0;it<128;++it){
        float u=beta_sample(state,a+1.0f,b);
        u=fminf(fmaxf(u,0.0f),1.0f);
        float x=c*u;
        float accept=(x<1.0e-7f)?1.0f:sinf(x)/x;
        if(rnd_uniform(state)<=accept) return x;
    }
    float u=beta_sample(state,a+1.0f,b);
    return c*fminf(fmaxf(u,0.0f),1.0f);
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

// Lightweight complex arithmetic for absorbing-particle Fresnel coefficients.
struct cpx { float re; float im; };
__device__ cpx C(float re,float im){cpx z;z.re=re;z.im=im;return z;}
__device__ cpx cadd(cpx a,cpx b){return C(a.re+b.re,a.im+b.im);}
__device__ cpx csub(cpx a,cpx b){return C(a.re-b.re,a.im-b.im);}
__device__ cpx cmul(cpx a,cpx b){return C(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);}
__device__ cpx cscale(cpx a,float s){return C(a.re*s,a.im*s);}
__device__ cpx cdiv(cpx a,cpx b){
    float d=b.re*b.re+b.im*b.im;
    if(d<1e-30f)d=1e-30f;
    return C((a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d);
}
__device__ float cabs2(cpx a){return a.re*a.re+a.im*a.im;}
__device__ cpx csqrtp(cpx z){
    float r=sqrtf(z.re*z.re+z.im*z.im);
    float u=sqrtf(fmaxf(0.0f,0.5f*(r+z.re)));
    float v=sqrtf(fmaxf(0.0f,0.5f*(r-z.re)));
    if(z.im<0.0f)v=-v;
    // choose branch with non-negative real part; for exactly-zero real part choose non-negative imag
    if(u<0.0f || (fabsf(u)<1e-20f && v<0.0f)){u=-u;v=-v;}
    return C(u,v);
}

// Unpolarised Fresnel reflectance with possibly complex incident/transmitted indices.
// Geometry/direction still uses real phase indices separately.
__device__ float fresnel_R_complex(float ci,float n1r,float n1k,float n2r,float n2k){
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float si2=fmaxf(0.0f,1.0f-ci*ci);
    cpx N1=C(n1r,n1k), N2=C(n2r,n2k);
    cpx eta=cdiv(N1,N2);
    cpx eta2=cmul(eta,eta);
    cpx one_minus=csub(C(1.0f,0.0f),cscale(eta2,si2));
    cpx ct=csqrtp(one_minus);
    cpx N1ci=cscale(N1,ci), N2ci=cscale(N2,ci);
    cpx N2ct=cmul(N2,ct), N1ct=cmul(N1,ct);
    cpx rs=cdiv(csub(N1ci,N2ct),cadd(N1ci,N2ct));
    cpx rp=cdiv(csub(N2ci,N1ct),cadd(N2ci,N1ct));
    float R=0.5f*(cabs2(rs)+cabs2(rp));
    return fminf(fmaxf(R,0.0f),1.0f);
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


// V24.16 retains the V24.15 free-surface models.
// FREE_SURFACE_MODEL: 0=flat, 1=legacy V24.14 volume-preserving paraboloid,
// 2=localized volume-preserving Scully vortex.
// DELTA_H is wall height minus centre height. CORE_R is the Scully core radius a.
__device__ float localized_scully_hinf(float R,float DELTA_H,float CORE_R){
    float a2=CORE_R*CORE_R, R2=R*R;
    return DELTA_H*(a2+R2)/R2;
}

__device__ float localized_scully_hbar(float R,float DELTA_H,float CORE_R){
    float a2=CORE_R*CORE_R, R2=R*R;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    return H*(1.0f-(a2/R2)*log1pf(R2/a2));
}

__device__ float free_surface_z(float x,float y,float ZMEAN,float R,int MODEL,float DELTA_H,float CORE_R){
    if(MODEL==0 || DELTA_H<=0.0f) return ZMEAN;
    float q=x*x+y*y;
    if(MODEL==1){
        return ZMEAN + DELTA_H*(q/(R*R)-0.5f);
    }
    float a2=CORE_R*CORE_R;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    float hbar=localized_scully_hbar(R,DELTA_H,CORE_R);
    return ZMEAN - hbar + H*q/(a2+q);
}


// V24.16 size-resolved stirred-sediment field.
// TRANSPORT_MODEL: 0=uniform historical suspension, 1=Stokes drift-diffusion.
// The local scale is relative to each bin's vessel-volume mean.  The peak is
// at the bottom outer wall, and the scaled exponent is constructed <=0.
__device__ double sediment_local_scale(
    const int TRANSPORT_MODEL,const int pidx,const double* SED_A,const double* SED_B,
    const double* SED_NORM_SCALED,const float x,const float y,const float z,
    const float R,const float ZMIN,const float CORE_R){
    if(TRANSPORT_MODEL==0) return 1.0;
    double rr2=(double)x*(double)x+(double)y*(double)y;
    double R2=(double)R*(double)R;if(rr2>R2)rr2=R2;
    double a2=(double)CORE_R*(double)CORE_R;
    double f=rr2/(a2+rr2),fR=R2/(a2+R2);
    double h=fmax((double)z-(double)ZMIN,0.0);
    double logrel=SED_A[pidx]*(f-fR)-SED_B[pidx]*h;
    double ns=SED_NORM_SCALED[pidx];
    if(!(ns>0.0)) return 0.0;
    return exp(logrel)/ns;
}

__device__ int sample_local_mu_bin(
    unsigned int* state,const int TRANSPORT_MODEL,const double* MU_BIN,const int n_particles,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,
    const float x,const float y,const float z,const float R,const float ZMIN,const float CORE_R,
    const double local_total){
    if(!(local_total>0.0))return 0;
    double target=(double)rnd_uniform(state)*local_total,cum=0.0;
    for(int i=0;i<n_particles;++i){
        double sc=sediment_local_scale(TRANSPORT_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,x,y,z,R,ZMIN,CORE_R);
        cum+=MU_BIN[i]*sc;
        if(target<=cum)return i;
    }
    return n_particles>0?n_particles-1:0;
}

__device__ double quadratic_min_positive(double A,double B,double C){
    const double eps=2.0e-7;
    if(fabs(A)<1e-24){
        if(fabs(B)<1e-24) return 1.0e300;
        double t=-C/B; return t>eps?t:1.0e300;
    }
    double D=B*B-4.0*A*C;
    if(D<0.0) return 1.0e300;
    double sd=sqrt(fmax(D,0.0));
    // Stable quadratic form, followed by the companion root from C/q.
    double q=-0.5*(B+copysign(sd,B));
    double t1=(fabs(q)>1e-300)?q/A:(-B-sd)/(2.0*A);
    double t2=(fabs(q)>1e-300)?C/q:(-B+sd)/(2.0*A);
    double best=1.0e300;
    if(t1>eps && t1<best)best=t1;
    if(t2>eps && t2<best)best=t2;
    return best;
}

__device__ double cubic_min_positive(double A3,double A2,double A1,double A0){
    const double eps=2.0e-7;
    double scale=fmax(fmax(fabs(A3),fabs(A2)),fmax(fabs(A1),fabs(A0)));
    if(scale<=0.0) return 1.0e300;
    // Directions are normalized, so |A3| is O(|vz|). Treat near-horizontal
    // rays as the mathematically limiting quadratic to avoid cubic conditioning loss.
    if(fabs(A3)<=1e-12) return quadratic_min_positive(A2,A1,A0);
    double a=A2/A3,b=A1/A3,c=A0/A3;
    double p=b-a*a/3.0;
    double q=2.0*a*a*a/27.0-a*b/3.0+c;
    double halfq=0.5*q, thirdp=p/3.0;
    double d1=halfq*halfq,d2=thirdp*thirdp*thirdp;
    double disc=d1+d2;
    // Relative discriminant tolerance is essential here: ray-distance roots are
    // O(1e-2..1e-1 m), so an absolute O(1) tolerance would collapse genuinely
    // three-real-root cases into the repeated-root branch.
    double tol=1e-12*(fabs(d1)+fabs(d2)+1e-300);
    double best=1.0e300;
    if(disc>tol){
        double sd=sqrt(disc);
        double u=cbrt(-halfq+sd),v=cbrt(-halfq-sd);
        double t=u+v-a/3.0;
        if(t>eps)best=t;
    }else if(fabs(p)<1e-24 && fabs(q)<1e-24){
        double t=-a/3.0;if(t>eps)best=t;
    }else if(disc>=-tol){
        double u=cbrt(-halfq);
        double t1=2.0*u-a/3.0;
        double t2=-u-a/3.0;
        if(t1>eps && t1<best)best=t1;
        if(t2>eps && t2<best)best=t2;
    }else{
        double m=2.0*sqrt(fmax(-thirdp,0.0));
        double den=sqrt(fmax(-(thirdp*thirdp*thirdp),1e-300));
        double ca=fmin(fmax(-halfq/den,-1.0),1.0);
        double phi=acos(ca);
        for(int k=0;k<3;++k){
            double t=m*cos((phi+2.0*PI_D*(double)k)/3.0)-a/3.0;
            if(t>eps && t<best)best=t;
        }
    }
    return best;
}

// First positive ray intersection with the selected free surface. For DELTA_H=0
// this is the exact legacy flat expression and consumes no random numbers.
__device__ float free_surface_hit(float x,float y,float z,float vx,float vy,float vz,
                                  float ZMEAN,float R,int MODEL,float DELTA_H,float CORE_R){
    if(MODEL==0 || DELTA_H<=0.0f){
        if(vz<=1e-14f) return INF_F;
        float t=(ZMEAN-z)/vz;
        return (t>EPS_F)?t:INF_F;
    }
    if(MODEL==1){
        float aa=DELTA_H/(R*R);
        double A=-(double)aa*((double)vx*vx+(double)vy*vy);
        double B=(double)vz-2.0*(double)aa*((double)x*vx+(double)y*vy);
        double C=(double)z-((double)ZMEAN-0.5*(double)DELTA_H)-(double)aa*((double)x*x+(double)y*y);
        double t=quadratic_min_positive(A,B,C);
        return (t<0.5*(double)INF_F)?(float)t:INF_F;
    }
    // Scully surface: z_s=C+H*q/(a^2+q), q=x^2+y^2.  Multiplication by
    // (a^2+q)>0 produces a cubic with no extraneous real roots.
    double X=x,Y=y,Z=z,VX=vx,VY=vy,VZ=vz;
    double RR=(double)R*(double)R, aa=(double)CORE_R*(double)CORE_R;
    double H=(double)DELTA_H*(aa+RR)/RR;
    double hbar=H*(1.0-(aa/RR)*log1p(RR/aa));
    double Csurf=(double)ZMEAN-hbar;
    double Q2=VX*VX+VY*VY;
    double Q1=2.0*(X*VX+Y*VY);
    double Q0=X*X+Y*Y;
    double L0=Z-Csurf;
    double c3=VZ*Q2;
    double c2=(L0-H)*Q2+VZ*Q1;
    double c1=(L0-H)*Q1+VZ*(aa+Q0);
    double c0=L0*aa+(L0-H)*Q0;
    double t=cubic_min_positive(c3,c2,c1,c0);
    if(!(t<0.5*(double)INF_F)) return INF_F;
    double xx=X+t*VX,yy=Y+t*VY;
    if(xx*xx+yy*yy>RR*(1.0+2e-6)) return INF_F;
    return (float)t;
}

// Unit normal pointing into the incident WATER medium.
__device__ void free_surface_normal_into_water(float x,float y,float R,int MODEL,float DELTA_H,float CORE_R,
                                               float* nx,float* ny,float* nz){
    if(MODEL==0 || DELTA_H<=0.0f){*nx=0.0f;*ny=0.0f;*nz=-1.0f;return;}
    if(MODEL==1){
        float aa=DELTA_H/(R*R);
        *nx=2.0f*aa*x;*ny=2.0f*aa*y;*nz=-1.0f;normalize3(nx,ny,nz);return;
    }
    float a2=CORE_R*CORE_R,q=x*x+y*y;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    float fac=2.0f*H*a2/((a2+q)*(a2+q));
    *nx=fac*x;*ny=fac*y;*nz=-1.0f;normalize3(nx,ny,nz);
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

// Beckmann microfacet roughness diagnostic.  alpha is the RMS magnitude
// of the tangent-plane slope vector.  Equivalently the user-facing RMS slope
// angle is atan(alpha).  Tangent slopes are independent Gaussians with
// sigma=alpha/sqrt(2), yielding E[sx^2+sy^2]=alpha^2.
//
// desired_dot_sign = -1 for an entry-side outward normal (incident ray has
// dot(v,n)<0), +1 for an exit-side outward normal (dot(v,n)>0).  Back-facing
// sampled facets are rejected.  alpha<=0 returns the base normal and consumes
// no RNG, preserving the V24.7 smooth-sphere random stream.
__device__ void beckmann_microfacet_normal(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float vx,float vy,float vz,int desired_dot_sign,
    float* mx,float* my,float* mz) {
    *mx=bx;*my=by;*mz=bz;
    if(alpha<=0.0f) return;
    float t1x,t1y,t1z,t2x,t2y,t2z;
    perpendicular_basis(bx,by,bz,&t1x,&t1y,&t1z,&t2x,&t2y,&t2z);
    const float sigma=0.7071067811865475f*alpha;
    for(int attempt=0;attempt<8;++attempt){
        float u1=fmaxf(rnd_uniform(state),1e-12f);
        float u2=rnd_uniform(state);
        float mag=sqrtf(-2.0f*logf(u1));
        float g1=mag*cosf(2.0f*PI_F*u2);
        float g2=mag*sinf(2.0f*PI_F*u2);
        float sx=sigma*g1, sy=sigma*g2;
        float nx=bx+sx*t1x+sy*t2x;
        float ny=by+sx*t1y+sy*t2y;
        float nz=bz+sx*t1z+sy*t2z;
        normalize3(&nx,&ny,&nz);
        float dv=dot3(vx,vy,vz,nx,ny,nz);
        if((desired_dot_sign<0 && dv<-1e-6f) || (desired_dot_sign>0 && dv>1e-6f)){
            *mx=nx;*my=ny;*mz=nz;return;
        }
    }
}

__device__ int sample_cdf_index(unsigned int* state,const double* cdf,int n) {
    if(n<=1) return 0;
    float u=rnd_uniform(state);
    int lo=0,hi=n-1;
    while(lo<hi){int mid=(lo+hi)>>1;if(u<=(float)cdf[mid])hi=mid;else lo=mid+1;}
    return lo;
}

__device__ float sample_mie_theta(unsigned int* state,const double* cdf_table,const double* theta_table,int ntheta,int pidx) {
    float u=rnd_uniform(state);int base=pidx*ntheta;int lo=0,hi=ntheta-1;
    while(lo<hi){int mid=(lo+hi)>>1;if(u<=(float)cdf_table[base+mid])hi=mid;else lo=mid+1;}
    int j=lo;if(j<=0)return (float)theta_table[0];
    float c0=(float)cdf_table[base+j-1],c1=(float)cdf_table[base+j];
    float t0=(float)theta_table[j-1],t1=(float)theta_table[j];
    float f=(c1>c0)?fminf(fmaxf((u-c0)/(c1-c0),0.0f),1.0f):0.0f;
    return t0+f*(t1-t0);
}

__device__ void mie_scatter_direction(unsigned int* state,const double* cdf_table,const double* theta_table,int ntheta,int pidx,
                                      float* vx,float* vy,float* vz) {
    float theta=sample_mie_theta(state,cdf_table,theta_table,ntheta,pidx);
    float phi=2.0f*PI_F*rnd_uniform(state);
    float e1x,e1y,e1z,e2x,e2y,e2z;perpendicular_basis(*vx,*vy,*vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float st=sinf(theta),ct=cosf(theta),cp=cosf(phi),sp=sinf(phi);
    float ox=ct*(*vx)+st*(cp*e1x+sp*e2x);
    float oy=ct*(*vy)+st*(cp*e1y+sp*e2y);
    float oz=ct*(*vz)+st*(cp*e1z+sp*e2z);
    *vx=ox;*vy=oy;*vz=oz;normalize3(vx,vy,vz);
}

__device__ int sphere_fresnel_interaction_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions) {
    // return 1=survived particle event, 2=absorbed in particle, 0=numerical/path failure.
    //
    // V24.8 morphology diagnostic: sphere geometry controls projected-area
    // interception and path length, while the local optical interface normal
    // may be perturbed by a Beckmann microfacet slope distribution.  This is
    // deliberately local: it extends rather than replaces the successful
    // V23/V24.5 3-D sphere Snell/Fresnel logic.
    if(radius<=0.0f) return 1;
    float rho=sqrtf(fminf(fmaxf(rnd_uniform(state),0.0f),0.99999994f));
    float ci=sqrtf(fmaxf(0.0f,1.0f-rho*rho));
    float phi=rnd_uniform(state)*2.0f*PI_F;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(*vx,*vy,*vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float qx=cosf(phi)*e1x+sinf(phi)*e2x;
    float qy=cosf(phi)*e1y+sinf(phi)*e2y;
    float qz=cosf(phi)*e1z+sinf(phi)*e2z;
    float gnx=-ci*(*vx)+rho*qx, gny=-ci*(*vy)+rho*qy, gnz=-ci*(*vz)+rho*qz;
    normalize3(&gnx,&gny,&gnz);
    float cx=*x-radius*gnx, cy=*y-radius*gny, cz=*z-radius*gnz;

    // Entry surface.  The geometric normal fixes the sphere centre; a rough
    // optical normal controls Fresnel/refraction.  If a very tilted microfacet
    // would refract away from the idealised sphere volume, fall back to the
    // geometric normal for that interface so the local microfacet approximation
    // remains compatible with the sphere path geometry.
    float onx=gnx,ony=gny,onz=gnz;
    beckmann_microfacet_normal(state,rough_alpha,gnx,gny,gnz,*vx,*vy,*vz,-1,&onx,&ony,&onz);
    float a,b,c;
    float ci_opt=(rough_alpha>0.0f)?fminf(fmaxf(-dot3(*vx,*vy,*vz,onx,ony,onz),0.0f),1.0f):ci;
    int refr_ok=refract3(*vx,*vy,*vz,onx,ony,onz,n_medium,n_particle,&a,&b,&c);
    if(!refr_ok || dot3(a,b,c,gnx,gny,gnz)>=-1e-7f){
        onx=gnx;ony=gny;onz=gnz;ci_opt=ci;
        refr_ok=refract3(*vx,*vy,*vz,onx,ony,onz,n_medium,n_particle,&a,&b,&c);
    }
    float R=fresnel_R_complex(ci_opt,n_medium,0.0f,n_particle,k_particle);
    if(rnd_uniform(state)<R){
        float ra,rb,rc; reflect3(*vx,*vy,*vz,onx,ony,onz,&ra,&rb,&rc);
        *vx=ra;*vy=rb;*vz=rc;(*p_reflections)++;(*p_entry_reflections)++;return 1;
    }
    if(!refr_ok) return 0;
    *vx=a;*vy=b;*vz=c;
    float alpha_abs=(wavelength>0.0f && k_particle>0.0f) ? (4.0f*PI_F*k_particle/wavelength) : 0.0f;

    for(int bounce=0;bounce<=max_bounces;++bounce){
        float rx=*x-cx,ry=*y-cy,rz=*z-cz;
        float chord=-2.0f*dot3(rx,ry,rz,*vx,*vy,*vz);
        if(chord<=1e-12f) return 0;

        if(alpha_abs>0.0f){
            float survival=expf(-alpha_abs*chord);
            if(rnd_uniform(state)>survival){
                *x+=chord*(*vx);*y+=chord*(*vy);*z+=chord*(*vz);*internal_path+=chord;
                (*p_absorptions)++;
                return 2;
            }
        }

        *x+=chord*(*vx);*y+=chord*(*vy);*z+=chord*(*vz);*internal_path+=chord;
        float nox=(*x-cx)/radius,noy=(*y-cy)/radius,noz=(*z-cz)/radius;normalize3(&nox,&noy,&noz);
        float cii_geom=fminf(fmaxf(dot3(*vx,*vy,*vz,nox,noy,noz),0.0f),1.0f);

        float rnx=nox,rny=noy,rnz=noz;
        beckmann_microfacet_normal(state,rough_alpha,nox,noy,noz,*vx,*vy,*vz,+1,&rnx,&rny,&rnz);
        float cii=(rough_alpha>0.0f)?fminf(fmaxf(dot3(*vx,*vy,*vz,rnx,rny,rnz),0.0f),1.0f):cii_geom;

        // Trial transmitted ray.  As at entry, reject a rough optical normal if
        // its transmitted ray points back into the idealised sphere geometry.
        float ta=0.0f,tb=0.0f,tc=0.0f;
        int refr_out=refract3(*vx,*vy,*vz,-rnx,-rny,-rnz,n_particle,n_medium,&ta,&tb,&tc);
        if(!refr_out || dot3(ta,tb,tc,nox,noy,noz)<=1e-7f){
            rnx=nox;rny=noy;rnz=noz;cii=cii_geom;
            refr_out=refract3(*vx,*vy,*vz,-rnx,-rny,-rnz,n_particle,n_medium,&ta,&tb,&tc);
        }

        float eta_real=n_particle/n_medium;
        float st2=eta_real*eta_real*fmaxf(0.0f,1.0f-cii*cii);
        int tir_real=(st2>=1.0f);
        float Ri=tir_real ? 1.0f : fresnel_R_complex(cii,n_particle,k_particle,n_medium,0.0f);
        if(!tir_real && refr_out && rnd_uniform(state)>=Ri){
            *vx=ta;*vy=tb;*vz=tc;return 1;
        }

        float ra,rb,rc; reflect3(*vx,*vy,*vz,rnx,rny,rnz,&ra,&rb,&rc);
        // A local microfacet approximation should send an internally reflected
        // ray back into the geometric particle volume.  Fall back to the smooth
        // geometric normal if a very tilted facet violates that condition.
        if(dot3(ra,rb,rc,nox,noy,noz)>=-1e-7f)
            reflect3(*vx,*vy,*vz,nox,noy,noz,&ra,&rb,&rc);
        *vx=ra;*vy=rb;*vz=rc;
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces) return 0;
    }
    return 0;
}

// ============================ INHERITED SPHEROID WAVE MORPHOLOGY ============================
// Random-orientation, volume-preserving spheroid used ONLY for the incoherent
// supplemental wave channel.  The geometric projected-area channel remains the
// frozen V23/V24.8 rough-sphere interaction so this is a one-variable physics
// diagnostic of the wave-event angular law.
//
// Let q=a/b be the symmetry-axis/equatorial-axis aspect ratio and r the
// volume-equivalent sphere radius.  CUDA evaluates
//      a = r*q^(2/3),  b = r*q^(-1/3),  hence a*b^2=r^3.
// q>1 is prolate; q<1 is oblate. V24.16 freezes q=1 and isotropic orientation, while the inherited sampler remains available for regression.

__device__ void random_unit_axis(unsigned int* state,float* ux,float* uy,float* uz){
    float cz=2.0f*rnd_uniform(state)-1.0f;
    float az=2.0f*PI_F*rnd_uniform(state);
    float sz=sqrtf(fmaxf(0.0f,1.0f-cz*cz));
    *ux=sz*cosf(az);*uy=sz*sinf(az);*uz=cz;
}

// Inherited axial orientation sampler.  The model code is:
//   0 = isotropic (legacy random_unit_axis)
//   1 = local-tangential, head-tail-symmetric von-Mises-Fisher mixture.
// For model 1 and kappa=0 this calls random_unit_axis directly, preserving the
// exact V24.12 RNG path.  For kappa>0, first sample the ordinary 3-D vMF law
// p(c)=kappa*exp(kappa*c)/(2*sinh(kappa)), c=cos(theta), by exact inverse CDF,
// then flip the sampled axis sign with probability 1/2.  The resulting axial
// density is proportional to cosh(kappa * u.flow), so +u and -u are equivalent.
__device__ void sample_spheroid_axis(
    unsigned int* state,int orientation_model,float kappa,
    float fx,float fy,float fz,float* ux,float* uy,float* uz){
    if(orientation_model==0 || !(kappa>1e-7f) || !isfinite(kappa)){
        random_unit_axis(state,ux,uy,uz);return;
    }
    normalize3(&fx,&fy,&fz);
    if(!(fx*fx+fy*fy+fz*fz>0.5f)){
        random_unit_axis(state,ux,uy,uz);return;
    }
    // Stable inverse CDF. exp(-2*kappa) safely underflows to zero at strong alignment.
    float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
    float em2=expf(-2.0f*kappa);
    float c=1.0f+logf(u+(1.0f-u)*em2)/kappa;
    c=fminf(fmaxf(c,-1.0f),1.0f);
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(fx,fy,fz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float phi=2.0f*PI_F*rnd_uniform(state);
    float st=sqrtf(fmaxf(0.0f,1.0f-c*c));
    float cp=cosf(phi),sp=sinf(phi);
    *ux=c*fx+st*(cp*e1x+sp*e2x);
    *uy=c*fy+st*(cp*e1y+sp*e2y);
    *uz=c*fz+st*(cp*e1z+sp*e2z);
    if(rnd_uniform(state)<0.5f){*ux=-*ux;*uy=-*uy;*uz=-*uz;}
    normalize3(ux,uy,uz);
}

__global__ void orientation_sampler_test_kernel(
    const int n,const unsigned int seed,const int orientation_model,const float kappa,
    float* ux_out,float* uy_out,float* uz_out){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;
    unsigned int state=seed+(unsigned int)tid*104729u+29u;
    float ux,uy,uz;
    sample_spheroid_axis(&state,orientation_model,kappa,1.0f,0.0f,0.0f,&ux,&uy,&uz);
    ux_out[tid]=ux;uy_out[tid]=uy;uz_out[tid]=uz;
}

__device__ float spheroid_F(float x,float y,float z,float ux,float uy,float uz,float invb2,float delta){
    float pu=x*ux+y*uy+z*uz;
    return invb2*(x*x+y*y+z*z)+delta*pu*pu-1.0f;
}

__device__ void spheroid_normal(float x,float y,float z,float ux,float uy,float uz,float invb2,float delta,
                                 float* nx,float* ny,float* nz){
    float pu=x*ux+y*uy+z*uz;
    *nx=invb2*x+delta*pu*ux;
    *ny=invb2*y+delta*pu*uy;
    *nz=invb2*z+delta*pu*uz;
    normalize3(nx,ny,nz);
}

__device__ int spheroid_line_roots(float ox,float oy,float oz,float dx,float dy,float dz,
                                   float ux,float uy,float uz,float invb2,float delta,
                                   float* t0,float* t1){
    float du=dx*ux+dy*uy+dz*uz;
    float ou=ox*ux+oy*uy+oz*uz;
    float A=invb2*(dx*dx+dy*dy+dz*dz)+delta*du*du;
    float B=2.0f*(invb2*(ox*dx+oy*dy+oz*dz)+delta*ou*du);
    float Cc=invb2*(ox*ox+oy*oy+oz*oz)+delta*ou*ou-1.0f;
    float disc=B*B-4.0f*A*Cc;
    if(!(A>0.0f) || disc<0.0f)return 0;
    float sd=sqrtf(fmaxf(0.0f,disc));
    float r0=(-B-sd)/(2.0f*A),r1=(-B+sd)/(2.0f*A);
    if(r0>r1){float q=r0;r0=r1;r1=q;}
    *t0=r0;*t1=r1;return 1;
}

__device__ int spheroid_sample_entry_projected(unsigned int* state,float vx,float vy,float vz,float a,float b,
                                                float ux,float uy,float uz,
                                                float* x,float* y,float* z,float* gnx,float* gny,float* gnz){
    float invb2=1.0f/(b*b);
    float delta=1.0f/(a*a)-invb2;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(vx,vy,vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float du=dot3(ux,uy,uz,vx,vy,vz);
    float rproj2=b*b+(a*a-b*b)*fmaxf(0.0f,1.0f-du*du);
    float rbound=sqrtf(fmaxf(b*b,rproj2));
    float rmax=fmaxf(a,b);
    // Rejection from a tight projected bounding disk gives an exactly uniform
    // offset over the spheroid's projected silhouette after conditioning on hit.
    for(int attempt=0;attempt<64;++attempt){
        float rr=rbound*sqrtf(rnd_uniform(state));
        float ph=2.0f*PI_F*rnd_uniform(state);
        float ox=rr*(cosf(ph)*e1x+sinf(ph)*e2x)-2.5f*rmax*vx;
        float oy=rr*(cosf(ph)*e1y+sinf(ph)*e2y)-2.5f*rmax*vy;
        float oz=rr*(cosf(ph)*e1z+sinf(ph)*e2z)-2.5f*rmax*vz;
        float t0,t1;
        if(!spheroid_line_roots(ox,oy,oz,vx,vy,vz,ux,uy,uz,invb2,delta,&t0,&t1))continue;
        float t=(t0>1e-9f*rmax)?t0:((t1>1e-9f*rmax)?t1:-1.0f);
        if(t<=0.0f)continue;
        *x=ox+t*vx;*y=oy+t*vy;*z=oz+t*vz;
        spheroid_normal(*x,*y,*z,ux,uy,uz,invb2,delta,gnx,gny,gnz);
        if(dot3(vx,vy,vz,*gnx,*gny,*gnz)<-1e-6f)return 1;
    }
    return 0;
}

__device__ float spheroid_next_boundary_distance(float x,float y,float z,float vx,float vy,float vz,
                                                  float ux,float uy,float uz,float invb2,float delta,float scale){
    float t0,t1;
    if(!spheroid_line_roots(x,y,z,vx,vy,vz,ux,uy,uz,invb2,delta,&t0,&t1))return -1.0f;
    float eps=fmaxf(1e-5f*scale,1e-12f);
    float t=INF_F;
    if(t0>eps)t=t0;
    if(t1>eps&&t1<t)t=t1;
    return t<INF_F*0.5f?t:-1.0f;
}

__device__ int spheroid_fresnel_scatter_direction_only_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,
    float r_equiv,float aspect_ratio,int orientation_model,float orientation_kappa,float flow_fx,float flow_fy,float flow_fz,
    int max_bounces,float* vx,float* vy,float* vz,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions){
    if(r_equiv<=0.0f)return 1;
    if(!(aspect_ratio>0.0f) || !isfinite(aspect_ratio))return 0;
    // Exact V24.9 regression path at q=1: same function and same RNG stream.
    if(fabsf(aspect_ratio-1.0f)<1e-7f){
        float lx=0.0f,ly=0.0f,lz=0.0f,lvx=*vx,lvy=*vy,lvz=*vz,ip=0.0f;
        int ps=sphere_fresnel_interaction_3d(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,r_equiv,max_bounces,
            &lx,&ly,&lz,&lvx,&lvy,&lvz,&ip,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions);
        if(ps==1){*vx=lvx;*vy=lvy;*vz=lvz;}return ps;
    }
    float q13=powf(aspect_ratio,1.0f/3.0f);
    float a=r_equiv*q13*q13;
    float b=r_equiv/q13;
    if(!(a>0.0f&&b>0.0f&&isfinite(a)&&isfinite(b)))return 0;
    float ux,uy,uz;sample_spheroid_axis(state,orientation_model,orientation_kappa,flow_fx,flow_fy,flow_fz,&ux,&uy,&uz);
    float x,y,z,gnx,gny,gnz;
    if(!spheroid_sample_entry_projected(state,*vx,*vy,*vz,a,b,ux,uy,uz,&x,&y,&z,&gnx,&gny,&gnz))return 0;
    float invb2=1.0f/(b*b),delta=1.0f/(a*a)-invb2,scale=fmaxf(a,b);

    float onx=gnx,ony=gny,onz=gnz;
    beckmann_microfacet_normal(state,rough_alpha,gnx,gny,gnz,*vx,*vy,*vz,-1,&onx,&ony,&onz);
    float ci_geom=fminf(fmaxf(-dot3(*vx,*vy,*vz,gnx,gny,gnz),0.0f),1.0f);
    float ci_opt=(rough_alpha>0.0f)?fminf(fmaxf(-dot3(*vx,*vy,*vz,onx,ony,onz),0.0f),1.0f):ci_geom;
    float tx,ty,tz;
    int refr_ok=refract3(*vx,*vy,*vz,onx,ony,onz,n_medium,n_particle,&tx,&ty,&tz);
    if(!refr_ok || dot3(tx,ty,tz,gnx,gny,gnz)>=-1e-7f){
        onx=gnx;ony=gny;onz=gnz;ci_opt=ci_geom;
        refr_ok=refract3(*vx,*vy,*vz,onx,ony,onz,n_medium,n_particle,&tx,&ty,&tz);
    }
    float R=fresnel_R_complex(ci_opt,n_medium,0.0f,n_particle,k_particle);
    if(rnd_uniform(state)<R){
        reflect3(*vx,*vy,*vz,onx,ony,onz,&tx,&ty,&tz);
        *vx=tx;*vy=ty;*vz=tz;(*p_reflections)++;(*p_entry_reflections)++;return 1;
    }
    if(!refr_ok)return 0;
    *vx=tx;*vy=ty;*vz=tz;
    float alpha_abs=(wavelength>0.0f&&k_particle>0.0f)?(4.0f*PI_F*k_particle/wavelength):0.0f;

    for(int bounce=0;bounce<=max_bounces;++bounce){
        float chord=spheroid_next_boundary_distance(x,y,z,*vx,*vy,*vz,ux,uy,uz,invb2,delta,scale);
        if(chord<=0.0f)return 0;
        if(alpha_abs>0.0f){
            float survival=expf(-alpha_abs*chord);
            if(rnd_uniform(state)>survival){(*p_absorptions)++;return 2;}
        }
        x+=chord*(*vx);y+=chord*(*vy);z+=chord*(*vz);
        float nox,noy,noz;spheroid_normal(x,y,z,ux,uy,uz,invb2,delta,&nox,&noy,&noz);
        float cii_geom=fminf(fmaxf(dot3(*vx,*vy,*vz,nox,noy,noz),0.0f),1.0f);
        float rnx=nox,rny=noy,rnz=noz;
        beckmann_microfacet_normal(state,rough_alpha,nox,noy,noz,*vx,*vy,*vz,+1,&rnx,&rny,&rnz);
        float cii=(rough_alpha>0.0f)?fminf(fmaxf(dot3(*vx,*vy,*vz,rnx,rny,rnz),0.0f),1.0f):cii_geom;
        float ox=0.0f,oy=0.0f,oz=0.0f;
        int refr_out=refract3(*vx,*vy,*vz,-rnx,-rny,-rnz,n_particle,n_medium,&ox,&oy,&oz);
        if(!refr_out || dot3(ox,oy,oz,nox,noy,noz)<=1e-7f){
            rnx=nox;rny=noy;rnz=noz;cii=cii_geom;
            refr_out=refract3(*vx,*vy,*vz,-rnx,-rny,-rnz,n_particle,n_medium,&ox,&oy,&oz);
        }
        float eta_real=n_particle/n_medium;
        float st2=eta_real*eta_real*fmaxf(0.0f,1.0f-cii*cii);
        int tir_real=(st2>=1.0f);
        float Ri=tir_real?1.0f:fresnel_R_complex(cii,n_particle,k_particle,n_medium,0.0f);
        if(!tir_real&&refr_out&&rnd_uniform(state)>=Ri){*vx=ox;*vy=oy;*vz=oz;return 1;}
        reflect3(*vx,*vy,*vz,rnx,rny,rnz,&ox,&oy,&oz);
        if(dot3(ox,oy,oz,nox,noy,noz)>=-1e-7f)reflect3(*vx,*vy,*vz,nox,noy,noz,&ox,&oy,&oz);
        *vx=ox;*vy=oy;*vz=oz;(*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces)return 0;
    }
    return 0;
}

// V24.9 incoherent wave-morphology angular event.  This reuses the
// V23/V24.5/V24.8 particle-interface physics to obtain a physically generated
// outgoing direction from a rough dielectric particle, but keeps the event at
// the macroscopic scattering point.  This avoids falsely translating a wave
// event by a physical sphere chord when sigma_wave exceeds projected area.
__device__ int rough_particle_scatter_direction_only_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* vx,float* vy,float* vz,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions) {
    float lx=0.0f,ly=0.0f,lz=0.0f;
    float lvx=*vx,lvy=*vy,lvz=*vz,internal_path=0.0f;
    int ps=sphere_fresnel_interaction_3d(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        &lx,&ly,&lz,&lvx,&lvy,&lvz,&internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions);
    if(ps==1){*vx=lvx;*vy=lvy;*vz=lvz;}
    return ps;
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

// V24.10.2: finite 3-mm acrylic bottom disc.  The ray arrives at z_inner from water.
// return 1 = back in water, 2 = transmitted below disc (lost), 0 = failure.
__device__ int bottom_acrylic_disc(
    unsigned int* state,float n_water,float n_acrylic,float n_external,
    float z_inner,float thickness,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* inner_reflections,int* outer_reflections,int* axial_tir) {
    float a,b,c; int tir=0;
    // water -> acrylic, with +z normal pointing into incident water.
    float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,1.0f),0.0f),1.0f);
    float R=fresnel_R(ci,n_water,n_acrylic,&tir);
    if(tir || rnd_uniform(state)<R || !refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_water,n_acrylic,&a,&b,&c)){
        reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;
        (*inner_reflections)++; if(tir)(*axial_tir)++; *z=z_inner+EPS_F; return 1;
    }
    *vx=a;*vy=b;*vz=c;*z=z_inner-EPS_F;
    float z_outer=z_inner-thickness;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(*vz< -1e-12f){
            float t=(z_outer-*z)/(*vz); if(t<=0.0f) return 0;
            *x+=t*(*vx);*y+=t*(*vy);*z=z_outer;*acrylic_path+=t;
            // acrylic -> external, +z normal points into incident acrylic.
            ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,1.0f),0.0f),1.0f);tir=0;
            R=fresnel_R(ci,n_acrylic,n_external,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_acrylic,n_external,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=z_outer-EPS_F;return 2;
            }
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;
            (*outer_reflections)++; if(tir)(*axial_tir)++; *z=z_outer+EPS_F;
        } else if(*vz>1e-12f){
            float t=(z_inner-*z)/(*vz); if(t<=0.0f) return 0;
            *x+=t*(*vx);*y+=t*(*vy);*z=z_inner;*acrylic_path+=t;
            // acrylic -> water, -z normal points into incident acrylic.
            ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,-1.0f),0.0f),1.0f);tir=0;
            R=fresnel_R(ci,n_acrylic,n_water,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,n_acrylic,n_water,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=z_inner+EPS_F;return 1;
            }
            reflect3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;
            (*inner_reflections)++; if(tir)(*axial_tir)++; *z=z_inner-EPS_F;
        } else return 0;
        if(bounce==max_bounces)return 0;
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



__global__ void build_theta_grid_kernel(double* theta,const int ntheta){
    int j=blockDim.x*blockIdx.x+threadIdx.x;if(j>=ntheta)return;
    if(j==0){theta[j]=0.0;return;}if(j==ntheta-1){theta[j]=PI_D;return;}
    double u=(double)j/(double)(ntheta-1);
    theta[j]=0.5*PI_D*(1.0-cos(PI_D*u));
}

__global__ void mie_coefficients_efficiencies_kernel(
    const double* diameter,const double* n_real,const double* k_imag,
    const double n_medium,const double wavelength,const int n_particles,
    double* D_re,double* D_im,double* a_re,double* a_im,double* b_re,double* b_im,
    int* nstop_out,double* size_parameter,double* qext,double* qsca,double* qabs,double* qback,double* asym_g,double* albedo,
    int* error_flag){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n_particles)return;
    double x=PI_D*n_medium*diameter[p]/wavelength;
    size_parameter[p]=x;
    if(!(x>0.0) || !(n_real[p]>0.0) || !(n_medium>0.0) || !(wavelength>0.0)){
        nstop_out[p]=0;qext[p]=qsca[p]=qabs[p]=qback[p]=asym_g[p]=0.0;albedo[p]=1.0;atomicExch(error_flag,2);return;
    }
    double mr=n_real[p]/n_medium,mi=k_imag[p]/n_medium;
    int nstop=(int)(x+4.0*cbrt(x)+2.0);if(nstop<1)nstop=1;
    double mz=hypot(mr*x,mi*x);
    int nmx=(int)fmax((double)nstop,mz)+15;if(nmx<nstop+1)nmx=nstop+1;
    if(nmx>=MIE_MAX_ORDER || nstop>=MIE_MAX_ORDER){nstop_out[p]=0;atomicExch(error_flag,1);return;}
    nstop_out[p]=nstop;
    int base=p*MIE_MAX_ORDER;
    D_re[base+nmx]=0.0;D_im[base+nmx]=0.0;
    cpxd z=CD(mr*x,mi*x);
    for(int n=nmx;n>=1;--n){
        cpxd nz=cd_div(CD((double)n,0.0),z);
        cpxd dn=CD(D_re[base+n],D_im[base+n]);
        cpxd val=cd_sub(nz,cd_div(CD(1.0,0.0),cd_add(dn,nz)));
        D_re[base+n-1]=val.re;D_im[base+n-1]=val.im;
    }
    double psi_nm1=sin(x),psi_n=sin(x)/x-cos(x);
    double chi_nm1=cos(x),chi_n=cos(x)/x+sin(x);
    cpxd m=CD(mr,mi);
    for(int j=0;j<nstop;++j){
        int n=j+1;
        cpxd xi_nm1=CD(psi_nm1,-chi_nm1),xi_n=CD(psi_n,-chi_n);
        cpxd dn=CD(D_re[base+n],D_im[base+n]);
        cpxd ca=cd_add(cd_div(dn,m),CD((double)n/x,0.0));
        cpxd cb=cd_add(cd_mul(m,dn),CD((double)n/x,0.0));
        cpxd aa=cd_div(cd_sub(cd_scale(ca,psi_n),CD(psi_nm1,0.0)),cd_sub(cd_mul(ca,xi_n),xi_nm1));
        cpxd bb=cd_div(cd_sub(cd_scale(cb,psi_n),CD(psi_nm1,0.0)),cd_sub(cd_mul(cb,xi_n),xi_nm1));
        a_re[base+j]=aa.re;a_im[base+j]=aa.im;b_re[base+j]=bb.re;b_im[base+j]=bb.im;
        double psi_np1=((2.0*n+1.0)/x)*psi_n-psi_nm1;
        double chi_np1=((2.0*n+1.0)/x)*chi_n-chi_nm1;
        psi_nm1=psi_n;psi_n=psi_np1;chi_nm1=chi_n;chi_n=chi_np1;
    }
    double se=0.0,ss=0.0,sbr=0.0,sbi=0.0,t1=0.0,t2=0.0;
    for(int j=0;j<nstop;++j){
        int n=j+1;double cn=2.0*n+1.0;
        cpxd aa=CD(a_re[base+j],a_im[base+j]),bb=CD(b_re[base+j],b_im[base+j]);
        se+=cn*(aa.re+bb.re);ss+=cn*(cd_abs2(aa)+cd_abs2(bb));
        double sign=(n&1)?-1.0:1.0;sbr+=cn*sign*(aa.re-bb.re);sbi+=cn*sign*(aa.im-bb.im);
        t2+=(cn/((double)n*(n+1.0)))*cd_re_mul_conj(aa,bb);
        if(j+1<nstop){
            cpxd an=CD(a_re[base+j+1],a_im[base+j+1]),bn=CD(b_re[base+j+1],b_im[base+j+1]);
            t1+=((double)n*(n+2.0)/(n+1.0))*(cd_re_mul_conj(aa,an)+cd_re_mul_conj(bb,bn));
        }
    }
    double x2=x*x;double qe=2.0*se/x2,qs=2.0*ss/x2,qb=(sbr*sbr+sbi*sbi)/x2;
    if(qe<0.0 && qe>-1e-12)qe=0.0;if(qs<0.0 && qs>-1e-12)qs=0.0;
    double qa=qe-qs;if(qa<0.0 && qa>-1e-10)qa=0.0;
    double gg=qs>0.0?4.0*(t1+t2)/(x2*qs):0.0;if(gg>1.0)gg=1.0;if(gg<-1.0)gg=-1.0;
    qext[p]=qe;qsca[p]=qs;qabs[p]=qa;qback[p]=qb;asym_g[p]=gg;albedo[p]=qe>0.0?qs/qe:1.0;
}

__global__ void mie_phase_pdf_kernel(
    const double* theta,const int ntheta,const int n_particles,const int* nstop,
    const double* a_re,const double* a_im,const double* b_re,const double* b_im,double* phase_pdf){
    long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)n_particles*ntheta;if(idx>=total)return;
    int p=(int)(idx/ntheta),j=(int)(idx-(long long)p*ntheta),base=p*MIE_MAX_ORDER;
    double th=theta[j],mu=cos(th),s1r=0.0,s1i=0.0,s2r=0.0,s2i=0.0;
    double pi_nm1=0.0,pi_n=1.0;
    for(int k=0;k<nstop[p];++k){
        int n=k+1;double nf=(double)n;
        double tau=nf*mu*pi_n-(nf+1.0)*pi_nm1;
        double c=(2.0*nf+1.0)/(nf*(nf+1.0));
        double ar=a_re[base+k],ai=a_im[base+k],br=b_re[base+k],bi=b_im[base+k];
        s1r+=c*(ar*pi_n+br*tau);s1i+=c*(ai*pi_n+bi*tau);
        s2r+=c*(ar*tau+br*pi_n);s2i+=c*(ai*tau+bi*pi_n);
        double pi_np1=((2.0*nf+1.0)/nf)*mu*pi_n-((nf+1.0)/nf)*pi_nm1;
        pi_nm1=pi_n;pi_n=pi_np1;
    }
    double intensity=0.5*(s1r*s1r+s1i*s1i+s2r*s2r+s2i*s2i);
    phase_pdf[idx]=intensity*sin(th);
}

__global__ void mie_phase_cdf_kernel(const double* theta,const double* pdf,const int ntheta,const int n_particles,double* cdf,int* error_flag){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n_particles)return;
    long long base=(long long)p*ntheta;cdf[base]=0.0;double s=0.0;
    for(int j=1;j<ntheta;++j){
        double area=0.5*(pdf[base+j]+pdf[base+j-1])*(theta[j]-theta[j-1]);
        if(area<0.0 && area>-1e-18)area=0.0;s+=area;cdf[base+j]=s;
    }
    if(!(s>0.0) || !isfinite(s)){atomicExch(error_flag,3);for(int j=0;j<ntheta;++j)cdf[base+j]=(double)j/(double)(ntheta-1);return;}
    double prev=0.0;for(int j=0;j<ntheta;++j){double v=cdf[base+j]/s;if(v<prev)v=prev;if(v>1.0)v=1.0;cdf[base+j]=v;prev=v;}cdf[base+ntheta-1]=1.0;
}

__global__ void dummy_phase_kernel(double* theta,double* cdf,const int n_particles){
    if(blockIdx.x==0&&threadIdx.x==0){theta[0]=0.0;theta[1]=PI_D;}
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p<n_particles){cdf[2*p]=0.0;cdf[2*p+1]=1.0;}
}

#define AGG_MAX_BINS 64

// Inherited orthokinetic compact-aggregation population balance (disabled in V24.16).
// Independent variable exposure = alpha_stick * G * tau, dimensionless.
// The dimensional shear kernel K=(4/3)G(ri+rj)^3 therefore becomes
// K_E=(4/3)(ri+rj)^3 with respect to exposure E. Concentration enters
// through the initial number densities, so the PSD evolution is generated
// by collision physics rather than by a fitted concentration multiplier.
__global__ void orthokinetic_compact_aggregation_kernel(
    const double* diameter,const double* base_weights,const int n_particles,
    const double concentration,const double density,const double exposure,
    const double step_safety,const int max_steps,
    double* effective_weights,double* aggregation_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(n_particles<1 || n_particles>AGG_MAX_BINS){atomicExch(error_flag,20);return;}
    if(!(density>0.0) || concentration<0.0 || exposure<0.0 || !(step_safety>0.0) || step_safety>0.25){atomicExch(error_flag,21);return;}
    double sw=0.0; for(int i=0;i<n_particles;++i) sw+=base_weights[i];
    if(!(sw>0.0)){atomicExch(error_flag,22);return;}
    double vol[AGG_MAX_BINS],mass[AGG_MAX_BINS],ncur[AGG_MAX_BINS],dn[AGG_MAX_BINS];
    double initial_n=0.0,initial_area=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i],v=(PI_D/6.0)*d*d*d,m=density*v,w=base_weights[i]/sw;
        vol[i]=v;mass[i]=m;
        double ni=(concentration>0.0)?concentration*w/m:0.0;
        ncur[i]=ni;initial_n+=ni;initial_area+=ni*PI_D*0.25*d*d;
    }
    if(exposure<=0.0 || concentration<=0.0){
        for(int i=0;i<n_particles;++i) effective_weights[i]=base_weights[i]/sw;
        aggregation_scalars[0]=exposure; aggregation_scalars[1]=initial_n; aggregation_scalars[2]=initial_n;
        aggregation_scalars[3]=0.0; aggregation_scalars[4]=0.0; aggregation_scalars[5]=initial_area; aggregation_scalars[6]=initial_area;
        aggregation_scalars[7]=base_weights[n_particles-1]/sw; aggregation_scalars[8]=0.0; aggregation_scalars[9]=0.0;
        return;
    }
    double s=0.0; int steps=0; double overflow_events=0.0,total_events=0.0;
    while(s<exposure && steps<max_steps){
        double maxloss=0.0;
        for(int i=0;i<n_particles;++i){
            double Li=0.0,ri=0.5*diameter[i];
            for(int j=0;j<n_particles;++j){double q=ri+0.5*diameter[j]; Li+=(4.0/3.0)*q*q*q*ncur[j];}
            if(Li>maxloss)maxloss=Li;
        }
        double h=exposure-s;
        if(maxloss>0.0){double hs=step_safety/maxloss;if(h>hs)h=hs;}
        if(!(h>0.0) || !isfinite(h)){atomicExch(error_flag,23);return;}
        for(int i=0;i<n_particles;++i)dn[i]=0.0;
        for(int i=0;i<n_particles;++i){
            double ri=0.5*diameter[i];
            for(int j=i;j<n_particles;++j){
                double q=ri+0.5*diameter[j]; double K=(4.0/3.0)*q*q*q;
                double rate=(i==j)?0.5*K*ncur[i]*ncur[i]:K*ncur[i]*ncur[j];
                if(!(rate>0.0))continue;
                total_events+=rate*h;
                if(i==j)dn[i]-=2.0*rate; else {dn[i]-=rate;dn[j]-=rate;}
                double vn=vol[i]+vol[j];
                if(vn>=vol[n_particles-1]){
                    dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);
                    if(vn>vol[n_particles-1]) overflow_events+=rate*h;
                }else{
                    int k=0; while(k+1<n_particles && vol[k+1]<vn)++k;
                    if(k+1>=n_particles){dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);}
                    else{
                        double f=(vn-vol[k])/(vol[k+1]-vol[k]); if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                        dn[k]+=rate*(1.0-f); dn[k+1]+=rate*f;
                    }
                }
            }
        }
        for(int i=0;i<n_particles;++i){
            double nv=ncur[i]+h*dn[i];
            if(nv<0.0 && nv>-1e-12*fmax(ncur[i],1.0))nv=0.0;
            if(nv<0.0 || !isfinite(nv)){atomicExch(error_flag,24);return;}
            ncur[i]=nv;
        }
        s+=h; ++steps;
    }
    if(s+1e-12<exposure){atomicExch(error_flag,25);return;}
    double final_mass=0.0,final_n=0.0,final_area=0.0;
    for(int i=0;i<n_particles;++i){final_mass+=ncur[i]*mass[i];final_n+=ncur[i];final_area+=ncur[i]*PI_D*0.25*diameter[i]*diameter[i];}
    if(!(final_mass>0.0)){atomicExch(error_flag,26);return;}
    for(int i=0;i<n_particles;++i)effective_weights[i]=(ncur[i]*mass[i])/final_mass;
    double mass_error=(concentration>0.0)?(final_mass-concentration)/concentration:0.0;
    double num_reduction=(initial_n>0.0)?1.0-final_n/initial_n:0.0;
    aggregation_scalars[0]=exposure; aggregation_scalars[1]=initial_n; aggregation_scalars[2]=final_n;
    aggregation_scalars[3]=num_reduction; aggregation_scalars[4]=mass_error; aggregation_scalars[5]=initial_area; aggregation_scalars[6]=final_area;
    aggregation_scalars[7]=effective_weights[n_particles-1]; aggregation_scalars[8]=(total_events>0.0)?overflow_events/total_events:0.0; aggregation_scalars[9]=(double)steps;
}


// Inherited coupled orthokinetic aggregation + shear-fragmentation population balance (disabled in V24.16).
//
// Aggregation uses the V24.11 Smoluchowski orthokinetic kernel with a common
// dimensionless exposure EA = alpha_stick * G * tau.
//
// Fragmentation is a first-order binary-breakage process with a common exposure
// EB defined at reference diameter d_ref:
//
//     B_i * tau = EB * (d_i / d_ref)^m
//
// Each resolved breakage event creates two equal-volume daughters. Daughter
// volume is conservatively mapped onto adjacent PSD volume bins. If the daughter
// size lies below the smallest resolved bin, the parent mass is retained in the
// smallest bin and the event is counted as a lower-grid underflow diagnostic.
//
// EA, EB, d_ref and m are shared across concentrations. Concentration enters
// only through number density in the aggregation term. Thus low concentration
// can be fragmentation-dominated while high concentration remains collision-
// dominated without any concentration-specific optical or PSD multiplier.
__global__ void orthokinetic_compact_aggregation_fragmentation_kernel(
    const double* diameter,const double* base_weights,const int n_particles,
    const double concentration,const double density,
    const double aggregation_exposure,const double breakup_exposure,
    const double breakup_reference_diameter,const double breakup_size_exponent,
    const double step_safety,const int max_steps,
    double* effective_weights,double* population_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(n_particles<1 || n_particles>AGG_MAX_BINS){atomicExch(error_flag,30);return;}
    if(!(density>0.0) || concentration<0.0 || aggregation_exposure<0.0 || breakup_exposure<0.0 ||
       !(breakup_reference_diameter>0.0) || breakup_size_exponent<0.0 ||
       !(step_safety>0.0) || step_safety>0.25){atomicExch(error_flag,31);return;}
    double sw=0.0; for(int i=0;i<n_particles;++i)sw+=base_weights[i];
    if(!(sw>0.0)){atomicExch(error_flag,32);return;}
    double vol[AGG_MAX_BINS],mass[AGG_MAX_BINS],ncur[AGG_MAX_BINS],dn[AGG_MAX_BINS];
    double initial_n=0.0,initial_area=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i],v=(PI_D/6.0)*d*d*d,mass_i=density*v,w=base_weights[i]/sw;
        vol[i]=v;mass[i]=mass_i;
        double ni=(concentration>0.0)?concentration*w/mass_i:0.0;
        ncur[i]=ni;initial_n+=ni;initial_area+=ni*PI_D*0.25*d*d;
    }
    if(concentration<=0.0 || (aggregation_exposure<=0.0 && breakup_exposure<=0.0)){
        for(int i=0;i<n_particles;++i)effective_weights[i]=base_weights[i]/sw;
        population_scalars[0]=aggregation_exposure;
        population_scalars[1]=breakup_exposure;
        population_scalars[2]=breakup_reference_diameter;
        population_scalars[3]=breakup_size_exponent;
        population_scalars[4]=initial_n;population_scalars[5]=initial_n;
        population_scalars[6]=0.0;population_scalars[7]=0.0;
        population_scalars[8]=initial_area;population_scalars[9]=initial_area;
        population_scalars[10]=base_weights[n_particles-1]/sw;
        population_scalars[11]=0.0;population_scalars[12]=0.0;
        population_scalars[13]=0.0;population_scalars[14]=0.0;population_scalars[15]=0.0;
        return;
    }
    double u=0.0;
    int steps=0;
    double aggregation_overflow_events=0.0,total_aggregation_events=0.0;
    double fragmentation_underflow_events=0.0,total_fragmentation_events=0.0;
    while(u<1.0 && steps<max_steps){
        double maxloss=0.0;
        for(int i=0;i<n_particles;++i){
            double ri=0.5*diameter[i],Lagg=0.0;
            if(aggregation_exposure>0.0){
                for(int j=0;j<n_particles;++j){
                    double q=ri+0.5*diameter[j];
                    Lagg+=(4.0/3.0)*q*q*q*ncur[j];
                }
                Lagg*=aggregation_exposure;
            }
            double Lbreak=0.0;
            if(i>0 && breakup_exposure>0.0){
                Lbreak=breakup_exposure*pow(diameter[i]/breakup_reference_diameter,breakup_size_exponent);
            }
            double Li=Lagg+Lbreak;if(Li>maxloss)maxloss=Li;
        }
        double h=1.0-u;
        if(maxloss>0.0){double hs=step_safety/maxloss;if(h>hs)h=hs;}
        if(!(h>0.0) || !isfinite(h)){atomicExch(error_flag,33);return;}
        for(int i=0;i<n_particles;++i)dn[i]=0.0;

        if(aggregation_exposure>0.0){
            for(int i=0;i<n_particles;++i){
                double ri=0.5*diameter[i];
                for(int j=i;j<n_particles;++j){
                    double q=ri+0.5*diameter[j];double K=(4.0/3.0)*q*q*q;
                    double rate=aggregation_exposure*((i==j)?0.5*K*ncur[i]*ncur[i]:K*ncur[i]*ncur[j]);
                    if(!(rate>0.0))continue;
                    total_aggregation_events+=rate*h;
                    if(i==j)dn[i]-=2.0*rate;else{dn[i]-=rate;dn[j]-=rate;}
                    double vn=vol[i]+vol[j];
                    if(vn>=vol[n_particles-1]){
                        dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);
                        if(vn>vol[n_particles-1])aggregation_overflow_events+=rate*h;
                    }else{
                        int k=0;while(k+1<n_particles && vol[k+1]<vn)++k;
                        if(k+1>=n_particles){dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);}
                        else{
                            double f=(vn-vol[k])/(vol[k+1]-vol[k]);if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                            dn[k]+=rate*(1.0-f);dn[k+1]+=rate*f;
                        }
                    }
                }
            }
        }

        if(breakup_exposure>0.0){
            for(int i=1;i<n_particles;++i){
                double B=breakup_exposure*pow(diameter[i]/breakup_reference_diameter,breakup_size_exponent);
                double rate=B*ncur[i];if(!(rate>0.0))continue;
                total_fragmentation_events+=rate*h;dn[i]-=rate;
                double child_vol=0.5*vol[i];
                if(child_vol<=vol[0]){
                    // The equal-volume daughters are below the resolved grid.
                    // Retain total mass as an equivalent number of minimum-bin particles.
                    dn[0]+=rate*(vol[i]/vol[0]);
                    fragmentation_underflow_events+=rate*h;
                }else{
                    int k=0;while(k+1<n_particles && vol[k+1]<child_vol)++k;
                    if(k+1>=n_particles){atomicExch(error_flag,34);return;}
                    double f=(child_vol-vol[k])/(vol[k+1]-vol[k]);if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                    dn[k]+=2.0*rate*(1.0-f);dn[k+1]+=2.0*rate*f;
                }
            }
        }

        for(int i=0;i<n_particles;++i){
            double nv=ncur[i]+h*dn[i];
            if(nv<0.0 && nv>-1e-12*fmax(ncur[i],1.0))nv=0.0;
            if(nv<0.0 || !isfinite(nv)){atomicExch(error_flag,35);return;}
            ncur[i]=nv;
        }
        u+=h;++steps;
    }
    if(u+1e-12<1.0){atomicExch(error_flag,36);return;}
    double final_mass=0.0,final_n=0.0,final_area=0.0;
    for(int i=0;i<n_particles;++i){
        final_mass+=ncur[i]*mass[i];final_n+=ncur[i];final_area+=ncur[i]*PI_D*0.25*diameter[i]*diameter[i];
    }
    if(!(final_mass>0.0)){atomicExch(error_flag,37);return;}
    for(int i=0;i<n_particles;++i)effective_weights[i]=(ncur[i]*mass[i])/final_mass;
    double mass_error=(concentration>0.0)?(final_mass-concentration)/concentration:0.0;
    double number_reduction=(initial_n>0.0)?1.0-final_n/initial_n:0.0;
    population_scalars[0]=aggregation_exposure;
    population_scalars[1]=breakup_exposure;
    population_scalars[2]=breakup_reference_diameter;
    population_scalars[3]=breakup_size_exponent;
    population_scalars[4]=initial_n;population_scalars[5]=final_n;
    population_scalars[6]=number_reduction;population_scalars[7]=mass_error;
    population_scalars[8]=initial_area;population_scalars[9]=final_area;
    population_scalars[10]=effective_weights[n_particles-1];
    population_scalars[11]=(total_aggregation_events>0.0)?aggregation_overflow_events/total_aggregation_events:0.0;
    population_scalars[12]=(total_fragmentation_events>0.0)?fragmentation_underflow_events/total_fragmentation_events:0.0;
    population_scalars[13]=(initial_n>0.0)?total_aggregation_events/initial_n:0.0;
    population_scalars[14]=(initial_n>0.0)?total_fragmentation_events/initial_n:0.0;
    population_scalars[15]=(double)steps;
}

__global__ void transport_precompute_kernel(
    const double* diameter,const double* weights,const double* qext,const double* qsca,const int n_particles,
    const double concentration,const double density,const int weight_mode,const int event_model,const double wave_scale,
    const double n_medium,const double wavelength,const double wave_rms_height_m,
    double* radius,double* particle_mass,double* source_weight,double* number_density,
    double* sigma_geom,double* sigma_sca,double* sigma_ext,double* sigma_wave,double* sigma_event,
    double* mu_geom_bin,double* mu_sca_bin,double* mu_ext_bin,double* mu_wave_bin,double* mu_event_bin,
    double* particle_event_weight,double* geometric_event_weight,double* wave_event_weight,
    double* geometric_cdf,double* wave_cdf,double* particle_event_cdf,double* mu_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    double sw=0.0;for(int i=0;i<n_particles;++i)sw+=weights[i];if(!(sw>0.0)){atomicExch(error_flag,4);return;}
    double avg_mass=0.0;
    for(int i=0;i<n_particles;++i){
        double r=0.5*diameter[i];radius[i]=r;source_weight[i]=weights[i]/sw;
        double m=density*(4.0/3.0)*PI_D*r*r*r;particle_mass[i]=m;avg_mass+=source_weight[i]*m;
    }
    for(int i=0;i<n_particles;++i){
        double nd=0.0;
        if(weight_mode==0) nd=concentration*source_weight[i]/particle_mass[i];
        else nd=(avg_mass>0.0?concentration/avg_mass:0.0)*source_weight[i];
        number_density[i]=nd;
        double sg=PI_D*radius[i]*radius[i],ss=qsca[i]*sg,se=qext[i]*sg,swv=(event_model==3?wave_scale*fmax(qsca[i]-1.0,0.0)*sg:0.0);
        sigma_geom[i]=sg;sigma_sca[i]=ss;sigma_ext[i]=se;sigma_wave[i]=swv;
        mu_geom_bin[i]=nd*sg;mu_sca_bin[i]=nd*ss;mu_ext_bin[i]=nd*se;mu_wave_bin[i]=nd*swv;
    }
    double mg=0.0,ms=0.0,me=0.0,mw=0.0;
    for(int i=0;i<n_particles;++i){mg+=mu_geom_bin[i];ms+=mu_sca_bin[i];me+=mu_ext_bin[i];mw+=mu_wave_bin[i];}
    double mt=0.0;
    for(int i=0;i<n_particles;++i){
        double mb;
        if(event_model==0)mb=mu_geom_bin[i];else if(event_model==1)mb=mu_sca_bin[i];else if(event_model==2)mb=mu_ext_bin[i];else mb=mu_geom_bin[i]+mu_wave_bin[i];
        mu_event_bin[i]=mb;mt+=mb;
        if(event_model==0)sigma_event[i]=sigma_geom[i];else if(event_model==1)sigma_event[i]=sigma_sca[i];else if(event_model==2)sigma_event[i]=sigma_ext[i];else sigma_event[i]=sigma_geom[i]+sigma_wave[i];
    }
    double cg=0.0,cw=0.0,ct=0.0;
    for(int i=0;i<n_particles;++i){
        double pg=(event_model==3?(mg>0.0?mu_geom_bin[i]/mg:0.0):(mt>0.0?mu_event_bin[i]/mt:0.0));
        double pw=(event_model==3&&mw>0.0?mu_wave_bin[i]/mw:0.0),pt=(mt>0.0?mu_event_bin[i]/mt:0.0);
        geometric_event_weight[i]=pg;wave_event_weight[i]=pw;particle_event_weight[i]=pt;
        cg+=pg;cw+=pw;ct+=pt;geometric_cdf[i]=cg;wave_cdf[i]=cw;particle_event_cdf[i]=ct;
    }
    if(n_particles>0){if(cg>0.0)geometric_cdf[n_particles-1]=1.0;if(cw>0.0)wave_cdf[n_particles-1]=1.0;if(ct>0.0)particle_event_cdf[n_particles-1]=1.0;}
    // [trace geometric channel, trace wave channel, physical geom, sca, ext, wave, total event, wave coherent fraction]
    mu_scalars[0]=(event_model==3)?mg:mt;mu_scalars[1]=(event_model==3)?mw:0.0;
    mu_scalars[2]=mg;mu_scalars[3]=ms;mu_scalars[4]=me;mu_scalars[5]=(event_model==3)?mw:0.0;mu_scalars[6]=mt;
    double cwave=1.0;
    if(event_model==3 && wave_rms_height_m>0.0 && wavelength>0.0 && n_medium>0.0){
        double a=4.0*PI_D*n_medium*wave_rms_height_m/wavelength;
        cwave=exp(-a*a);
        if(cwave<0.0)cwave=0.0;if(cwave>1.0)cwave=1.0;
    }
    mu_scalars[7]=cwave;
}


// Generate the V24.16 drift-diffusion field entirely on CUDA.  One thread is
// intentional: there are only 37/38 PSD bins and this deterministic startup
// integration is negligible beside ray tracing while avoiding reduction races.
__global__ void sediment_transport_precompute_kernel(
    const double* diameter,const int n_particles,const int TRANSPORT_MODEL,
    const double particle_density,const double water_density,const double water_dynamic_viscosity,
    const double gravity,const double eddy_diffusivity,const double R,const double ZMIN,const double ZMEAN,
    const int FREE_SURFACE_MODEL,const double VORTEX_DELTA_H,const double VORTEX_CORE_RADIUS,const int NQUAD,
    const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    double* RESPONSE_TIME,double* SETTLING_VELOCITY,double* SED_A,double* SED_B,double* SED_NORM_SCALED,double* SED_MAX_SCALE,
    double* SCALE_SENSOR_CENTER,double* SCALE_SENSOR_WALL,double* SCALE_BOTTOM_CENTER,double* SCALE_BOTTOM_WALL,
    double* MAJORANTS,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(TRANSPORT_MODEL==0){
        double mg=0.0,mw=0.0;
        for(int i=0;i<n_particles;++i){RESPONSE_TIME[i]=0.0;SETTLING_VELOCITY[i]=0.0;SED_A[i]=0.0;SED_B[i]=0.0;SED_NORM_SCALED[i]=1.0;SED_MAX_SCALE[i]=1.0;SCALE_SENSOR_CENTER[i]=1.0;SCALE_SENSOR_WALL[i]=1.0;SCALE_BOTTOM_CENTER[i]=1.0;SCALE_BOTTOM_WALL[i]=1.0;mg+=TRACE_GEOM_BIN[i];mw+=TRACE_WAVE_BIN[i];}
        MAJORANTS[0]=mg;MAJORANTS[1]=mw;MAJORANTS[2]=mg+mw;MAJORANTS[3]=0.0;return;
    }
    if(TRANSPORT_MODEL!=1 || !(particle_density>water_density) || !(water_density>0.0) || !(water_dynamic_viscosity>0.0) || !(gravity>0.0) || !(eddy_diffusivity>0.0) || !(R>0.0) || !(VORTEX_CORE_RADIUS>0.0) || NQUAD<64){atomicExch(error_flag,31);return;}
    if(VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL!=2){atomicExch(error_flag,32);return;}
    double R2=R*R,a2=VORTEX_CORE_RADIUS*VORTEX_CORE_RADIUS;
    double Hinf=(VORTEX_DELTA_H>0.0)?VORTEX_DELTA_H*(a2+R2)/R2:0.0;
    double hbar=(VORTEX_DELTA_H>0.0)?Hinf*(1.0-(a2/R2)*log1p(R2/a2)):0.0;
    double omega2=(VORTEX_DELTA_H>0.0)?2.0*gravity*Hinf/a2:0.0;
    double fR=R2/(a2+R2),Hmean=ZMEAN-ZMIN,dr=R/(double)NQUAD;
    if(!(Hmean>0.0)){atomicExch(error_flag,33);return;}
    double major_g=0.0,major_w=0.0,maxA=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i];
        double tau=particle_density*d*d/(18.0*water_dynamic_viscosity);
        double ws=(particle_density-water_density)*gravity*d*d/(18.0*water_dynamic_viscosity);
        double A=tau*omega2*a2/(2.0*eddy_diffusivity);
        double B=ws/eddy_diffusivity;
        RESPONSE_TIME[i]=tau;SETTLING_VELOCITY[i]=ws;SED_A[i]=A;SED_B[i]=B;if(A>maxA)maxA=A;
        double integ=0.0;
        for(int j=0;j<NQUAD;++j){
            double r=((double)j+0.5)*dr,rr2=r*r,f=rr2/(a2+rr2);
            double zsurf=ZMEAN-hbar+Hinf*rr2/(a2+rr2);
            double H=fmax(zsurf-ZMIN,0.0);
            double radial=exp(A*(f-fR));
            double iz=(fabs(B)<1e-14)?radial*H:radial*(-expm1(-B*H))/B;
            integ+=r*iz;
        }
        double norm=(2.0/(R2*Hmean))*integ*dr;
        if(!(norm>0.0) || !isfinite(norm)){atomicExch(error_flag,34);return;}
        double mx=1.0/norm;SED_NORM_SCALED[i]=norm;SED_MAX_SCALE[i]=mx;
        // Four fixed diagnostic points.  Wall sample is kept just inside the acrylic interface.
        double f0=0.0,fw=(0.95*0.95*R2)/(a2+0.95*0.95*R2);
        double hs=fmax(0.0-ZMIN,0.0);
        SCALE_SENSOR_CENTER[i]=exp(A*(f0-fR)-B*hs)/norm;
        SCALE_SENSOR_WALL[i]=exp(A*(fw-fR)-B*hs)/norm;
        SCALE_BOTTOM_CENTER[i]=exp(A*(f0-fR))/norm;
        SCALE_BOTTOM_WALL[i]=exp(A*(fw-fR))/norm;
        major_g+=TRACE_GEOM_BIN[i]*mx;major_w+=TRACE_WAVE_BIN[i]*mx;
    }
    MAJORANTS[0]=major_g;MAJORANTS[1]=major_w;MAJORANTS[2]=major_g+major_w;MAJORANTS[3]=maxA;
    if(!(MAJORANTS[2]>=0.0) || !isfinite(MAJORANTS[2]))atomicExch(error_flag,35);
}

__global__ void sediment_field_test_kernel(
    const float* xyz,const int n_points,const int pidx,const int TRANSPORT_MODEL,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,
    const float R,const float ZMIN,const float CORE_R,float* out_scale){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n_points)return;
    out_scale[tid]=(float)sediment_local_scale(TRANSPORT_MODEL,pidx,SED_A,SED_B,SED_NORM_SCALED,xyz[3*tid],xyz[3*tid+1],xyz[3*tid+2],R,ZMIN,CORE_R);
}

__device__ int hardware_detector_id(float x,float y,float z,float vx,float vy,float vz,int mirror,
    float ring_in,float throat_out,float ring_out,float throat_r,float counterbore_r){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;float rth2=throat_r*throat_r,rcb2=counterbore_r*counterbore_r;
    for(int j=0;j<18;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1e-10f)continue;
        float pdot=x*ux+y*uy;
        float t0=(ring_in-pdot)/den,t1=(throat_out-pdot)/den,t2=(ring_out-pdot)/den;
        if(!(t0>0.0f&&t1>t0&&t2>t1))continue;
        float q0x=x+t0*vx-ring_in*ux,q0y=y+t0*vy-ring_in*uy,q0z=z+t0*vz;
        float q1x=x+t1*vx-throat_out*ux,q1y=y+t1*vy-throat_out*uy,q1z=z+t1*vz;
        float q2x=x+t2*vx-ring_out*ux,q2y=y+t2*vy-ring_out*uy,q2z=z+t2*vz;
        if(q0x*q0x+q0y*q0y+q0z*q0z<=rth2 && q1x*q1x+q1y*q1y+q1z*q1z<=rth2 && q2x*q2x+q2y*q2y+q2z*q2z<=rcb2){
            if(t0<best_t){best_t=t0;best=j;}
        }
    }
    return best;
}

__global__ void score_detectors_kernel(
    const float* air_x,const float* air_y,const float* air_z,const float* air_vx,const float* air_vy,const float* air_vz,
    const int* status,const int* interaction_count,const int n,
    const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,const float legacy_accept_deg,
    int* hardware_native,int* hardware_mirror,int* legacy_native,int* legacy_mirror,int* native_id,int* mirror_id){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;native_id[tid]=-1;mirror_id[tid]=-1;
    if(status[tid]!=1)return;
    float x=air_x[tid],y=air_y[tid],z=air_z[tid],vx=air_vx[tid],vy=air_vy[tid],vz=air_vz[tid];
    if(!(isfinite(x)&&isfinite(y)&&isfinite(z)&&isfinite(vx)&&isfinite(vy)&&isfinite(vz)))return;
    int ni=hardware_detector_id(x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    int mi=hardware_detector_id(x,y,z,vx,vy,vz,1,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    native_id[tid]=ni;mirror_id[tid]=mi;if(ni>=0)atomicAdd(&hardware_native[ni],1);if(mi>=0)atomicAdd(&hardware_mirror[mi],1);
    for(int mirror=0;mirror<2;++mirror){
        float xx=mirror?-x:x;float ang=atan2f(xx,y)*(180.0f/PI_F);if(ang<0.0f)ang+=360.0f;
        if(ang<0.0f||ang>180.0f)continue;
        for(int j=0;j<18;++j){float deg=10.0f*j;if(interaction_count[tid]<=0&&deg>=90.0f)continue;if(fabsf(ang-deg)<=legacy_accept_deg){if(mirror)atomicAdd(&legacy_mirror[j],1);else atomicAdd(&legacy_native[j],1);}}
    }
}

__global__ void free_surface_geometry_test_kernel(
    const float* rays6,const int N,const float ZMEAN,const float R,const int MODEL,
    const float DELTA_H,const float CORE_R,
    float* surface_z_start,float* t_hit,float* hit_surface_z,float* normal_x,float* normal_y,float* normal_z){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=N)return;
    const float* q=rays6+6*tid;
    float x=q[0],y=q[1],z=q[2],vx=q[3],vy=q[4],vz=q[5];
    surface_z_start[tid]=free_surface_z(x,y,ZMEAN,R,MODEL,DELTA_H,CORE_R);
    float t=free_surface_hit(x,y,z,vx,vy,vz,ZMEAN,R,MODEL,DELTA_H,CORE_R);
    t_hit[tid]=t;
    if(t<INF_F*0.5f){
        float hx=x+t*vx,hy=y+t*vy;
        hit_surface_z[tid]=free_surface_z(hx,hy,ZMEAN,R,MODEL,DELTA_H,CORE_R);
        free_surface_normal_into_water(hx,hy,R,MODEL,DELTA_H,CORE_R,&normal_x[tid],&normal_y[tid],&normal_z[tid]);
    }else{
        hit_surface_z[tid]=INF_F;normal_x[tid]=0.0f;normal_y[tid]=0.0f;normal_z[tid]=0.0f;
    }
}

__global__ void trace_kernel(
    const float MAX_ITERATIONS,const double* MU_CHANNELS,
    const float N_WATER,const float N_ACRYLIC,const float N_AIR,const float PARTICLE_WAVELENGTH,const float PARTICLE_SURFACE_ROUGH_ALPHA,const float WAVE_SPHEROID_ASPECT_RATIO,
    const int WAVE_SPHEROID_ORIENTATION_MODEL,const float WAVE_SPHEROID_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const int MAX_CELL_BOUNCES,
    const float RIN,const float ROUT,const float ZMIN,const float ZMAX,const int FREE_SURFACE_MODEL,const float VORTEX_DELTA_H,const float VORTEX_CORE_RADIUS,const float BOTTOM_DISC_THICKNESS,const float BOTTOM_EXTERNAL_N,const int MAX_AXIAL_BOUNCES,
    const float RING_IN_R,const float THROAT_OUT_R,const float RING_OUT_R,const float SOURCE_R,
    const float THROAT_R,const float COUNTERBORE_R,const float BEAM_SIGMA,
    const int VIS_SIZE,const float SOURCE_ALPHA1,const float SOURCE_ALPHA2,const int SOURCE_ANGULAR_MODEL,
    const int N_rays,
    const double* geometric_cdf,const double* wave_cdf,const double* mie_phase_cdf,const double* mie_theta,const int mie_ntheta,
    const double* particle_n,const double* particle_k,const double* particle_radius,const int n_particles,
    const int SEDIMENT_TRANSPORT_MODEL,const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_MAJORANTS,
    float* heat_xy,float* heat_zy,
    float* air_x,float* air_y,float* air_z,float* air_vx,float* air_vy,float* air_vz,
    float* water_path,float* acrylic_path,
    int* interaction_count,int* geometric_interactions,int* wave_interactions,int* coherent_wave_interactions,int* morphology_wave_interactions,
    int* particle_reflections,int* particle_entry_reflections,int* particle_internal_reflections,int* particle_absorptions,
    int* cell_inner_reflections,int* cell_outer_reflections,int* cell_tir,
    int* top_reflections,int* top_tir,int* bottom_inner_reflections,int* bottom_outer_reflections,int* bottom_tir,
    int* entered_water,int* status,
    unsigned int seed0,unsigned int seed1,unsigned int seed2,const unsigned int ray_offset) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x; if(tid>=N_rays)return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    unsigned int state=seed0+gid*74729u+13u, stateOPT=seed1+gid*104729u+29u, stateSRC=seed2+gid*130363u+43u;
    float MU_GEOM=(float)MU_CHANNELS[0],MU_WAVE=(float)MU_CHANNELS[1];
    float MU_MAJOR_GEOM=(SEDIMENT_TRANSPORT_MODEL==0)?MU_GEOM:(float)SED_MAJORANTS[0];
    float MU_MAJOR_WAVE=(SEDIMENT_TRANSPORT_MODEL==0)?MU_WAVE:(float)SED_MAJORANTS[1];
    float WAVE_COHERENT_FRACTION=fminf(fmaxf((float)MU_CHANNELS[7],0.0f),1.0f);
    int ic=0,gic=0,wic=0,wcoh=0,wmorph=0,pr=0,per=0,pir=0,pabs=0,cir=0,cor=0,ctir=0,tr=0,ttir=0,bir=0,bor=0,btir=0,entered=0,st=0;
    float wp=0.0f,ap=0.0f;

    // TARDIIS LED module is at 180 degrees. V24.4 reconstructs the sensor-ring
    // bore from the engineering drawing: ring OD 131 mm / ID 101 mm, an
    // 8.7-mm counterbore 8 mm deep, then a 4-mm through-bore to the inner face.
    // The emitting spot itself remains the pre-existing narrow Gaussian source;
    // only the mechanical launch/collimation geometry is changed here.
    float u1=fmaxf(rnd_uniform(&stateSRC),1e-12f),u2=rnd_uniform(&stateSRC);
    float mag=sqrtf(-2.0f*logf(u1));
    float x=BEAM_SIGMA*mag*cosf(2.0f*PI_F*u2);
    float z=BEAM_SIGMA*mag*sinf(2.0f*PI_F*u2);
    float y=-SOURCE_R;
    float polar=0.0f,az=0.0f,vx=0.0f,vy=1.0f,vz=0.0f;
    if(SOURCE_ANGULAR_MODEL==0){
        // Regression control: V24.10.1 sampled the Beta law directly as the
        // polar-angle probability density.
        polar=beta_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2)*(PI_F/2.0f);
        az=2.0f*PI_F*rnd_uniform(&stateSRC);
        float sp=sinf(polar);vx=sp*cosf(az);vy=cosf(polar);vz=sp*sinf(az);
    } else if(SOURCE_ANGULAR_MODEL==1){
        // Production correction: the Beta law specifies angular radiance;
        // include the 3-D solid-angle Jacobian before uniform azimuth sampling.
        polar=beta_radiance_polar_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2);
        az=2.0f*PI_F*rnd_uniform(&stateSRC);
        float sp=sinf(polar);vx=sp*cosf(az);vy=cosf(polar);vz=sp*sinf(az);
    } else {
        // Diagnostic reproduction of the thesis CLARITAS planar source:
        // signed horizontal Beta-distributed angle and zero vertical angle.
        float h=beta_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2)*(PI_F/2.0f);
        if(rnd_uniform(&stateSRC)<0.5f) h=-h;
        vx=sinf(h);vy=cosf(h);vz=0.0f;
    }
    normalize3(&vx,&vy,&vz);
    if(vy<=1e-12f){st=3;goto WRITE_OUT;}

    // At the outer ring face the source must lie inside the 8.7-mm counterbore.
    if(x*x+z*z>COUNTERBORE_R*COUNTERBORE_R){st=3;goto WRITE_OUT;}

    // The narrow 4-mm throat occupies RING_IN_R .. THROAT_OUT_R. A straight
    // ray is inside the cylindrical throat throughout iff both end-plane
    // transverse radii are inside the throat circle (norm is convex along a line).
    {
        float t=(-THROAT_OUT_R-y)/vy;
        if(t<-EPS_F){st=3;goto WRITE_OUT;}
        if(t<0.0f)t=0.0f;
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
        float zsurf_here=free_surface_z(x,y,ZMAX,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(z<=ZMIN || z>=zsurf_here){st=2;break;}
        float tside=cylinder_hit(x,y,vx,vy,RIN);
        float ttop=INF_F,tbottom=INF_F;
        ttop=free_surface_hit(x,y,z,vx,vy,vz,ZMAX,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(vz<-1e-14f) tbottom=(ZMIN-z)/vz;
        if(ttop<=EPS_F) ttop=INF_F;
        if(tbottom<=EPS_F) tbottom=INF_F;
        int axial_top=(ttop<=tbottom && ttop<tside);
        int axial_bottom=(tbottom<ttop && tbottom<tside);
        int axial=axial_top||axial_bottom;
        float tbound=axial_top?ttop:(axial_bottom?tbottom:tside);
        if(tbound>=INF_F*0.5f){st=4;break;}
        float freep=INF_F;
        float MU_TOTAL=MU_GEOM+MU_WAVE;
        float MU_TRACK=(SEDIMENT_TRANSPORT_MODEL==0)?MU_TOTAL:(MU_MAJOR_GEOM+MU_MAJOR_WAVE);
        if(MU_TRACK>0.0f){float u=fmaxf(rnd_uniform(&state),1e-12f);freep=-logf(u)/MU_TRACK;}
        int do_particle=freep<tbound;
        float travel=do_particle?freep:tbound;
        float x0=x,y0=y,z0=z;
        float zvismax=free_surface_z(RIN,0.0f,ZMAX,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        raster_water_segment(x0,y0,z0,vx,vy,vz,travel,VIS_SIZE,RIN,ZMIN,zvismax,heat_xy,heat_zy);
        x+=travel*vx;y+=travel*vy;z+=travel*vz;wp+=travel;
        if(do_particle){
            float LOCAL_GEOM=MU_GEOM,LOCAL_WAVE=MU_WAVE,LOCAL_TOTAL=MU_TOTAL;
            if(SEDIMENT_TRANSPORT_MODEL!=0){
                double lg=0.0,lw=0.0;
                for(int i=0;i<n_particles;++i){
                    double sc=sediment_local_scale(SEDIMENT_TRANSPORT_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS);
                    lg+=TRACE_GEOM_BIN[i]*sc;lw+=TRACE_WAVE_BIN[i]*sc;
                }
                LOCAL_GEOM=(float)lg;LOCAL_WAVE=(float)lw;LOCAL_TOTAL=LOCAL_GEOM+LOCAL_WAVE;
                if(!(LOCAL_TOTAL>0.0f))continue;
                float pacc=fminf(LOCAL_TOTAL/fmaxf(MU_TRACK,1e-30f),1.0f);
                if(rnd_uniform(&state)>=pacc)continue; // Woodcock null collision.
            }
            ic++;
            int use_wave=(LOCAL_WAVE>0.0f && rnd_uniform(&state)*LOCAL_TOTAL>=LOCAL_GEOM);
            if(use_wave){
                wic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,wave_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,TRACE_WAVE_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_WAVE);
                // sigma_h=0 gives C=1 and deliberately consumes no extra RNG,
                // preserving the V24.8.1 coherent-Mie random stream.
                int coherent=(WAVE_COHERENT_FRACTION>=1.0f) || (rnd_uniform(&stateOPT)<WAVE_COHERENT_FRACTION);
                if(coherent){
                    wcoh++;
                    mie_scatter_direction(&stateOPT,mie_phase_cdf,mie_theta,mie_ntheta,pidx,&vx,&vy,&vz);
                } else {
                    wmorph++;
                    float fr=sqrtf(x*x+y*y);
                    float ffx=1.0f,ffy=0.0f,ffz=0.0f;
                    if(fr>1e-12f){ffx=-y/fr;ffy=x/fr;}
                    int ps=spheroid_fresnel_scatter_direction_only_3d(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,
                        (float)particle_radius[pidx],WAVE_SPHEROID_ASPECT_RATIO,WAVE_SPHEROID_ORIENTATION_MODEL,WAVE_SPHEROID_ORIENTATION_KAPPA,ffx,ffy,ffz,
                        MAX_PARTICLE_BOUNCES,&vx,&vy,&vz,&pr,&per,&pir,&pabs);
                    if(ps==2){st=5;break;}
                    if(ps!=1){st=4;break;}
                }
            } else {
                gic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,geometric_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,TRACE_GEOM_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_GEOM);
                float internal=0.0f;
                int ps=sphere_fresnel_interaction_3d(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,(float)particle_radius[pidx],
                    MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&internal,&pr,&per,&pir,&pabs);
                if(ps==2){st=5;break;}
                if(ps!=1){st=4;break;}
            }
            continue;
        }
        if(axial){
            float a,b,c; int tir=0;
            if(axial_top){
                // Water -> air at the local flat/parabolic/localized free surface.  The normal
                // is defined into the incident water medium, matching refract3().
                float snx=0.0f,sny=0.0f,snz=-1.0f;
                free_surface_normal_into_water(x,y,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,&snx,&sny,&snz);
                float ci=fminf(fmaxf(-dot3(vx,vy,vz,snx,sny,snz),0.0f),1.0f);
                float R=fresnel_R(ci,N_WATER,N_AIR,&tir);
                if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,snx,sny,snz,N_WATER,N_AIR,&a,&b,&c)){
                    reflect3(vx,vy,vz,snx,sny,snz,&a,&b,&c);vx=a;vy=b;vz=c;tr++;if(tir)ttir++;
                    x+=EPS_F*snx;y+=EPS_F*sny;z+=EPS_F*snz;continue;
                }
                // V24.16 retains the established headspace treatment:
                // transmission through the free surface is counted as a top exit.
                // Reflected/TIR paths remain fully ray-traced and may hit the curved
                // interface repeatedly.
                st=6;break;
            } else {
                // Water -> finite acrylic bottom disc, including repeated acrylic-slab Fresnel/TIR.
                int where=bottom_acrylic_disc(&state,N_WATER,N_ACRYLIC,BOTTOM_EXTERNAL_N,ZMIN,BOTTOM_DISC_THICKNESS,MAX_AXIAL_BOUNCES,
                    &x,&y,&z,&vx,&vy,&vz,&ap,&bir,&bor,&btir);
                if(where==1) continue;
                if(where==2){st=7;break;}
                st=4;break;
            }
        }

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
    interaction_count[tid]=ic;geometric_interactions[tid]=gic;wave_interactions[tid]=wic;coherent_wave_interactions[tid]=wcoh;morphology_wave_interactions[tid]=wmorph;particle_reflections[tid]=pr;particle_entry_reflections[tid]=per;particle_internal_reflections[tid]=pir;particle_absorptions[tid]=pabs;
    cell_inner_reflections[tid]=cir;cell_outer_reflections[tid]=cor;cell_tir[tid]=ctir;
    top_reflections[tid]=tr;top_tir[tid]=ttir;bottom_inner_reflections[tid]=bir;bottom_outer_reflections[tid]=bor;bottom_tir[tid]=btir;
    entered_water[tid]=entered;status[tid]=st;
}
}
'''

def get_material_psd_raw(material: str):
    """Return authoritative diameter bins and raw source weights without host normalization."""
    m=material.strip().lower()
    if m=='loess': return LOESS_DIAMETER_M.copy(),LOESS_WEIGHTS.copy()
    if m=='kaolin': return KAOLIN_DIAMETER_M.copy(),KAOLIN_WEIGHTS.copy()
    raise ValueError("material must be 'loess' or 'kaolin'")


def get_material_psd(material: str):
    """Compatibility view with normalized weights; forward CUDA normalizes its own inputs."""
    d,w=get_material_psd_raw(material)
    return d,w/w.sum()



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
    particle_event_model: str
    particle_surface_rms_slope_deg: float
    particle_surface_beckmann_alpha: float
    wave_surface_rms_height_nm: float
    wave_spheroid_aspect_ratio: float
    wave_spheroid_orientation_model: str
    wave_spheroid_orientation_kappa: float
    aggregation_model: str
    aggregation_collision_exposure: float
    aggregation_number_reduction_fraction: float
    aggregation_mass_conservation_error: float
    aggregation_projected_area_ratio: float
    aggregation_largest_bin_mass_fraction: float
    aggregation_overflow_event_fraction: float
    aggregation_steps: int
    population_number_reduction_fraction: float
    fragmentation_breakup_exposure: float
    fragmentation_reference_diameter_m: float
    fragmentation_size_exponent: float
    fragmentation_underflow_event_fraction: float
    aggregation_events_per_initial_particle: float
    fragmentation_events_per_initial_particle: float
    wave_coherent_fraction_nominal_cuda: float
    mean_coherent_wave_interactions: float
    mean_morphology_wave_interactions: float
    morphology_fraction_of_wave_interactions: float
    mu_geom_per_m: float
    mu_wave_per_m: float
    mu_event_per_m: float
    mu_sca_per_m: float
    mu_ext_per_m: float
    geometric_optical_depth_diameter: float
    optical_depth_diameter: float
    mean_interactions: float
    mean_geometric_interactions: float
    mean_wave_interactions: float
    median_interactions: float
    ballistic_fraction: float
    mean_fresnel_reflections: float
    entry_reflection_fraction: float
    internal_reflection_fraction: float
    particle_absorption_fraction: float
    particle_absorbed_ray_count: int
    mean_cell_inner_reflections: float
    mean_cell_outer_reflections: float
    cell_tir_fraction: float
    mean_top_surface_reflections: float
    top_surface_tir_fraction: float
    mean_top_surface_interactions: float
    free_surface_model: str
    vortex_wall_center_height_difference_mm: float
    vortex_core_radius_mm: float
    vortex_asymptotic_height_parameter_mm: float
    vortex_area_mean_uncorrected_rise_mm: float
    vortex_equivalent_core_rpm: float
    vortex_equivalent_rigid_body_rpm: float
    vortex_center_surface_z_mm: float
    vortex_wall_surface_z_mm: float
    sediment_transport_model: str
    sediment_eddy_diffusivity_m2_s: float
    sediment_majorant_geom_per_m: float
    sediment_majorant_wave_per_m: float
    sediment_majorant_total_per_m: float
    sediment_max_bin_density_scale: float
    mean_bottom_inner_reflections: float
    mean_bottom_outer_reflections: float
    bottom_tir_fraction: float
    top_exit_fraction: float
    bottom_exit_fraction: float
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
    particle_diagnostics: Optional[Dict[str,np.ndarray]]=None
    ray_data: Optional[Dict[str,np.ndarray]]=None
    heatmap_xy: Optional[np.ndarray]=None
    heatmap_zy: Optional[np.ndarray]=None
    def to_dict(self):
        d={k:v for k,v in self.__dict__.items() if k not in {'particle_diagnostics','ray_data','heatmap_xy','heatmap_zy'}}
        for k,v in list(d.items()):
            if isinstance(v,np.ndarray): d[k]=v.tolist()
            elif isinstance(v,np.generic): d[k]=v.item()
        return d




class TardiisForwardModel:
    """CUDA-authoritative V24.16 vortex sediment drift-diffusion diagnostic.

    Host code performs configuration, array transfer, diagnostics aggregation and
    I/O only. Mie/event/phase/source/detector physics that determine ray outcomes
    are implemented in CUDA kernels in ``CUDA_SRC``.
    """
    MIE_MAX_ORDER=4096
    EVENT_MODEL_CODES={'geometric':0,'mie_qsca':1,'mie_qext':2,'hybrid_mie_excess':3}
    WEIGHT_MODE_CODES={'mass_fraction':0,'number_fraction':1}
    SOURCE_ANGULAR_MODEL_CODES={'legacy_polar_pdf':0,'beta_radiance_3d':1,'thesis_planar_beta':2}
    SPHEROID_ORIENTATION_MODEL_CODES={'isotropic':0,'local_tangential_axial_vmf':1}
    FREE_SURFACE_MODEL_CODES={'flat':0,'parabolic_vortex':1,'localized_scully_vortex':2}
    SEDIMENT_TRANSPORT_MODEL_CODES={'uniform':0,'stokes_drift_diffusion':1}

    def __init__(self,n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,
                 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
                 water_height_m=0.142,sensor_height_above_bottom_m=0.093,
                 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,
                 counterbore_depth_m=0.008,through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,
                 source_launch_radius_m=0.0655,legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1.0e-5,
                 particle_wavelength_m=DEFAULT_PARTICLE_WAVELENGTH_M,particle_optical_constants_by_material=None,
                 particle_event_model='geometric',hybrid_wave_scale=1.0,mie_phase_grid_size=2049,
                 particle_surface_rms_slope_deg=25.0,wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.5,
                 wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
                 free_surface_model='localized_scully_vortex',vortex_wall_center_height_difference_m=0.050,vortex_core_radius_m=0.020,
                 sediment_transport_model='uniform',sediment_eddy_diffusivity_m2_s=5.0e-4,water_density_kg_per_m3=997.0,water_dynamic_viscosity_pa_s=8.9e-4,gravity_m_s2=9.81,sediment_transport_quadrature_points=4096,
                 aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
                 fragmentation_reference_diameter_m=100e-6,fragmentation_size_exponent=1.0,
                 aggregation_step_safety=0.2,aggregation_max_steps=200000,
                 alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d',density_kg_per_m3=2600.0,
                 bottom_disc_thickness_m=0.003,bottom_external_refractive_index=1.0,max_axial_bounces=64,
                 max_iterations=int(1e6),max_internal_bounces=64,max_cell_bounces=64,
                 chunk_size=250000,statistics_batch_rays=250000,
                 symmetry_average_exact=True):
        try: import cupy as cp
        except Exception as exc: raise RuntimeError('CuPy/CUDA is required for CLARITAS V24.16.') from exc
        self.cp=cp;self.n_water=float(n_water);self.n_particle=float(n_particle);self.n_acrylic=float(n_acrylic);self.n_air=float(n_air)
        self.rin=float(tube_inner_radius_m);self.rout=float(tube_outer_radius_m)
        self.water_height=float(water_height_m);self.sensor_height=float(sensor_height_above_bottom_m)
        self.zmin=-self.sensor_height;self.zmax=self.water_height-self.sensor_height
        self.ring_in=float(sensor_ring_inner_radius_m);self.ring_out=float(sensor_ring_outer_radius_m)
        self.counterbore_depth=float(counterbore_depth_m);self.throat_out=self.ring_out-self.counterbore_depth
        self.throat_radius=0.5*float(through_bore_diameter_m);self.counterbore_radius=0.5*float(counterbore_diameter_m)
        self.source_launch_radius=float(source_launch_radius_m)
        self.legacy_acceptance_deg=float(legacy_detector_acceptance_deg);self.beam_sigma=float(source_beam_sigma_m)
        self.particle_wavelength=float(particle_wavelength_m);self.particle_optical_constants_by_material=particle_optical_constants_by_material or {}
        self.particle_event_model=str(particle_event_model).strip().lower()
        self.hybrid_wave_scale=float(hybrid_wave_scale);self.mie_phase_grid_size=int(mie_phase_grid_size)
        self.particle_surface_rms_slope_deg=float(particle_surface_rms_slope_deg)
        self.particle_surface_beckmann_alpha=float(np.tan(np.deg2rad(self.particle_surface_rms_slope_deg)))
        self.wave_surface_rms_height_m=float(wave_surface_rms_height_m)
        self.wave_spheroid_aspect_ratio=float(wave_spheroid_aspect_ratio)
        self.wave_spheroid_orientation_model=str(wave_spheroid_orientation_model).strip().lower()
        self.wave_spheroid_orientation_kappa=float(wave_spheroid_orientation_kappa)
        self.free_surface_model=str(free_surface_model).strip().lower()
        self.vortex_delta_h=float(vortex_wall_center_height_difference_m)
        self.vortex_core_radius=float(vortex_core_radius_m)
        self.sediment_transport_model=str(sediment_transport_model).strip().lower()
        self.sediment_eddy_diffusivity=float(sediment_eddy_diffusivity_m2_s)
        self.water_density=float(water_density_kg_per_m3)
        self.water_dynamic_viscosity=float(water_dynamic_viscosity_pa_s)
        self.gravity=float(gravity_m_s2)
        self.sediment_transport_quadrature_points=int(sediment_transport_quadrature_points)
        self.aggregation_model=str(aggregation_model).strip().lower()
        self.aggregation_collision_exposure=float(aggregation_collision_exposure)
        self.fragmentation_breakup_exposure=float(fragmentation_breakup_exposure)
        self.fragmentation_reference_diameter_m=float(fragmentation_reference_diameter_m)
        self.fragmentation_size_exponent=float(fragmentation_size_exponent)
        self.aggregation_step_safety=float(aggregation_step_safety)
        self.aggregation_max_steps=int(aggregation_max_steps)
        self.alpha1=float(alpha1);self.alpha2=float(alpha2)
        self.source_angular_model=str(source_angular_model).strip().lower()
        self.density=float(density_kg_per_m3)
        self.bottom_disc_thickness=float(bottom_disc_thickness_m);self.bottom_external_n=float(bottom_external_refractive_index);self.max_axial_bounces=int(max_axial_bounces)
        self.max_iterations=int(max_iterations);self.max_internal_bounces=int(max_internal_bounces);self.max_cell_bounces=int(max_cell_bounces)
        self.chunk_size=int(chunk_size);self.statistics_batch_rays=int(statistics_batch_rays)
        self.symmetry_average_exact=bool(symmetry_average_exact)
        if self.chunk_size<1 or self.statistics_batch_rays<1: raise ValueError('chunk/batch sizes must be positive')
        if not (0<self.rin<self.rout<self.ring_in<self.throat_out<=self.source_launch_radius<=self.ring_out): raise ValueError('TARDIIS radial/source geometry is inconsistent')
        if not (0<self.throat_radius<self.counterbore_radius): raise ValueError('through-bore must be narrower than counterbore')
        if not (0.0<self.legacy_acceptance_deg<90.0): raise ValueError('legacy detector acceptance must be between 0 and 90 degrees')
        if self.particle_wavelength<=0: raise ValueError('particle_wavelength_m must be positive')
        if self.particle_event_model not in self.EVENT_MODEL_CODES: raise ValueError('invalid particle_event_model')
        if self.hybrid_wave_scale<0 or not np.isfinite(self.hybrid_wave_scale): raise ValueError('hybrid_wave_scale must be finite and >=0')
        if self.mie_phase_grid_size<257: raise ValueError('mie_phase_grid_size must be >=257')
        if not np.isfinite(self.particle_surface_rms_slope_deg) or not (0.0<=self.particle_surface_rms_slope_deg<80.0): raise ValueError('particle_surface_rms_slope_deg must be finite and in [0,80) degrees')
        if not np.isfinite(self.wave_surface_rms_height_m) or self.wave_surface_rms_height_m<0.0: raise ValueError('wave_surface_rms_height_m must be finite and >=0')
        if not np.isfinite(self.wave_spheroid_aspect_ratio) or not (0.2<=self.wave_spheroid_aspect_ratio<=5.0): raise ValueError('wave_spheroid_aspect_ratio must be finite and in [0.2,5.0]')
        if self.wave_spheroid_orientation_model not in self.SPHEROID_ORIENTATION_MODEL_CODES:
            raise ValueError("wave_spheroid_orientation_model must be 'isotropic' or 'local_tangential_axial_vmf'")
        if not np.isfinite(self.wave_spheroid_orientation_kappa) or not (0.0<=self.wave_spheroid_orientation_kappa<=100.0):
            raise ValueError('wave_spheroid_orientation_kappa must be finite and in [0,100]')
        if self.free_surface_model not in self.FREE_SURFACE_MODEL_CODES:
            raise ValueError("free_surface_model must be 'flat', 'parabolic_vortex', or 'localized_scully_vortex'")
        if not np.isfinite(self.vortex_delta_h) or self.vortex_delta_h<0.0:
            raise ValueError('vortex_wall_center_height_difference_m must be finite and >=0')
        if not np.isfinite(self.vortex_core_radius) or self.vortex_core_radius<=0.0:
            raise ValueError('vortex_core_radius_m must be finite and >0')
        if self.sediment_transport_model not in self.SEDIMENT_TRANSPORT_MODEL_CODES:
            raise ValueError("sediment_transport_model must be 'uniform' or 'stokes_drift_diffusion'")
        if not np.isfinite(self.sediment_eddy_diffusivity) or self.sediment_eddy_diffusivity<=0.0:
            raise ValueError('sediment_eddy_diffusivity_m2_s must be finite and >0')
        if not (np.isfinite(self.water_density) and 0.0<self.water_density<self.density):
            raise ValueError('water_density_kg_per_m3 must be positive and below particle density')
        if not np.isfinite(self.water_dynamic_viscosity) or self.water_dynamic_viscosity<=0.0:
            raise ValueError('water_dynamic_viscosity_pa_s must be finite and >0')
        if not np.isfinite(self.gravity) or self.gravity<=0.0:
            raise ValueError('gravity_m_s2 must be finite and >0')
        if self.sediment_transport_quadrature_points<64:
            raise ValueError('sediment_transport_quadrature_points must be >=64')
        if self.sediment_transport_model!='uniform' and self.vortex_delta_h>0.0 and self.free_surface_model!='localized_scully_vortex':
            raise ValueError('V24.16 stokes_drift_diffusion with non-zero vortex depth requires localized_scully_vortex')
        vg=vortex_surface_geometry(self.free_surface_model,self.zmax,self.rin,self.vortex_delta_h,self.vortex_core_radius)
        if vg['center_z'] <= self.zmin:
            raise ValueError('vortex depression reaches or passes the acrylic bottom; reduce vortex height difference or increase core radius')
        if self.aggregation_model not in {'none','orthokinetic_compact','orthokinetic_compact_fragmentation'}:
            raise ValueError("aggregation_model must be 'none', 'orthokinetic_compact', or 'orthokinetic_compact_fragmentation'")
        if not np.isfinite(self.aggregation_collision_exposure) or self.aggregation_collision_exposure<0.0: raise ValueError('aggregation_collision_exposure must be finite and >=0')
        if not np.isfinite(self.fragmentation_breakup_exposure) or self.fragmentation_breakup_exposure<0.0: raise ValueError('fragmentation_breakup_exposure must be finite and >=0')
        if not np.isfinite(self.fragmentation_reference_diameter_m) or self.fragmentation_reference_diameter_m<=0.0: raise ValueError('fragmentation_reference_diameter_m must be finite and >0')
        if not np.isfinite(self.fragmentation_size_exponent) or not (0.0<=self.fragmentation_size_exponent<=6.0): raise ValueError('fragmentation_size_exponent must be finite and in [0,6]')
        if self.aggregation_model!='orthokinetic_compact_fragmentation' and self.fragmentation_breakup_exposure>0.0:
            raise ValueError('fragmentation_breakup_exposure requires aggregation_model=orthokinetic_compact_fragmentation')
        if not (0.0<self.aggregation_step_safety<=0.25): raise ValueError('aggregation_step_safety must be in (0,0.25]')
        if self.aggregation_max_steps<1: raise ValueError('aggregation_max_steps must be >=1')
        if not (self.alpha1>0.0 and self.alpha2>0.0): raise ValueError('source Beta alpha1/alpha2 must be positive')
        if self.source_angular_model not in self.SOURCE_ANGULAR_MODEL_CODES:
            raise ValueError("source_angular_model must be one of: legacy_polar_pdf, beta_radiance_3d, thesis_planar_beta")
        if not (self.zmin<0<self.zmax): raise ValueError('sensor plane must lie within water column')
        if self.bottom_disc_thickness<=0.0: raise ValueError('bottom_disc_thickness_m must be positive')
        if self.bottom_external_n<=0.0: raise ValueError('bottom_external_refractive_index must be positive')
        if self.max_axial_bounces<1: raise ValueError('max_axial_bounces must be >=1')
        self.module=cp.RawModule(code=CUDA_SRC,options=('-std=c++11',))
        names=['trace_kernel','free_surface_geometry_test_kernel','sediment_field_test_kernel','orientation_sampler_test_kernel','build_theta_grid_kernel','mie_coefficients_efficiencies_kernel','mie_phase_pdf_kernel','mie_phase_cdf_kernel','dummy_phase_kernel','orthokinetic_compact_aggregation_kernel','orthokinetic_compact_aggregation_fragmentation_kernel','transport_precompute_kernel','sediment_transport_precompute_kernel','score_detectors_kernel']
        self.kernels={n:self.module.get_function(n) for n in names}
        self.kernel=self.kernels['trace_kernel']

    def _prepare_gpu_particle_physics(self,material,concentration_g_per_L,weights=None,weight_mode='mass_fraction'):
        cp=self.cp;d,bw=get_material_psd_raw(material);w=bw if weights is None else np.asarray(weights,dtype=np.float64)
        if w.ndim!=1 or w.size!=d.size: raise ValueError(f'{material} requires {d.size} PSD weights')
        if np.any(~np.isfinite(w)) or np.any(w<0.0) or not np.any(w>0.0): raise ValueError('invalid PSD weights')
        if weight_mode not in self.WEIGHT_MODE_CODES: raise ValueError("weight_mode must be 'mass_fraction' or 'number_fraction'")
        conc=float(concentration_g_per_L)
        if not np.isfinite(conc) or conc<0.0: raise ValueError('concentration must be finite and >=0')
        nidx,kidx=get_material_optical_constants(material,self.particle_optical_constants_by_material,self.n_particle)
        n=int(d.size);M=self.MIE_MAX_ORDER
        dd=cp.asarray(d,dtype=cp.float64);wd_input=cp.asarray(w,dtype=cp.float64);nd=cp.asarray(nidx,dtype=cp.float64);kd=cp.asarray(kidx,dtype=cp.float64)
        wd=cp.empty_like(wd_input)
        agg_scalars=cp.zeros(10,dtype=cp.float64)
        population_scalars=cp.zeros(16,dtype=cp.float64)
        agg_err=cp.zeros(1,dtype=cp.int32)
        used_coupled_population_kernel=False
        if self.aggregation_model=='orthokinetic_compact':
            if weight_mode!='mass_fraction': raise ValueError('orthokinetic_compact aggregation currently requires mass_fraction PSD input')
            self.kernels['orthokinetic_compact_aggregation_kernel']((1,),(1,),(dd,wd_input,np.int32(n),np.float64(conc),np.float64(self.density),np.float64(self.aggregation_collision_exposure),np.float64(self.aggregation_step_safety),np.int32(self.aggregation_max_steps),wd,agg_scalars,agg_err))
            cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(agg_err)[0])
            if ecode: raise RuntimeError(f'CUDA aggregation precompute failed with error code {ecode}')
        elif self.aggregation_model=='orthokinetic_compact_fragmentation':
            if weight_mode!='mass_fraction': raise ValueError('orthokinetic_compact_fragmentation currently requires mass_fraction PSD input')
            # Preserve the exact V24.11 aggregation path when breakup exposure is zero.
            # This gives a clean release regression while the coupled CUDA kernel is used
            # whenever fragmentation is actually enabled.
            if self.fragmentation_breakup_exposure<=0.0:
                self.kernels['orthokinetic_compact_aggregation_kernel']((1,),(1,),(dd,wd_input,np.int32(n),np.float64(conc),np.float64(self.density),np.float64(self.aggregation_collision_exposure),np.float64(self.aggregation_step_safety),np.int32(self.aggregation_max_steps),wd,agg_scalars,agg_err))
            else:
                used_coupled_population_kernel=True
                self.kernels['orthokinetic_compact_aggregation_fragmentation_kernel']((1,),(1,),(
                    dd,wd_input,np.int32(n),np.float64(conc),np.float64(self.density),
                    np.float64(self.aggregation_collision_exposure),np.float64(self.fragmentation_breakup_exposure),
                    np.float64(self.fragmentation_reference_diameter_m),np.float64(self.fragmentation_size_exponent),
                    np.float64(self.aggregation_step_safety),np.int32(self.aggregation_max_steps),wd,population_scalars,agg_err))
            cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(agg_err)[0])
            if ecode: raise RuntimeError(f'CUDA aggregation-fragmentation precompute failed with error code {ecode}')
        else:
            self.kernels['orthokinetic_compact_aggregation_kernel']((1,),(1,),(dd,wd_input,np.int32(n),np.float64(conc),np.float64(self.density),np.float64(0.0),np.float64(self.aggregation_step_safety),np.int32(self.aggregation_max_steps),wd,agg_scalars,agg_err))
            cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(agg_err)[0])
            if ecode: raise RuntimeError(f'CUDA PSD normalization precompute failed with error code {ecode}')
        # Mie coefficient scratch and outputs. Physics is computed only by CUDA.
        Dre=cp.empty(n*M,dtype=cp.float64);Dim=cp.empty_like(Dre);ar=cp.empty_like(Dre);ai=cp.empty_like(Dre);br=cp.empty_like(Dre);bi=cp.empty_like(Dre)
        nstop=cp.zeros(n,dtype=cp.int32);x=cp.empty(n,dtype=cp.float64);qext=cp.empty_like(x);qsca=cp.empty_like(x);qabs=cp.empty_like(x);qback=cp.empty_like(x);g=cp.empty_like(x);albedo=cp.empty_like(x)
        err=cp.zeros(1,dtype=cp.int32);threads=128;blocks=(n+threads-1)//threads
        self.kernels['mie_coefficients_efficiencies_kernel']((blocks,),(threads,),(dd,nd,kd,np.float64(self.n_water),np.float64(self.particle_wavelength),np.int32(n),Dre,Dim,ar,ai,br,bi,nstop,x,qext,qsca,qabs,qback,g,albedo,err))
        cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0])
        if ecode: raise RuntimeError(f'CUDA Mie precompute failed with error code {ecode}; MIE_MAX_ORDER={M}')
        if self.particle_event_model=='hybrid_mie_excess':
            nt=int(self.mie_phase_grid_size);theta=cp.empty(nt,dtype=cp.float64);phase_pdf=cp.empty(n*nt,dtype=cp.float64);phase_cdf=cp.empty(n*nt,dtype=cp.float64)
            tb=(nt+threads-1)//threads;self.kernels['build_theta_grid_kernel']((tb,),(threads,),(theta,np.int32(nt)))
            total=n*nt;pb=(total+threads-1)//threads;self.kernels['mie_phase_pdf_kernel']((pb,),(threads,),(theta,np.int32(nt),np.int32(n),nstop,ar,ai,br,bi,phase_pdf))
            self.kernels['mie_phase_cdf_kernel']((n,),(1,),(theta,phase_pdf,np.int32(nt),np.int32(n),phase_cdf,err))
        else:
            nt=2;theta=cp.empty(2,dtype=cp.float64);phase_cdf=cp.empty(2*n,dtype=cp.float64);self.kernels['dummy_phase_kernel']((max(1,(n+threads-1)//threads),),(threads,),(theta,phase_cdf,np.int32(n)))
        # Transport arrays/CDFs.
        arr=lambda:cp.empty(n,dtype=cp.float64)
        radius=arr();pmass=arr();srcw=arr();nden=arr();sg=arr();ssca=arr();sext=arr();swave=arr();sevent=arr()
        mug=arr();mus=arr();mue=arr();muw=arr();muev=arr();pew=arr();gew=arr();wew=arr();gcdf=arr();wcdf=arr();pcdf=arr();muscal=cp.empty(8,dtype=cp.float64)
        self.kernels['transport_precompute_kernel']((1,),(1,),(dd,wd,qext,qsca,np.int32(n),np.float64(conc),np.float64(self.density),np.int32(self.WEIGHT_MODE_CODES[weight_mode]),np.int32(self.EVENT_MODEL_CODES[self.particle_event_model]),np.float64(self.hybrid_wave_scale),np.float64(self.n_water),np.float64(self.particle_wavelength),np.float64(self.wave_surface_rms_height_m),radius,pmass,srcw,nden,sg,ssca,sext,swave,sevent,mug,mus,mue,muw,muev,pew,gew,wew,gcdf,wcdf,pcdf,muscal,err))
        cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0])
        if ecode: raise RuntimeError(f'CUDA particle transport precompute failed with error code {ecode}')
        # V24.16 size-resolved spatial sediment field.  For non-hybrid event laws
        # the historical trace channel is mu_event; for the production hybrid it
        # remains geometric plus the separate Mie-excess wave channel.
        trace_geom_bin=mug if self.particle_event_model=='hybrid_mie_excess' else muev
        trace_wave_bin=muw if self.particle_event_model=='hybrid_mie_excess' else cp.zeros_like(muw)
        sed_tau=arr();sed_ws=arr();sed_A=arr();sed_B=arr();sed_norm=arr();sed_max=arr()
        sed_sc=arr();sed_sw=arr();sed_bc=arr();sed_bw=arr();sed_major=cp.zeros(4,dtype=cp.float64)
        err.fill(0)
        self.kernels['sediment_transport_precompute_kernel']((1,),(1,),(
            dd,np.int32(n),np.int32(self.SEDIMENT_TRANSPORT_MODEL_CODES[self.sediment_transport_model]),
            np.float64(self.density),np.float64(self.water_density),np.float64(self.water_dynamic_viscosity),np.float64(self.gravity),np.float64(self.sediment_eddy_diffusivity),
            np.float64(self.rin),np.float64(self.zmin),np.float64(self.zmax),np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),np.float64(self.vortex_delta_h),np.float64(self.vortex_core_radius),np.int32(self.sediment_transport_quadrature_points),
            trace_geom_bin,trace_wave_bin,sed_tau,sed_ws,sed_A,sed_B,sed_norm,sed_max,sed_sc,sed_sw,sed_bc,sed_bw,sed_major,err))
        cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0])
        if ecode: raise RuntimeError(f'CUDA sediment transport precompute failed with error code {ecode}')
        # Copy only diagnostic products; device fields/rates remain authoritative inputs to ray transport.
        keys={'diameter_m':dd,'input_source_weight':wd_input,'effective_aggregation_weight':wd,'source_weight':srcw,'radius_m':radius,'particle_mass_kg':pmass,'number_density_by_bin':nden,
              'n_real':nd,'k_imag':kd,'size_parameter':x,'qext':qext,'qsca':qsca,'qabs':qabs,'qback':qback,'asymmetry_g':g,'single_scattering_albedo':albedo,
              'geometric_cross_section_m2':sg,'scattering_cross_section_m2':ssca,'extinction_cross_section_m2':sext,'wave_cross_section_m2':swave,'event_cross_section_m2':sevent,
              'mu_geom_by_bin':mug,'mu_sca_by_bin':mus,'mu_ext_by_bin':mue,'mu_wave_by_bin':muw,'mu_event_by_bin':muev,
              'particle_event_weights':pew,'geometric_event_weights':gew,'wave_event_weights':wew,
              'sediment_response_time_s':sed_tau,'sediment_settling_velocity_m_s':sed_ws,'sediment_radial_log_coefficient_A':sed_A,'sediment_vertical_inverse_length_B_per_m':sed_B,
              'sediment_normalization_scaled':sed_norm,'sediment_max_density_scale':sed_max,'sediment_scale_sensor_center':sed_sc,'sediment_scale_sensor_wall_r0p95R':sed_sw,'sediment_scale_bottom_center':sed_bc,'sediment_scale_bottom_wall_r0p95R':sed_bw}
        host={k:cp.asnumpy(v) for k,v in keys.items()};mu_host=cp.asnumpy(muscal);agg_host=cp.asnumpy(agg_scalars)
        if used_coupled_population_kernel:
            pop_host=cp.asnumpy(population_scalars)
        else:
            # Compatibility mapping from the unchanged V24.11 aggregation kernel.
            # Fragmentation fields are zero because the coupled kernel was not active.
            pop_host=np.zeros(16,dtype=np.float64)
            pop_host[0]=float(agg_host[0]);pop_host[1]=0.0
            pop_host[2]=self.fragmentation_reference_diameter_m;pop_host[3]=self.fragmentation_size_exponent
            pop_host[4]=float(agg_host[1]);pop_host[5]=float(agg_host[2]);pop_host[6]=float(agg_host[3]);pop_host[7]=float(agg_host[4])
            pop_host[8]=float(agg_host[5]);pop_host[9]=float(agg_host[6]);pop_host[10]=float(agg_host[7]);pop_host[11]=float(agg_host[8]);pop_host[12]=0.0
            pop_host[13]=0.0;pop_host[14]=0.0;pop_host[15]=float(agg_host[9])
        input_sw=float(np.sum(host['input_source_weight']))
        if input_sw>0: host['input_source_weight']=host['input_source_weight']/input_sw
        host['effective_population_weight']=host['effective_aggregation_weight'].copy()
        host.update({'mu_trace_geometric':float(mu_host[0]),'mu_trace_wave':float(mu_host[1]),'mu_geom':float(mu_host[2]),'mu_sca':float(mu_host[3]),'mu_ext':float(mu_host[4]),'mu_wave':float(mu_host[5]),'mu_event':float(mu_host[6]),'wave_coherent_fraction_nominal_cuda':float(mu_host[7]),'event_model':self.particle_event_model,'wave_scale':self.hybrid_wave_scale,'wave_surface_rms_height_m':self.wave_surface_rms_height_m,
                     'aggregation_model':self.aggregation_model,'aggregation_collision_exposure':float(pop_host[0]),
                     'fragmentation_breakup_exposure':float(pop_host[1]),'fragmentation_reference_diameter_m':float(pop_host[2]),'fragmentation_size_exponent':float(pop_host[3]),
                     'aggregation_initial_number_density':float(pop_host[4]),'aggregation_final_number_density':float(pop_host[5]),
                     'aggregation_number_reduction_fraction':float(pop_host[6]),'population_number_reduction_fraction':float(pop_host[6]),
                     'aggregation_mass_conservation_error':float(pop_host[7]),'aggregation_initial_projected_area_per_m':float(pop_host[8]),'aggregation_final_projected_area_per_m':float(pop_host[9]),
                     'aggregation_projected_area_ratio':float(pop_host[9]/pop_host[8]) if pop_host[8]>0 else 1.0,
                     'aggregation_largest_bin_mass_fraction':float(pop_host[10]),'aggregation_overflow_event_fraction':float(pop_host[11]),
                     'fragmentation_underflow_event_fraction':float(pop_host[12]),'aggregation_events_per_initial_particle':float(pop_host[13]),
                     'fragmentation_events_per_initial_particle':float(pop_host[14]),'aggregation_steps':int(round(float(pop_host[15]))),
                     'population_kernel_coupled_fragmentation_active':bool(used_coupled_population_kernel)})
        sed_major_host=cp.asnumpy(sed_major)
        host.update({'sediment_transport_model':self.sediment_transport_model,'sediment_eddy_diffusivity_m2_s':self.sediment_eddy_diffusivity,
                     'sediment_water_density_kg_m3':self.water_density,'sediment_water_dynamic_viscosity_pa_s':self.water_dynamic_viscosity,'sediment_gravity_m_s2':self.gravity,
                     'sediment_transport_quadrature_points':self.sediment_transport_quadrature_points,'sediment_majorant_geom_per_m':float(sed_major_host[0]),
                     'sediment_majorant_wave_per_m':float(sed_major_host[1]),'sediment_majorant_total_per_m':float(sed_major_host[2]),'sediment_max_radial_log_coefficient':float(sed_major_host[3])})
        return {'d':d,'w':w,'nidx':nidx,'kidx':kidx,'n':n,'device':{'radius':radius,'n':nd,'k':kd,'gcdf':gcdf,'wcdf':wcdf,'phase_cdf':phase_cdf,'theta':theta,'mu_channels':muscal,
                'trace_geom_bin':trace_geom_bin,'trace_wave_bin':trace_wave_bin,'sed_A':sed_A,'sed_B':sed_B,'sed_norm':sed_norm,'sed_majorants':sed_major},'host':host,'ntheta':nt}

    def particle_physics_diagnostics(self,material,concentration_g_per_L,weights=None,weight_mode='mass_fraction'):
        """Run CUDA precompute and return diagnostic copies without tracing rays."""
        return self._prepare_gpu_particle_physics(material,concentration_g_per_L,weights,weight_mode)['host']

    def phase_table_diagnostics(self,material,concentration_g_per_L,weights=None,weight_mode='mass_fraction'):
        """Diagnostic-only copy of the CUDA-generated Mie theta/CDF table."""
        prep=self._prepare_gpu_particle_physics(material,concentration_g_per_L,weights,weight_mode)
        cp=self.cp;nt=prep['ntheta'];n=prep['n']
        return cp.asnumpy(prep['device']['theta']),cp.asnumpy(prep['device']['phase_cdf']).reshape(n,nt)

    def simulate(self,material,concentration_g_per_L,weights=None,n_rays=100000,seed=12345,
                 weight_mode='mass_fraction',collect_rays=False,heatmap_size=0,
                 target_detector_score=None,min_rays=None,max_rays=None,
                 stability_l1_tolerance=None,stability_window=2):
        cp=self.cp;prep=self._prepare_gpu_particle_physics(material,concentration_g_per_L,weights,weight_mode);ph=prep['host'];dev=prep['device'];npart=prep['n'];ntheta=prep['ntheta']
        requested=int(n_rays);adaptive=target_detector_score is not None
        if adaptive:
            target=float(target_detector_score)
            if target<=0: raise ValueError('target_detector_score must be >0')
            maxN=int(max_rays if max_rays is not None else max(requested,self.statistics_batch_rays));minN=int(min_rays if min_rays is not None else min(self.statistics_batch_rays,maxN))
            if minN<0 or maxN<minN: raise ValueError('invalid min_rays/max_rays')
        else: target=None;maxN=requested;minN=requested
        if maxN<1: raise ValueError('number of rays must be positive')
        vis=int(heatmap_size);hxy=cp.zeros(max(1,vis*vis),dtype=cp.float32);hzy=cp.zeros(max(1,vis*vis),dtype=cp.float32)
        seed0=np.uint32((int(seed)*1664525+1013904223)&0x7fffffff or 1);seed1=np.uint32((int(seed)*22695477+1)&0x7fffffff or 7);seed2=np.uint32((int(seed)*1103515245+12345)&0x7fffffff or 11)
        threads=256
        native_total=np.zeros(18,dtype=np.int64);mirror_total=np.zeros(18,dtype=np.int64);sym_total=np.zeros(18,dtype=np.float64)
        legacy_native_total=np.zeros(18,dtype=np.int64);legacy_mirror_total=np.zeros(18,dtype=np.int64);legacy_sym_total=np.zeros(18,dtype=np.float64)
        batch_scores=[];legacy_batch_scores=[];batch_rows=[];prev_cum=None;recent_l1=[]
        total_rays=0;valid_air_total=0;entered_total=0;source_accept_total=0;axial_total=0;top_exit_total=0;bottom_exit_total=0
        sum_ic=sum_gic=sum_wic=sum_wcoh=sum_wmorph=0.0;ballistic_total=0;ic_hist=np.zeros(1,dtype=np.int64)
        sum_pr=0.0;rays_entry_pr=rays_internal_pr=particle_absorbed_total=0;sum_cir=sum_cor=0.0;rays_ctir=0
        sum_top_ref=sum_bottom_inner=sum_bottom_outer=0.0;rays_top_tir=rays_bottom_tir=0
        ray_chunks=[] if collect_rays else None;stop_reason='fixed_n_rays'
        while total_rays<maxN:
            batchN=min(self.statistics_batch_rays,maxN-total_rays)
            batch_native=np.zeros(18,dtype=np.int64);batch_mirror=np.zeros(18,dtype=np.int64);batch_legacy_native=np.zeros(18,dtype=np.int64);batch_legacy_mirror=np.zeros(18,dtype=np.int64)
            batch_start=total_rays
            for local_start in range(0,batchN,self.chunk_size):
                local_end=min(batchN,local_start+self.chunk_size);n=local_end-local_start;global_start=batch_start+local_start
                outf=[cp.full(n,cp.nan,dtype=cp.float32) for _ in range(8)];outi=[cp.zeros(n,dtype=cp.int32) for _ in range(19)]
                blocks=(n+threads-1)//threads
                self.kernel((blocks,),(threads,),(
                    np.float32(self.max_iterations),dev['mu_channels'],np.float32(self.n_water),np.float32(self.n_acrylic),np.float32(self.n_air),np.float32(self.particle_wavelength),np.float32(self.particle_surface_beckmann_alpha),np.float32(self.wave_spheroid_aspect_ratio),
                    np.int32(self.SPHEROID_ORIENTATION_MODEL_CODES[self.wave_spheroid_orientation_model]),np.float32(self.wave_spheroid_orientation_kappa),
                    np.int32(self.max_internal_bounces),np.int32(self.max_cell_bounces),np.float32(self.rin),np.float32(self.rout),np.float32(self.zmin),np.float32(self.zmax),np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),np.float32(self.vortex_delta_h),np.float32(self.vortex_core_radius),np.float32(self.bottom_disc_thickness),np.float32(self.bottom_external_n),np.int32(self.max_axial_bounces),
                    np.float32(self.ring_in),np.float32(self.throat_out),np.float32(self.ring_out),np.float32(self.source_launch_radius),np.float32(self.throat_radius),np.float32(self.counterbore_radius),np.float32(self.beam_sigma),np.int32(vis),np.float32(self.alpha1),np.float32(self.alpha2),np.int32(self.SOURCE_ANGULAR_MODEL_CODES[self.source_angular_model]),np.int32(n),
                    dev['gcdf'],dev['wcdf'],dev['phase_cdf'],dev['theta'],np.int32(ntheta),dev['n'],dev['k'],dev['radius'],np.int32(npart),
                    np.int32(self.SEDIMENT_TRANSPORT_MODEL_CODES[self.sediment_transport_model]),dev['trace_geom_bin'],dev['trace_wave_bin'],dev['sed_A'],dev['sed_B'],dev['sed_norm'],dev['sed_majorants'],
                    hxy,hzy,*outf,*outi,seed0,seed1,seed2,np.uint32(global_start)))
                # Detector geometry/scoring is CUDA-resident. Only counts/indices are copied back.
                hnat=cp.zeros(18,dtype=cp.int32);hmir=cp.zeros(18,dtype=cp.int32);lnat=cp.zeros(18,dtype=cp.int32);lmir=cp.zeros(18,dtype=cp.int32);nid=cp.empty(n,dtype=cp.int32);mid=cp.empty(n,dtype=cp.int32)
                self.kernels['score_detectors_kernel']((blocks,),(threads,),(outf[0],outf[1],outf[2],outf[3],outf[4],outf[5],outi[18],outi[0],np.int32(n),np.float32(self.ring_in),np.float32(self.throat_out),np.float32(self.ring_out),np.float32(self.throat_radius),np.float32(self.counterbore_radius),np.float32(self.legacy_acceptance_deg),hnat,hmir,lnat,lmir,nid,mid))
                cp.cuda.Stream.null.synchronize()
                batch_native+=cp.asnumpy(hnat).astype(np.int64);batch_mirror+=cp.asnumpy(hmir).astype(np.int64);batch_legacy_native+=cp.asnumpy(lnat).astype(np.int64);batch_legacy_mirror+=cp.asnumpy(lmir).astype(np.int64)
                af=[cp.asnumpy(a) for a in outf];ai=[cp.asnumpy(a) for a in outi]
                arrays=dict(zip(['air_x','air_y','air_z','air_vx','air_vy','air_vz','water_path','acrylic_path'],af));ints=dict(zip(['interaction_count','geometric_interactions','wave_interactions','coherent_wave_interactions','morphology_wave_interactions','particle_reflections','particle_entry_reflections','particle_internal_reflections','particle_absorptions','cell_inner_reflections','cell_outer_reflections','cell_tir','top_reflections','top_tir','bottom_inner_reflections','bottom_outer_reflections','bottom_tir','entered_water','status'],ai))
                valid_air=(ints['status']==1)&np.isfinite(arrays['air_x'])&np.isfinite(arrays['air_vx']);entered=ints['entered_water']>0;ic=ints['interaction_count'][entered]
                valid_air_total+=int(valid_air.sum());entered_total+=int(entered.sum());source_accept_total+=int(np.count_nonzero(ints['status']!=3));top_exit_total+=int(np.count_nonzero(ints['status']==6));bottom_exit_total+=int(np.count_nonzero(ints['status']==7));axial_total+=int(np.count_nonzero((ints['status']==6)|(ints['status']==7)))
                if ic.size:
                    sum_ic+=float(ic.sum());sum_gic+=float(ints['geometric_interactions'][entered].sum());sum_wic+=float(ints['wave_interactions'][entered].sum());sum_wcoh+=float(ints['coherent_wave_interactions'][entered].sum());sum_wmorph+=float(ints['morphology_wave_interactions'][entered].sum());ballistic_total+=int(np.count_nonzero(ic==0))
                    bc=np.bincount(ic.astype(np.int64));
                    if bc.size>ic_hist.size: ic_hist=np.pad(ic_hist,(0,bc.size-ic_hist.size))
                    ic_hist[:bc.size]+=bc;pr=ints['particle_reflections'][entered];per=ints['particle_entry_reflections'][entered];pir=ints['particle_internal_reflections'][entered]
                    particle_absorbed_total+=int(np.count_nonzero(ints['particle_absorptions'][entered]>0));cir=ints['cell_inner_reflections'][entered];cor=ints['cell_outer_reflections'][entered];ct=ints['cell_tir'][entered]
                    sum_pr+=float(pr.sum());rays_entry_pr+=int(np.count_nonzero(per>0));rays_internal_pr+=int(np.count_nonzero(pir>0));sum_cir+=float(cir.sum());sum_cor+=float(cor.sum());rays_ctir+=int(np.count_nonzero(ct>0));sum_top_ref+=float(ints['top_reflections'][entered].sum());rays_top_tir+=int(np.count_nonzero(ints['top_tir'][entered]>0));sum_bottom_inner+=float(ints['bottom_inner_reflections'][entered].sum());sum_bottom_outer+=float(ints['bottom_outer_reflections'][entered].sum());rays_bottom_tir+=int(np.count_nonzero(ints['bottom_tir'][entered]>0))
                if collect_rays:
                    ray_chunks.append({**arrays,**ints,'native_detector_index':cp.asnumpy(nid),'mirror_detector_index':cp.asnumpy(mid)})
                del outf,outi,af,ai,hnat,hmir,lnat,lmir,nid,mid
            total_rays+=batchN;native_total+=batch_native;mirror_total+=batch_mirror;legacy_native_total+=batch_legacy_native;legacy_mirror_total+=batch_legacy_mirror
            batch_sym=0.5*(batch_native.astype(float)+batch_mirror.astype(float)) if self.symmetry_average_exact else batch_native.astype(float);batch_legacy_sym=0.5*(batch_legacy_native.astype(float)+batch_legacy_mirror.astype(float)) if self.symmetry_average_exact else batch_legacy_native.astype(float)
            sym_total+=batch_sym;legacy_sym_total+=batch_legacy_sym;batch_scores.append(batch_sym.copy());legacy_batch_scores.append(batch_legacy_sym.copy())
            den=float(sym_total.sum());cum=sym_total/den if den>0 else np.zeros_like(sym_total);l1=float(np.sum(np.abs(cum-prev_cum))) if prev_cum is not None else float('nan')
            if np.isfinite(l1):recent_l1.append(l1)
            prev_cum=cum.copy();batch_rows.append({'batch':len(batch_rows)+1,'batch_rays':batchN,'cumulative_rays':total_rays,'batch_native_hits':int(batch_native.sum()),'batch_mirror_hits':int(batch_mirror.sum()),'batch_symmetry_score':float(batch_sym.sum()),'cumulative_symmetry_score':float(sym_total.sum()),'batch_legacy_channel_score':float(batch_legacy_sym.sum()),'cumulative_legacy_channel_score':float(legacy_sym_total.sum()),'cumulative_l1_change':l1})
            if adaptive and total_rays>=minN and sym_total.sum()>=target:
                stable=True
                if stability_l1_tolerance is not None:
                    wdw=max(1,int(stability_window));stable=len(recent_l1)>=wdw and all(q<=float(stability_l1_tolerance) for q in recent_l1[-wdw:])
                if stable:stop_reason='target_detector_score_and_stability' if stability_l1_tolerance is not None else 'target_detector_score';break
        if adaptive and total_rays>=maxN and sym_total.sum()<target:stop_reason='max_rays_before_target'
        elif adaptive and total_rays>=maxN and stop_reason.startswith('fixed'):stop_reason='max_rays'
        nsum=float(native_total.sum());msum=float(mirror_total.sum());ssum=float(sym_total.sum());nn=native_total.astype(float)/nsum if nsum>0 else np.zeros(18);nm=mirror_total.astype(float)/msum if msum>0 else np.zeros(18);ns=sym_total/ssum if ssum>0 else np.zeros(18)
        lnsum=float(legacy_native_total.sum());lmsum=float(legacy_mirror_total.sum());lssum=float(legacy_sym_total.sum());lnn=legacy_native_total.astype(float)/lnsum if lnsum>0 else np.zeros(18);lnm=legacy_mirror_total.astype(float)/lmsum if lmsum>0 else np.zeros(18);lns=legacy_sym_total/lssum if lssum>0 else np.zeros(18)
        se=_jackknife_normalized_se(batch_scores);legacy_se=_jackknife_normalized_se(legacy_batch_scores);denom_enter=max(entered_total,1);ray_data=None
        if collect_rays and ray_chunks:
            keys=ray_chunks[0].keys();ray_data={k:np.concatenate([q[k] for q in ray_chunks]) for k in keys};ni=ray_data['native_detector_index'];mi=ray_data['mirror_detector_index'];ray_data['native_detector_deg']=np.where(ni>=0,DETECTOR_ANGLES_DEG[np.clip(ni,0,17)],np.nan);ray_data['mirror_detector_deg']=np.where(mi>=0,DETECTOR_ANGLES_DEG[np.clip(mi,0,17)],np.nan)
        hmxy=cp.asnumpy(hxy).reshape(vis,vis) if vis>0 else None;hmzy=cp.asnumpy(hzy).reshape(vis,vis) if vis>0 else None
        vg=vortex_surface_geometry(self.free_surface_model,self.zmax,self.rin,self.vortex_delta_h,self.vortex_core_radius)
        return ForwardResult(detector_angles_deg=DETECTOR_ANGLES_DEG.copy(),raw_hits=native_total.copy(),normalized_response=ns,native_exact_hits=native_total.copy(),mirror_exact_hits=mirror_total.copy(),symmetry_exact_scores=sym_total.copy(),normalized_native_exact_response=nn,normalized_mirror_exact_response=nm,normalized_symmetry_exact_response=ns,normalized_response_jackknife_se=se,
            hardware_native_hits=native_total.copy(),hardware_mirror_hits=mirror_total.copy(),hardware_symmetry_scores=sym_total.copy(),normalized_hardware_native_response=nn,normalized_hardware_mirror_response=nm,normalized_hardware_symmetry_response=ns,hardware_normalized_jackknife_se=se,
            legacy_native_channel_counts=legacy_native_total.copy(),legacy_mirror_channel_counts=legacy_mirror_total.copy(),legacy_symmetry_channel_scores=legacy_sym_total.copy(),normalized_legacy_native_response=lnn,normalized_legacy_mirror_response=lnm,normalized_legacy_symmetry_response=lns,legacy_normalized_jackknife_se=legacy_se,legacy_acceptance_deg=self.legacy_acceptance_deg,
            valid_exit_count=valid_air_total,n_rays=total_rays,requested_n_rays=requested,stopped_adaptively=bool(adaptive and total_rays<maxN),stop_reason=stop_reason,particle_event_model=self.particle_event_model,particle_surface_rms_slope_deg=self.particle_surface_rms_slope_deg,particle_surface_beckmann_alpha=self.particle_surface_beckmann_alpha,wave_surface_rms_height_nm=self.wave_surface_rms_height_m*1e9,wave_spheroid_aspect_ratio=self.wave_spheroid_aspect_ratio,wave_spheroid_orientation_model=self.wave_spheroid_orientation_model,wave_spheroid_orientation_kappa=self.wave_spheroid_orientation_kappa,aggregation_model=ph['aggregation_model'],aggregation_collision_exposure=ph['aggregation_collision_exposure'],aggregation_number_reduction_fraction=ph['aggregation_number_reduction_fraction'],aggregation_mass_conservation_error=ph['aggregation_mass_conservation_error'],aggregation_projected_area_ratio=ph['aggregation_projected_area_ratio'],aggregation_largest_bin_mass_fraction=ph['aggregation_largest_bin_mass_fraction'],aggregation_overflow_event_fraction=ph['aggregation_overflow_event_fraction'],aggregation_steps=ph['aggregation_steps'],population_number_reduction_fraction=ph['population_number_reduction_fraction'],fragmentation_breakup_exposure=ph['fragmentation_breakup_exposure'],fragmentation_reference_diameter_m=ph['fragmentation_reference_diameter_m'],fragmentation_size_exponent=ph['fragmentation_size_exponent'],fragmentation_underflow_event_fraction=ph['fragmentation_underflow_event_fraction'],aggregation_events_per_initial_particle=ph['aggregation_events_per_initial_particle'],fragmentation_events_per_initial_particle=ph['fragmentation_events_per_initial_particle'],wave_coherent_fraction_nominal_cuda=ph['wave_coherent_fraction_nominal_cuda'],mean_coherent_wave_interactions=sum_wcoh/denom_enter if entered_total else float('nan'),mean_morphology_wave_interactions=sum_wmorph/denom_enter if entered_total else float('nan'),morphology_fraction_of_wave_interactions=sum_wmorph/sum_wic if sum_wic>0 else 0.0,
            mu_geom_per_m=ph['mu_geom'],mu_wave_per_m=ph['mu_wave'],mu_event_per_m=ph['mu_event'],mu_sca_per_m=ph['mu_sca'],mu_ext_per_m=ph['mu_ext'],geometric_optical_depth_diameter=ph['mu_geom']*(2*self.rin),optical_depth_diameter=ph['mu_event']*(2*self.rin),
            mean_interactions=sum_ic/denom_enter if entered_total else float('nan'),mean_geometric_interactions=sum_gic/denom_enter if entered_total else float('nan'),mean_wave_interactions=sum_wic/denom_enter if entered_total else float('nan'),median_interactions=_histogram_median(ic_hist),ballistic_fraction=ballistic_total/denom_enter if entered_total else float('nan'),mean_fresnel_reflections=sum_pr/denom_enter if entered_total else float('nan'),entry_reflection_fraction=rays_entry_pr/denom_enter if entered_total else float('nan'),internal_reflection_fraction=rays_internal_pr/denom_enter if entered_total else float('nan'),particle_absorption_fraction=particle_absorbed_total/denom_enter if entered_total else float('nan'),particle_absorbed_ray_count=particle_absorbed_total,
            mean_cell_inner_reflections=sum_cir/denom_enter if entered_total else float('nan'),mean_cell_outer_reflections=sum_cor/denom_enter if entered_total else float('nan'),cell_tir_fraction=rays_ctir/denom_enter if entered_total else float('nan'),mean_top_surface_reflections=sum_top_ref/denom_enter if entered_total else float('nan'),top_surface_tir_fraction=rays_top_tir/denom_enter if entered_total else float('nan'),mean_top_surface_interactions=(sum_top_ref+top_exit_total)/denom_enter if entered_total else float('nan'),free_surface_model=self.free_surface_model,vortex_wall_center_height_difference_mm=self.vortex_delta_h*1e3,vortex_core_radius_mm=self.vortex_core_radius*1e3,vortex_asymptotic_height_parameter_mm=vg['h_inf']*1e3,vortex_area_mean_uncorrected_rise_mm=vg['h_bar']*1e3,vortex_equivalent_core_rpm=vg['equivalent_core_rpm'],vortex_equivalent_rigid_body_rpm=vg['equivalent_rigid_body_rpm'],vortex_center_surface_z_mm=vg['center_z']*1e3,vortex_wall_surface_z_mm=vg['wall_z']*1e3,sediment_transport_model=self.sediment_transport_model,sediment_eddy_diffusivity_m2_s=self.sediment_eddy_diffusivity,sediment_majorant_geom_per_m=ph['sediment_majorant_geom_per_m'],sediment_majorant_wave_per_m=ph['sediment_majorant_wave_per_m'],sediment_majorant_total_per_m=ph['sediment_majorant_total_per_m'],sediment_max_bin_density_scale=float(np.max(ph['sediment_max_density_scale'])) if len(ph['sediment_max_density_scale']) else 1.0,mean_bottom_inner_reflections=sum_bottom_inner/denom_enter if entered_total else float('nan'),mean_bottom_outer_reflections=sum_bottom_outer/denom_enter if entered_total else float('nan'),bottom_tir_fraction=rays_bottom_tir/denom_enter if entered_total else float('nan'),top_exit_fraction=top_exit_total/max(total_rays,1),bottom_exit_fraction=bottom_exit_total/max(total_rays,1),source_collimator_acceptance_fraction=source_accept_total/max(total_rays,1),entered_water_fraction=entered_total/max(total_rays,1),air_exit_fraction=valid_air_total/max(total_rays,1),axial_loss_fraction=axial_total/max(total_rays,1),detector_detection_fraction=ssum/max(total_rays,1),native_detector_detection_fraction=nsum/max(total_rays,1),mirror_detector_detection_fraction=msum/max(total_rays,1),legacy_channel_score_per_ray=lssum/max(total_rays,1),symmetry_variance_reduction=self.symmetry_average_exact,batch_diagnostics=batch_rows,particle_diagnostics=ph,ray_data=ray_data,heatmap_xy=hmxy,heatmap_zy=hmzy)

