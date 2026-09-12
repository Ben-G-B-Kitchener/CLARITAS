#!/usr/bin/env python3
"""CLARITAS V24.46 candidate: V24.42 Smith-Beckmann boundary physics retained with V24.46 finite-cell topology and state-aware near-boundary intersection semantics, plus full-production failure observability and pre-authorization transport qualification on frozen V24.29 event-rate transport.

V24.29 is a tightly controlled physics experiment. It preserves the full V24.27
per-event angular treatment, including the 25-degree specular-microfacet geometric
branch and the coherent-Mie / morphology-wave excess branch, but corrects the
transport event rate bin-by-bin so that the total scattering-event cross-section is
exactly the CUDA Mie scattering cross-section:

    sigma_event(d) = sigma_sca,Mie(d) = Qsca(d) * pi r^2.

For Qsca >= 1, the V24.27 split is unchanged: one projected-area geometric branch
plus the Mie-excess wave branch. For Qsca < 1, the geometric branch is thinned to
Qsca*pi*r^2 and the wave branch is zero. No angular law, Fresnel law, morphology
model, PSD, concentration, apparatus optic, source law, or detector model is changed.
No H170/material/PSD/concentration multiplier is introduced.

V24.27 causal first/last-event and path-length diagnostics are retained verbatim so
loess acts as a strong regression control while kaolin isolates the event-rate fix.
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

# V24.46 full-production integrity diagnostics.  These codes report execution failure
# stages only; they do not alter ray probabilities or physical outcomes.  The legacy
# high-level status array is retained for compatibility.
PRODUCTION_FAILURE_NAMES = {
    0: 'PROD_OK',
    1: 'PARTICLE_MICROSURFACE_NUMERICAL',
    2: 'PARTICLE_MICROSURFACE_WATCHDOG',
    3: 'PARTICLE_MACRO_GEOMETRY_FAILURE',
    4: 'PARTICLE_MACRO_BOUNCE_LIMIT',
    5: 'ACRYLIC_ANNULUS_NO_INTERSECTION',
    6: 'ACRYLIC_ANNULUS_BOUNCE_LIMIT',
    7: 'BOTTOM_DISC_GEOMETRY_FAILURE',
    8: 'BOTTOM_DISC_BOUNCE_LIMIT',
    9: 'WATER_BOUNDARY_NO_INTERSECTION',
    10: 'TRACE_ITERATION_LIMIT',
    11: 'NONFINITE_POSITION',
    12: 'NONFINITE_DIRECTION',
    13: 'INVALID_MEDIUM_STATE',
    14: 'INVALID_EVENT_STATE',
    15: 'OTHER_EXPLICIT_FAILURE',
    16: 'SPHEROID_GEOMETRY_FAILURE',
}
PRODUCTION_MEDIUM_NAMES = {0:'unknown/pre-water',1:'water',2:'acrylic_annulus',3:'bottom_acrylic',4:'particle'}
PRODUCTION_EVENT_CHANNEL_NAMES = {0:'none',1:'geometric_particle',2:'morphology_wave_particle'}
PRODUCTION_WATCHDOG_CODES = frozenset((2,4,6,8,10))
PRODUCTION_GEOMETRY_FAILURE_CODES = frozenset((3,5,7,9,13,16))
PRODUCTION_NONFINITE_CODES = frozenset((11,12))
PRODUCTION_FAILURE_SUBCODE_NAMES = {
    0:'NONE',
    10:'VNDF_PROJECTED_AREA_INVALID',11:'VNDF_CDF_INVALID',12:'VNDF_INVERSE_CDF_INVALID',13:'VNDF_VISIBILITY_FAILURE',14:'VNDF_NORMALIZATION_FAILURE',
    20:'HEIGHT_CDF_INVALID',21:'HEIGHT_SAMPLE_INVALID',22:'FREE_PATH_INVALID',23:'HEIGHT_LAMBDA_INVALID',24:'HEIGHT_C0_INVALID',25:'HEIGHT_LOG_C0_INVALID',
    26:'HEIGHT_G1_INVALID',27:'HEIGHT_ESCAPE_PROB_INVALID',28:'HEIGHT_RANDOM_INVALID',29:'HEIGHT_LOG_CNEXT_INVALID',30:'MICROFACET_NORMAL_INVALID',
    31:'REFLECTION_VECTOR_INVALID',32:'REFRACTION_VECTOR_INVALID',33:'SIDE_STATE_INVALID',34:'NONFINITE_DIRECTION',35:'FRESNEL_INVALID',36:'HEIGHT_CNEXT_RANGE_INVALID',
    37:'HEIGHT_INVERSE_CDF_INVALID',38:'HEIGHT_OUTPUT_INVALID',39:'HEIGHT_FREE_PATH_INVALID',40:'MICRO_WATCHDOG',41:'MACRO_WATCHDOG_LEGACY_MICRO_ENUM',
    1001:'SPHERE_ENTRY_SIDE',1002:'SPHERE_CHORD',1003:'SPHERE_EXIT_SIDE',1004:'SPHERE_INTERNAL_SIDE',1005:'SPHERE_BOUNCE_LIMIT',
    1101:'SPHEROID_ASPECT',1102:'SPHEROID_AXES',1103:'SPHEROID_ENTRY',1104:'SPHEROID_CHORD',1105:'SPHEROID_EXIT_SIDE',1106:'SPHEROID_INTERNAL_SIDE',1107:'SPHEROID_BOUNCE_LIMIT',
    2001:'ACRYLIC_NO_INTERSECTION',2002:'ACRYLIC_BOUNCE_LIMIT',2101:'BOTTOM_NONPOSITIVE_T',2102:'BOTTOM_TANGENTIAL',2103:'BOTTOM_BOUNCE_LIMIT',2104:'BOTTOM_TOPOLOGY_INVALID',
    3001:'WATER_NO_BOUNDARY',3002:'TRACE_ITERATION_LIMIT',3003:'NONFINITE_POSITION',3004:'NONFINITE_DIRECTION',3005:'WATER_OUTSIDE_INNER_RADIUS',
}

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


def meridional_velocity_host(r_m, z_m, zmean, zmin, radius, delta_h, core_radius,
                             circulation_ratio, gravity_m_s2=9.81):
    """Axisymmetric secondary-circulation field used by V24.20.

    The streamfunction is

        psi = -A r^2 (1-r^2/R^2)^2 eta(1-eta),
        eta = (z-z_bottom)/(z_surface(r)-z_bottom),

    with ``A = 2 U_m``.  ``U_m`` is tied to the Scully vortex rather than an
    arbitrary absolute speed: ``U_m = chi * max(v_theta)`` and
    ``max(v_theta)=Omega_c*a/2``.  Positive ``chi`` gives outward flow near
    the bottom, upward return in the outer annulus, inward flow near the free
    surface and downward axial return.  Because psi is constant on r=0, r=R,
    z=z_bottom and z=z_surface(r), the field is incompressible and tangent to
    all vessel boundaries in the continuous model.
    """
    r=np.asarray(r_m,dtype=np.float64); z=np.asarray(z_m,dtype=np.float64)
    R=float(radius);a=float(core_radius);chi=float(circulation_ratio);g=float(gravity_m_s2)
    if R<=0 or a<=0 or chi<0 or g<=0:
        raise ValueError('invalid V24.20 meridional-circulation parameters')
    vg=vortex_surface_geometry('localized_scully_vortex',float(zmean),R,float(delta_h),a)
    omega=math.sqrt(max(2.0*g*vg['h_inf']/(a*a),0.0)) if delta_h>0 else 0.0
    vtheta_peak=0.5*omega*a
    U=chi*vtheta_peak; Apsi=2.0*U
    rr=np.clip(r,0.0,R); rr2=rr*rr; a2=a*a
    zsurf=float(zmean)-vg['h_bar']+vg['h_inf']*rr2/(a2+rr2)
    H=np.maximum(zsurf-float(zmin),1e-15)
    eta=np.clip((z-float(zmin))/H,0.0,1.0)
    s=(rr/R)**2; f=eta*(1.0-eta); fp=1.0-2.0*eta
    hp=2.0*vg['h_inf']*a2*rr/np.square(a2+rr2)
    ur=Apsi*rr*np.square(1.0-s)*fp/H
    uz=-Apsi*(2.0*(1.0-s)*(1.0-3.0*s)*f
               -rr*np.square(1.0-s)*fp*eta*hp/H)
    return ur,uz,dict(vtheta_peak_m_s=vtheta_peak,meridional_peak_speed_m_s=U,
                      streamfunction_velocity_amplitude_m_s=Apsi,omega_core_rad_s=omega)


def _schiller_naumann_velocity_host(stokes_velocity_m_s, diameter_m, water_density_kg_m3, water_dynamic_viscosity_pa_s):
    """Solve the Schiller-Naumann corrected slip speed for a Stokes driving speed.

    The monotone equation is ``w*(1+0.15*Re(w)**0.687)=w_stokes``.  Newton's
    method is used with the Stokes value as an upper-bound initial estimate.
    The returned drag factor is ``w_stokes/w`` and tends exactly to one as
    Reynolds number tends to zero.
    """
    v0=np.asarray(stokes_velocity_m_s,dtype=np.float64)
    d=np.asarray(diameter_m,dtype=np.float64)
    rho=float(water_density_kg_m3);mu=float(water_dynamic_viscosity_pa_s)
    if rho<=0 or mu<=0: raise ValueError('water density and viscosity must be positive')
    v=np.maximum(v0,0.0).copy()
    K=rho*d/mu
    for _ in range(24):
        Re=np.maximum(K*v,0.0)
        corr=0.15*np.power(Re,0.687,where=Re>0,out=np.zeros_like(Re))
        f=v*(1.0+corr)-v0
        df=1.0+1.687*corr
        vn=np.maximum(v-f/df,0.0)
        if np.all(np.abs(vn-v)<=2e-14*np.maximum(1.0,np.abs(v))):
            v=vn;break
        v=vn
    Re=np.maximum(K*v,0.0)
    drag=1.0+0.15*np.power(Re,0.687,where=Re>0,out=np.zeros_like(Re))
    return v,Re,drag


def sediment_transport_coefficients_host(diameter_m, zmean, zmin, radius, delta_h, core_radius,
                                         eddy_diffusivity_m2_s, particle_density_kg_m3=2600.0,
                                         water_density_kg_m3=997.0, water_dynamic_viscosity_pa_s=8.9e-4,
                                         gravity_m_s2=9.81, quadrature_points=8192, radial_lookup_points=2049,
                                         transport_model='schiller_naumann_drift_diffusion'):
    """Independent host reference for inherited analytic transport coefficients.

    ``uniform`` retains the historical homogeneous suspension. The legacy
    ``stokes_drift_diffusion`` branch reproduces V24.16 and the
    ``schiller_naumann_drift_diffusion`` branch reproduces the V24.17/V24.18
    finite-Re analytic field. V24.20 model 3 uses these finite-Re coefficients
    as inputs/initialisation, but its 2-D advection-diffusion field is solved on
    CUDA and is therefore not returned by this host helper.
    """
    d=np.asarray(diameter_m,dtype=np.float64)
    tm=str(transport_model).strip().lower()
    if tm=='uniform':
        one=np.ones_like(d);zero=np.zeros_like(d)
        return dict(response_time_s=zero,settling_velocity_m_s=zero,settling_reynolds=zero,
                    settling_drag_factor=one,radial_log_coefficient=zero,vertical_inverse_length_m=zero,
                    normalization_scaled=one,max_density_scale=one,radial_max_slip_velocity_m_s=zero,
                    radial_max_reynolds=zero,radial_center_log_factor=zero,radial_log_lookup=np.zeros((d.size,2)),
                    radial_lookup_r_m=np.array([0.0,float(radius)]),omega_rad_s=0.0)
    if tm not in {'stokes_drift_diffusion','schiller_naumann_drift_diffusion'}:
        raise ValueError(f'unknown sediment transport model {transport_model!r}')
    Dt=float(eddy_diffusivity_m2_s);rho_p=float(particle_density_kg_m3);rho_w=float(water_density_kg_m3)
    mu=float(water_dynamic_viscosity_pa_s);g=float(gravity_m_s2);R=float(radius);a=float(core_radius)
    if not (Dt>0 and rho_p>rho_w>0 and mu>0 and g>0 and R>0 and a>0):
        raise ValueError('invalid V24.20 drift-diffusion physical parameters')
    vg=vortex_surface_geometry('localized_scully_vortex',float(zmean),R,float(delta_h),a)
    omega2=(2.0*g*vg['h_inf']/(a*a)) if delta_h>0 else 0.0
    tau=rho_p*d*d/(18.0*mu)
    ws_stokes=(rho_p-rho_w)*g*d*d/(18.0*mu)
    nr=max(65,int(radial_lookup_points))
    rnodes=np.linspace(0.0,R,nr,dtype=np.float64)
    a2=a*a
    accel_r=np.zeros_like(rnodes)
    if omega2>0:
        accel_r=omega2*rnodes/np.square(1.0+(rnodes/a)**2)
    w0=tau[:,None]*accel_r[None,:]
    fR=R*R/(a2+R*R)
    if tm=='stokes_drift_diffusion':
        ws=ws_stokes.copy();sett_re=rho_w*d*ws/mu;sett_drag=np.ones_like(d)
        wr=w0;radial_re=rho_w*d[:,None]*wr/mu
        Aeq=tau*omega2*a2/(2.0*Dt)
        fnode=rnodes*rnodes/(a2+rnodes*rnodes)
        radial_log=Aeq[:,None]*(fnode[None,:]-fR)
        radial_log[:,-1]=0.0
    else:
        ws,sett_re,sett_drag=_schiller_naumann_velocity_host(ws_stokes,d,rho_w,mu)
        wr,radial_re,_=_schiller_naumann_velocity_host(w0,d[:,None],rho_w,mu)
        dr=R/(nr-1)
        cum=np.zeros_like(wr)
        if nr>1:
            cum[:,1:]=np.cumsum(0.5*(wr[:,1:]+wr[:,:-1])*dr,axis=1)
        radial_log=(cum-cum[:,-1,None])/Dt
        radial_log[:,-1]=0.0
        Aeq=np.where(fR>0,-radial_log[:,0]/fR,0.0)
    B=ws/Dt
    nquad=max(64,int(quadrature_points));dq=R/nquad
    rr=(np.arange(nquad,dtype=np.float64)+0.5)*dq
    if tm=='stokes_drift_diffusion':
        fmid=rr*rr/(a2+rr*rr)
        radial_mid=Aeq[:,None]*(fmid[None,:]-fR)
    else:
        # interpolate the finite-Re radial potential onto the volume quadrature
        radial_mid=np.vstack([np.interp(rr,rnodes,radial_log[i]) for i in range(d.size)])
    Hinf=vg['h_inf'];hbar=vg['h_bar']
    zsurf=float(zmean)-hbar+Hinf*rr*rr/(a2+rr*rr)
    H=np.maximum(zsurf-float(zmin),0.0)
    Hmean=float(zmean)-float(zmin)
    norm=np.empty_like(d)
    for i in range(d.size):
        radial=np.exp(np.minimum(radial_mid[i],0.0))
        if abs(B[i])<1e-14:
            iz=radial*H
        else:
            iz=radial*(-np.expm1(-B[i]*H))/B[i]
        norm[i]=(2.0/(R*R*Hmean))*np.sum(rr*iz)*dq
    return dict(response_time_s=tau,settling_velocity_m_s=ws,settling_reynolds=sett_re,
                settling_drag_factor=sett_drag,radial_log_coefficient=Aeq,
                vertical_inverse_length_m=B,normalization_scaled=norm,max_density_scale=1.0/norm,
                radial_max_slip_velocity_m_s=np.max(wr,axis=1),radial_max_reynolds=np.max(radial_re,axis=1),
                radial_center_log_factor=radial_log[:,0],radial_log_lookup=radial_log,
                radial_lookup_r_m=rnodes,omega_rad_s=math.sqrt(max(omega2,0.0)))


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


// V24.20/V24.23 collimator and inner-ring-face stray-light models.
// The TARDIIS sensor ring was machined from nylon.  Model 0 reproduces the
// historical perfectly absorbing bore walls.  Model 1 treats the air/nylon
// bore wall as a smooth opaque dielectric: unpolarised Fresnel reflection is
// traced exactly and the transmitted component is terminated in the bulk ring.
// This deliberately adds no empirical diffuse-albedo multiplier.
__device__ float bore_side_hit(float q,float w,float vq,float vw,float radius){
    float A=vq*vq+vw*vw;
    if(A<=1.0e-20f)return INF_F;
    float B=2.0f*(q*vq+w*vw);
    float C=q*q+w*w-radius*radius;
    float D=B*B-4.0f*A*C;
    if(D<0.0f)return INF_F;
    float root=sqrtf(fmaxf(D,0.0f));
    float t1=(-B-root)/(2.0f*A),t2=(-B+root)/(2.0f*A);
    float t=INF_F;
    if(t1>EPS_F)t=t1;
    if(t2>EPS_F && t2<t)t=t2;
    return t;
}

__device__ int bore_wall_reflect(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,
    float q,float w,float* vs,float* vq,float* vw,int* interactions,int* reflections){
    (*interactions)++;
    if(surface_model==0)return 0;
    float rr=sqrtf(fmaxf(q*q+w*w,1.0e-30f));
    // Normal points from solid into the incident air cavity.
    float ns=0.0f,nq=-q/rr,nw=-w/rr;
    float ci=fminf(fmaxf(-((*vs)*ns+(*vq)*nq+(*vw)*nw),0.0f),1.0f);
    int tir=0;float R=fresnel_R(ci,n_air,n_bore,&tir);
    if(rnd_uniform(state)>=R)return 0; // transmitted into thick nylon -> loss
    float os,oq,ow;reflect3(*vs,*vq,*vw,ns,nq,nw,&os,&oq,&ow);
    *vs=os;*vq=oq;*vw=ow;(*reflections)++;
    return 1;
}

__device__ int trace_bore_segment(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    float s_end,const float radius,float* s,float* q,float* w,float* vs,float* vq,float* vw,
    int* interactions,int* reflections){
    for(int b=0;b<=max_bounces;++b){
        if(*vs<=1.0e-12f)return 0;
        float tp=(s_end-*s)/(*vs);
        if(tp<-EPS_F)return 0;
        if(tp<0.0f)tp=0.0f;
        float twall=bore_side_hit(*q,*w,*vq,*vw,radius);
        if(tp<=twall){
            *q+=tp*(*vq);*w+=tp*(*vw);*s=s_end;
            return ((*q)*(*q)+(*w)*(*w)<=radius*radius*(1.0f+2.0e-5f));
        }
        if(twall>=INF_F*0.5f)return 0;
        *q+=twall*(*vq);*w+=twall*(*vw);*s+=twall*(*vs);
        if(!bore_wall_reflect(state,surface_model,n_air,n_bore,*q,*w,vs,vq,vw,interactions,reflections))return 0;
        // Move back into the air cavity after the reflection.
        float rr=sqrtf(fmaxf((*q)*(*q)+(*w)*(*w),1.0e-30f));
        *q-=EPS_F*(*q)/rr;*w-=EPS_F*(*w)/rr;
    }
    return 0;
}

__device__ int source_collimator_transport(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    const float ring_in,const float throat_out,const float source_r,const float throat_r,const float counterbore_r,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,int* interactions,int* reflections){
    // Local source axis is +y (toward the cell); q=x,w=z.
    if(*vy<=1.0e-12f)return 0;
    float s=*y,q=*x,w=*z,vs=*vy,vq=*vx,vw=*vz;
    if(q*q+w*w>counterbore_r*counterbore_r)return 0;
    // Traverse only the portion of the counterbore between the actual luminous
    // plane and the shoulder.  SOURCE_R may equal THROAT_OUT_R.
    float shoulder=-throat_out;
    if(s<shoulder-EPS_F){
        if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,shoulder,counterbore_r,
                               &s,&q,&w,&vs,&vq,&vw,interactions,reflections))return 0;
    } else if(s>shoulder+EPS_F){
        return 0;
    }
    // The annular shoulder is opaque.  A ray must geometrically enter the 4-mm throat.
    if(q*q+w*w>throat_r*throat_r)return 0;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,-ring_in,throat_r,
                           &s,&q,&w,&vs,&vq,&vw,interactions,reflections))return 0;
    *x=q;*y=s;*z=w;*vx=vq;*vy=vs;*vz=vw;
    return 1;
}

__device__ int detector_collimator_path(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,int mirror,
    const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,
    const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    int* bounce_count){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;
    // First identify the physically first detector entrance aperture reached.
    for(int j=0;j<18;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1.0e-10f)continue;
        float pdot=x*ux+y*uy;float t=(ring_in-pdot)/den;if(t<=0.0f)continue;
        float etx=cosf(th),ety=-sinf(th);
        float hx=x+t*vx,hy=y+t*vy,hz=z+t*vz;
        float q=hx*etx+hy*ety,w=hz;
        if(q*q+w*w<=throat_r*throat_r && t<best_t){best_t=t;best=j;}
    }
    if(best<0)return -1;
    float th=(10.0f*(float)best)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th),etx=cosf(th),ety=-sinf(th);
    float hx=x+best_t*vx,hy=y+best_t*vy;
    float s=ring_in,q=hx*etx+hy*ety,w=z+best_t*vz;
    float vs=vx*ux+vy*uy,vq=vx*etx+vy*ety,vw=vz;
    int interactions=0,reflections=0;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,throat_out,throat_r,
                           &s,&q,&w,&vs,&vq,&vw,&interactions,&reflections))return -1;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,ring_out,counterbore_r,
                           &s,&q,&w,&vs,&vq,&vw,&interactions,&reflections))return -1;
    *bounce_count=reflections;return best;
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

// V24.46 cell-geometry intersection.  Geometric root validity is deliberately
// separate from the post-interface transport displacement EPS_F.  Inputs are float
// production state, but the quadratic and roots are evaluated in double precision.
// A strictly positive root is a physical forward hit even when t<EPS_F; exact t=0
// is suppressed as the surface the ray is already on.  Interface code continues to
// use side-aware EPS_F displacements after reflection/refraction to prevent self hits.
__device__ double cell_cylinder_hit_forward_d(float x,float y,float vx,float vy,float R) {
    double X=(double)x,Y=(double)y,VX=(double)vx,VY=(double)vy,RR=(double)R;
    double a=VX*VX+VY*VY;
    if(!(a>1e-30) || !isfinite(a)) return 1.0e300;
    double b=X*VX+Y*VY;
    double c=X*X+Y*Y-RR*RR;
    double d=b*b-a*c;
    double dscale=fabs(b*b)+fabs(a*c)+1e-300;
    double dtol=16.0*2.220446049250313080847263336181640625e-16*dscale;
    if(d<0.0){if(d>=-dtol)d=0.0;else return 1.0e300;}
    double sd=sqrt(fmax(d,0.0));
    // Stable roots for a*t^2 + 2*b*t + c = 0.
    double q=-b-copysign(sd,b);
    double t1=(fabs(q)>1e-300)?q/a:(-b-sd)/a;
    double t2=(fabs(q)>1e-300)?c/q:(-b+sd)/a;
    double best=1.0e300;
    if(t1>0.0 && t1<best)best=t1;
    if(t2>0.0 && t2<best)best=t2;
    return best;
}
__device__ float cell_cylinder_hit_forward(float x,float y,float vx,float vy,float R){
    double t=cell_cylinder_hit_forward_d(x,y,vx,vy,R);
    return (t<0.5*(double)INF_F)?(float)t:INF_F;
}

// Shared finite-bottom topology helpers used by both production and V24.46 geometry
// validation. Surface codes: 1=top plane, 2=bottom plane, 3=outer cylinder.
__device__ int bottom_disc_next_surface(float x,float y,float z,float vx,float vy,float vz,
                                         float Rout,float z_inner,float z_outer,double* t_out){
    double best=1.0e300;int surf=0;
    double to=cell_cylinder_hit_forward_d(x,y,vx,vy,Rout);
    if(to>0.0 && to<best){best=to;surf=3;}
    if(vz>0.0f){double tt=((double)z_inner-(double)z)/(double)vz;if(tt>0.0 && tt<best){best=tt;surf=1;}}
    if(vz<0.0f){double tb=((double)z_outer-(double)z)/(double)vz;if(tb>0.0 && tb<best){best=tb;surf=2;}}
    if(t_out)*t_out=best;return surf;
}
// At the top of the solid bottom disc the water aperture is strictly r<Rin.
// Rin<=r<=Rout is contiguous acrylic sidewall and therefore NOT an optical interface.
__device__ int bottom_disc_topology_at_top(float x,float y,float Rin,float Rout){
    double r2=(double)x*(double)x+(double)y*(double)y;
    double ri2=(double)Rin*(double)Rin,ro2=(double)Rout*(double)Rout;
    if(r2<ri2)return 1;
    if(r2<=ro2*(1.0+8.0*2.220446049250313080847263336181640625e-16))return 2;
    return 0;
}

__device__ void radial_normal(float x,float y,float* nx,float* ny) {
    float r=sqrtf(fmaxf(x*x+y*y,1e-30f)); *nx=x/r; *ny=y/r;
}


// V24.20 retains the V24.15 free-surface models.
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


// V24.20 size-resolved stirred-sediment field.
// TRANSPORT_MODEL: 0=uniform, 1=legacy V24.16 Stokes drift-diffusion,
// 2=V24.17 Schiller-Naumann finite-Re drift-diffusion,
// 3=V24.20 finite-Re + explicit incompressible meridional advection-diffusion.
// Model 3 samples a CUDA-solved axisymmetric r-z finite-volume field.  The
// rectangular solver grid is cell-centred; inactive cells above the curved
// free surface are extended from the nearest liquid cell only for interpolation.
__device__ double sediment_grid_sample(
    const int pidx,const double* GRID,const int NR,const int NZ,
    const float x,const float y,const float z,const float R,const float ZMIN,const float ZGRIDMAX){
    if(NR<2 || NZ<2 || !(ZGRIDMAX>ZMIN)) return 1.0;
    double rr=sqrt(fmax((double)x*(double)x+(double)y*(double)y,0.0));
    if(rr<0.0)rr=0.0;if(rr>(double)R)rr=(double)R;
    double dr=(double)R/(double)NR,dz=((double)ZGRIDMAX-(double)ZMIN)/(double)NZ;
    double ur=rr/dr-0.5,uz=((double)z-(double)ZMIN)/dz-0.5;
    if(ur<0.0)ur=0.0;if(uz<0.0)uz=0.0;
    if(ur>(double)(NR-1))ur=(double)(NR-1);
    if(uz>(double)(NZ-1))uz=(double)(NZ-1);
    int i=(int)floor(ur),k=(int)floor(uz);double tr=ur-(double)i,tz=uz-(double)k;
    if(i>=NR-1){i=NR-2;tr=1.0;}if(k>=NZ-1){k=NZ-2;tz=1.0;}
    long long plane=(long long)NR*(long long)NZ;
    long long b=(long long)pidx*plane;
    double q00=GRID[b+(long long)i*NZ+k];
    double q10=GRID[b+(long long)(i+1)*NZ+k];
    double q01=GRID[b+(long long)i*NZ+k+1];
    double q11=GRID[b+(long long)(i+1)*NZ+k+1];
    double q0=(1.0-tr)*q00+tr*q10,q1=(1.0-tr)*q01+tr*q11;
    return fmax((1.0-tz)*q0+tz*q1,0.0);
}

__device__ double sediment_local_scale(
    const int TRANSPORT_MODEL,const int pidx,const double* SED_A,const double* SED_B,
    const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float x,const float y,const float z,const float R,const float ZMIN,const float CORE_R){
    if(TRANSPORT_MODEL==0) return 1.0;
    if(TRANSPORT_MODEL==3)
        return sediment_grid_sample(pidx,SED_GRID,GRID_NR,GRID_NZ,x,y,z,R,ZMIN,GRID_ZMAX);
    double rr2=(double)x*(double)x+(double)y*(double)y;
    double R2=(double)R*(double)R;if(rr2>R2)rr2=R2;
    double radial_log=0.0;
    if(TRANSPORT_MODEL==1){
        double a2=(double)CORE_R*(double)CORE_R;
        double f=rr2/(a2+rr2),fR=R2/(a2+R2);
        radial_log=SED_A[pidx]*(f-fR);
    } else {
        double rr=sqrt(fmax(rr2,0.0));
        double u=(R>0.0)?rr/(double)R*(double)(N_RADIAL-1):0.0;
        int j=(int)floor(u);if(j<0)j=0;if(j>N_RADIAL-2)j=N_RADIAL-2;
        double t=u-(double)j;
        long long base=(long long)pidx*(long long)N_RADIAL;
        radial_log=(1.0-t)*SED_RADIAL_LOG[base+j]+t*SED_RADIAL_LOG[base+j+1];
        if(radial_log>0.0)radial_log=0.0;
    }
    double h=fmax((double)z-(double)ZMIN,0.0);
    double logrel=radial_log-SED_B[pidx]*h;
    double ns=SED_NORM_SCALED[pidx];
    if(!(ns>0.0)) return 0.0;
    return exp(logrel)/ns;
}

__device__ int sample_local_mu_bin(
    unsigned int* state,const int TRANSPORT_MODEL,const double* MU_BIN,const int n_particles,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float x,const float y,const float z,const float R,const float ZMIN,const float CORE_R,
    const double local_total){
    if(!(local_total>0.0))return 0;
    double target=(double)rnd_uniform(state)*local_total,cum=0.0;
    for(int i=0;i<n_particles;++i){
        double sc=sediment_local_scale(TRANSPORT_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,N_RADIAL,SED_GRID,GRID_NR,GRID_NZ,GRID_ZMAX,x,y,z,R,ZMIN,CORE_R);
        cum+=MU_BIN[i]*sc;
        if(target<=cum)return i;
    }
    return n_particles>0?n_particles-1:0;
}

__device__ double quadratic_min_positive(double A,double B,double C){
    const double eps=0.0; // V24.46: geometric validity is independent of EPS_F displacement.
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
    const double eps=0.0; // V24.46: geometric validity is independent of EPS_F displacement.
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
        return (t>0.0f)?t:INF_F;
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

// ============================ V24.46 CORRECTED ROUGH BOUNDARY ============================
// V24.46 freezes the V24.42 boundary correction: the Smith-Beckmann dielectric microsurface
// random-walk process, stable Lambda/log-height law, full-domain signed VNDF distribution,
// and direct double-precision stretched-slope support are retained.  V24.46 changes the
// surrounding full-production trace only to expose explicit failure stage/subcodes and to
// support fail-closed production qualification before sediment authorization.  The fixed TOP-oriented microsurface frame is used
// throughout.  At a lower/inside-side encounter the phase normal is sampled as
//     wm = -sampleD_wi(-wi)
// rather than flipping the nominal normal and rejecting a valid encounter.
//
// Physical microsurface escape, intersection, hard numerical failure and watchdog failure
// are explicit disjoint states.  No NaN/Inf/invalid denominator can masquerade as escape.
// No geometric-normal stochastic fallback exists in corrected production; the geometric
// normal appears only in the physical alpha->0 smooth-interface limit.
//
// Basis: Heitz, Hanika, d'Eon & Dachsbacher (2016), "Multiple-Scattering Microfacet BSDFs
// with the Smith Model", ACM TOG 35(4), DOI 10.1145/2897824.2925943, especially the
// supplemental signed Beckmann slope sampler (theta_i in [0,pi)) and dielectric vertical
// flip/random-walk algorithms.  RMS slope remains the inherited frozen alpha=tan(25 deg).

enum {
    MICRO_INTERSECTION=1,
    MICRO_ESCAPE=2,
    MICRO_NUMERICAL_FAILURE=3,
    MICRO_WATCHDOG_FAILURE=4
};

// V24.46 diagnostic subcodes.  These never alter physics; they identify the exact
// numerical branch if the fail-closed walker returns MICRO_NUMERICAL_FAILURE.
enum {
    MICRO_FAIL_NONE=0,
    VNDF_PROJECTED_AREA_INVALID=10,
    VNDF_CDF_INVALID=11,
    VNDF_INVERSE_CDF_INVALID=12,
    VNDF_VISIBILITY_FAILURE=13,
    VNDF_NORMALIZATION_FAILURE=14,
    HEIGHT_CDF_INVALID=20,              // historical aggregate code; validation-only compatibility
    HEIGHT_SAMPLE_INVALID=21,
    FREE_PATH_INVALID=22,
    HEIGHT_LAMBDA_INVALID=23,
    HEIGHT_C0_INVALID=24,
    HEIGHT_LOG_C0_INVALID=25,
    HEIGHT_G1_INVALID=26,
    HEIGHT_ESCAPE_PROB_INVALID=27,
    HEIGHT_RANDOM_INVALID=28,
    HEIGHT_LOG_CNEXT_INVALID=29,
    MICROFACET_NORMAL_INVALID=30,
    REFLECTION_VECTOR_INVALID=31,
    REFRACTION_VECTOR_INVALID=32,
    SIDE_STATE_INVALID=33,
    NONFINITE_DIRECTION=34,
    FRESNEL_INVALID=35,
    HEIGHT_CNEXT_RANGE_INVALID=36,
    HEIGHT_INVERSE_CDF_INVALID=37,
    HEIGHT_OUTPUT_INVALID=38,
    HEIGHT_FREE_PATH_INVALID=39,
    MICRO_WATCHDOG=40,
    MACRO_WATCHDOG=41
};
__device__ void micro_set_failure(int* code,int value){if(code)*code=value;}


// V24.46 full-production execution-integrity codes.  These are reporting-only and never
// convert a failure into escape/absorption/success.  Existing top-level status=4 is retained
// for compatibility while production_failure_code identifies the exact failure stage.
enum {
    PROD_OK=0,
    PROD_PARTICLE_MICROSURFACE_NUMERICAL=1,
    PROD_PARTICLE_MICROSURFACE_WATCHDOG=2,
    PROD_PARTICLE_MACRO_GEOMETRY_FAILURE=3,
    PROD_PARTICLE_MACRO_BOUNCE_LIMIT=4,
    PROD_ACRYLIC_ANNULUS_NO_INTERSECTION=5,
    PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT=6,
    PROD_BOTTOM_DISC_GEOMETRY_FAILURE=7,
    PROD_BOTTOM_DISC_BOUNCE_LIMIT=8,
    PROD_WATER_BOUNDARY_NO_INTERSECTION=9,
    PROD_TRACE_ITERATION_LIMIT=10,
    PROD_NONFINITE_POSITION=11,
    PROD_NONFINITE_DIRECTION=12,
    PROD_INVALID_MEDIUM_STATE=13,
    PROD_INVALID_EVENT_STATE=14,
    PROD_OTHER_EXPLICIT_FAILURE=15,
    PROD_SPHEROID_GEOMETRY_FAILURE=16
};
enum {
    PROD_SUB_NONE=0,
    PROD_SUB_SPHERE_ENTRY_SIDE=1001,
    PROD_SUB_SPHERE_CHORD=1002,
    PROD_SUB_SPHERE_EXIT_SIDE=1003,
    PROD_SUB_SPHERE_INTERNAL_SIDE=1004,
    PROD_SUB_SPHERE_BOUNCE_LIMIT=1005,
    PROD_SUB_SPHEROID_ASPECT=1101,
    PROD_SUB_SPHEROID_AXES=1102,
    PROD_SUB_SPHEROID_ENTRY=1103,
    PROD_SUB_SPHEROID_CHORD=1104,
    PROD_SUB_SPHEROID_EXIT_SIDE=1105,
    PROD_SUB_SPHEROID_INTERNAL_SIDE=1106,
    PROD_SUB_SPHEROID_BOUNCE_LIMIT=1107,
    PROD_SUB_ACRYLIC_NO_INTERSECTION=2001,
    PROD_SUB_ACRYLIC_BOUNCE_LIMIT=2002,
    PROD_SUB_BOTTOM_NONPOSITIVE_T=2101,
    PROD_SUB_BOTTOM_TANGENTIAL=2102,
    PROD_SUB_BOTTOM_BOUNCE_LIMIT=2103,
    PROD_SUB_BOTTOM_TOPOLOGY_INVALID=2104,
    PROD_SUB_WATER_NO_BOUNDARY=3001,
    PROD_SUB_TRACE_ITERATION_LIMIT=3002,
    PROD_SUB_NONFINITE_POSITION=3003,
    PROD_SUB_NONFINITE_DIRECTION=3004,
    PROD_SUB_WATER_OUTSIDE_INNER_RADIUS=3005
};
__device__ void prod_set_failure(int* code,int* subcode,int value,int sub){if(code)*code=value;if(subcode)*subcode=sub;}

// V24.46.0 NVRTC-portable IEEE-754 sentinels.  CuPy RawModule/NVRTC does not
// guarantee the host C NaN/infinity macros are defined.  These helpers encode exactly
// the same quiet-NaN/+Inf values and do not change any physical branch or probability.
__device__ double v41_qnan_d(){return __longlong_as_double(0x7ff8000000000000LL);}
__device__ double v41_pos_inf_d(){return __longlong_as_double(0x7ff0000000000000LL);}
__device__ float v41_qnan_f(){return __int_as_float(0x7fc00000);}
__device__ float v41_pos_inf_f(){return __int_as_float(0x7f800000);}

// Open-interval 32-bit uniform for the numerically delicate VNDF inversion.  The global
// legacy rnd_uniform() returns float and can round the largest uint32 values to exactly 1.0f.
// Using the same xorshift state word with a half-ulp bin-centre mapping preserves the RNG
// sequence while guaranteeing 0<U<1 without clipping any physical VNDF tail.
__device__ double beckmann_open_uniform(unsigned int* state){
    if(!state)return -1.0;
    unsigned int r=xorshift32_state(state);
    return ((double)r+0.5)*2.3283064365386962890625e-10; // 2^-32
}

// B(x)=1-x*sqrt(pi)*erfcx(x), x>=0.  For deep Beckmann backside incidence the
// individual projected-area terms are exponentially tiny and nearly cancel.  Evaluating
// their ratio through B avoids both erff(a)+1 cancellation and underflow.  The large-x
// branch is the standard asymptotic expansion of erfcx and contains no probability cutoff.
__device__ double beckmann_tail_B(double x){
    if(!(x>=0.0)||!isfinite(x))return -1.0;
    const double SQRT_PI=1.77245385090551602729816748334114518;
    if(x<5.0){
        double e=exp(x*x)*erfc(x);
        double b=1.0-x*SQRT_PI*e;
        return (isfinite(b)&&b>0.0)?b:-1.0;
    }
    double y=1.0/(x*x);
    // 1/(2x^2)-3/(4x^4)+15/(8x^6)-... ; seven terms are ample at x>=5.
    double b=0.5*y-0.75*y*y+1.875*y*y*y-6.5625*y*y*y*y
            +29.53125*y*y*y*y*y-162.421875*y*y*y*y*y*y
            +1055.7421875*y*y*y*y*y*y*y;
    return (isfinite(b)&&b>0.0)?b:-1.0;
}

// Marginal CDF of the alpha=1 visible Beckmann slope for the numerically ordinary
// domain slope_i>=-1.  The complementary error function is used so that the negative
// Gaussian tail is never evaluated as 1+erf(x).  No arbitrary erf-domain clipping exists.
__device__ int beckmann_visible_cdf_ordinary(double slope,double slope_i,double* out){
    if(!out||!isfinite(slope)||!isfinite(slope_i))return 0;
    if(slope>=slope_i){*out=1.0;return 1;}
    const double INV_2_SQRT_PI_D=0.28209479177387814347403972578038629;
    double q=INV_2_SQRT_PI_D*exp(-slope_i*slope_i)+0.5*slope_i*erfc(-slope_i);
    if(!(q>0.0)||!isfinite(q))return 0;
    double n=INV_2_SQRT_PI_D*exp(-slope*slope)+0.5*slope_i*erfc(-slope);
    double f=n/q;
    if(!isfinite(f))return 0;
    if(f<0.0&&f>-1.0e-13)f=0.0;
    if(f>1.0&&f<1.0+1.0e-13)f=1.0;
    if(f<0.0||f>1.0)return 0;
    *out=f;return 1;
}

// Numerically complete full-incidence isotropic Beckmann visible-slope sampler for alpha=1.
// The underlying density is exactly the Heitz/Jakob sampleP22_11 visible-slope density.
// V24.39 inherited the supplemental code's erf_min=-0.9999 convenience bound; that bound
// truncates the physically required backside support.  V24.41 instead inverts the same CDF
// directly.  For slope_i<-1 it uses a scaled tail variable u where slope=-(x+u), x=-slope_i:
//   F = exp(-(2*x*u+u^2)) * [u/(x+u)+x/(x+u) B(x+u)] / B(x)
// which is algebraically identical to the Beckmann CDF but remains finite as theta->pi.
// Full-incidence alpha=1 visible-slope sampler parameterised directly by signed cot(theta).
// V24.46 removes the V24.41.1 direction->acos->float-theta->cot round trip: the visible
// support is defined from the stretched direction in double precision.  This changes only
// numerical representation of the same Heitz/Beckmann VNDF, never its probability law.
__device__ int beckmann_sample11_signed_slope_d_ex(double slope_i,double U1,double U2,double* slope_x,double* slope_y,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!slope_x||!slope_y||!isfinite(U1)||!isfinite(U2)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    double u1=U1,u2=U2;
    if(!(u1>0.0&&u1<1.0)||!(u2>0.0&&u2<1.0)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    // Preserve the inherited V24.41.1 near-+normal numerical limit (theta<1e-4 rad),
    // expressed directly in cot(theta) so corrected production need not reconstruct theta.
    // In this limit the visible distribution is numerically indistinguishable from the NDF.
    const double COT_1E4=9999.999966666666;
    if(isinf(slope_i)||slope_i>COT_1E4){
        if(slope_i<0.0){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
        double r=sqrt(-log(u1)),phi=2.0*PI_D*u2;
        double sx=r*cos(phi),sy=r*sin(phi);
        if(!isfinite(sx)||!isfinite(sy)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        *slope_x=sx;*slope_y=sy;return 1;
    }
    if(!isfinite(slope_i)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}

    double sx=0.0;
    if(slope_i < -1.0){
        // Deep-backside branch: invert the scaled complementary-tail CDF in u>=0.
        double x=-slope_i,Bx=beckmann_tail_B(x);
        if(!(Bx>0.0)||!isfinite(Bx)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
        double target=log(u1),lo=0.0,hi=fmax(1.0/x,1.0e-12);
        int bracketed=0;
        for(int it=0;it<64;++it){
            double y=x+hi,By=beckmann_tail_B(y);
            if(!(By>0.0)||!isfinite(By)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double C=hi/y+(x/y)*By;
            if(!(C>0.0)||!isfinite(C)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double lr=-(2.0*x*hi+hi*hi)+log(C/Bx);
            if(!isfinite(lr)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            if(lr<=target){bracketed=1;break;}
            hi*=2.0;
            if(!isfinite(hi)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        }
        if(!bracketed){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        for(int it=0;it<52;++it){
            double mid=0.5*(lo+hi),y=x+mid,By=beckmann_tail_B(y);
            if(!(By>0.0)||!isfinite(By)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double C=mid/y+(x/y)*By;
            if(!(C>0.0)||!isfinite(C)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double lr=-(2.0*x*mid+mid*mid)+log(C/Bx);
            if(!isfinite(lr)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            if(lr>target)lo=mid;else hi=mid;
        }
        sx=-(x+0.5*(lo+hi));
    }else{
        // Ordinary/front-side branch: monotone direct inversion in slope space.
        double lo=fmin(-1.0,slope_i-1.0),hi=slope_i,flo=1.0;
        int bracketed=0;
        for(int it=0;it<64;++it){
            if(!beckmann_visible_cdf_ordinary(lo,slope_i,&flo)){
                micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
            }
            if(flo<=u1){bracketed=1;break;}
            lo=(lo<0.0)?2.0*lo:lo-1.0;
            if(!isfinite(lo)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        }
        if(!bracketed){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        for(int it=0;it<48;++it){
            double mid=0.5*(lo+hi),fm=0.0;
            if(!beckmann_visible_cdf_ordinary(mid,slope_i,&fm)){
                micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
            }
            if(fm<u1)lo=mid;else hi=mid;
        }
        sx=0.5*(lo+hi);
    }
    double ey=2.0*u2-1.0;
    if(!(ey>-1.0&&ey<1.0)||!isfinite(ey)){
        micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;
    }
    double sy=erfinv(ey);
    if(!isfinite(sx)||!isfinite(sy)){
        micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;
    }
    *slope_x=sx;*slope_y=sy;return 1;
}

// Compatibility angle API.  Theta remains double end-to-end; corrected production bypasses
// this wrapper and supplies signed cot(theta) directly from the stretched direction.
__device__ int beckmann_sample11_signed_d_ex(double theta_i,double U1,double U2,double* slope_x,double* slope_y,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!isfinite(theta_i)||!(theta_i>=0.0&&theta_i<PI_D)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    if(theta_i==0.0)return beckmann_sample11_signed_slope_d_ex(v41_pos_inf_d(),U1,U2,slope_x,slope_y,failure_code);
    double st=sin(theta_i),ct=cos(theta_i);
    if(!(st>0.0)||!isfinite(st)||!isfinite(ct)){
        micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;
    }
    return beckmann_sample11_signed_slope_d_ex(ct/st,U1,U2,slope_x,slope_y,failure_code);
}
__device__ int beckmann_sample11_signed_ex(float theta_i,double U1,double U2,float* slope_x,float* slope_y,int* failure_code){
    double sx=0.0,sy=0.0;int ok=beckmann_sample11_signed_d_ex(theta_i,U1,U2,&sx,&sy,failure_code);
    if(!ok)return 0;*slope_x=(float)sx;*slope_y=(float)sy;
    if(!isfinite(*slope_x)||!isfinite(*slope_y)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}return 1;
}
__device__ int beckmann_sample11_signed(float theta_i,float U1,float U2,float* slope_x,float* slope_y){
    int fc=MICRO_FAIL_NONE;double u1=(double)U1,u2=(double)U2;if(u1<=0.0)u1=nextafter(0.0,1.0);if(u2<=0.0)u2=nextafter(0.0,1.0);if(u1>=1.0)u1=nextafter(1.0,0.0);if(u2>=1.0)u2=nextafter(1.0,0.0);return beckmann_sample11_signed_ex(theta_i,u1,u2,slope_x,slope_y,&fc);
}

// Signed/full-incidence Beckmann VNDF in a fixed TOP-oriented frame b.  The returned
// microfacet normal belongs to the top-oriented NDF and is visible from w.  The normal is
// constructed with double intermediates in the deep backside tail to avoid float cancellation.
// Double-precision reconstruction used by the V24.46 production walker.  Sampling
// probabilities are unchanged from V24.41.1; the stretched visible-slope support is now passed
// directly in double precision and representation precision is retained through visibility/Fresnel
// geometry so an ultra-grazing positive dot product cannot flip sign.
__device__ int beckmann_vndf_normal_signed_d_ex(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float wx,float wy,float wz,
    double* mx,double* my,double* mz,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!(alpha>=0.0f)||!isfinite(alpha)||!state||!mx||!my||!mz){
        micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;
    }
    normalize3(&bx,&by,&bz);normalize3(&wx,&wy,&wz);
    if(!isfinite(bx)||!isfinite(by)||!isfinite(bz)||!isfinite(wx)||!isfinite(wy)||!isfinite(wz)){
        micro_set_failure(failure_code,NONFINITE_DIRECTION);return 0;
    }
    if(alpha<=1.0e-8f){
        double vd=(double)wx*bx+(double)wy*by+(double)wz*bz;
        if(!(vd>0.0)){micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return 0;}
        *mx=(double)bx;*my=(double)by;*mz=(double)bz;return 1;
    }
    float t1x,t1y,t1z,t2x,t2y,t2z;
    perpendicular_basis(bx,by,bz,&t1x,&t1y,&t1z,&t2x,&t2y,&t2z);
    double lx=(double)wx*t1x+(double)wy*t1y+(double)wz*t1z;
    double ly=(double)wx*t2x+(double)wy*t2y+(double)wz*t2z;
    double lz=(double)wx*bx+(double)wy*by+(double)wz*bz; // SIGN IS PHYSICAL: never clamp to positive.
    double sxv=(double)alpha*lx,syv=(double)alpha*ly,szv=lz;
    double vl=sqrt(sxv*sxv+syv*syv+szv*szv);
    if(!(vl>0.0)||!isfinite(vl)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    sxv/=vl;syv/=vl;szv/=vl;
    double radial=hypot(sxv,syv),slope_i=0.0;
    if(radial==0.0){
        if(szv>0.0)slope_i=v41_pos_inf_d();
        else{micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
    }else{
        slope_i=szv/radial;
        if(!isfinite(slope_i)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
    }
    double sx0=0.0,sy0=0.0;int fc=MICRO_FAIL_NONE;
    double U1=beckmann_open_uniform(state),U2=beckmann_open_uniform(state);
    if(!beckmann_sample11_signed_slope_d_ex(slope_i,U1,U2,&sx0,&sy0,&fc)){
        micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return 0;
    }
    double phi=atan2(syv,sxv),cp=cos(phi),sp=sin(phi);
    double tx=cp*sx0-sp*sy0,ty=sp*sx0+cp*sy0;
    double sx=(double)alpha*tx,sy=(double)alpha*ty;
    if(!isfinite(sx)||!isfinite(sy)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    double invn=1.0/sqrt(sx*sx+sy*sy+1.0);
    if(!(invn>0.0)||!isfinite(invn)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    double nlx=-sx*invn,nly=-sy*invn,nlz=invn;
    double nx=nlx*t1x+nly*t2x+nlz*bx;
    double ny=nlx*t1y+nly*t2y+nlz*by;
    double nz=nlx*t1z+nly*t2z+nlz*bz;
    double nn=sqrt(nx*nx+ny*ny+nz*nz);
    if(!(nn>0.0)||!isfinite(nn)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    nx/=nn;ny/=nn;nz/=nn;
    double top=nx*bx+ny*by+nz*bz;
    double vis=nx*wx+ny*wy+nz*wz;
    if(!(top>0.0)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    if(!(vis>0.0)){micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return 0;}
    *mx=nx;*my=ny;*mz=nz;return 1;
}
__device__ int beckmann_vndf_normal_signed_ex(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float wx,float wy,float wz,
    float* mx,float* my,float* mz,int* failure_code){
    double nx=0.0,ny=0.0,nz=0.0;int ok=beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,wx,wy,wz,&nx,&ny,&nz,failure_code);
    if(!ok)return 0;*mx=(float)nx;*my=(float)ny;*mz=(float)nz;
    if(!isfinite(*mx)||!isfinite(*my)||!isfinite(*mz)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    return 1;
}
__device__ int beckmann_vndf_normal_signed(
    unsigned int* state,float alpha,float bx,float by,float bz,float wx,float wy,float wz,float* mx,float* my,float* mz){
    int fc=MICRO_FAIL_NONE;return beckmann_vndf_normal_signed_ex(state,alpha,bx,by,bz,wx,wy,wz,mx,my,mz,&fc);
}

// Compatibility name for audit/non-production call sites.  It is the same V24.46 signed
// VNDF and never reverts to a raw NDF or geometric-normal fallback.
__device__ int beckmann_microfacet_normal(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float vx,float vy,float vz,int desired_dot_sign,
    float* mx,float* my,float* mz){
    float wx=(desired_dot_sign<0)?-vx:vx,wy=(desired_dot_sign<0)?-vy:vy,wz=(desired_dot_sign<0)?-vz:vz;
    return beckmann_vndf_normal_signed(state,alpha,bx,by,bz,wx,wy,wz,mx,my,mz);
}

__device__ float smith_uniform_C1(float h){return fminf(fmaxf(0.5f*(h+1.0f),0.0f),1.0f);}
__device__ float smith_uniform_invC1(float u){return 2.0f*fminf(fmaxf(u,0.0f),1.0f)-1.0f;}

// Numerically stable signed Beckmann Smith Lambda over the full direction sphere.
// For a>0 the standard Beckmann expression
//   Lambda = -0.5*erfc(a) + exp(-a^2)/(2*sqrt(pi)*a)
// is factored as exp(-a^2)/(2*sqrt(pi)*a) * B(a), where
//   B(a)=1-a*sqrt(pi)*erfcx(a).
// beckmann_tail_B evaluates B without the catastrophic erf/exp cancellation that caused
// V24.40 HEIGHT_CDF_INVALID failures.  Signed symmetry is exact: Lambda(-a)=-1-Lambda(a).
// No epsilon clamp is used.  The exact vertical limits are Lambda(+z)=0 and Lambda(-z)=-1.
__device__ double smith_beckmann_lambda_d(float wx,float wy,float wz,float bx,float by,float bz,float alpha){
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c)||!isfinite(alpha)||!(alpha>=0.0f))return v41_qnan_d();
    c=fmax(-1.0,fmin(1.0,c));
    double ss2=fmax(0.0,1.0-c*c);
    if(ss2==0.0)return c>=0.0?0.0:-1.0;
    if(c==0.0)return copysign(v41_pos_inf_d(),c);
    if(alpha==0.0f)return c>=0.0?0.0:-1.0;
    double aa=fabs(c)/((double)alpha*sqrt(ss2));
    if(!(aa>0.0)||!isfinite(aa))return c>=0.0?0.0:-1.0;
    const double SQRT_PI=1.77245385090551602729816748334114518;
    double B=beckmann_tail_B(aa);
    if(!(B>0.0)||!isfinite(B))return v41_qnan_d();
    double logpref=-(aa*aa)-log(2.0*SQRT_PI*aa);
    // Direct exp() underflow simply rounds a physically positive Lambda below the smallest
    // representable double to zero; no empirical cutoff or epsilon clamp is introduced.
    double lp=exp(logpref)*B;
    if(!(lp>=0.0)||!isfinite(lp))return v41_qnan_d();
    return c>=0.0?lp:(-1.0-lp);
}
__device__ float smith_beckmann_lambda(float wx,float wy,float wz,float bx,float by,float bz,float alpha){
    double l=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(isnan(l))return v41_qnan_f();
    if(isinf(l))return l>0.0?v41_pos_inf_f():-v41_pos_inf_f();
    return (float)l;
}

__device__ double smith_uniform_C1_d(double h){return fmin(1.0,fmax(0.0,0.5*(h+1.0)));}
__device__ double smith_uniform_invC1_d(double u){return 2.0*fmin(1.0,fmax(0.0,u))-1.0;}

__device__ int smith_G1_height_checked(float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_G1){
    if(!out_G1||!isfinite(h))return 0;
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c))return 0;
    if(c<=0.0){*out_G1=0.0f;return 1;}
    double C1=smith_uniform_C1_d((double)h);
    if(C1<=0.0){*out_G1=0.0f;return 1;}
    if(C1>=1.0){*out_G1=1.0f;return 1;}
    double lam=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(!isfinite(lam)||lam<0.0)return 0;
    if(lam==0.0){*out_G1=1.0f;return 1;}
    double lg=lam*log(C1),g=exp(lg);
    if(!(g>=0.0&&g<=1.0)||!isfinite(g))return 0;
    *out_G1=(float)g;return 1;
}

// Stable Smith height/free-path sampler.  It is algebraically identical to the Heitz
// uniform-height process but evaluates the escape probability as -expm1(Lambda*log(C0))
// and the next CDF in log domain.  This removes both 1-G cancellation and the unstable
// pow(1-U,1/Lambda) inversion.  Numerical failure and physical escape remain disjoint.
__device__ int smith_sample_height_u_ex(
    double U,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!out_h||!isfinite(U)||!isfinite(wx)||!isfinite(wy)||!isfinite(wz)||!isfinite(h)||!(alpha>=0.0f)||!isfinite(alpha)){
        micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;
    }
    if(!(U>0.0&&U<1.0)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c)){micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;}
    c=fmax(-1.0,fmin(1.0,c));

    double C0=smith_uniform_C1_d((double)h);
    if(!(C0>=0.0&&C0<=1.0)||!isfinite(C0)){
        micro_set_failure(failure_code,HEIGHT_C0_INVALID);return MICRO_NUMERICAL_FAILURE;
    }

    // Exact vertical limits of the Smith process; no finite-angle exclusion cone.
    double ss2=fmax(0.0,1.0-c*c);
    if(ss2==0.0){
        if(c>0.0)return MICRO_ESCAPE;
        if(c<0.0){
            double Cnext=U*C0,hn=smith_uniform_invC1_d(Cnext);
            if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
            *out_h=(float)hn;return MICRO_INTERSECTION;
        }
        *out_h=h;return MICRO_INTERSECTION;
    }
    // Exact horizontal limit: Lambda magnitude diverges and the next interaction is at h.
    if(c==0.0){*out_h=h;return MICRO_INTERSECTION;}

    double lam=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(isnan(lam)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}

    if(c>0.0){
        if(!(lam>=0.0)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}
        if(C0<=0.0)return MICRO_ESCAPE;
        if(C0>=1.0)return MICRO_ESCAPE;
        if(lam==0.0)return MICRO_ESCAPE; // exact representable Lambda->0 limit
        double logC0=log(C0);
        if(!isfinite(logC0)){micro_set_failure(failure_code,HEIGHT_LOG_C0_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double logG1=lam*logC0;
        if(!(logG1<=0.0)||!isfinite(logG1)){micro_set_failure(failure_code,HEIGHT_G1_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double p_intersect=-expm1(logG1);
        if(!(p_intersect>=0.0&&p_intersect<=1.0)||!isfinite(p_intersect)){
            micro_set_failure(failure_code,HEIGHT_ESCAPE_PROB_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        if(U>p_intersect)return MICRO_ESCAPE;
        double log1mU=log1p(-U);
        if(!isfinite(log1mU)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double logCnext=logC0-log1mU/lam;
        if(!isfinite(logCnext)){micro_set_failure(failure_code,HEIGHT_LOG_CNEXT_INVALID);return MICRO_NUMERICAL_FAILURE;}
        // Exact arithmetic guarantees logCnext<=0 for U<=p_intersect.  Permit only a
        // few double ulps of boundary-rounding before projecting the representable endpoint.
        const double BOUND_TOL=64.0*2.2204460492503130808472633361816e-16;
        if(logCnext>BOUND_TOL){micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;}
        if(logCnext>0.0)logCnext=0.0;
        double Cnext=exp(logCnext);
        if(!(Cnext>=0.0&&Cnext<=1.0)||!isfinite(Cnext)){
            micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        double hn=smith_uniform_invC1_d(Cnext);
        if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
        *out_h=(float)hn;
        if(!isfinite(*out_h)){micro_set_failure(failure_code,HEIGHT_OUTPUT_INVALID);return MICRO_NUMERICAL_FAILURE;}
        return MICRO_INTERSECTION;
    }

    // Downward/signed side: Lambda<=-1, intersection is certain and the same log-CDF
    // inversion naturally moves C downward.  The exact negative-vertical limit is handled above.
    if(!(lam<=-1.0)||!isfinite(lam)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(C0<=0.0){*out_h=-1.0f;return MICRO_INTERSECTION;}
    double logC0=log(C0);
    if(!isfinite(logC0)){micro_set_failure(failure_code,HEIGHT_LOG_C0_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double log1mU=log1p(-U);
    if(!isfinite(log1mU)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double logCnext=logC0-log1mU/lam;
    if(!isfinite(logCnext)){micro_set_failure(failure_code,HEIGHT_LOG_CNEXT_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(logCnext>0.0){micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double Cnext=exp(logCnext);
    if(!(Cnext>=0.0&&Cnext<=C0*(1.0+8.0e-15))||!isfinite(Cnext)){
        micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;
    }
    double hn=smith_uniform_invC1_d(Cnext);
    if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
    *out_h=(float)hn;
    if(!isfinite(*out_h)){micro_set_failure(failure_code,HEIGHT_OUTPUT_INVALID);return MICRO_NUMERICAL_FAILURE;}
    return MICRO_INTERSECTION;
}
__device__ int smith_sample_height_u(
    float U,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h){
    int fc=MICRO_FAIL_NONE;double u=(double)U;
    if(u<=0.0)u=nextafter(0.0,1.0);if(u>=1.0)u=nextafter(1.0,0.0);
    return smith_sample_height_u_ex(u,wx,wy,wz,bx,by,bz,alpha,h,out_h,&fc);
}
__device__ int smith_sample_height_ex(
    unsigned int* state,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h,int* failure_code){
    double U=beckmann_open_uniform(state);
    if(!(U>0.0&&U<1.0)||!isfinite(U)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    return smith_sample_height_u_ex(U,wx,wy,wz,bx,by,bz,alpha,h,out_h,failure_code);
}
__device__ int smith_sample_height(
    unsigned int* state,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h){
    int fc=MICRO_FAIL_NONE;return smith_sample_height_ex(state,wx,wy,wz,bx,by,bz,alpha,h,out_h,&fc);
}

// Complete local rough-dielectric Smith microsurface random walk.  The V24.40 physical
// process is retained; V24.46 keeps V24.41 Lambda/height arithmetic and changes only ultra-grazing
// representation precision, not the stochastic distribution or Fresnel branch law.
__device__ int smith_microsurface_dielectric_walk_ex(
    unsigned int* state,float alpha,int micro_watchdog,
    float bx,float by,float bz,float n_top,float k_top,float n_bottom,float k_bottom,
    float* dx,float* dy,float* dz,int* out_top,int* scatter_order,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    normalize3(&bx,&by,&bz);normalize3(dx,dy,dz);*scatter_order=0;
    if(!isfinite(bx)||!isfinite(by)||!isfinite(bz)||!isfinite(*dx)||!isfinite(*dy)||!isfinite(*dz)){
        micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
    }
    if(dot3(*dx,*dy,*dz,bx,by,bz)>=-1.0e-7f){micro_set_failure(failure_code,SIDE_STATE_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(micro_watchdog<1){micro_set_failure(failure_code,MICRO_WATCHDOG);return MICRO_WATCHDOG_FAILURE;}

    // Exact physical smooth-interface limit; not a stochastic fallback.
    if(alpha<=1.0e-7f){
        float wi_x=-(*dx),wi_y=-(*dy),wi_z=-(*dz);
        float ci=fminf(fmaxf(dot3(wi_x,wi_y,wi_z,bx,by,bz),0.0f),1.0f);
        float eta=n_top/n_bottom;
        if(!(n_bottom>0.0f)||!isfinite(eta)){micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        int tir=(eta*eta*fmaxf(0.0f,1.0f-ci*ci)>=1.0f);
        float R=tir?1.0f:fresnel_R_complex(ci,n_top,k_top,n_bottom,k_bottom);
        if(!isfinite(R)){micro_set_failure(failure_code,FRESNEL_INVALID);return MICRO_NUMERICAL_FAILURE;}
        float ox,oy,oz;
        if(tir||rnd_uniform(state)<R){
            reflect3(*dx,*dy,*dz,bx,by,bz,&ox,&oy,&oz);*out_top=1;
            if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){micro_set_failure(failure_code,REFLECTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        }else{
            if(!refract3(*dx,*dy,*dz,bx,by,bz,n_top,n_bottom,&ox,&oy,&oz)){
                micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
            *out_top=0;
        }
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;}
        *dx=ox;*dy=oy;*dz=oz;*scatter_order=1;return MICRO_ESCAPE;
    }

    float h=1.0f+smith_uniform_invC1(0.999f);int outside=1,order=0;
    for(int it=0;it<micro_watchdog;++it){
        float nh=0.0f;int hs,fc=MICRO_FAIL_NONE;
        if(outside){
            hs=smith_sample_height_ex(state,*dx,*dy,*dz,bx,by,bz,alpha,h,&nh,&fc);
            if(hs==MICRO_ESCAPE){*out_top=1;*scatter_order=order;return MICRO_ESCAPE;}
        }else{
            float tmp=0.0f;
            hs=smith_sample_height_ex(state,-(*dx),-(*dy),-(*dz),bx,by,bz,alpha,-h,&tmp,&fc);
            if(hs==MICRO_ESCAPE){*out_top=0;*scatter_order=order;return MICRO_ESCAPE;}
            if(hs==MICRO_INTERSECTION)nh=-tmp;
        }
        if(hs!=MICRO_INTERSECTION||!isfinite(nh)){
            *scatter_order=order;micro_set_failure(failure_code,fc?fc:HEIGHT_SAMPLE_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        h=nh;++order;
        float wix=-(*dx),wiy=-(*dy),wiz=-(*dz);double mx,my,mz;fc=MICRO_FAIL_NONE;
        if(outside){
            if(!beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,wix,wiy,wiz,&mx,&my,&mz,&fc)){
                *scatter_order=order;micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
        }else{
            double tx,ty,tz;
            if(!beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,-wix,-wiy,-wiz,&tx,&ty,&tz,&fc)){
                *scatter_order=order;micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
            mx=-tx;my=-ty;mz=-tz;
        }
        double vis=(double)wix*mx+(double)wiy*my+(double)wiz*mz;
        if(!isfinite(mx)||!isfinite(my)||!isfinite(mz)){
            *scatter_order=order;micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        if(!(vis>0.0)){
            *scatter_order=order;micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return MICRO_NUMERICAL_FAILURE;
        }
        double ci_d=fmin(1.0,fmax(0.0,vis));float ci=(float)ci_d;
        float n1=outside?n_top:n_bottom,k1=outside?k_top:k_bottom,n2=outside?n_bottom:n_top,k2=outside?k_bottom:k_top;
        if(!(n1>0.0f&&n2>0.0f)){micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double eta=(double)n1/(double)n2;double kt=1.0-eta*eta*fmax(0.0,1.0-ci_d*ci_d);int tir=(kt<=0.0);
        float R=tir?1.0f:fresnel_R_complex(ci,n1,k1,n2,k2);
        if(!isfinite(R)){micro_set_failure(failure_code,FRESNEL_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double ox_d,oy_d,oz_d;
        if(tir||rnd_uniform(state)<R){
            double dd=(double)(*dx)*mx+(double)(*dy)*my+(double)(*dz)*mz;
            ox_d=(double)(*dx)-2.0*dd*mx;oy_d=(double)(*dy)-2.0*dd*my;oz_d=(double)(*dz)-2.0*dd*mz;
        }else{
            double ct=sqrt(fmax(0.0,kt));
            ox_d=eta*(double)(*dx)+(eta*ci_d-ct)*mx;
            oy_d=eta*(double)(*dy)+(eta*ci_d-ct)*my;
            oz_d=eta*(double)(*dz)+(eta*ci_d-ct)*mz;
            outside=1-outside;
        }
        double qn=sqrt(ox_d*ox_d+oy_d*oy_d+oz_d*oz_d);
        if(!(qn>0.0)||!isfinite(qn)||!isfinite(ox_d)||!isfinite(oy_d)||!isfinite(oz_d)){
            *scatter_order=order;micro_set_failure(failure_code,tir?REFLECTION_VECTOR_INVALID:REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        ox_d/=qn;oy_d/=qn;oz_d/=qn;float ox=(float)ox_d,oy=(float)oy_d,oz=(float)oz_d;
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){
            *scatter_order=order;micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
        }
        normalize3(&ox,&oy,&oz);
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){
            *scatter_order=order;micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
        }
        *dx=ox;*dy=oy;*dz=oz;
    }
    *scatter_order=order;micro_set_failure(failure_code,MICRO_WATCHDOG);return MICRO_WATCHDOG_FAILURE;
}
__device__ int smith_microsurface_dielectric_walk(
    unsigned int* state,float alpha,int micro_watchdog,float bx,float by,float bz,
    float n_top,float k_top,float n_bottom,float k_bottom,float* dx,float* dy,float* dz,int* out_top,int* scatter_order){
    int fc=MICRO_FAIL_NONE;return smith_microsurface_dielectric_walk_ex(state,alpha,micro_watchdog,bx,by,bz,n_top,k_top,n_bottom,k_bottom,dx,dy,dz,out_top,scatter_order,&fc);
}

__device__ void sphere_sample_geometry(
    unsigned int* state,const float radius,const float x,const float y,const float z,
    const float vx,const float vy,const float vz,
    float* rho_out,float* phi_out,float* gnx_out,float* gny_out,float* gnz_out,
    float* cx_out,float* cy_out,float* cz_out){
    float rho=sqrtf(fminf(fmaxf(rnd_uniform(state),0.0f),0.99999994f));
    float ci=sqrtf(fmaxf(0.0f,1.0f-rho*rho));
    float phi=rnd_uniform(state)*2.0f*PI_F;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(vx,vy,vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float qx=cosf(phi)*e1x+sinf(phi)*e2x;
    float qy=cosf(phi)*e1y+sinf(phi)*e2y;
    float qz=cosf(phi)*e1z+sinf(phi)*e2z;
    float gnx=-ci*vx+rho*qx,gny=-ci*vy+rho*qy,gnz=-ci*vz+rho*qz;
    normalize3(&gnx,&gny,&gnz);
    if(rho_out)*rho_out=rho;if(phi_out)*phi_out=phi;
    if(gnx_out)*gnx_out=gnx;if(gny_out)*gny_out=gny;if(gnz_out)*gnz_out=gnz;
    if(cx_out)*cx_out=x-radius*gnx;if(cy_out)*cy_out=y-radius*gny;if(cz_out)*cz_out=z-radius*gnz;
}

// V24.46 finite-particle accessibility uses the ACTUAL sampled particle centre.
// The localized free-surface models are axisymmetric and monotone increasing with r,
// so the lowest free-surface height over a spherical footprint occurs at r=max(0,rc-radius).
__device__ int sphere_particle_accessible_in_water(
    const float cx,const float cy,const float cz,const float radius,
    const float Rin,const float Zmin,const float Zmean,const int free_surface_model,
    const float vortex_delta_h,const float vortex_core_radius,
    float* center_radial_clearance,float* finite_wall_clearance,float* bottom_clearance,float* top_clearance){
    double rc=sqrt((double)cx*(double)cx+(double)cy*(double)cy);
    double scale=fmax((double)Rin,fmax(fabs((double)cz),(double)radius));
    double tol=8.0*1.1920928955078125e-7*fmax(scale,1.0e-6);
    double cclear=(double)Rin-rc;
    double wall=cclear-(double)radius;
    double bottom=(double)cz-(double)radius-(double)Zmin;
    double rmin=fmax(0.0,rc-(double)radius);
    float zsurf=free_surface_z((float)rmin,0.0f,Zmean,Rin,free_surface_model,vortex_delta_h,vortex_core_radius);
    double top=(double)zsurf-((double)cz+(double)radius);
    if(center_radial_clearance)*center_radial_clearance=(float)cclear;
    if(finite_wall_clearance)*finite_wall_clearance=(float)wall;
    if(bottom_clearance)*bottom_clearance=(float)bottom;
    if(top_clearance)*top_clearance=(float)top;
    return (wall>=-tol && bottom>=-tol && top>=-tol)?1:0;
}

__device__ int sphere_fresnel_interaction_3d_sampled_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    const float gnx,const float gny,const float gnz,const float cx,const float cy,const float cz,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order) {
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OK,PROD_SUB_NONE);
    if(out_macro_bounces)*out_macro_bounces=0;if(out_micro_order)*out_micro_order=0;
    if(radius<=0.0f)return 1;
    const int MICRO_WATCHDOG_LIMIT=4096;
    int top=1,order=0,mfc=MICRO_FAIL_NONE;
    int ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,gnx,gny,gnz,
            n_medium,0.0f,n_particle,k_particle,vx,vy,vz,&top,&order,&mfc);
    if(out_micro_order)*out_micro_order=order;
    if(ms!=MICRO_ESCAPE){
        prod_set_failure(prod_failure_code,prod_failure_subcode,
            ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);
        return 0;
    }
    if(top){(*p_reflections)++;(*p_entry_reflections)++;return 1;}
    if(dot3(*vx,*vy,*vz,gnx,gny,gnz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_ENTRY_SIDE);return 0;}

    float alpha_abs=(wavelength>0.0f && k_particle>0.0f)?(4.0f*PI_F*k_particle/wavelength):0.0f;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(out_macro_bounces)*out_macro_bounces=bounce;
        float rx=*x-cx,ry=*y-cy,rz=*z-cz;
        float chord=-2.0f*dot3(rx,ry,rz,*vx,*vy,*vz);
        if(chord<=1.0e-12f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        if(alpha_abs>0.0f){
            float survival=expf(-alpha_abs*chord);
            if(rnd_uniform(state)>survival){
                *x+=chord*(*vx);*y+=chord*(*vy);*z+=chord*(*vz);*internal_path+=chord;(*p_absorptions)++;return 2;
            }
        }
        *x+=chord*(*vx);*y+=chord*(*vy);*z+=chord*(*vz);*internal_path+=chord;
        float nox=(*x-cx)/radius,noy=(*y-cy)/radius,noz=(*z-cz)/radius;normalize3(&nox,&noy,&noz);
        float bx=-nox,by=-noy,bz=-noz;top=1;order=0;mfc=MICRO_FAIL_NONE;
        ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,bx,by,bz,
                n_particle,k_particle,n_medium,0.0f,vx,vy,vz,&top,&order,&mfc);
        if(out_micro_order)*out_micro_order=order;
        if(ms!=MICRO_ESCAPE){
            prod_set_failure(prod_failure_code,prod_failure_subcode,
                ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);
            return 0;
        }
        if(!top){
            if(dot3(*vx,*vy,*vz,nox,noy,noz)<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_EXIT_SIDE);return 0;}
            return 1;
        }
        if(dot3(*vx,*vy,*vz,nox,noy,noz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_INTERNAL_SIDE);return 0;}
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_BOUNCE_LIMIT,PROD_SUB_SPHERE_BOUNCE_LIMIT);return 0;}
    }
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OTHER_EXPLICIT_FAILURE,PROD_SUB_NONE);return 0;
}

__device__ int sphere_fresnel_interaction_3d_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order) {
    float rho=0.0f,phi=0.0f,gnx=0.0f,gny=0.0f,gnz=0.0f,cx=*x,cy=*y,cz=*z;
    sphere_sample_geometry(state,radius,*x,*y,*z,*vx,*vy,*vz,&rho,&phi,&gnx,&gny,&gnz,&cx,&cy,&cz);
    return sphere_fresnel_interaction_3d_sampled_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        gnx,gny,gnz,cx,cy,cz,x,y,z,vx,vy,vz,internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,
        prod_failure_code,prod_failure_subcode,out_macro_bounces,out_micro_order);
}
__device__ int sphere_fresnel_interaction_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions) {
    int pc=PROD_OK,ps=PROD_SUB_NONE,mb=0,mo=0;
    return sphere_fresnel_interaction_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        x,y,z,vx,vy,vz,internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,&pc,&ps,&mb,&mo);
}

// ========================== END V24.46 CORRECTED ROUGH BOUNDARY ==========================

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


// ============================ INHERITED SPHEROID WAVE MORPHOLOGY ============================
// Random-orientation, volume-preserving spheroid used ONLY for the incoherent
// supplemental wave channel.  The geometric projected-area channel remains the
// frozen V23/V24.8 rough-sphere interaction so this is a one-variable physics
// diagnostic of the wave-event angular law.
//
// Let q=a/b be the symmetry-axis/equatorial-axis aspect ratio and r the
// volume-equivalent sphere radius.  CUDA evaluates
//      a = r*q^(2/3),  b = r*q^(-1/3),  hence a*b^2=r^3.
// q>1 is prolate; q<1 is oblate. V24.20 freezes q=1 and isotropic orientation, while the inherited sampler remains available for regression.

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

__device__ int spheroid_fresnel_scatter_direction_only_3d_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,
    float r_equiv,float aspect_ratio,int orientation_model,float orientation_kappa,float flow_fx,float flow_fy,float flow_fz,
    int max_bounces,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order){
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OK,PROD_SUB_NONE);
    if(out_macro_bounces)*out_macro_bounces=0;if(out_micro_order)*out_micro_order=0;
    if(r_equiv<=0.0f)return 1;
    if(!(aspect_ratio>0.0f) || !isfinite(aspect_ratio)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ASPECT);return 0;}
    if(fabsf(aspect_ratio-1.0f)<1e-7f){
        float lx=0.0f,ly=0.0f,lz=0.0f,lvx=*vx,lvy=*vy,lvz=*vz,ip=0.0f;
        int ps=sphere_fresnel_interaction_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,r_equiv,max_bounces,
            &lx,&ly,&lz,&lvx,&lvy,&lvz,&ip,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,
            prod_failure_code,prod_failure_subcode,out_macro_bounces,out_micro_order);
        *internal_path += ip;if(ps==1){*vx=lvx;*vy=lvy;*vz=lvz;}return ps;
    }
    const int MICRO_WATCHDOG_LIMIT=4096;
    float q13=powf(aspect_ratio,1.0f/3.0f);float a=r_equiv*q13*q13;float b=r_equiv/q13;
    if(!(a>0.0f&&b>0.0f&&isfinite(a)&&isfinite(b))){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_AXES);return 0;}
    float ux,uy,uz;sample_spheroid_axis(state,orientation_model,orientation_kappa,flow_fx,flow_fy,flow_fz,&ux,&uy,&uz);
    float x,y,z,gnx,gny,gnz;
    if(!spheroid_sample_entry_projected(state,*vx,*vy,*vz,a,b,ux,uy,uz,&x,&y,&z,&gnx,&gny,&gnz)){
        prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ENTRY);return 0;
    }
    float invb2=1.0f/(b*b),delta=1.0f/(a*a)-invb2,scale=fmaxf(a,b);
    int top=1,order=0,mfc=MICRO_FAIL_NONE;
    int ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,gnx,gny,gnz,
            n_medium,0.0f,n_particle,k_particle,vx,vy,vz,&top,&order,&mfc);
    if(out_micro_order)*out_micro_order=order;
    if(ms!=MICRO_ESCAPE){prod_set_failure(prod_failure_code,prod_failure_subcode,ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);return 0;}
    if(top){(*p_reflections)++;(*p_entry_reflections)++;return 1;}
    if(dot3(*vx,*vy,*vz,gnx,gny,gnz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ENTRY);return 0;}
    float alpha_abs=(wavelength>0.0f&&k_particle>0.0f)?(4.0f*PI_F*k_particle/wavelength):0.0f;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(out_macro_bounces)*out_macro_bounces=bounce;
        float chord=spheroid_next_boundary_distance(x,y,z,*vx,*vy,*vz,ux,uy,uz,invb2,delta,scale);
        if(chord<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_CHORD);return 0;}
        if(alpha_abs>0.0f){float survival=expf(-alpha_abs*chord);if(rnd_uniform(state)>survival){*internal_path+=chord;(*p_absorptions)++;return 2;}}
        x+=chord*(*vx);y+=chord*(*vy);z+=chord*(*vz);*internal_path+=chord;
        float nox,noy,noz;spheroid_normal(x,y,z,ux,uy,uz,invb2,delta,&nox,&noy,&noz);
        float bx=-nox,by=-noy,bz=-noz;top=1;order=0;mfc=MICRO_FAIL_NONE;
        ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,bx,by,bz,
                n_particle,k_particle,n_medium,0.0f,vx,vy,vz,&top,&order,&mfc);
        if(out_micro_order)*out_micro_order=order;
        if(ms!=MICRO_ESCAPE){prod_set_failure(prod_failure_code,prod_failure_subcode,ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);return 0;}
        if(!top){if(dot3(*vx,*vy,*vz,nox,noy,noz)<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_EXIT_SIDE);return 0;}return 1;}
        if(dot3(*vx,*vy,*vz,nox,noy,noz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_INTERNAL_SIDE);return 0;}
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_BOUNCE_LIMIT,PROD_SUB_SPHEROID_BOUNCE_LIMIT);return 0;}
    }
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OTHER_EXPLICIT_FAILURE,PROD_SUB_NONE);return 0;
}
__device__ int spheroid_fresnel_scatter_direction_only_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,
    float r_equiv,float aspect_ratio,int orientation_model,float orientation_kappa,float flow_fx,float flow_fy,float flow_fz,
    int max_bounces,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions){
    int pc=PROD_OK,ps=PROD_SUB_NONE,mb=0,mo=0;
    return spheroid_fresnel_scatter_direction_only_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,r_equiv,aspect_ratio,
        orientation_model,orientation_kappa,flow_fx,flow_fy,flow_fz,max_bounces,vx,vy,vz,internal_path,
        p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,&pc,&ps,&mb,&mo);
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
__device__ int acrylic_annulus_next_surface(
    const float x,const float y,const float z,const float vx,const float vy,const float vz,
    const float Rin,const float Rout,const float Zlower,const float Zupper,double* t_out){
    double best=1.0e300;int surf=0;
    double ti=cell_cylinder_hit_forward_d(x,y,vx,vy,Rin);
    double to=cell_cylinder_hit_forward_d(x,y,vx,vy,Rout);
    if(ti>0.0 && ti<best){best=ti;surf=1;}
    if(to>0.0 && to<best){best=to;surf=2;}
    if(vz>0.0f){double t=((double)Zupper-(double)z)/(double)vz;if(t>0.0&&isfinite(t)&&t<best){best=t;surf=3;}}
    if(vz<0.0f){double t=((double)Zlower-(double)z)/(double)vz;if(t>0.0&&isfinite(t)&&t<best){best=t;surf=4;}}
    if(t_out)*t_out=best;return surf;
}

// V24.46: finite acrylic annulus.  The annulus spans the complete acrylic sidewall,
// including the contiguous bottom-disc thickness, and therefore has BOTH lower and
// upper axial external faces.  The lower bottom-disc/sidewall seam is not an optical
// interface; the finite volume simply continues down to Zlower.
// return: 1=water, 2=external/air, 0=failure.
__device__ int acrylic_annulus_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_air,
    float Rin,float Rout,float Zlower,float Zupper,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* inner_reflections,int* outer_reflections,int* cell_tir,int* failure_subcode,int* out_bounces) {
    if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=0;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(out_bounces)*out_bounces=bounce;
        double td=1.0e300;int surf=acrylic_annulus_next_surface(*x,*y,*z,*vx,*vy,*vz,Rin,Rout,Zlower,Zupper,&td);
        if(!surf || !(td>0.0) || !isfinite(td) || td>=1.0e299){if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_NO_INTERSECTION;return 0;}
        float t=(float)td;*x+=t*(*vx);*y+=t*(*vy);*z+=t*(*vz);*acrylic_path+=t;
        float a,b,c; int tir=0;
        if(surf==3){
            *z=Zupper;
            float ci=fminf(fmaxf(*vz,0.0f),1.0f);float R=fresnel_R(ci,n_acrylic,n_air,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,n_acrylic,n_air,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=Zupper+EPS_F;return 2;
            }
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*outer_reflections)++;if(tir)(*cell_tir)++;*z=Zupper-EPS_F;
        }else if(surf==4){
            *z=Zlower;
            float ci=fminf(fmaxf(-(*vz),0.0f),1.0f);float R=fresnel_R(ci,n_acrylic,n_air,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_acrylic,n_air,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=Zlower-EPS_F;return 2;
            }
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*outer_reflections)++;if(tir)(*cell_tir)++;*z=Zlower+EPS_F;
        }else{
            float nx,ny; radial_normal(*x,*y,&nx,&ny);
            if(surf==1){
                float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);float R=fresnel_R(ci,n_acrylic,n_water,&tir);
                if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,nx,ny,0.0f,n_acrylic,n_water,&a,&b,&c)){
                    *vx=a;*vy=b;*vz=c;*x-=EPS_F*nx;*y-=EPS_F*ny;return 1;
                }
                reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*inner_reflections)++;if(tir)(*cell_tir)++;*x+=EPS_F*nx;*y+=EPS_F*ny;
            }else{
                float ci=fminf(fmaxf(dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);float R=fresnel_R(ci,n_acrylic,n_air,&tir);
                if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,-nx,-ny,0.0f,n_acrylic,n_air,&a,&b,&c)){
                    *vx=a;*vy=b;*vz=c;*x+=EPS_F*nx;*y+=EPS_F*ny;return 2;
                }
                reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*outer_reflections)++;if(tir)(*cell_tir)++;*x-=EPS_F*nx;*y-=EPS_F*ny;
            }
        }
        if(bounce==max_bounces){if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_BOUNCE_LIMIT;return 0;}
    }
    if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_BOUNCE_LIMIT;return 0;
}
__device__ int acrylic_annulus(
    unsigned int* state,float n_water,float n_acrylic,float n_air,float Rin,float Rout,float Zlower,float Zupper,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,int* inner_reflections,int* outer_reflections,int* cell_tir){
    int fs=PROD_SUB_NONE,bb=0;return acrylic_annulus_ex(state,n_water,n_acrylic,n_air,Rin,Rout,Zlower,Zupper,max_bounces,x,y,z,vx,vy,vz,acrylic_path,inner_reflections,outer_reflections,cell_tir,&fs,&bb);
}

// V24.46: genuinely finite acrylic bottom disc.  The ray arrives at z_inner from water.
// The solid disc occupies r<=Rout, z_outer<=z<=z_inner.  At its top face only
// r<Rin is an acrylic/water interface; Rin<=r<=Rout is the contiguous acrylic
// sidewall seam and consumes no Fresnel event or RNG.  The outer cylindrical edge
// is a real acrylic/external boundary.
// return 1 = back in water, 2 = transmitted to external medium (lost),
//        3 = crossed the contiguous acrylic seam into the annular sidewall, 0 = failure.
__device__ int bottom_acrylic_disc_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_bottom_external,float n_side_external,
    float Rin,float Rout,float z_inner,float thickness,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* inner_reflections,int* outer_reflections,int* bottom_tir,int* failure_subcode,int* out_bounces) {
    if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=0;
    float a,b,c; int tir=0;
    // Initial water -> acrylic event exists only over the real water aperture.
    if(bottom_disc_topology_at_top(*x,*y,Rin,Rout)!=1){if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_TOPOLOGY_INVALID;return 0;}
    float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,1.0f),0.0f),1.0f);float R=fresnel_R(ci,n_water,n_acrylic,&tir);
    if(tir || rnd_uniform(state)<R || !refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_water,n_acrylic,&a,&b,&c)){
        reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*inner_reflections)++; if(tir)(*bottom_tir)++; *z=z_inner+EPS_F; return 1;
    }
    *vx=a;*vy=b;*vz=c;*z=z_inner-EPS_F;float z_outer=z_inner-thickness;
    int reflections=0;
    for(int guard=0;guard<2*(max_bounces+1)+8;++guard){
        if(out_bounces)*out_bounces=reflections;
        double td=1.0e300;int surf=bottom_disc_next_surface(*x,*y,*z,*vx,*vy,*vz,Rout,z_inner,z_outer,&td);
        if(surf==0 || !(td>0.0) || !isfinite(td)){if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_NONPOSITIVE_T;return 0;}
        float t=(float)td;*x+=t*(*vx);*y+=t*(*vy);*z+=t*(*vz);*acrylic_path+=t;
        if(surf==1){
            *z=z_inner;int topo=bottom_disc_topology_at_top(*x,*y,Rin,Rout);
            if(topo==2){
                // Same acrylic material across the disc/sidewall seam: no Fresnel and no RNG.
                *z=nextafterf(z_inner,v41_pos_inf_f());return 3;
            }
            if(topo!=1){if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_TOPOLOGY_INVALID;return 0;}
            ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,-1.0f),0.0f),1.0f);tir=0;R=fresnel_R(ci,n_acrylic,n_water,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,n_acrylic,n_water,&a,&b,&c)){*vx=a;*vy=b;*vz=c;*z=z_inner+EPS_F;return 1;}
            reflect3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*inner_reflections)++;if(tir)(*bottom_tir)++;*z=z_inner-EPS_F;
        }else if(surf==2){
            *z=z_outer;ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,1.0f),0.0f),1.0f);tir=0;R=fresnel_R(ci,n_acrylic,n_bottom_external,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_acrylic,n_bottom_external,&a,&b,&c)){*vx=a;*vy=b;*vz=c;*z=z_outer-EPS_F;return 2;}
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*outer_reflections)++;if(tir)(*bottom_tir)++;*z=z_outer+EPS_F;
        }else if(surf==3){
            // Finite radial edge of the bottom disc: acrylic -> configured external medium.
            float nx,ny;radial_normal(*x,*y,&nx,&ny);ci=fminf(fmaxf(dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);tir=0;R=fresnel_R(ci,n_acrylic,n_side_external,&tir);
            if(!tir && rnd_uniform(state)>=R && refract3(*vx,*vy,*vz,-nx,-ny,0.0f,n_acrylic,n_side_external,&a,&b,&c)){*vx=a;*vy=b;*vz=c;*x+=EPS_F*nx;*y+=EPS_F*ny;return 2;}
            reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*outer_reflections)++;if(tir)(*bottom_tir)++;*x-=EPS_F*nx;*y-=EPS_F*ny;
        }else{
            if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_TOPOLOGY_INVALID;return 0;
        }
        reflections++;if(reflections>max_bounces){if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_BOUNCE_LIMIT;if(out_bounces)*out_bounces=reflections;return 0;}
    }
    if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_BOUNCE_LIMIT;return 0;
}
__device__ int bottom_acrylic_disc(
    unsigned int* state,float n_water,float n_acrylic,float n_bottom_external,float n_side_external,float Rin,float Rout,float z_inner,float thickness,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,int* inner_reflections,int* outer_reflections,int* bottom_tir){
    int fs=PROD_SUB_NONE,bb=0;return bottom_acrylic_disc_ex(state,n_water,n_acrylic,n_bottom_external,n_side_external,Rin,Rout,z_inner,thickness,max_bounces,x,y,z,vx,vy,vz,acrylic_path,inner_reflections,outer_reflections,bottom_tir,&fs,&bb);
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

// Inherited orthokinetic compact-aggregation population balance (disabled in V24.20).
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


// Inherited coupled orthokinetic aggregation + shear-fragmentation population balance (disabled in V24.20).
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
        double sg=PI_D*radius[i]*radius[i],ss=fmax(qsca[i],0.0)*sg,se=fmax(qext[i],0.0)*sg;
        double swv=0.0;
        if(event_model==3) swv=wave_scale*fmax(qsca[i]-1.0,0.0)*sg;
        else if(event_model==4 || event_model==5) swv=fmax(ss-sg,0.0);
        sigma_geom[i]=sg;sigma_sca[i]=ss;sigma_ext[i]=se;sigma_wave[i]=swv;
        mu_geom_bin[i]=nd*sg;mu_sca_bin[i]=nd*ss;mu_ext_bin[i]=nd*se;mu_wave_bin[i]=nd*swv;
    }
    double mg=0.0,ms=0.0,me=0.0,mw=0.0;
    for(int i=0;i<n_particles;++i){mg+=mu_geom_bin[i];ms+=mu_sca_bin[i];me+=mu_ext_bin[i];mw+=mu_wave_bin[i];}
    double mt=0.0;
    for(int i=0;i<n_particles;++i){
        double mb;
        if(event_model==0)mb=mu_geom_bin[i];
        else if(event_model==1)mb=mu_sca_bin[i];
        else if(event_model==2)mb=mu_ext_bin[i];
        else if(event_model==3)mb=mu_geom_bin[i]+mu_wave_bin[i];
        else mb=mu_sca_bin[i]; // V24.29 exact-Mie total event rate.
        mu_event_bin[i]=mb;mt+=mb;
        if(event_model==0)sigma_event[i]=sigma_geom[i];
        else if(event_model==1)sigma_event[i]=sigma_sca[i];
        else if(event_model==2)sigma_event[i]=sigma_ext[i];
        else if(event_model==3)sigma_event[i]=sigma_geom[i]+sigma_wave[i];
        else sigma_event[i]=sigma_sca[i];
    }
    // For V24.29 the geometric transport coefficient is the exact-rate remainder
    // mu_sca-mu_wave = nd*min(sigma_sca,sigma_geom).  The stored mu_geom_bin remains
    // the physical projected-area coefficient for diagnostics/CSA accounting.
    double mtrace_geom=((event_model==4)||(event_model==5))?fmax(mt-mw,0.0):((event_model==3)?mg:mt);
    double mtrace_wave=((event_model==3)||(event_model==4)||(event_model==5))?mw:0.0;
    double cg=0.0,cw=0.0,ct=0.0;
    for(int i=0;i<n_particles;++i){
        double mgi=((event_model==4)||(event_model==5))?fmax(mu_event_bin[i]-mu_wave_bin[i],0.0):mu_geom_bin[i];
        double pg=((event_model==3)||(event_model==4)||(event_model==5))?(mtrace_geom>0.0?mgi/mtrace_geom:0.0):(mt>0.0?mu_event_bin[i]/mt:0.0);
        double pw=(((event_model==3)||(event_model==4)||(event_model==5))&&mtrace_wave>0.0?mu_wave_bin[i]/mtrace_wave:0.0),pt=(mt>0.0?mu_event_bin[i]/mt:0.0);
        geometric_event_weight[i]=pg;wave_event_weight[i]=pw;particle_event_weight[i]=pt;
        cg+=pg;cw+=pw;ct+=pt;geometric_cdf[i]=cg;wave_cdf[i]=cw;particle_event_cdf[i]=ct;
    }
    if(n_particles>0){if(cg>0.0)geometric_cdf[n_particles-1]=1.0;if(cw>0.0)wave_cdf[n_particles-1]=1.0;if(ct>0.0)particle_event_cdf[n_particles-1]=1.0;}
    // [trace geometric channel, trace wave channel, physical geom, exact sca, ext, wave, total event, wave coherent fraction]
    mu_scalars[0]=mtrace_geom;mu_scalars[1]=mtrace_wave;
    mu_scalars[2]=mg;mu_scalars[3]=ms;mu_scalars[4]=me;mu_scalars[5]=mtrace_wave;mu_scalars[6]=mt;
    double cwave=1.0;
    if((event_model==3 || event_model==4 || event_model==5) && wave_rms_height_m>0.0 && wavelength>0.0 && n_medium>0.0){
        double a=4.0*PI_D*n_medium*wave_rms_height_m/wavelength;
        cwave=exp(-a*a);
        if(cwave<0.0)cwave=0.0;if(cwave>1.0)cwave=1.0;
    }
    mu_scalars[7]=cwave;
}


// Generate the V24.20 drift-diffusion field entirely on CUDA.  One thread is
// intentional: there are only 37/38 PSD bins and this deterministic startup
// integration is negligible beside ray tracing while avoiding reduction races.
__device__ double schiller_naumann_velocity(const double v0,const double d,const double rho_w,const double mu,double* out_Re,double* out_drag){
    if(!(v0>0.0)){if(out_Re)*out_Re=0.0;if(out_drag)*out_drag=1.0;return 0.0;}
    double v=v0,K=rho_w*d/mu;
    for(int it=0;it<24;++it){
        double Re=fmax(K*v,0.0);double corr=0.15*pow(Re,0.687);
        double f=v*(1.0+corr)-v0;double df=1.0+1.687*corr;
        double vn=fmax(v-f/df,0.0);
        if(fabs(vn-v)<=2e-14*fmax(1.0,fabs(v))){v=vn;break;}v=vn;
    }
    double Re=fmax(K*v,0.0),drag=1.0+0.15*pow(Re,0.687);
    if(out_Re)*out_Re=Re;if(out_drag)*out_drag=drag;return v;
}

__global__ void sediment_transport_precompute_kernel(
    const double* diameter,const int n_particles,const int TRANSPORT_MODEL,
    const double particle_density,const double water_density,const double water_dynamic_viscosity,
    const double gravity,const double eddy_diffusivity,const double R,const double ZMIN,const double ZMEAN,
    const int FREE_SURFACE_MODEL,const double VORTEX_DELTA_H,const double VORTEX_CORE_RADIUS,const int NQUAD,const int N_RADIAL,
    const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    double* RESPONSE_TIME,double* SETTLING_VELOCITY,double* SETTLING_RE,double* SETTLING_DRAG,
    double* SED_A,double* SED_B,double* SED_NORM_SCALED,double* SED_MAX_SCALE,
    double* RADIAL_MAX_SLIP,double* RADIAL_MAX_RE,double* RADIAL_CENTER_LOG,double* SED_RADIAL_LOG,double* SED_RADIAL_WR,
    double* SCALE_SENSOR_CENTER,double* SCALE_SENSOR_WALL,double* SCALE_BOTTOM_CENTER,double* SCALE_BOTTOM_WALL,
    double* MAJORANTS,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(TRANSPORT_MODEL==0){
        double mg=0.0,mw=0.0;
        for(int i=0;i<n_particles;++i){
            RESPONSE_TIME[i]=0.0;SETTLING_VELOCITY[i]=0.0;SETTLING_RE[i]=0.0;SETTLING_DRAG[i]=1.0;
            SED_A[i]=0.0;SED_B[i]=0.0;SED_NORM_SCALED[i]=1.0;SED_MAX_SCALE[i]=1.0;
            RADIAL_MAX_SLIP[i]=0.0;RADIAL_MAX_RE[i]=0.0;RADIAL_CENTER_LOG[i]=0.0;
            SCALE_SENSOR_CENTER[i]=1.0;SCALE_SENSOR_WALL[i]=1.0;SCALE_BOTTOM_CENTER[i]=1.0;SCALE_BOTTOM_WALL[i]=1.0;
            for(int j=0;j<N_RADIAL;++j){SED_RADIAL_LOG[(long long)i*N_RADIAL+j]=0.0;SED_RADIAL_WR[(long long)i*N_RADIAL+j]=0.0;}
            mg+=TRACE_GEOM_BIN[i];mw+=TRACE_WAVE_BIN[i];
        }
        MAJORANTS[0]=mg;MAJORANTS[1]=mw;MAJORANTS[2]=mg+mw;MAJORANTS[3]=0.0;return;
    }
    if((TRANSPORT_MODEL!=1 && TRANSPORT_MODEL!=2 && TRANSPORT_MODEL!=3) || !(particle_density>water_density) || !(water_density>0.0) || !(water_dynamic_viscosity>0.0) || !(gravity>0.0) || !(eddy_diffusivity>0.0) || !(R>0.0) || !(VORTEX_CORE_RADIUS>0.0) || NQUAD<64 || N_RADIAL<65){atomicExch(error_flag,31);return;}
    if(VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL!=2){atomicExch(error_flag,32);return;}
    double R2=R*R,a2=VORTEX_CORE_RADIUS*VORTEX_CORE_RADIUS;
    double Hinf=(VORTEX_DELTA_H>0.0)?VORTEX_DELTA_H*(a2+R2)/R2:0.0;
    double hbar=(VORTEX_DELTA_H>0.0)?Hinf*(1.0-(a2/R2)*log1p(R2/a2)):0.0;
    double omega2=(VORTEX_DELTA_H>0.0)?2.0*gravity*Hinf/a2:0.0;
    double fR=R2/(a2+R2),Hmean=ZMEAN-ZMIN,dr=R/(double)NQUAD,drad=R/(double)(N_RADIAL-1);
    if(!(Hmean>0.0)){atomicExch(error_flag,33);return;}
    double major_g=0.0,major_w=0.0,maxAeq=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i];
        double tau=particle_density*d*d/(18.0*water_dynamic_viscosity);
        double ws0=(particle_density-water_density)*gravity*d*d/(18.0*water_dynamic_viscosity);
        double sre=0.0,sdrag=1.0;
        double ws=(TRANSPORT_MODEL==2 || TRANSPORT_MODEL==3)?schiller_naumann_velocity(ws0,d,water_density,water_dynamic_viscosity,&sre,&sdrag):ws0;
        if(TRANSPORT_MODEL==1){sre=water_density*d*ws/water_dynamic_viscosity;sdrag=1.0;}
        RESPONSE_TIME[i]=tau;SETTLING_VELOCITY[i]=ws;SETTLING_RE[i]=sre;SETTLING_DRAG[i]=sdrag;
        double B=ws/eddy_diffusivity;SED_B[i]=B;
        double maxwr=0.0,maxre=0.0;
        long long base=(long long)i*(long long)N_RADIAL;
        if(TRANSPORT_MODEL==1){
            double A=tau*omega2*a2/(2.0*eddy_diffusivity);SED_A[i]=A;if(A>maxAeq)maxAeq=A;
            for(int j=0;j<N_RADIAL;++j){double r=(double)j*drad,rr2=r*r,f=rr2/(a2+rr2);SED_RADIAL_LOG[base+j]=A*(f-fR);}
            for(int j=0;j<N_RADIAL;++j){
                double r=(double)j*drad;double acc=(r>0.0)?omega2*r/pow(1.0+r*r/a2,2.0):0.0;double wr=tau*acc;double re=water_density*d*wr/water_dynamic_viscosity;
                SED_RADIAL_WR[base+j]=wr;if(wr>maxwr)maxwr=wr;if(re>maxre)maxre=re;
            }
        } else {
            // First pass: integrate the corrected radial slip from centre to wall.
            double total=0.0,prev=0.0;
            for(int j=0;j<N_RADIAL;++j){
                double r=(double)j*drad;double acc=(r>0.0)?omega2*r/pow(1.0+r*r/a2,2.0):0.0;double w0=tau*acc,re=0.0,drag=1.0;
                double wr=schiller_naumann_velocity(w0,d,water_density,water_dynamic_viscosity,&re,&drag);
                SED_RADIAL_WR[base+j]=wr;if(j>0)total+=0.5*(prev+wr)*drad;prev=wr;if(wr>maxwr)maxwr=wr;if(re>maxre)maxre=re;
            }
            // Second pass: store psi(r)=[integral_0^r w ds - integral_0^R w ds]/Dt.
            double cum=0.0;prev=0.0;
            for(int j=0;j<N_RADIAL;++j){
                double wr=SED_RADIAL_WR[base+j];
                if(j>0)cum+=0.5*(prev+wr)*drad;prev=wr;
                double psi=(cum-total)/eddy_diffusivity;if(psi>0.0)psi=0.0;SED_RADIAL_LOG[base+j]=psi;
            }
            SED_RADIAL_LOG[base+N_RADIAL-1]=0.0;
            double Aeq=(fR>0.0)?-SED_RADIAL_LOG[base]/fR:0.0;SED_A[i]=Aeq;if(Aeq>maxAeq)maxAeq=Aeq;
        }
        RADIAL_MAX_SLIP[i]=maxwr;RADIAL_MAX_RE[i]=maxre;RADIAL_CENTER_LOG[i]=SED_RADIAL_LOG[base];
        double integ=0.0;
        for(int j=0;j<NQUAD;++j){
            double r=((double)j+0.5)*dr,rr2=r*r;
            double zsurf=ZMEAN-hbar+Hinf*rr2/(a2+rr2);double H=fmax(zsurf-ZMIN,0.0);
            double radial_log;
            if(TRANSPORT_MODEL==1){double f=rr2/(a2+rr2);radial_log=SED_A[i]*(f-fR);}else{
                double u=r/R*(double)(N_RADIAL-1);int k=(int)floor(u);if(k<0)k=0;if(k>N_RADIAL-2)k=N_RADIAL-2;double t=u-(double)k;
                radial_log=(1.0-t)*SED_RADIAL_LOG[base+k]+t*SED_RADIAL_LOG[base+k+1];if(radial_log>0.0)radial_log=0.0;
            }
            double radial=exp(radial_log);double iz=(fabs(B)<1e-14)?radial*H:radial*(-expm1(-B*H))/B;integ+=r*iz;
        }
        double norm=(2.0/(R2*Hmean))*integ*dr;
        if(!(norm>0.0) || !isfinite(norm)){atomicExch(error_flag,34);return;}
        double mx=1.0/norm;SED_NORM_SCALED[i]=norm;SED_MAX_SCALE[i]=mx;
        double rwall=0.95*R,uwall=rwall/R*(double)(N_RADIAL-1);int kw=(int)floor(uwall);if(kw>N_RADIAL-2)kw=N_RADIAL-2;double tw=uwall-(double)kw;
        double log0=(TRANSPORT_MODEL==1)?SED_A[i]*(0.0-fR):SED_RADIAL_LOG[base];
        double rrw2=rwall*rwall,fw=rrw2/(a2+rrw2);double logw=(TRANSPORT_MODEL==1)?SED_A[i]*(fw-fR):((1.0-tw)*SED_RADIAL_LOG[base+kw]+tw*SED_RADIAL_LOG[base+kw+1]);
        double hs=fmax(0.0-ZMIN,0.0);
        SCALE_SENSOR_CENTER[i]=exp(log0-B*hs)/norm;SCALE_SENSOR_WALL[i]=exp(logw-B*hs)/norm;
        SCALE_BOTTOM_CENTER[i]=exp(log0)/norm;SCALE_BOTTOM_WALL[i]=exp(logw)/norm;
        major_g+=TRACE_GEOM_BIN[i]*mx;major_w+=TRACE_WAVE_BIN[i]*mx;
    }
    MAJORANTS[0]=major_g;MAJORANTS[1]=major_w;MAJORANTS[2]=major_g+major_w;MAJORANTS[3]=maxAeq;
    if(!(MAJORANTS[2]>=0.0) || !isfinite(MAJORANTS[2]))atomicExch(error_flag,35);
}

// ---------------- V24.20 explicit secondary meridional circulation ----------------
// The fluid streamfunction is
//   psi = -A r^2(1-r^2/R^2)^2 eta(1-eta),  eta=(z-ZMIN)/(zsurf(r)-ZMIN)
// with A=2*UPEAK.  UPEAK is chi times the maximum Scully azimuthal speed.
// This gives outward bottom flow, outer-annulus upflow, inward surface flow and
// axial downflow while making psi constant on every continuous vessel boundary.
__device__ void meridional_velocity_cuda(
    const double r,const double z,const double ZMIN,const double ZMEAN,const double R,
    const double DELTA_H,const double CORE_R,const double UPEAK,double* ur,double* uz){
    *ur=0.0;*uz=0.0;
    if(!(UPEAK>0.0) || !(R>0.0) || !(CORE_R>0.0) || !(DELTA_H>0.0))return;
    double rr=fmin(fmax(r,0.0),R),a2=CORE_R*CORE_R,R2=R*R,rr2=rr*rr;
    double Hinf=DELTA_H*(a2+R2)/R2;
    double hbar=Hinf*(1.0-(a2/R2)*log1p(R2/a2));
    double zs=ZMEAN-hbar+Hinf*rr2/(a2+rr2),H=zs-ZMIN;
    if(!(H>0.0))return;
    double eta=(z-ZMIN)/H;if(eta<0.0)eta=0.0;if(eta>1.0)eta=1.0;
    double s=rr2/R2,f=eta*(1.0-eta),fp=1.0-2.0*eta;
    double hp=2.0*Hinf*a2*rr/((a2+rr2)*(a2+rr2));
    double Apsi=2.0*UPEAK,om=1.0-s;
    *ur=Apsi*rr*om*om*fp/H;
    *uz=-Apsi*(2.0*om*(1.0-3.0*s)*f-rr*om*om*fp*eta*hp/H);
}

__device__ double radial_slip_lookup_cuda(const int pidx,const double r,const double R,
    const double* WR,const int N_RADIAL){
    if(N_RADIAL<2 || !(R>0.0))return 0.0;
    double q=fmin(fmax(r/R,0.0),1.0)*(double)(N_RADIAL-1);
    int j=(int)floor(q);double t=q-(double)j;if(j>=N_RADIAL-1){j=N_RADIAL-2;t=1.0;}
    long long b=(long long)pidx*(long long)N_RADIAL;
    return (1.0-t)*WR[b+j]+t*WR[b+j+1];
}

__device__ int meridional_physical_fluid(const double r,const double z,const double ZMIN,const double ZMEAN,
    const double R,const double DELTA_H,const double CORE_R){
    if(z<ZMIN)return 0;
    double a2=CORE_R*CORE_R,R2=R*R,rr=fmin(fmax(r,0.0),R),rr2=rr*rr;
    double Hinf=DELTA_H*(a2+R2)/R2;
    double hbar=Hinf*(1.0-(a2/R2)*log1p(R2/a2));
    double zs=ZMEAN-hbar+Hinf*rr2/(a2+rr2);
    return z<zs;
}

__global__ void meridional_active_mask_kernel(
    const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const double DELTA_H,const double CORE_R,unsigned char* ACTIVE){
    int t=blockDim.x*blockIdx.x+threadIdx.x,plane=NR*NZ;if(t>=plane)return;
    int i=t/NZ,k=t-i*NZ;double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double r=((double)i+0.5)*dr,z=ZMIN+((double)k+0.5)*dz;
    ACTIVE[t]=(unsigned char)(meridional_physical_fluid(r,z,ZMIN,ZMEAN,R,DELTA_H,CORE_R)?1:0);
}

__global__ void meridional_grid_initialize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double CORE_R,
    const double* SED_A,const double* SED_B,const double* SED_NORM,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const unsigned char* ACTIVE,double* GRID){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    if(!ACTIVE[c]){GRID[t]=0.0;return;}
    double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    float x=(float)(((double)i+0.5)*dr),z=(float)(ZMIN+((double)k+0.5)*dz);
    GRID[t]=sediment_local_scale(2,p,SED_A,SED_B,SED_NORM,SED_RADIAL_LOG,N_RADIAL,GRID,NR,NZ,(float)ZGRIDMAX,x,0.0f,z,(float)R,(float)ZMIN,(float)CORE_R);
}

__global__ void meridional_build_coefficients_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const double DELTA_H,const double CORE_R,const double UPEAK,const double EDDY_DIFFUSIVITY,
    const double* SETTLING_VELOCITY,const double* RADIAL_WR,const int N_RADIAL,const unsigned char* ACTIVE,
    double* AP,double* AE,double* AW,double* AN,double* AS){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    AP[t]=AE[t]=AW[t]=AN[t]=AS[t]=0.0;if(!ACTIVE[c])return;
    double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double rc=((double)i+0.5)*dr,zc=ZMIN+((double)k+0.5)*dz;
    double rw=(double)i*dr,re=(double)(i+1)*dr;
    double Az=0.5*(re*re-rw*rw),ap=0.0,ur=0.0,uz=0.0,V=0.0,F=0.0,diff=0.0;
    // East radial face.
    if(i+1<NR && ACTIVE[(i+1)*NZ+k] && meridional_physical_fluid(re,zc,ZMIN,ZMEAN,R,DELTA_H,CORE_R)){
        meridional_velocity_cuda(re,zc,ZMIN,ZMEAN,R,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=ur+radial_slip_lookup_cuda(p,re,R,RADIAL_WR,N_RADIAL);F=V*re*dz;diff=EDDY_DIFFUSIVITY*re*dz/dr;
        AE[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // West radial face; outward normal is -r.
    if(i>0 && ACTIVE[(i-1)*NZ+k] && meridional_physical_fluid(rw,zc,ZMIN,ZMEAN,R,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rw,zc,ZMIN,ZMEAN,R,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=ur+radial_slip_lookup_cuda(p,rw,R,RADIAL_WR,N_RADIAL);F=-V*rw*dz;diff=EDDY_DIFFUSIVITY*rw*dz/dr;
        AW[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // Top horizontal face.
    double zf=zc+0.5*dz;
    if(k+1<NZ && ACTIVE[i*NZ+k+1] && meridional_physical_fluid(rc,zf,ZMIN,ZMEAN,R,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rc,zf,ZMIN,ZMEAN,R,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=uz-SETTLING_VELOCITY[p];F=V*Az;diff=EDDY_DIFFUSIVITY*Az/dz;
        AN[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // Bottom horizontal face; outward normal is -z.
    zf=zc-0.5*dz;
    if(k>0 && ACTIVE[i*NZ+k-1] && meridional_physical_fluid(rc,zf,ZMIN,ZMEAN,R,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rc,zf,ZMIN,ZMEAN,R,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=uz-SETTLING_VELOCITY[p];F=-V*Az;diff=EDDY_DIFFUSIVITY*Az/dz;
        AS[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    AP[t]=ap;
}

__global__ void meridional_relax_kernel(
    const int n_particles,const int NR,const int NZ,const int parity,const double OMEGA,
    const unsigned char* ACTIVE,const double* AP,const double* AE,const double* AW,const double* AN,const double* AS,double* GRID){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    if(!ACTIVE[c] || ((i+k)&1)!=parity || !(AP[t]>0.0))return;
    long long b=(long long)p*plane;double num=0.0;
    if(i+1<NR)num+=AE[t]*GRID[b+(long long)(i+1)*NZ+k];
    if(i>0)num+=AW[t]*GRID[b+(long long)(i-1)*NZ+k];
    if(k+1<NZ)num+=AN[t]*GRID[b+(long long)i*NZ+k+1];
    if(k>0)num+=AS[t]*GRID[b+(long long)i*NZ+k-1];
    double target=num/AP[t],old=GRID[t],nv=(1.0-OMEGA)*old+OMEGA*target;
    if(!isfinite(nv) || nv<1e-14)nv=1e-14;GRID[t]=nv;
}

__global__ void meridional_normalize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const unsigned char* ACTIVE,double* GRID,int* error_flag){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;
    long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double mass=0.0;
    for(int i=0;i<NR;++i){double rw=(double)i*dr,re=(double)(i+1)*dr,vol=0.5*(re*re-rw*rw)*dz;
        for(int k=0;k<NZ;++k){int c=i*NZ+k;if(ACTIVE[c])mass+=GRID[b+c]*vol;}}
    double target=0.5*R*R*(ZMEAN-ZMIN);if(!(mass>0.0) || !(target>0.0) || !isfinite(mass)){atomicExch(error_flag,41);return;}
    double scale=target/mass;
    for(int i=0;i<NR;++i)for(int k=0;k<NZ;++k){int c=i*NZ+k;if(ACTIVE[c])GRID[b+c]*=scale;}
}

__global__ void meridional_residual_kernel(
    const int n_particles,const int NR,const int NZ,const unsigned char* ACTIVE,
    const double* AP,const double* AE,const double* AW,const double* AN,const double* AS,const double* GRID,double* RESIDUAL){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;double mr=0.0;
    for(int i=0;i<NR;++i)for(int k=0;k<NZ;++k){int c=i*NZ+k;long long t=b+c;if(!ACTIVE[c] || !(AP[t]>0.0))continue;
        double num=0.0;if(i+1<NR)num+=AE[t]*GRID[b+(long long)(i+1)*NZ+k];if(i>0)num+=AW[t]*GRID[b+(long long)(i-1)*NZ+k];
        if(k+1<NZ)num+=AN[t]*GRID[b+(long long)i*NZ+k+1];if(k>0)num+=AS[t]*GRID[b+(long long)i*NZ+k-1];
        double den=fmax(fabs(AP[t]*GRID[t]),1e-30),q=fabs(AP[t]*GRID[t]-num)/den;if(q>mr)mr=q;}
    RESIDUAL[p]=mr;
}

__global__ void meridional_extend_grid_kernel(
    const int n_particles,const int NR,const int NZ,const unsigned char* ACTIVE,double* GRID){
    int t=blockDim.x*blockIdx.x+threadIdx.x,total=n_particles*NR;if(t>=total)return;int p=t/NR,i=t-p*NR;
    long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane+(long long)i*NZ;int top=-1;
    for(int k=NZ-1;k>=0;--k){if(ACTIVE[i*NZ+k]){top=k;break;}}
    if(top<0){for(int k=0;k<NZ;++k)GRID[b+k]=1.0;return;}double q=GRID[b+top];for(int k=top+1;k<NZ;++k)GRID[b+k]=q;
}

__global__ void meridional_finalize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,
    const unsigned char* ACTIVE,const double* GRID,double* SED_NORM,double* SED_MAX,double* SED_MIN,
    double* SCALE_SENSOR_CENTER,double* SCALE_SENSOR_WALL,double* SCALE_BOTTOM_CENTER,double* SCALE_BOTTOM_WALL){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;
    double mx=0.0,mn=1e300;for(int c=0;c<NR*NZ;++c)if(ACTIVE[c]){double q=GRID[b+c];if(q>mx)mx=q;if(q<mn)mn=q;}
    if(!(mx>0.0)){mx=1.0;mn=1.0;}SED_NORM[p]=1.0;SED_MAX[p]=mx;SED_MIN[p]=mn;
    SCALE_SENSOR_CENTER[p]=sediment_grid_sample(p,GRID,NR,NZ,0.0f,0.0f,0.0f,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_SENSOR_WALL[p]=sediment_grid_sample(p,GRID,NR,NZ,(float)(0.95*R),0.0f,0.0f,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_BOTTOM_CENTER[p]=sediment_grid_sample(p,GRID,NR,NZ,0.0f,0.0f,(float)ZMIN,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_BOTTOM_WALL[p]=sediment_grid_sample(p,GRID,NR,NZ,(float)(0.95*R),0.0f,(float)ZMIN,(float)R,(float)ZMIN,(float)ZGRIDMAX);
}

__global__ void meridional_majorants_kernel(const int n_particles,const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    const double* SED_MAX,double* MAJORANTS){
    if(blockIdx.x||threadIdx.x)return;double mg=0.0,mw=0.0;for(int i=0;i<n_particles;++i){mg+=TRACE_GEOM_BIN[i]*SED_MAX[i];mw+=TRACE_WAVE_BIN[i]*SED_MAX[i];}
    MAJORANTS[0]=mg;MAJORANTS[1]=mw;MAJORANTS[2]=mg+mw;
}

__global__ void meridional_velocity_test_kernel(const float* rz,const int n,const float ZMIN,const float ZMEAN,const float R,
    const float DELTA_H,const float CORE_R,const float UPEAK,float* ur,float* uz){
    int t=blockDim.x*blockIdx.x+threadIdx.x;if(t>=n)return;double a=0.0,b=0.0;meridional_velocity_cuda((double)rz[2*t],(double)rz[2*t+1],ZMIN,ZMEAN,R,DELTA_H,CORE_R,UPEAK,&a,&b);ur[t]=(float)a;uz[t]=(float)b;
}


__global__ void sediment_field_test_kernel(
    const float* xyz,const int n_points,const int pidx,const int TRANSPORT_MODEL,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float R,const float ZMIN,const float CORE_R,float* out_scale){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n_points)return;
    out_scale[tid]=(float)sediment_local_scale(TRANSPORT_MODEL,pidx,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,N_RADIAL,SED_GRID,GRID_NR,GRID_NZ,GRID_ZMAX,xyz[3*tid],xyz[3*tid+1],xyz[3*tid+2],R,ZMIN,CORE_R);
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


// V24.23 external air-gap stray-light cavity.
// The 0.5-mm annulus between the acrylic tube outer radius and the nylon
// sensor-ring inner radius is traced explicitly only for rays that fail the
// established direct detector-bore acceptance.  The production baseline
// (RING_FACE_SURFACE_MODEL==0) bypasses this code exactly.
//
// Surface model 1 treats the broad inner cylindrical nylon face between bore
// openings as a smooth opaque dielectric.  Air->nylon Fresnel reflection is
// sampled; transmission into the thick nylon ring is loss.  After a face
// reflection the ray encounters the acrylic outer cylinder.  Air->acrylic
// Fresnel reflection keeps it in the external cavity; transmission into
// acrylic leaves this direct external-stray branch and is conservatively
// terminated rather than being misclassified as apparatus-only stray light.
// No diffuse albedo or fitted BRDF parameter is introduced.

// V24.23 SFH213 packaged-photodiode angular response.
// TARDIIS uses the OSRAM/ams SFH213 5-mm clear-epoxy PIN photodiode.
// The manufacturer specifies a typical half angle phi_1/2 = 10 degrees and
// plots a relative directional response of approximately 0.10 at 20 degrees.
// We represent that published package-integrated response (lens + 1x1 mm chip)
// with the two-anchor generalized-Lorentz fit
//   S(phi) = 1 / [1 + (|phi|/10 deg)^p],  p = ln(9)/ln(2)
// which gives S(0)=1, S(10)=0.5 and S(20)=0.1 exactly.  No additional cos(phi)
// factor is applied because the datasheet directional curve already includes
// projected-area and package-lens effects.
__device__ float sfh213_relative_response_deg(float phi_deg){
    const float HALF_DEG=10.0f;
    const float P=3.169925001442312f; // ln(9)/ln(2)
    float a=fabsf(phi_deg);
    if(a>=90.0f)return 0.0f;
    float u=a/HALF_DEG;
    return 1.0f/(1.0f+powf(u,P));
}

__device__ float detector_incidence_angle_deg(float vx,float vy,float vz,int detector_id,int mirror){
    // NVRTC-portable invalid sentinel.  Do not rely on the host NAN macro: raw
    // CUDA source is compiled without a C math header by CuPy RawModule.
    if(detector_id<0||detector_id>=18)return -1.0f;
    if(mirror)vx=-vx;
    float th=(10.0f*(float)detector_id)*(PI_F/180.0f);
    float ux=sinf(th),uy=cosf(th);
    float c=vx*ux+vy*uy;
    c=fminf(fmaxf(c,-1.0f),1.0f);
    return acosf(c)*(180.0f/PI_F);
}

__device__ int hardware_throat_opening_id(float x,float y,float z,float vx,float vy,float vz,int mirror,
    float ring_in,float throat_r){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;float r2=throat_r*throat_r;
    for(int j=0;j<19;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1.0e-10f)continue;
        float pdot=x*ux+y*uy;float t=(ring_in-pdot)/den;if(t<=0.0f)continue;
        float qx=x+t*vx-ring_in*ux,qy=y+t*vy-ring_in*uy,qz=z+t*vz;
        if(qx*qx+qy*qy+qz*qz<=r2 && t<best_t){best_t=t;best=j;}
    }
    return best; // 0..17 detector throat, 18 source throat, -1 solid face
}

__device__ int detector_once_from_air(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,
    const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,
    const int detector_bore_surface_model,const float n_air,const float n_bore,const int max_bore_bounces,
    int* bore_bounces){
    *bore_bounces=0;
    if(detector_bore_surface_model==0)
        return hardware_detector_id(x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    return detector_collimator_path(state,x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r,
                                    detector_bore_surface_model,n_air,n_bore,max_bore_bounces,bore_bounces);
}

__device__ int ring_face_cavity_detector_path(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,int mirror,
    const float acrylic_outer_r,const float ring_in,const float throat_out,const float ring_out,
    const float throat_r,const float counterbore_r,
    const int ring_face_surface_model,const float n_air,const float n_ring,const float n_acrylic,const int max_ring_face_bounces,
    const int detector_bore_surface_model,const float n_bore,const int max_bore_bounces,
    int* bore_bounces,int* face_reflections){
    if(mirror){x=-x;vx=-vx;}
    *bore_bounces=0;*face_reflections=0;
    if(ring_face_surface_model==0)return -1;
    // The established direct path was already tested by the caller.  If this
    // leg enters any 4-mm throat but fails the complete bore acceptance, it is
    // absorbed by that bore rather than being allowed to reflect from solid face.
    if(hardware_throat_opening_id(x,y,z,vx,vy,vz,0,ring_in,throat_r)>=0)return -1;
    for(int b=0;b<max_ring_face_bounces;++b){
        float tr=cylinder_hit(x,y,vx,vy,ring_in);
        if(tr>=INF_F*0.5f)return -1;
        x+=tr*vx;y+=tr*vy;z+=tr*vz;
        float nx,ny;radial_normal(x,y,&nx,&ny);
        // Normal from nylon into the incident air cavity is -radial.
        float ci=fminf(fmaxf(vx*nx+vy*ny,0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,n_air,n_ring,&tir);
        if(rnd_uniform(state)>=R)return -1; // transmitted into bulk nylon -> loss
        float a,c,d;reflect3(vx,vy,vz,-nx,-ny,0.0f,&a,&c,&d);vx=a;vy=c;vz=d;(*face_reflections)++;
        x-=EPS_F*nx;y-=EPS_F*ny;

        float ta=cylinder_hit(x,y,vx,vy,acrylic_outer_r);
        if(ta>=INF_F*0.5f)return -1;
        x+=ta*vx;y+=ta*vy;z+=ta*vz;
        radial_normal(x,y,&nx,&ny);
        // At the acrylic outer surface, +radial points into incident air.
        ci=fminf(fmaxf(-dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        R=fresnel_R(ci,n_air,n_acrylic,&tir);
        if(rnd_uniform(state)>=R)return -1; // re-enters acrylic: not an external-cavity-only stray path
        reflect3(vx,vy,vz,nx,ny,0.0f,&a,&c,&d);vx=a;vy=c;vz=d;
        x+=EPS_F*nx;y+=EPS_F*ny;

        int db=0;
        int det=detector_once_from_air(state,x,y,z,vx,vy,vz,ring_in,throat_out,ring_out,throat_r,counterbore_r,
                                       detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&db);
        if(det>=0){*bore_bounces=db;return det;}
        // Detector/source throat interception without full acceptance is an
        // absorbing bore hit, not a reflection from the broad nylon face.
        if(hardware_throat_opening_id(x,y,z,vx,vy,vz,0,ring_in,throat_r)>=0)return -1;
    }
    return -1;
}

__global__ void score_detectors_kernel(
    const float* air_x,const float* air_y,const float* air_z,const float* air_vx,const float* air_vy,const float* air_vz,
    const int* status,const int* interaction_count,const int n,
    const float acrylic_outer_r,const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,const float legacy_accept_deg,
    const int detector_bore_surface_model,const float n_air,const float n_bore,const int max_bore_bounces,
    const int ring_face_surface_model,const float n_ring_face,const float n_acrylic,const int max_ring_face_bounces,
    const int photodiode_response_model,
    const unsigned int detector_seed,const unsigned int ray_offset,
    int* hardware_native,int* hardware_mirror,int* hardware_native_direct,int* hardware_mirror_direct,
    int* hardware_native_stray,int* hardware_mirror_stray,int* hardware_native_face_stray,int* hardware_mirror_face_stray,
    float* photodiode_native,float* photodiode_mirror,float* incidence_native_sum,float* incidence_mirror_sum,
    int* legacy_native,int* legacy_mirror,int* native_id,int* mirror_id,int* native_bounces,int* mirror_bounces,int* native_face_bounces,int* mirror_face_bounces){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;
    native_id[tid]=-1;mirror_id[tid]=-1;native_bounces[tid]=0;mirror_bounces[tid]=0;native_face_bounces[tid]=0;mirror_face_bounces[tid]=0;
    if(status[tid]!=1)return;
    float x=air_x[tid],y=air_y[tid],z=air_z[tid],vx=air_vx[tid],vy=air_vy[tid],vz=air_vz[tid];
    if(!(isfinite(x)&&isfinite(y)&&isfinite(z)&&isfinite(vx)&&isfinite(vy)&&isfinite(vz)))return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    unsigned int stateN=detector_seed+gid*2246822519u+3266489917u;
    unsigned int stateM=detector_seed+gid*3266489917u+668265263u;
    int nb=0,mb=0,nfb=0,mfb=0,ni=-1,mi=-1;
    // Exact V24.21 direct branch first.  Therefore RING_FACE_SURFACE_MODEL=0 is
    // a strict regression path and enabling the face cavity can only add paths
    // after the established direct detector acceptance has failed.
    if(detector_bore_surface_model==0){
        ni=hardware_detector_id(x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r);
        mi=hardware_detector_id(x,y,z,vx,vy,vz,1,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    }else{
        ni=detector_collimator_path(&stateN,x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r,detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&nb);
        mi=detector_collimator_path(&stateM,x,y,z,vx,vy,vz,1,ring_in,throat_out,ring_out,throat_r,counterbore_r,detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&mb);
    }
    if(ni<0 && ring_face_surface_model!=0)
        ni=ring_face_cavity_detector_path(&stateN,x,y,z,vx,vy,vz,0,acrylic_outer_r,ring_in,throat_out,ring_out,throat_r,counterbore_r,
            ring_face_surface_model,n_air,n_ring_face,n_acrylic,max_ring_face_bounces,detector_bore_surface_model,n_bore,max_bore_bounces,&nb,&nfb);
    if(mi<0 && ring_face_surface_model!=0)
        mi=ring_face_cavity_detector_path(&stateM,x,y,z,vx,vy,vz,1,acrylic_outer_r,ring_in,throat_out,ring_out,throat_r,counterbore_r,
            ring_face_surface_model,n_air,n_ring_face,n_acrylic,max_ring_face_bounces,detector_bore_surface_model,n_bore,max_bore_bounces,&mb,&mfb);
    native_id[tid]=ni;mirror_id[tid]=mi;native_bounces[tid]=nb;mirror_bounces[tid]=mb;native_face_bounces[tid]=nfb;mirror_face_bounces[tid]=mfb;
    if(ni>=0){
        atomicAdd(&hardware_native[ni],1);if(nb>0||nfb>0)atomicAdd(&hardware_native_stray[ni],1);else atomicAdd(&hardware_native_direct[ni],1);if(nfb>0)atomicAdd(&hardware_native_face_stray[ni],1);
        float phi=detector_incidence_angle_deg(vx,vy,vz,ni,0);
        if(phi>=0.0f && isfinite(phi)){
            float resp=(photodiode_response_model==0)?1.0f:sfh213_relative_response_deg(phi);
            atomicAdd(&photodiode_native[ni],resp);atomicAdd(&incidence_native_sum[ni],phi);
        }
    }
    if(mi>=0){
        atomicAdd(&hardware_mirror[mi],1);if(mb>0||mfb>0)atomicAdd(&hardware_mirror_stray[mi],1);else atomicAdd(&hardware_mirror_direct[mi],1);if(mfb>0)atomicAdd(&hardware_mirror_face_stray[mi],1);
        float phi=detector_incidence_angle_deg(vx,vy,vz,mi,1);
        if(phi>=0.0f && isfinite(phi)){
            float resp=(photodiode_response_model==0)?1.0f:sfh213_relative_response_deg(phi);
            atomicAdd(&photodiode_mirror[mi],resp);atomicAdd(&incidence_mirror_sum[mi],phi);
        }
    }
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


// ================= V24.36 Qsca>=1 MIE-TWO-MOMENT ANGULAR CONSTRAINT BEGIN =================
__device__ void v2436_apply_moment_constraint(
    const float vinx,const float viny,const float vinz,
    float* vx,float* vy,float* vz,const float A,const float B){
    float c=fminf(fmaxf(dot3(vinx,viny,vinz,*vx,*vy,*vz),-1.0f),1.0f);
    float d=fminf(fmaxf(1.0f-c,0.0f),2.0f);
    float dm=fminf(fmaxf(A*d+B*d*d,0.0f),2.0f);
    float cp=1.0f-dm;
    float px=*vx-c*vinx,py=*vy-c*viny,pz=*vz-c*vinz;
    float pn=sqrtf(fmaxf(px*px+py*py+pz*pz,0.0f));
    if(pn>1.0e-8f){px/=pn;py/=pn;pz/=pn;}
    else{
        float e1x,e1y,e1z,e2x,e2y,e2z;
        perpendicular_basis(vinx,viny,vinz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
        px=e1x;py=e1y;pz=e1z;
    }
    float sp=sqrtf(fmaxf(1.0f-cp*cp,0.0f));
    *vx=cp*vinx+sp*px;*vy=cp*viny+sp*py;*vz=cp*vinz+sp*pz;normalize3(vx,vy,vz);
}
// ================= V24.36 Qsca>=1 MIE-TWO-MOMENT ANGULAR CONSTRAINT END =================

__global__ void trace_kernel(
    const float MAX_ITERATIONS,const double* MU_CHANNELS,
    const float N_WATER,const float N_ACRYLIC,const float N_AIR,const float PARTICLE_WAVELENGTH,const float PARTICLE_SURFACE_ROUGH_ALPHA,const float WAVE_SPHEROID_ASPECT_RATIO,
    const int WAVE_SPHEROID_ORIENTATION_MODEL,const float WAVE_SPHEROID_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const int MAX_CELL_BOUNCES,
    const float RIN,const float ROUT,const float ZMIN,const float ZMAX,const int FREE_SURFACE_MODEL,const float VORTEX_DELTA_H,const float VORTEX_CORE_RADIUS,const float BOTTOM_DISC_THICKNESS,const float BOTTOM_EXTERNAL_N,const int MAX_AXIAL_BOUNCES,
    const float RING_IN_R,const float THROAT_OUT_R,const float RING_OUT_R,const float SOURCE_R,
    const float THROAT_R,const float COUNTERBORE_R,const float BEAM_SIGMA,
    const int VIS_SIZE,const float SOURCE_ALPHA1,const float SOURCE_ALPHA2,const int SOURCE_ANGULAR_MODEL,
    const int SOURCE_BORE_SURFACE_MODEL,const float BORE_N,const int MAX_BORE_BOUNCES,
    const int N_rays,
    const double* geometric_cdf,const double* wave_cdf,const double* mie_phase_cdf,const double* mie_theta,const int mie_ntheta,
    const double* particle_n,const double* particle_k,const double* particle_radius,const int n_particles,
    const double* MOMENT_A,const double* MOMENT_B,const unsigned char* MOMENT_MODE,
    const int SEDIMENT_TRANSPORT_MODEL,const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int SED_N_RADIAL,const double* SED_MAJORANTS,
    const double* SED_GRID,const int SED_GRID_NR,const int SED_GRID_NZ,const float SED_GRID_ZMAX,
    float* heat_xy,float* heat_zy,
    float* air_x,float* air_y,float* air_z,float* air_vx,float* air_vy,float* air_vz,
    float* water_path,float* acrylic_path,
    float* first_particle_deflection_deg,float* last_particle_deflection_deg,float* internal_particle_path_total,float* internal_particle_opl_total,float* water_path_after_last_particle_event,
    float* failure_x,float* failure_y,float* failure_z,float* failure_vx,float* failure_vy,float* failure_vz,
    float* particle_diag_radius,float* particle_diag_rho,float* particle_diag_phi,float* particle_diag_cx,float* particle_diag_cy,float* particle_diag_cz,
    float* particle_diag_entry_x,float* particle_diag_entry_y,float* particle_diag_entry_z,float* particle_diag_exit_x,float* particle_diag_exit_y,float* particle_diag_exit_z,
    float* particle_diag_pre_x,float* particle_diag_pre_y,float* particle_diag_pre_z,float* particle_diag_pre_vx,float* particle_diag_pre_vy,float* particle_diag_pre_vz,
    float* particle_diag_post_x,float* particle_diag_post_y,float* particle_diag_post_z,float* particle_diag_post_vx,float* particle_diag_post_vy,float* particle_diag_post_vz,
    float* particle_diag_center_radial_clearance,float* particle_diag_finite_wall_clearance,float* particle_diag_bottom_clearance,float* particle_diag_top_clearance,
    int* interaction_count,int* geometric_interactions,int* wave_interactions,int* coherent_wave_interactions,int* morphology_wave_interactions,
    int* particle_reflections,int* particle_entry_reflections,int* particle_internal_reflections,int* particle_absorptions,
    int* cell_inner_reflections,int* cell_outer_reflections,int* cell_tir,
    int* top_reflections,int* top_tir,int* bottom_inner_reflections,int* bottom_outer_reflections,int* bottom_tir,
    int* source_bore_interactions,int* source_bore_reflections,int* entered_water,int* status,
    int* production_failure_code,int* production_failure_subcode,int* production_failure_trace_iteration,int* production_failure_macro_bounces,int* production_failure_micro_order,int* production_failure_medium,int* production_failure_event_channel,
    int* particle_diag_bin,int* particle_diag_previous_event,int* particle_diag_overlap_class,int* particle_diag_proposal_accepted,int* particle_proposal_count,int* particle_accessible_count,int* particle_null_count,
    int* first_particle_event_type,int* last_particle_event_type,int* last_particle_event_order,int* first_particle_backscatter,int* first_particle_source_hemisphere,
    unsigned int seed0,unsigned int seed1,unsigned int seed2,const unsigned int ray_offset) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x; if(tid>=N_rays)return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    unsigned int state=seed0+gid*74729u+13u, stateOPT=seed1+gid*104729u+29u, stateSRC=seed2+gid*130363u+43u;
    float MU_GEOM=(float)MU_CHANNELS[0],MU_WAVE=(float)MU_CHANNELS[1];
    float MU_MAJOR_GEOM=(SEDIMENT_TRANSPORT_MODEL==0)?MU_GEOM:(float)SED_MAJORANTS[0];
    float MU_MAJOR_WAVE=(SEDIMENT_TRANSPORT_MODEL==0)?MU_WAVE:(float)SED_MAJORANTS[1];
    float WAVE_COHERENT_FRACTION=fminf(fmaxf((float)MU_CHANNELS[7],0.0f),1.0f);
    int ic=0,gic=0,wic=0,wcoh=0,wmorph=0,pr=0,per=0,pir=0,pabs=0,cir=0,cor=0,ctir=0,tr=0,ttir=0,bir=0,bor=0,btir=0,sbi=0,sbr=0,entered=0,st=0;
    int pf=PROD_OK,pfs=PROD_SUB_NONE,pfstep=-1,pfmacro=-1,pfmicro=-1,pfmedium=0,pfevent=0;
    int pd_bin=-1,pd_prev_evt=0,pd_overlap=0,pd_accept=-1,pd_proposals=0,pd_accessible=0,pd_null=0;
    float pd_radius=0.0f,pd_rho=0.0f,pd_phi=0.0f,pd_cx=0.0f,pd_cy=0.0f,pd_cz=0.0f;
    float pd_ex=0.0f,pd_ey=0.0f,pd_ez=0.0f,pd_xx=0.0f,pd_xy=0.0f,pd_xz=0.0f;
    float pd_prex=0.0f,pd_prey=0.0f,pd_prez=0.0f,pd_prevx=0.0f,pd_prevy=0.0f,pd_prevz=0.0f;
    float pd_postx=0.0f,pd_posty=0.0f,pd_postz=0.0f,pd_postvx=0.0f,pd_postvy=0.0f,pd_postvz=0.0f;
    float pd_center_clear=0.0f,pd_wall_clear=0.0f,pd_bottom_clear=0.0f,pd_top_clear=0.0f;
    // V24.29 retained V24.27 diagnostic state. Event type codes are host-documented and do not alter transport:
    // 1 geom external reflection, 2 geom transmit/refract, 3 geom internal-reflection path,
    // 4 coherent Mie wave, 5 morphology-wave external reflection,
    // 6 morphology-wave transmit/refract, 7 morphology-wave internal-reflection path.
    int first_evt=0,last_evt=0,last_evt_order=0,first_back=0,first_source_hemi=0;
    float first_defl=-1.0f,last_defl=-1.0f,internal_path_total=0.0f,internal_opl_total=0.0f,wp_at_last_evt=-1.0f;
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

    // V24.20: retain the exact V24.19 hard-aperture source branch as control.
    // The reflective branch explicitly propagates through the piecewise cylindrical
    // source collimator and traces air/nylon Fresnel wall reflections.
    if(SOURCE_BORE_SURFACE_MODEL==0){
        if(x*x+z*z>COUNTERBORE_R*COUNTERBORE_R){st=3;goto WRITE_OUT;}
        float t=(-THROAT_OUT_R-y)/vy;
        if(t<-EPS_F){st=3;goto WRITE_OUT;}
        if(t<0.0f)t=0.0f;
        float xt=x+t*vx,zt=z+t*vz;
        if(xt*xt+zt*zt>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
        t=(-RING_IN_R-y)/vy;
        if(t<=0.0f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        if(x*x+z*z>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
    }else{
        if(!source_collimator_transport(&stateSRC,SOURCE_BORE_SURFACE_MODEL,N_AIR,BORE_N,MAX_BORE_BOUNCES,
                RING_IN_R,THROAT_OUT_R,SOURCE_R,THROAT_R,COUNTERBORE_R,
                &x,&y,&z,&vx,&vy,&vz,&sbi,&sbr)){st=3;goto WRITE_OUT;}
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
        int ahsub=PROD_SUB_NONE,ahb=0;int where=acrylic_annulus_ex(&state,N_WATER,N_ACRYLIC,N_AIR,RIN,ROUT,ZMIN-BOTTOM_DISC_THICKNESS,ZMAX,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&ctir,&ahsub,&ahb);
        if(where==2){st=1;goto AIR_EXIT;}
        if(where!=1){pf=(ahsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_ACRYLIC_ANNULUS_NO_INTERSECTION;pfs=ahsub;pfmacro=ahb;pfmedium=2;st=4;goto WRITE_OUT;}
        entered=1;
    }

    for(int step=0;step<(int)MAX_ITERATIONS;++step){
        if(!isfinite(x)||!isfinite(y)||!isfinite(z)){pf=PROD_NONFINITE_POSITION;pfs=PROD_SUB_NONFINITE_POSITION;pfstep=step;pfmedium=1;st=4;break;}
        if(!isfinite(vx)||!isfinite(vy)||!isfinite(vz)){pf=PROD_NONFINITE_DIRECTION;pfs=PROD_SUB_NONFINITE_DIRECTION;pfstep=step;pfmedium=1;st=4;break;}
        float zsurf_here=free_surface_z(x,y,ZMAX,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(z<=ZMIN || z>=zsurf_here){st=2;break;}
        // Defensive topology check.  Correct interface handling keeps water rays inside
        // Rin; a materially outside state is reported, never silently snapped inward.
        double rr2w=(double)x*(double)x+(double)y*(double)y,ri2w=(double)RIN*(double)RIN;
        double water_tol=8.0*1.1920928955078125e-7*ri2w;
        if(rr2w>ri2w+water_tol){pf=PROD_INVALID_MEDIUM_STATE;pfs=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;pfstep=step;pfmedium=1;st=4;break;}
        float tside=cell_cylinder_hit_forward(x,y,vx,vy,RIN);
        float ttop=INF_F,tbottom=INF_F;
        ttop=free_surface_hit(x,y,z,vx,vy,vz,ZMAX,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(vz<-1e-14f) tbottom=(ZMIN-z)/vz;
        if(!(ttop>0.0f)) ttop=INF_F;
        if(!(tbottom>0.0f)) tbottom=INF_F;
        int axial_top=(ttop<=tbottom && ttop<tside);
        int axial_bottom=(tbottom<ttop && tbottom<tside);
        int axial=axial_top||axial_bottom;
        float tbound=axial_top?ttop:(axial_bottom?tbottom:tside);
        if(tbound>=INF_F*0.5f){pf=PROD_WATER_BOUNDARY_NO_INTERSECTION;pfs=PROD_SUB_WATER_NO_BOUNDARY;pfstep=step;pfmedium=1;st=4;break;}
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
                    double sc=sediment_local_scale(SEDIMENT_TRANSPORT_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS);
                    lg+=TRACE_GEOM_BIN[i]*sc;lw+=TRACE_WAVE_BIN[i]*sc;
                }
                LOCAL_GEOM=(float)lg;LOCAL_WAVE=(float)lw;LOCAL_TOTAL=LOCAL_GEOM+LOCAL_WAVE;
                if(!(LOCAL_TOTAL>0.0f))continue;
                float pacc=fminf(LOCAL_TOTAL/fmaxf(MU_TRACK,1e-30f),1.0f);
                if(rnd_uniform(&state)>=pacc)continue; // Woodcock null collision.
            }
            ic++;
            const float vinx=vx,viny=vy,vinz=vz;
            const int per_before=per,pir_before=pir;
            int event_type=0,event_pidx=-1;
            float event_internal_path=0.0f;
            int use_wave=(LOCAL_WAVE>0.0f && rnd_uniform(&state)*LOCAL_TOTAL>=LOCAL_GEOM);
            if(use_wave){
                wic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,wave_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,TRACE_WAVE_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_WAVE);
                event_pidx=pidx;
                // sigma_h=0 gives C=1 and deliberately consumes no extra RNG,
                // preserving the V24.8.1 coherent-Mie random stream.
                int coherent=(WAVE_COHERENT_FRACTION>=1.0f) || (rnd_uniform(&stateOPT)<WAVE_COHERENT_FRACTION);
                if(coherent){
                    wcoh++;
                    mie_scatter_direction(&stateOPT,mie_phase_cdf,mie_theta,mie_ntheta,pidx,&vx,&vy,&vz);
                    event_type=4;
                } else {
                    wmorph++;
                    float fr=sqrtf(x*x+y*y);
                    float ffx=1.0f,ffy=0.0f,ffz=0.0f;
                    if(fr>1e-12f){ffx=-y/fr;ffy=x/fr;}
                    int ppc=PROD_OK,ppsub=PROD_SUB_NONE,ppmb=0,ppmo=0;
                    int ps=spheroid_fresnel_scatter_direction_only_3d_ex(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,
                        (float)particle_radius[pidx],WAVE_SPHEROID_ASPECT_RATIO,WAVE_SPHEROID_ORIENTATION_MODEL,WAVE_SPHEROID_ORIENTATION_KAPPA,ffx,ffy,ffz,
                        MAX_PARTICLE_BOUNCES,&vx,&vy,&vz,&event_internal_path,&pr,&per,&pir,&pabs,&ppc,&ppsub,&ppmb,&ppmo);
                    if(ps==2){st=5;break;}
                    if(ps!=1){pf=ppc?ppc:PROD_OTHER_EXPLICIT_FAILURE;pfs=ppsub;pfstep=step;pfmacro=ppmb;pfmicro=ppmo;pfmedium=4;pfevent=2;st=4;break;}
                    event_type=(per>per_before)?5:((pir>pir_before)?7:6);
                }
            } else {
                gic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,geometric_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,TRACE_GEOM_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_GEOM);
                event_pidx=pidx;
                int ppc=PROD_OK,ppsub=PROD_SUB_NONE,ppmb=0,ppmo=0;
                // V24.46: sample the finite sphere placement ONCE, validate the ACTUAL sampled centre,
                // then execute the interaction with exactly that geometry.  Inaccessible proposals are
                // null collisions in the frozen V24.29 bulk proposal process.
                float proposed_radius=(float)particle_radius[pidx];
                float srho=0.0f,sphi=0.0f,sgnx=0.0f,sgny=0.0f,sgnz=0.0f,scx=x,scy=y,scz=z;
                const float prepx=x,prepy=y,prepz=z,prepvx=vx,prevpy=vy,prevpz=vz;
                sphere_sample_geometry(&stateOPT,proposed_radius,x,y,z,vx,vy,vz,&srho,&sphi,&sgnx,&sgny,&sgnz,&scx,&scy,&scz);
                float center_clear=0.0f,wall_clear=0.0f,bottom_clear=0.0f,top_clear=0.0f;
                int accessible=sphere_particle_accessible_in_water(scx,scy,scz,proposed_radius,RIN,ZMIN,ZMAX,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,
                    &center_clear,&wall_clear,&bottom_clear,&top_clear);
                pd_bin=pidx;pd_prev_evt=last_evt;pd_overlap=accessible?0:1;pd_accept=accessible?1:0;pd_proposals++;
                pd_radius=proposed_radius;pd_rho=srho;pd_phi=sphi;pd_cx=scx;pd_cy=scy;pd_cz=scz;
                pd_ex=x;pd_ey=y;pd_ez=z;pd_prex=prepx;pd_prey=prepy;pd_prez=prepz;pd_prevx=prepvx;pd_prevy=prevpy;pd_prevz=prevpz;
                pd_center_clear=center_clear;pd_wall_clear=wall_clear;pd_bottom_clear=bottom_clear;pd_top_clear=top_clear;
                if(!accessible){pd_null++;pd_xx=x;pd_xy=y;pd_xz=z;pd_postx=x;pd_posty=y;pd_postz=z;pd_postvx=vx;pd_postvy=vy;pd_postvz=vz;ic--;gic--;continue;}
                pd_accessible++;
                int ps=sphere_fresnel_interaction_3d_sampled_ex(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,proposed_radius,
                    MAX_PARTICLE_BOUNCES,sgnx,sgny,sgnz,scx,scy,scz,&x,&y,&z,&vx,&vy,&vz,&event_internal_path,&pr,&per,&pir,&pabs,&ppc,&ppsub,&ppmb,&ppmo);
                pd_xx=x;pd_xy=y;pd_xz=z;pd_postx=x;pd_posty=y;pd_postz=z;pd_postvx=vx;pd_postvy=vy;pd_postvz=vz;
                if(ps==2){st=5;break;}
                if(ps!=1){pf=ppc?ppc:PROD_OTHER_EXPLICIT_FAILURE;pfs=ppsub;pfstep=step;pfmacro=ppmb;pfmicro=ppmo;pfmedium=4;pfevent=1;st=4;break;}
                event_type=(per>per_before)?1:((pir>pir_before)?3:2);pd_prev_evt=event_type;
            }
            // V24.36 changes only the post-outcome angular magnitude for bins whose runtime
            // physical calibration selected a Mie-two-moment constraint. Event rate, branch
            // selection, Fresnel outcome, internal path and event code are inherited V24.29.
            if(event_pidx>=0 && MOMENT_MODE[event_pidx]){
                v2436_apply_moment_constraint(vinx,viny,vinz,&vx,&vy,&vz,(float)MOMENT_A[event_pidx],(float)MOMENT_B[event_pidx]);
            }
            // Causal diagnostic is evaluated only after a survived particle event.
            float dcos=fminf(fmaxf(dot3(vinx,viny,vinz,vx,vy,vz),-1.0f),1.0f);
            float defl=acosf(dcos)*(180.0f/PI_F);
            internal_path_total+=event_internal_path;
            if(event_pidx>=0) internal_opl_total+=event_internal_path*(float)particle_n[event_pidx];
            last_evt=event_type;last_evt_order=ic;last_defl=defl;wp_at_last_evt=wp;
            if(ic==1){first_evt=event_type;first_defl=defl;first_back=(dcos<0.0f)?1:0;first_source_hemi=(vy<0.0f)?1:0;}
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
                // V24.20 retains the established headspace treatment:
                // transmission through the free surface is counted as a top exit.
                // Reflected/TIR paths remain fully ray-traced and may hit the curved
                // interface repeatedly.
                st=6;break;
            } else {
                // Water -> finite acrylic bottom disc, including repeated acrylic-slab Fresnel/TIR.
                int bhsub=PROD_SUB_NONE,bhb=0;int where=bottom_acrylic_disc_ex(&state,N_WATER,N_ACRYLIC,BOTTOM_EXTERNAL_N,N_AIR,RIN,ROUT,ZMIN,BOTTOM_DISC_THICKNESS,MAX_AXIAL_BOUNCES,
                    &x,&y,&z,&vx,&vy,&vz,&ap,&bir,&bor,&btir,&bhsub,&bhb);
                if(where==1) continue;
                if(where==2){st=7;break;}
                if(where==3){
                    // Contiguous bottom-disc -> sidewall acrylic seam: no optical event.
                    int ahsub=PROD_SUB_NONE,ahb=0;int awhere=acrylic_annulus_ex(&state,N_WATER,N_ACRYLIC,N_AIR,RIN,ROUT,ZMIN-BOTTOM_DISC_THICKNESS,ZMAX,MAX_CELL_BOUNCES,
                        &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&ctir,&ahsub,&ahb);
                    if(awhere==1)continue;
                    if(awhere==2){st=1;goto AIR_EXIT;}
                    pf=(ahsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_ACRYLIC_ANNULUS_NO_INTERSECTION;pfs=ahsub;pfstep=step;pfmacro=ahb;pfmedium=2;st=4;break;
                }
                pf=(bhsub==PROD_SUB_BOTTOM_BOUNCE_LIMIT)?PROD_BOTTOM_DISC_BOUNCE_LIMIT:PROD_BOTTOM_DISC_GEOMETRY_FAILURE;pfs=bhsub;pfstep=step;pfmacro=bhb;pfmedium=3;st=4;break;
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
        int ahsub=PROD_SUB_NONE,ahb=0;int where=acrylic_annulus_ex(&state,N_WATER,N_ACRYLIC,N_AIR,RIN,ROUT,ZMIN-BOTTOM_DISC_THICKNESS,ZMAX,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&ctir,&ahsub,&ahb);
        if(where==1) continue;
        if(where==2){st=1;goto AIR_EXIT;}
        pf=(ahsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_ACRYLIC_ANNULUS_NO_INTERSECTION;pfs=ahsub;pfstep=step;pfmacro=ahb;pfmedium=2;st=4;break;
    }
    if(st==0){pf=PROD_TRACE_ITERATION_LIMIT;pfs=PROD_SUB_TRACE_ITERATION_LIMIT;pfstep=(int)MAX_ITERATIONS;pfmedium=1;st=4;}
    goto WRITE_OUT;

AIR_EXIT:
    air_x[tid]=x;air_y[tid]=y;air_z[tid]=z;air_vx[tid]=vx;air_vy[tid]=vy;air_vz[tid]=vz;
WRITE_OUT:
    water_path[tid]=wp;acrylic_path[tid]=ap;
    production_failure_code[tid]=pf;production_failure_subcode[tid]=pfs;production_failure_trace_iteration[tid]=pfstep;production_failure_macro_bounces[tid]=pfmacro;production_failure_micro_order[tid]=pfmicro;production_failure_medium[tid]=pfmedium;production_failure_event_channel[tid]=pfevent;
    if(pf!=PROD_OK){failure_x[tid]=x;failure_y[tid]=y;failure_z[tid]=z;failure_vx[tid]=vx;failure_vy[tid]=vy;failure_vz[tid]=vz;}
    particle_diag_radius[tid]=pd_radius;particle_diag_rho[tid]=pd_rho;particle_diag_phi[tid]=pd_phi;particle_diag_cx[tid]=pd_cx;particle_diag_cy[tid]=pd_cy;particle_diag_cz[tid]=pd_cz;
    particle_diag_entry_x[tid]=pd_ex;particle_diag_entry_y[tid]=pd_ey;particle_diag_entry_z[tid]=pd_ez;particle_diag_exit_x[tid]=pd_xx;particle_diag_exit_y[tid]=pd_xy;particle_diag_exit_z[tid]=pd_xz;
    particle_diag_pre_x[tid]=pd_prex;particle_diag_pre_y[tid]=pd_prey;particle_diag_pre_z[tid]=pd_prez;particle_diag_pre_vx[tid]=pd_prevx;particle_diag_pre_vy[tid]=pd_prevy;particle_diag_pre_vz[tid]=pd_prevz;
    particle_diag_post_x[tid]=pd_postx;particle_diag_post_y[tid]=pd_posty;particle_diag_post_z[tid]=pd_postz;particle_diag_post_vx[tid]=pd_postvx;particle_diag_post_vy[tid]=pd_postvy;particle_diag_post_vz[tid]=pd_postvz;
    particle_diag_center_radial_clearance[tid]=pd_center_clear;particle_diag_finite_wall_clearance[tid]=pd_wall_clear;particle_diag_bottom_clearance[tid]=pd_bottom_clear;particle_diag_top_clearance[tid]=pd_top_clear;
    particle_diag_bin[tid]=pd_bin;particle_diag_previous_event[tid]=pd_prev_evt;particle_diag_overlap_class[tid]=pd_overlap;particle_diag_proposal_accepted[tid]=pd_accept;particle_proposal_count[tid]=pd_proposals;particle_accessible_count[tid]=pd_accessible;particle_null_count[tid]=pd_null;
    first_particle_deflection_deg[tid]=first_defl;last_particle_deflection_deg[tid]=last_defl;internal_particle_path_total[tid]=internal_path_total;internal_particle_opl_total[tid]=internal_opl_total;
    water_path_after_last_particle_event[tid]=(last_evt_order>0 && wp_at_last_evt>=0.0f)?fmaxf(wp-wp_at_last_evt,0.0f):-1.0f;
    interaction_count[tid]=ic;geometric_interactions[tid]=gic;wave_interactions[tid]=wic;coherent_wave_interactions[tid]=wcoh;morphology_wave_interactions[tid]=wmorph;particle_reflections[tid]=pr;particle_entry_reflections[tid]=per;particle_internal_reflections[tid]=pir;particle_absorptions[tid]=pabs;
    cell_inner_reflections[tid]=cir;cell_outer_reflections[tid]=cor;cell_tir[tid]=ctir;
    top_reflections[tid]=tr;top_tir[tid]=ttir;bottom_inner_reflections[tid]=bir;bottom_outer_reflections[tid]=bor;bottom_tir[tid]=btir;
    source_bore_interactions[tid]=sbi;source_bore_reflections[tid]=sbr;entered_water[tid]=entered;status[tid]=st;
    first_particle_event_type[tid]=first_evt;last_particle_event_type[tid]=last_evt;last_particle_event_order[tid]=last_evt_order;first_particle_backscatter[tid]=first_back;first_particle_source_hemisphere[tid]=first_source_hemi;
}
// ================= V24.35 DIAGNOSTIC-ONLY SINGLE-PARTICLE AUDIT BEGIN =================
// Samples the inherited V24.27/V24.29 particle angular law in isolation with a fixed +y
// incident direction. It is never called by trace_kernel and cannot alter production transport.
__global__ void single_particle_angular_audit_kernel(
    const int SAMPLES_PER_BIN,const int N_PARTICLES,
    const float N_MEDIUM,const float WAVELENGTH,const float ROUGH_ALPHA,
    const float WAVE_SPHEROID_ASPECT_RATIO,const int WAVE_ORIENTATION_MODEL,const float WAVE_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const double* PARTICLE_N,const double* PARTICLE_K,const double* PARTICLE_RADIUS,
    const double* QSCA,const double* MIE_PHASE_CDF,const double* MIE_THETA,const int MIE_NTHETA,
    const float WAVE_COHERENT_FRACTION,const unsigned int SEED,
    float* INTERFACE_COS,unsigned char* INTERFACE_TYPE,unsigned char* INTERFACE_STATUS,
    float* PRODUCTION_COS,unsigned char* PRODUCTION_TYPE,unsigned char* PRODUCTION_STATUS){
    long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)SAMPLES_PER_BIN*(long long)N_PARTICLES;if(tid>=total)return;
    int p=(int)(tid/(long long)SAMPLES_PER_BIN);
    unsigned int sid=(unsigned int)(tid-(long long)p*(long long)SAMPLES_PER_BIN);
    unsigned int stateI=SEED+(unsigned int)p*2654435761u+sid*74729u+13u;
    unsigned int stateP=SEED^0x9e3779b9u;stateP+=(unsigned int)p*2246822519u+sid*104729u+29u;

    // Pure inherited finite-particle interface kernel.
    float vx=0.0f,vy=1.0f,vz=0.0f,x=0.0f,y=0.0f,z=0.0f,ip=0.0f;int pr=0,per=0,pir=0,pabs=0;
    int ps=sphere_fresnel_interaction_3d(&stateI,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
        (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
    INTERFACE_STATUS[tid]=(unsigned char)ps;
    if(ps==1){INTERFACE_COS[tid]=fminf(fmaxf(vy,-1.0f),1.0f);INTERFACE_TYPE[tid]=(unsigned char)((per>0)?1:((pir>0)?3:2));}
    else {INTERFACE_COS[tid]=0.0f;INTERFACE_TYPE[tid]=0;} // status!=1 makes payload invalid

    // Exact V24.29 per-bin branch mixture. For Qsca<1 the exact event-rate remainder is
    // entirely the geometric/interface branch. For Qsca>=1 the projected-area branch has
    // probability 1/Qsca and the Mie-excess wave branch has probability (Qsca-1)/Qsca.
    float pvx=0.0f,pvy=1.0f,pvz=0.0f;int ppr=0,pper=0,ppir=0,ppabs=0;float pip=0.0f;
    float q=(float)fmax(QSCA[p],0.0);float geom_fraction=(q>1.0f)?(1.0f/q):1.0f;
    int use_wave=(q>1.0f && rnd_uniform(&stateP)>=geom_fraction);
    if(!use_wave){
        float px=0.0f,py=0.0f,pz=0.0f;
        ps=sphere_fresnel_interaction_3d(&stateP,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
            (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&px,&py,&pz,&pvx,&pvy,&pvz,&pip,&ppr,&pper,&ppir,&ppabs);
        PRODUCTION_STATUS[tid]=(unsigned char)ps;
        if(ps==1){PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=(unsigned char)((pper>0)?1:((ppir>0)?3:2));}
        else {PRODUCTION_COS[tid]=0.0f;PRODUCTION_TYPE[tid]=0;} // status!=1 makes payload invalid
        return;
    }
    int coherent=(WAVE_COHERENT_FRACTION>=1.0f)||(rnd_uniform(&stateP)<WAVE_COHERENT_FRACTION);
    if(coherent){
        mie_scatter_direction(&stateP,MIE_PHASE_CDF,MIE_THETA,MIE_NTHETA,p,&pvx,&pvy,&pvz);
        PRODUCTION_STATUS[tid]=1;PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=4;return;
    }
    ps=spheroid_fresnel_scatter_direction_only_3d(&stateP,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
        (float)PARTICLE_RADIUS[p],WAVE_SPHEROID_ASPECT_RATIO,WAVE_ORIENTATION_MODEL,WAVE_ORIENTATION_KAPPA,1.0f,0.0f,0.0f,
        MAX_PARTICLE_BOUNCES,&pvx,&pvy,&pvz,&pip,&ppr,&pper,&ppir,&ppabs);
    PRODUCTION_STATUS[tid]=(unsigned char)ps;
    if(ps==1){PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=(unsigned char)((pper>0)?5:((ppir>0)?7:6));}
    else {PRODUCTION_COS[tid]=0.0f;PRODUCTION_TYPE[tid]=0;} // status!=1 makes payload invalid
}
// ================= V24.35 DIAGNOSTIC-ONLY SINGLE-PARTICLE AUDIT END =================

// ================= V24.36 MOMENT-CONSTRAINED SINGLE-PARTICLE AUDIT BEGIN =================
__global__ void single_particle_moment_constrained_audit_kernel(
    const int SAMPLES_PER_BIN,const int N_PARTICLES,
    const float N_MEDIUM,const float WAVELENGTH,const float ROUGH_ALPHA,
    const float WAVE_SPHEROID_ASPECT_RATIO,const int WAVE_ORIENTATION_MODEL,const float WAVE_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const double* PARTICLE_N,const double* PARTICLE_K,const double* PARTICLE_RADIUS,
    const double* QSCA,const double* MIE_PHASE_CDF,const double* MIE_THETA,const int MIE_NTHETA,
    const float WAVE_COHERENT_FRACTION,const unsigned int SEED,
    const double* MOMENT_A,const double* MOMENT_B,const unsigned char* MOMENT_MODE,
    float* BASE_COS,float* CONSTRAINED_COS,unsigned char* EVENT_TYPE,unsigned char* STATUS){
    long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)SAMPLES_PER_BIN*(long long)N_PARTICLES;if(tid>=total)return;
    int p=(int)(tid/(long long)SAMPLES_PER_BIN);
    unsigned int sid=(unsigned int)(tid-(long long)p*(long long)SAMPLES_PER_BIN);
    unsigned int state=SEED^0x517cc1b7u;state+=(unsigned int)p*2246822519u+sid*104729u+31u;
    float vx=0.0f,vy=1.0f,vz=0.0f;int pr=0,per=0,pir=0,pabs=0;float ip=0.0f;
    float q=(float)fmax(QSCA[p],0.0);float geom_fraction=(q>1.0f)?(1.0f/q):1.0f;
    int use_wave=(q>1.0f && rnd_uniform(&state)>=geom_fraction);int ps=0;unsigned char typ=0;
    if(!use_wave){
        float x=0.0f,y=0.0f,z=0.0f;
        ps=sphere_fresnel_interaction_3d(&state,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
            (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
        typ=(unsigned char)((per>0)?1:((pir>0)?3:2));
    } else {
        int coherent=(WAVE_COHERENT_FRACTION>=1.0f)||(rnd_uniform(&state)<WAVE_COHERENT_FRACTION);
        if(coherent){mie_scatter_direction(&state,MIE_PHASE_CDF,MIE_THETA,MIE_NTHETA,p,&vx,&vy,&vz);ps=1;typ=4;}
        else{
            ps=spheroid_fresnel_scatter_direction_only_3d(&state,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
                (float)PARTICLE_RADIUS[p],WAVE_SPHEROID_ASPECT_RATIO,WAVE_ORIENTATION_MODEL,WAVE_ORIENTATION_KAPPA,1.0f,0.0f,0.0f,
                MAX_PARTICLE_BOUNCES,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
            typ=(unsigned char)((per>0)?5:((pir>0)?7:6));
        }
    }
    STATUS[tid]=(unsigned char)ps;
    if(ps!=1){BASE_COS[tid]=0.0f;CONSTRAINED_COS[tid]=0.0f;EVENT_TYPE[tid]=0;return;}
    float base=fminf(fmaxf(vy,-1.0f),1.0f);BASE_COS[tid]=base;EVENT_TYPE[tid]=typ;
    if(MOMENT_MODE[p]){float ox=vx,oy=vy,oz=vz;v2436_apply_moment_constraint(0.0f,1.0f,0.0f,&ox,&oy,&oz,(float)MOMENT_A[p],(float)MOMENT_B[p]);CONSTRAINED_COS[tid]=fminf(fmaxf(oy,-1.0f),1.0f);}
    else CONSTRAINED_COS[tid]=base;
}
// ================= V24.36 MOMENT-CONSTRAINED SINGLE-PARTICLE AUDIT END =================

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



# ============================ V24.27 EVENT-HISTORY DECOMPOSITION ============================
def _v2427_event_history_masks(ints):
    """Return exact per-ray mechanism/order masks from CUDA-emitted primitive counters.

    The masks are diagnostic only.  They do not modify ray weights or transport.  The
    interaction-order bins are mutually exclusive.  The single-event mechanism partition
    is also mutually exclusive for detected, non-absorbed rays:
      external entry reflection | internal-reflection path | surface transmission/refraction
      | coherent Mie-wave event.
    Morphology-wave events use the same explicit particle-surface Fresnel primitive as the
    geometric branch and therefore enter the surface categories.
    """
    ic=np.asarray(ints['interaction_count'])
    gic=np.asarray(ints['geometric_interactions'])
    wic=np.asarray(ints['wave_interactions'])
    wcoh=np.asarray(ints['coherent_wave_interactions'])
    wmorph=np.asarray(ints['morphology_wave_interactions'])
    per=np.asarray(ints['particle_entry_reflections'])
    pir=np.asarray(ints['particle_internal_reflections'])
    pabs=np.asarray(ints['particle_absorptions'])
    single=(ic==1)
    multiple=(ic>=2)
    surface_touch=(gic>0)|(wmorph>0)
    masks={
        # Existing broad categories retained for regression/continuity.
        'any_particle':ic>0,
        'any_geometric':gic>0,
        'any_wave':wic>0,
        'any_coherent_wave':wcoh>0,
        'any_morphology_wave':wmorph>0,
        'any_entry_reflection':per>0,
        'any_internal_reflection':pir>0,
        'ballistic':ic==0,
        'single_interaction':single,
        'single_entry_reflection':single&(per>0),
        'multiple_interaction':multiple,
        # V24.27 exact interaction order.
        'interaction_order_2':ic==2,
        'interaction_order_3':ic==3,
        'interaction_order_4plus':ic>=4,
        # V24.27 single-event branch identity.
        'exactly_one_geometric':single&(gic==1)&(wic==0),
        'exactly_one_wave':single&(gic==0)&(wic==1),
        'single_coherent_wave_escape':single&(wcoh==1),
        'single_external_entry_reflection_escape':single&(per>0),
        'single_internal_reflection_escape':single&(pir>0),
        'single_surface_transmission_escape':single&surface_touch&(per==0)&(pir==0)&(pabs==0),
        # V24.27 multi-event branch identity.
        'mixed_geometric_wave':multiple&(gic>0)&(wic>0),
        'geometric_only_multiple':multiple&(gic>0)&(wic==0),
        'wave_only_multiple':multiple&(gic==0)&(wic>0),
        'multiple_with_entry_reflection':multiple&(per>0),
        'multiple_with_internal_reflection':multiple&(pir>0),
    }
    return masks


V2427_PARTICLE_EVENT_TYPE_NAMES={
    0:'none',
    1:'geometric_external_entry_reflection',
    2:'geometric_surface_transmission_refraction',
    3:'geometric_internal_reflection_path',
    4:'coherent_mie_wave',
    5:'morphology_wave_external_entry_reflection',
    6:'morphology_wave_surface_transmission_refraction',
    7:'morphology_wave_internal_reflection_path',
}

def _v2427_safe_fraction(num,den):
    num=np.asarray(num,dtype=np.float64);den=np.asarray(den,dtype=np.float64)
    return np.divide(num,den,out=np.zeros_like(num),where=den>0)


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
    numerical_failure_count: int                    # compatibility: total production integrity failures
    numerical_failure_fraction: float
    production_failure_count: int
    production_failure_code_counts: dict
    production_failure_subcode_counts: dict
    production_watchdog_count: int
    production_geometry_failure_count: int
    production_nonfinite_count: int
    production_failure_diagnostics: dict
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
    meridional_circulation_ratio: float
    meridional_peak_speed_m_s: float
    meridional_grid_nr: int
    meridional_grid_nz: int
    meridional_solver_iterations: int
    meridional_solver_max_residual: float
    mean_bottom_inner_reflections: float
    mean_bottom_outer_reflections: float
    bottom_tir_fraction: float
    top_exit_fraction: float
    bottom_exit_fraction: float
    source_bore_surface_model: str
    detector_bore_surface_model: str
    bore_refractive_index: float
    ring_inner_face_surface_model: str
    ring_inner_face_refractive_index: float
    max_ring_face_bounces: int
    source_collimator_acceptance_fraction: float
    source_direct_acceptance_fraction: float
    source_reflected_acceptance_fraction: float
    source_bore_interactions_per_ray: float
    source_bore_reflections_per_ray: float
    hardware_native_direct_hits: np.ndarray
    hardware_mirror_direct_hits: np.ndarray
    hardware_symmetry_direct_scores: np.ndarray
    hardware_native_stray_hits: np.ndarray
    hardware_mirror_stray_hits: np.ndarray
    hardware_symmetry_stray_scores: np.ndarray
    detector_stray_fraction_by_channel: np.ndarray
    detector_stray_detection_fraction: float
    hardware_native_ring_face_stray_hits: np.ndarray
    hardware_mirror_ring_face_stray_hits: np.ndarray
    hardware_symmetry_ring_face_stray_scores: np.ndarray
    detector_ring_face_stray_fraction_by_channel: np.ndarray
    detector_ring_face_stray_detection_fraction: float
    photodiode_response_model: str
    sfh213_half_angle_deg: float
    sfh213_directional_fit_exponent: float
    photodiode_native_weighted_scores: np.ndarray
    photodiode_mirror_weighted_scores: np.ndarray
    photodiode_symmetry_weighted_scores: np.ndarray
    normalized_photodiode_native_response: np.ndarray
    normalized_photodiode_mirror_response: np.ndarray
    normalized_photodiode_symmetry_response: np.ndarray
    photodiode_relative_response_by_channel: np.ndarray
    detector_mean_incidence_angle_deg_by_channel: np.ndarray
    hardware_candidate_detection_fraction: float
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
    detector_particle_path_diagnostics: Optional[Dict[str,np.ndarray]]=None
    ray_data: Optional[Dict[str,np.ndarray]]=None
    heatmap_xy: Optional[np.ndarray]=None
    heatmap_zy: Optional[np.ndarray]=None
    def to_dict(self):
        d={k:v for k,v in self.__dict__.items() if k not in {'particle_diagnostics','detector_particle_path_diagnostics','ray_data','heatmap_xy','heatmap_zy'}}
        for k,v in list(d.items()):
            if isinstance(v,np.ndarray): d[k]=v.tolist()
            elif isinstance(v,np.generic): d[k]=v.item()
        return d




class TardiisForwardModel:
    """CUDA-authoritative V24.36 kernel: V24.29 event rate plus Qsca>=1 two-moment angular constraint.

    Host code performs configuration, array transfer, diagnostics aggregation and
    I/O only. Mie/event/phase/source/detector physics that determine ray outcomes
    are implemented in CUDA kernels in ``CUDA_SRC``.
    """
    MIE_MAX_ORDER=4096
    EVENT_MODEL_CODES={'geometric':0,'mie_qsca':1,'mie_qext':2,'hybrid_mie_excess':3,'hybrid_mie_exact_rate':4,'hybrid_mie_exact_rate_moment_constrained':5}
    WEIGHT_MODE_CODES={'mass_fraction':0,'number_fraction':1}
    SOURCE_ANGULAR_MODEL_CODES={'legacy_polar_pdf':0,'beta_radiance_3d':1,'thesis_planar_beta':2}
    BORE_SURFACE_MODEL_CODES={'absorbing':0,'smooth_fresnel_nylon':1}
    RING_FACE_SURFACE_MODEL_CODES={'absorbing':0,'smooth_fresnel_nylon':1}
    PHOTODIODE_RESPONSE_MODEL_CODES={'ideal_hard_aperture':0,'sfh213_datasheet_fit':1}
    SPHEROID_ORIENTATION_MODEL_CODES={'isotropic':0,'local_tangential_axial_vmf':1}
    FREE_SURFACE_MODEL_CODES={'flat':0,'parabolic_vortex':1,'localized_scully_vortex':2}
    SEDIMENT_TRANSPORT_MODEL_CODES={'uniform':0,'stokes_drift_diffusion':1,'schiller_naumann_drift_diffusion':2,'schiller_naumann_meridional_advection_diffusion':3}

    def __init__(self,n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,
                 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
                 water_height_m=0.142,sensor_height_above_bottom_m=0.093,
                 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,
                 counterbore_depth_m=0.008,through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,
                 source_launch_radius_m=0.0655,legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1.0e-5,
                 source_bore_surface_model='absorbing',detector_bore_surface_model='absorbing',bore_refractive_index=1.53,max_bore_bounces=16,
                 ring_inner_face_surface_model='absorbing',ring_inner_face_refractive_index=1.53,max_ring_face_bounces=16,
                 photodiode_response_model='ideal_hard_aperture',
                 particle_wavelength_m=DEFAULT_PARTICLE_WAVELENGTH_M,particle_optical_constants_by_material=None,
                 particle_event_model='geometric',hybrid_wave_scale=1.0,mie_phase_grid_size=2049,moment_constraint_by_material=None,
                 particle_surface_rms_slope_deg=25.0,wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.5,
                 wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
                 free_surface_model='localized_scully_vortex',vortex_wall_center_height_difference_m=0.050,vortex_core_radius_m=0.020,
                 sediment_transport_model='schiller_naumann_drift_diffusion',sediment_eddy_diffusivity_m2_s=1.0e-3,water_density_kg_per_m3=997.0,water_dynamic_viscosity_pa_s=8.9e-4,gravity_m_s2=9.81,sediment_transport_quadrature_points=4096,sediment_radial_lookup_points=2049,
                 meridional_circulation_ratio=0.10,meridional_grid_nr=48,meridional_grid_nz=96,meridional_solver_max_iterations=8000,meridional_solver_check_interval=250,meridional_solver_tolerance=1.0e-6,meridional_solver_sor=1.75,
                 aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
                 fragmentation_reference_diameter_m=100e-6,fragmentation_size_exponent=1.0,
                 aggregation_step_safety=0.2,aggregation_max_steps=200000,
                 alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d',density_kg_per_m3=2600.0,
                 bottom_disc_thickness_m=0.003,bottom_external_refractive_index=1.0,max_axial_bounces=64,
                 max_iterations=int(1e6),max_internal_bounces=64,max_cell_bounces=64,
                 chunk_size=250000,statistics_batch_rays=250000,
                 symmetry_average_exact=True):
        try: import cupy as cp
        except Exception as exc: raise RuntimeError('CuPy/CUDA is required for CLARITAS V24.29.') from exc
        self.cp=cp;self.n_water=float(n_water);self.n_particle=float(n_particle);self.n_acrylic=float(n_acrylic);self.n_air=float(n_air)
        self.rin=float(tube_inner_radius_m);self.rout=float(tube_outer_radius_m)
        self.water_height=float(water_height_m);self.sensor_height=float(sensor_height_above_bottom_m)
        self.zmin=-self.sensor_height;self.zmax=self.water_height-self.sensor_height
        self.ring_in=float(sensor_ring_inner_radius_m);self.ring_out=float(sensor_ring_outer_radius_m)
        self.counterbore_depth=float(counterbore_depth_m);self.throat_out=self.ring_out-self.counterbore_depth
        self.throat_radius=0.5*float(through_bore_diameter_m);self.counterbore_radius=0.5*float(counterbore_diameter_m)
        self.source_launch_radius=float(source_launch_radius_m)
        self.legacy_acceptance_deg=float(legacy_detector_acceptance_deg);self.beam_sigma=float(source_beam_sigma_m)
        self.source_bore_surface_model=str(source_bore_surface_model).strip().lower();self.detector_bore_surface_model=str(detector_bore_surface_model).strip().lower();self.bore_refractive_index=float(bore_refractive_index);self.max_bore_bounces=int(max_bore_bounces)
        self.ring_inner_face_surface_model=str(ring_inner_face_surface_model).strip().lower();self.ring_inner_face_refractive_index=float(ring_inner_face_refractive_index);self.max_ring_face_bounces=int(max_ring_face_bounces)
        self.photodiode_response_model=str(photodiode_response_model).strip().lower()
        self.particle_wavelength=float(particle_wavelength_m);self.particle_optical_constants_by_material=particle_optical_constants_by_material or {}
        self.particle_event_model=str(particle_event_model).strip().lower()
        self.hybrid_wave_scale=float(hybrid_wave_scale);self.mie_phase_grid_size=int(mie_phase_grid_size)
        self.moment_constraint_by_material=moment_constraint_by_material or {}
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
        self.sediment_radial_lookup_points=int(sediment_radial_lookup_points)
        self.meridional_circulation_ratio=float(meridional_circulation_ratio)
        self.meridional_grid_nr=int(meridional_grid_nr);self.meridional_grid_nz=int(meridional_grid_nz)
        self.meridional_solver_max_iterations=int(meridional_solver_max_iterations);self.meridional_solver_check_interval=int(meridional_solver_check_interval)
        self.meridional_solver_tolerance=float(meridional_solver_tolerance);self.meridional_solver_sor=float(meridional_solver_sor)
        self.meridional_peak_speed=0.0;self.meridional_vtheta_peak=0.0
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
        if self.photodiode_response_model not in self.PHOTODIODE_RESPONSE_MODEL_CODES: raise ValueError(f'unknown photodiode_response_model {self.photodiode_response_model!r}')
        if self.photodiode_response_model!='ideal_hard_aperture' and (self.detector_bore_surface_model!='absorbing' or self.ring_inner_face_surface_model!='absorbing'):
            raise ValueError('SFH213 package-response audit requires absorbing detector bores and absorbing ring inner face so incidence angle is defined on the validated direct path')
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
            raise ValueError("sediment_transport_model must be 'uniform', 'stokes_drift_diffusion', 'schiller_naumann_drift_diffusion', or 'schiller_naumann_meridional_advection_diffusion'")
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
        if self.sediment_radial_lookup_points<65:
            raise ValueError('sediment_radial_lookup_points must be >=65')
        if not np.isfinite(self.meridional_circulation_ratio) or not (0.0<=self.meridional_circulation_ratio<=0.5):
            raise ValueError('meridional_circulation_ratio must be finite and in [0,0.5]')
        if self.meridional_grid_nr<16 or self.meridional_grid_nz<32:
            raise ValueError('meridional solver grid must be at least 16 x 32')
        if self.meridional_solver_max_iterations<250 or self.meridional_solver_check_interval<10:
            raise ValueError('invalid meridional solver iteration controls')
        if not np.isfinite(self.meridional_solver_tolerance) or not (0.0<self.meridional_solver_tolerance<1e-2):
            raise ValueError('meridional_solver_tolerance must be in (0,1e-2)')
        if not np.isfinite(self.meridional_solver_sor) or not (1.0<=self.meridional_solver_sor<1.95):
            raise ValueError('meridional_solver_sor must be in [1,1.95)')
        if self.sediment_transport_model!='uniform' and self.vortex_delta_h>0.0 and self.free_surface_model!='localized_scully_vortex':
            raise ValueError('V24.20 non-uniform drift-diffusion with non-zero vortex depth requires localized_scully_vortex')
        vg=vortex_surface_geometry(self.free_surface_model,self.zmax,self.rin,self.vortex_delta_h,self.vortex_core_radius)
        if self.free_surface_model=='localized_scully_vortex' and self.vortex_delta_h>0.0:
            omega=math.sqrt(max(2.0*self.gravity*vg['h_inf']/(self.vortex_core_radius*self.vortex_core_radius),0.0))
            self.meridional_vtheta_peak=0.5*omega*self.vortex_core_radius
            self.meridional_peak_speed=self.meridional_circulation_ratio*self.meridional_vtheta_peak
        if self.sediment_transport_model=='schiller_naumann_meridional_advection_diffusion' and (self.free_surface_model!='localized_scully_vortex' or self.vortex_delta_h<=0.0):
            raise ValueError('V24.20 meridional transport requires a non-zero localized_scully_vortex')
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
        if self.source_bore_surface_model not in self.BORE_SURFACE_MODEL_CODES or self.detector_bore_surface_model not in self.BORE_SURFACE_MODEL_CODES:
            raise ValueError("bore surface model must be absorbing or smooth_fresnel_nylon")
        if self.ring_inner_face_surface_model not in self.RING_FACE_SURFACE_MODEL_CODES:
            raise ValueError("ring inner face surface model must be absorbing or smooth_fresnel_nylon")
        if not (self.ring_inner_face_refractive_index>0.0 and np.isfinite(self.ring_inner_face_refractive_index)):
            raise ValueError('ring_inner_face_refractive_index must be finite and positive')
        if self.max_ring_face_bounces<1: raise ValueError('max_ring_face_bounces must be >=1')
        if not (self.bore_refractive_index>1.0): raise ValueError('bore_refractive_index must be >1')
        if self.max_bore_bounces<0: raise ValueError('max_bore_bounces must be >=0')
        if not (self.zmin<0<self.zmax): raise ValueError('sensor plane must lie within water column')
        if self.bottom_disc_thickness<=0.0: raise ValueError('bottom_disc_thickness_m must be positive')
        if self.bottom_external_n<=0.0: raise ValueError('bottom_external_refractive_index must be positive')
        if self.max_axial_bounces<1: raise ValueError('max_axial_bounces must be >=1')
        self.module=cp.RawModule(code=CUDA_SRC,options=('-std=c++11',))
        names=['trace_kernel','free_surface_geometry_test_kernel','sediment_field_test_kernel','meridional_velocity_test_kernel','meridional_active_mask_kernel','meridional_grid_initialize_kernel','meridional_build_coefficients_kernel','meridional_relax_kernel','meridional_normalize_kernel','meridional_residual_kernel','meridional_extend_grid_kernel','meridional_finalize_kernel','meridional_majorants_kernel','orientation_sampler_test_kernel','build_theta_grid_kernel','mie_coefficients_efficiencies_kernel','mie_phase_pdf_kernel','mie_phase_cdf_kernel','dummy_phase_kernel','orthokinetic_compact_aggregation_kernel','orthokinetic_compact_aggregation_fragmentation_kernel','transport_precompute_kernel','sediment_transport_precompute_kernel','score_detectors_kernel','single_particle_angular_audit_kernel','single_particle_moment_constrained_audit_kernel']
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
        if self.particle_event_model in {'hybrid_mie_excess','hybrid_mie_exact_rate','hybrid_mie_exact_rate_moment_constrained'}:
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
        # V24.36 per-bin angular constraint arrays. These are runtime-derived from the
        # inherited V24.29 kernel and exact Mie phase moments, never from detector data.
        moment_a=cp.ones(n,dtype=cp.float64);moment_b=cp.zeros(n,dtype=cp.float64);moment_mode=cp.zeros(n,dtype=cp.uint8)
        if self.particle_event_model=='hybrid_mie_exact_rate_moment_constrained':
            key=str(material).strip().lower();spec=self.moment_constraint_by_material.get(key)
            if spec is None: raise RuntimeError(f'V24.36 moment constraints missing for material {key!r}')
            ah=np.asarray(spec.get('a'),dtype=np.float64);bh=np.asarray(spec.get('b'),dtype=np.float64);mh=np.asarray(spec.get('mode'),dtype=np.uint8)
            if ah.shape!=(n,) or bh.shape!=(n,) or mh.shape!=(n,): raise RuntimeError(f'V24.36 moment constraint shape mismatch for {key}: {(ah.shape,bh.shape,mh.shape)} expected {(n,)}')
            if np.any(~np.isfinite(ah)) or np.any(~np.isfinite(bh)): raise RuntimeError('V24.36 moment constraints contain non-finite coefficients')
            if np.any((mh!=0)&(mh!=1)): raise RuntimeError('V24.36 moment constraint mode must be 0/1')
            qh=cp.asnumpy(qsca)
            if np.any(mh[qh<1.0]!=0): raise RuntimeError('V24.36 Qsca<1 bins must remain unconstrained')
            # Monotone bounded quadratic map T(d)=a*d+b*d^2 over d in [0,2].
            active=mh.astype(bool)
            if np.any(ah[active] < -1e-12) or np.any(ah[active]+4.0*bh[active] < -1e-10): raise RuntimeError('V24.36 moment map is non-monotone on [0,2]')
            if np.any(2.0*ah[active]+4.0*bh[active] > 2.000001): raise RuntimeError('V24.36 moment map exceeds d=2 bound before CUDA clipping')
            moment_a=cp.asarray(ah);moment_b=cp.asarray(bh);moment_mode=cp.asarray(mh)
        # V24.20 size-resolved spatial sediment field.  For non-hybrid event laws
        # the historical trace channel is mu_event; for the production hybrid it
        # remains geometric plus the separate Mie-excess wave channel.
        if self.particle_event_model=='hybrid_mie_excess':
            trace_geom_bin=mug;trace_wave_bin=muw
        elif self.particle_event_model in {'hybrid_mie_exact_rate','hybrid_mie_exact_rate_moment_constrained'}:
            trace_geom_bin=cp.maximum(muev-muw,0.0);trace_wave_bin=muw
        else:
            trace_geom_bin=muev;trace_wave_bin=cp.zeros_like(muw)
        sed_tau=arr();sed_ws=arr();sed_sre=arr();sed_sdrag=arr();sed_A=arr();sed_B=arr();sed_norm=arr();sed_max=arr();sed_min=arr()
        sed_rmax=arr();sed_rre=arr();sed_rcenter=arr();sed_radial_log=cp.empty(n*self.sediment_radial_lookup_points,dtype=cp.float64);sed_radial_wr=cp.empty_like(sed_radial_log)
        sed_sc=arr();sed_sw=arr();sed_bc=arr();sed_bw=arr();sed_major=cp.zeros(4,dtype=cp.float64)
        sed_min.fill(np.nan)
        err.fill(0)
        self.kernels['sediment_transport_precompute_kernel']((1,),(1,),(
            dd,np.int32(n),np.int32(self.SEDIMENT_TRANSPORT_MODEL_CODES[self.sediment_transport_model]),
            np.float64(self.density),np.float64(self.water_density),np.float64(self.water_dynamic_viscosity),np.float64(self.gravity),np.float64(self.sediment_eddy_diffusivity),
            np.float64(self.rin),np.float64(self.zmin),np.float64(self.zmax),np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),np.float64(self.vortex_delta_h),np.float64(self.vortex_core_radius),np.int32(self.sediment_transport_quadrature_points),np.int32(self.sediment_radial_lookup_points),
            trace_geom_bin,trace_wave_bin,sed_tau,sed_ws,sed_sre,sed_sdrag,sed_A,sed_B,sed_norm,sed_max,sed_rmax,sed_rre,sed_rcenter,sed_radial_log,sed_radial_wr,sed_sc,sed_sw,sed_bc,sed_bw,sed_major,err))
        cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0])
        if ecode: raise RuntimeError(f'CUDA sediment transport precompute failed with error code {ecode}')

        # V24.20 model 3: solve the steady axisymmetric advection-diffusion field
        # on CUDA.  The finite-Re radial/settling slips above remain the particle
        # relative velocities; the new incompressible streamfunction adds bulk
        # meridional advection.  The solver is conservative finite-volume,
        # first-order upwind in advection and centred in diffusion, with no-flux
        # boundaries and exact per-bin mass renormalisation.
        vg=vortex_surface_geometry(self.free_surface_model,self.zmax,self.rin,self.vortex_delta_h,self.vortex_core_radius)
        sed_grid=cp.ones(1,dtype=cp.float64);grid_nr=2;grid_nz=2;grid_zmax=float(vg['wall_z'])
        solver_iterations=0;solver_residual=np.zeros(n,dtype=np.float64)
        if self.sediment_transport_model=='schiller_naumann_meridional_advection_diffusion':
            grid_nr=self.meridional_grid_nr;grid_nz=self.meridional_grid_nz;plane=grid_nr*grid_nz;total=n*plane
            sed_grid=cp.empty(total,dtype=cp.float64);active=cp.empty(plane,dtype=cp.uint8)
            ap=cp.empty(total,dtype=cp.float64);ae=cp.empty_like(ap);aw=cp.empty_like(ap);an=cp.empty_like(ap);ass=cp.empty_like(ap)
            residual=cp.empty(n,dtype=cp.float64);threads_solver=256;blocks_plane=(plane+threads_solver-1)//threads_solver;blocks_total=(total+threads_solver-1)//threads_solver
            self.kernels['meridional_active_mask_kernel']((blocks_plane,),(threads_solver,),(
                np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.zmax),np.float64(self.vortex_delta_h),np.float64(self.vortex_core_radius),active))
            self.kernels['meridional_grid_initialize_kernel']((blocks_total,),(threads_solver,),(
                np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.vortex_core_radius),sed_A,sed_B,sed_norm,sed_radial_log,np.int32(self.sediment_radial_lookup_points),active,sed_grid))
            self.kernels['meridional_build_coefficients_kernel']((blocks_total,),(threads_solver,),(
                np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.zmax),np.float64(self.vortex_delta_h),np.float64(self.vortex_core_radius),np.float64(self.meridional_peak_speed),np.float64(self.sediment_eddy_diffusivity),sed_ws,sed_radial_wr,np.int32(self.sediment_radial_lookup_points),active,ap,ae,aw,an,ass))
            err.fill(0)
            self.kernels['meridional_normalize_kernel']((n,),(1,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.zmax),active,sed_grid,err))
            cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0])
            if ecode: raise RuntimeError(f'CUDA V24.20 meridional initial normalization failed with error code {ecode}')
            min_solver_iterations=min(1000,self.meridional_solver_max_iterations)
            last_checked=0
            for it in range(1,self.meridional_solver_max_iterations+1):
                self.kernels['meridional_relax_kernel']((blocks_total,),(threads_solver,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.int32(0),np.float64(self.meridional_solver_sor),active,ap,ae,aw,an,ass,sed_grid))
                self.kernels['meridional_relax_kernel']((blocks_total,),(threads_solver,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.int32(1),np.float64(self.meridional_solver_sor),active,ap,ae,aw,an,ass,sed_grid))
                if it%20==0:
                    self.kernels['meridional_normalize_kernel']((n,),(1,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.zmax),active,sed_grid,err))
                if it%self.meridional_solver_check_interval==0 or it==self.meridional_solver_max_iterations:
                    self.kernels['meridional_normalize_kernel']((n,),(1,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),np.float64(self.zmax),active,sed_grid,err))
                    self.kernels['meridional_residual_kernel']((n,),(1,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),active,ap,ae,aw,an,ass,sed_grid,residual))
                    cp.cuda.Stream.null.synchronize();ecode=int(cp.asnumpy(err)[0]);solver_residual=cp.asnumpy(residual);last_checked=it
                    if ecode: raise RuntimeError(f'CUDA V24.20 meridional normalization failed with error code {ecode}')
                    if it>=min_solver_iterations and float(np.max(solver_residual))<=self.meridional_solver_tolerance:
                        solver_iterations=it;break
            if solver_iterations==0: solver_iterations=last_checked or self.meridional_solver_max_iterations
            if float(np.max(solver_residual))>self.meridional_solver_tolerance:
                raise RuntimeError(f'V24.20 meridional transport solver did not converge: max residual={float(np.max(solver_residual)):.6g} after {solver_iterations} iterations')
            # Extend only for interpolation above the curved boundary; active-cell
            # values and mass are unchanged. Then rebuild majorants/point diagnostics.
            self.kernels['meridional_extend_grid_kernel'](((n*grid_nr+threads_solver-1)//threads_solver,),(threads_solver,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),active,sed_grid))
            self.kernels['meridional_finalize_kernel']((n,),(1,),(np.int32(n),np.int32(grid_nr),np.int32(grid_nz),np.float64(self.rin),np.float64(self.zmin),np.float64(grid_zmax),active,sed_grid,sed_norm,sed_max,sed_min,sed_sc,sed_sw,sed_bc,sed_bw))
            self.kernels['meridional_majorants_kernel']((1,),(1,),(np.int32(n),trace_geom_bin,trace_wave_bin,sed_max,sed_major))
            cp.cuda.Stream.null.synchronize()
        # Copy only diagnostic products; device fields/rates remain authoritative inputs to ray transport.
        keys={'diameter_m':dd,'input_source_weight':wd_input,'effective_aggregation_weight':wd,'source_weight':srcw,'radius_m':radius,'particle_mass_kg':pmass,'number_density_by_bin':nden,
              'n_real':nd,'k_imag':kd,'size_parameter':x,'qext':qext,'qsca':qsca,'qabs':qabs,'qback':qback,'asymmetry_g':g,'single_scattering_albedo':albedo,
              'geometric_cross_section_m2':sg,'scattering_cross_section_m2':ssca,'extinction_cross_section_m2':sext,'wave_cross_section_m2':swave,'event_cross_section_m2':sevent,
              'mu_geom_by_bin':mug,'mu_sca_by_bin':mus,'mu_ext_by_bin':mue,'mu_wave_by_bin':muw,'mu_event_by_bin':muev,
              'particle_event_weights':pew,'geometric_event_weights':gew,'wave_event_weights':wew,
              'sediment_response_time_s':sed_tau,'sediment_settling_velocity_m_s':sed_ws,'sediment_settling_reynolds':sed_sre,'sediment_settling_drag_factor':sed_sdrag,'sediment_radial_log_coefficient_A':sed_A,'sediment_vertical_inverse_length_B_per_m':sed_B,'sediment_radial_max_slip_velocity_m_s':sed_rmax,'sediment_radial_max_reynolds':sed_rre,'sediment_radial_center_log_factor':sed_rcenter,
              'sediment_normalization_scaled':sed_norm,'sediment_max_density_scale':sed_max,'sediment_min_density_scale':sed_min,'sediment_scale_sensor_center':sed_sc,'sediment_scale_sensor_wall_r0p95R':sed_sw,'sediment_scale_bottom_center':sed_bc,'sediment_scale_bottom_wall_r0p95R':sed_bw}
        host={k:cp.asnumpy(v) for k,v in keys.items()};mu_host=cp.asnumpy(muscal);agg_host=cp.asnumpy(agg_scalars)
        # V24.24 host-only diagnostic: integrate the already-CUDA-generated Mie
        # angular CDF over narrow near-backscatter bands.  This does not alter
        # event selection or ray directions; it exposes the existing phase law.
        if self.particle_event_model=='hybrid_mie_excess' and nt>2:
            theta_host=cp.asnumpy(theta)
            cdf_host=cp.asnumpy(phase_cdf).reshape(n,nt)
            def _cdf_at_deg(deg):
                q=math.radians(float(deg))
                return np.asarray([np.interp(q,theta_host,row) for row in cdf_host],dtype=np.float64)
            host['mie_phase_probability_165_175_by_bin']=np.maximum(_cdf_at_deg(175.0)-_cdf_at_deg(165.0),0.0)
            host['mie_phase_probability_169_171_by_bin']=np.maximum(_cdf_at_deg(171.0)-_cdf_at_deg(169.0),0.0)
        else:
            host['mie_phase_probability_165_175_by_bin']=np.zeros(n,dtype=np.float64)
            host['mie_phase_probability_169_171_by_bin']=np.zeros(n,dtype=np.float64)
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
        if self.particle_event_model in {'hybrid_mie_exact_rate','hybrid_mie_exact_rate_moment_constrained'}:
            host['mu_effective_geometric_event_by_bin']=np.maximum(host['mu_event_by_bin']-host['mu_wave_by_bin'],0.0)
            host['effective_geometric_event_cross_section_m2']=np.divide(host['mu_effective_geometric_event_by_bin'],host['number_density_by_bin'],out=np.zeros(n,dtype=np.float64),where=host['number_density_by_bin']>0)
            host['legacy_hybrid_mu_event_by_bin']=host['mu_geom_by_bin']+np.maximum(host['qsca']-1.0,0.0)*host['mu_geom_by_bin']
            host['legacy_hybrid_event_overcount_by_bin']=host['legacy_hybrid_mu_event_by_bin']-host['mu_sca_by_bin']
            host['qsca_lt1_mask']=(host['qsca']<1.0).astype(np.float64)
        else:
            host['mu_effective_geometric_event_by_bin']=host['mu_geom_by_bin'].copy()
            host['effective_geometric_event_cross_section_m2']=host['geometric_cross_section_m2'].copy()
            host['legacy_hybrid_mu_event_by_bin']=host['mu_event_by_bin'].copy()
            host['legacy_hybrid_event_overcount_by_bin']=np.zeros(n,dtype=np.float64)
            host['qsca_lt1_mask']=np.zeros(n,dtype=np.float64)
        host['moment_constraint_a_by_bin']=cp.asnumpy(moment_a);host['moment_constraint_b_by_bin']=cp.asnumpy(moment_b);host['moment_constraint_mode_by_bin']=cp.asnumpy(moment_mode).astype(np.int32)
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
                     'sediment_transport_quadrature_points':self.sediment_transport_quadrature_points,'sediment_radial_lookup_points':self.sediment_radial_lookup_points,'sediment_majorant_geom_per_m':float(sed_major_host[0]),
                     'sediment_majorant_wave_per_m':float(sed_major_host[1]),'sediment_majorant_total_per_m':float(sed_major_host[2]),'sediment_max_radial_log_coefficient':float(sed_major_host[3]),
                     'meridional_circulation_ratio':self.meridional_circulation_ratio,'meridional_vtheta_peak_m_s':self.meridional_vtheta_peak,'meridional_peak_speed_m_s':self.meridional_peak_speed,
                     'meridional_grid_nr':int(grid_nr),'meridional_grid_nz':int(grid_nz),'meridional_grid_zmax_m':float(grid_zmax),
                     'meridional_solver_iterations':int(solver_iterations),'meridional_solver_tolerance':self.meridional_solver_tolerance,
                     'meridional_solver_max_residual':float(np.max(solver_residual)) if solver_residual.size else 0.0,'meridional_solver_residual_by_bin':np.asarray(solver_residual,dtype=np.float64)})
        return {'d':d,'w':w,'nidx':nidx,'kidx':kidx,'n':n,'device':{'radius':radius,'n':nd,'k':kd,'gcdf':gcdf,'wcdf':wcdf,'phase_cdf':phase_cdf,'theta':theta,'qsca':qsca,'moment_a':moment_a,'moment_b':moment_b,'moment_mode':moment_mode,'mu_channels':muscal,
                'trace_geom_bin':trace_geom_bin,'trace_wave_bin':trace_wave_bin,'sed_A':sed_A,'sed_B':sed_B,'sed_norm':sed_norm,'sed_radial_log':sed_radial_log,'sed_majorants':sed_major,
                'sed_grid':sed_grid,'sed_grid_nr':int(grid_nr),'sed_grid_nz':int(grid_nz),'sed_grid_zmax':float(grid_zmax)},'host':host,'ntheta':nt}


    def single_particle_angular_audit(self,material,samples_per_bin=100000,seed=243500,concentration_g_per_L=1.0,weight_mode='mass_fraction'):
        """Diagnostic-only Monte-Carlo audit of the inherited per-event angular kernel.

        The fixed incident direction is +y. For a spherical/isotropic kernel this loses no
        generality. Returned samples contain cos(theta) relative to the incident direction,
        event mechanism codes, status codes, and the CUDA-generated exact-Mie phase table.
        trace_kernel is not invoked.
        """
        cp=self.cp;samples=int(samples_per_bin)
        if samples<1: raise ValueError('samples_per_bin must be >=1')
        if samples>2000000: raise ValueError('samples_per_bin must be <=2000000')
        prep=self._prepare_gpu_particle_physics(material,float(concentration_g_per_L),None,weight_mode)
        n=prep['n'];dev=prep['device'];total=n*samples
        ic=cp.empty(total,dtype=cp.float32);it=cp.zeros(total,dtype=cp.uint8);ist=cp.zeros(total,dtype=cp.uint8)
        pc=cp.empty(total,dtype=cp.float32);pt=cp.zeros(total,dtype=cp.uint8);pst=cp.zeros(total,dtype=cp.uint8)
        coh=np.float32(prep['host']['wave_coherent_fraction_nominal_cuda'])
        threads=128;blocks=(total+threads-1)//threads
        self.kernels['single_particle_angular_audit_kernel']((blocks,),(threads,),(
            np.int32(samples),np.int32(n),np.float32(self.n_water),np.float32(self.particle_wavelength),np.float32(self.particle_surface_beckmann_alpha),
            np.float32(self.wave_spheroid_aspect_ratio),np.int32(self.SPHEROID_ORIENTATION_MODEL_CODES[self.wave_spheroid_orientation_model]),np.float32(self.wave_spheroid_orientation_kappa),
            np.int32(self.max_internal_bounces),dev['n'],dev['k'],dev['radius'],dev['qsca'],dev['phase_cdf'],dev['theta'],np.int32(prep['ntheta']),coh,np.uint32(seed),
            ic,it,ist,pc,pt,pst))
        cp.cuda.Stream.null.synchronize()
        theta=cp.asnumpy(dev['theta']);cdf=cp.asnumpy(dev['phase_cdf']).reshape(n,prep['ntheta'])
        return {'material':material,'samples_per_bin':samples,'host':prep['host'],'theta':theta,'mie_cdf':cdf,
                'interface_cos':cp.asnumpy(ic).reshape(n,samples),'interface_type':cp.asnumpy(it).reshape(n,samples),'interface_status':cp.asnumpy(ist).reshape(n,samples),
                'production_cos':cp.asnumpy(pc).reshape(n,samples),'production_type':cp.asnumpy(pt).reshape(n,samples),'production_status':cp.asnumpy(pst).reshape(n,samples)}


    def single_particle_moment_constrained_audit(self,material,samples_per_bin=50000,seed=243600,concentration_g_per_L=1.0,weight_mode='mass_fraction'):
        """Independent CUDA audit of the V24.36 post-outcome map using the same base V24.29 sampler."""
        cp=self.cp;samples=int(samples_per_bin)
        if self.particle_event_model!='hybrid_mie_exact_rate_moment_constrained': raise RuntimeError('constrained audit requires V24.36 moment-constrained event model')
        if samples<1 or samples>2000000: raise ValueError('samples_per_bin out of range')
        prep=self._prepare_gpu_particle_physics(material,float(concentration_g_per_L),None,weight_mode)
        n=prep['n'];dev=prep['device'];total=n*samples
        bc=cp.empty(total,dtype=cp.float32);cc=cp.empty(total,dtype=cp.float32);typ=cp.zeros(total,dtype=cp.uint8);st=cp.zeros(total,dtype=cp.uint8)
        coh=np.float32(prep['host']['wave_coherent_fraction_nominal_cuda']);threads=128;blocks=(total+threads-1)//threads
        self.kernels['single_particle_moment_constrained_audit_kernel']((blocks,),(threads,),(
            np.int32(samples),np.int32(n),np.float32(self.n_water),np.float32(self.particle_wavelength),np.float32(self.particle_surface_beckmann_alpha),
            np.float32(self.wave_spheroid_aspect_ratio),np.int32(self.SPHEROID_ORIENTATION_MODEL_CODES[self.wave_spheroid_orientation_model]),np.float32(self.wave_spheroid_orientation_kappa),
            np.int32(self.max_internal_bounces),dev['n'],dev['k'],dev['radius'],dev['qsca'],dev['phase_cdf'],dev['theta'],np.int32(prep['ntheta']),coh,np.uint32(seed),
            dev['moment_a'],dev['moment_b'],dev['moment_mode'],bc,cc,typ,st))
        cp.cuda.Stream.null.synchronize();theta=cp.asnumpy(dev['theta']);cdf=cp.asnumpy(dev['phase_cdf']).reshape(n,prep['ntheta'])
        return {'material':material,'samples_per_bin':samples,'host':prep['host'],'theta':theta,'mie_cdf':cdf,
                'base_cos':cp.asnumpy(bc).reshape(n,samples),'constrained_cos':cp.asnumpy(cc).reshape(n,samples),'event_type':cp.asnumpy(typ).reshape(n,samples),'status':cp.asnumpy(st).reshape(n,samples)}

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
                 target_detector_score=None,min_rays=None,max_rays=None,min_rear_channel_score=None,min_h170_score=None,
                 stability_l1_tolerance=None,stability_window=3):
        cp=self.cp;prep=self._prepare_gpu_particle_physics(material,concentration_g_per_L,weights,weight_mode);ph=prep['host'];dev=prep['device'];npart=prep['n'];ntheta=prep['ntheta']
        requested=int(n_rays);adaptive=target_detector_score is not None
        if adaptive:
            target=float(target_detector_score)
            if target<=0: raise ValueError('target_detector_score must be >0')
            rear_target=None if min_rear_channel_score is None else float(min_rear_channel_score)
            h170_target=None if min_h170_score is None else float(min_h170_score)
            if rear_target is not None and rear_target<=0: raise ValueError('min_rear_channel_score must be >0')
            if h170_target is not None and h170_target<=0: raise ValueError('min_h170_score must be >0')
            maxN=int(max_rays if max_rays is not None else max(requested,self.statistics_batch_rays));minN=int(min_rays if min_rays is not None else min(self.statistics_batch_rays,maxN))
            if minN<0 or maxN<minN: raise ValueError('invalid min_rays/max_rays')
        else: target=None;rear_target=None;h170_target=None;maxN=requested;minN=requested
        if maxN<1: raise ValueError('number of rays must be positive')
        vis=int(heatmap_size);hxy=cp.zeros(max(1,vis*vis),dtype=cp.float32);hzy=cp.zeros(max(1,vis*vis),dtype=cp.float32)
        seed0=np.uint32((int(seed)*1664525+1013904223)&0x7fffffff or 1);seed1=np.uint32((int(seed)*22695477+1)&0x7fffffff or 7);seed2=np.uint32((int(seed)*1103515245+12345)&0x7fffffff or 11)
        threads=256
        native_total=np.zeros(18,dtype=np.int64);mirror_total=np.zeros(18,dtype=np.int64);sym_total=np.zeros(18,dtype=np.float64)
        native_direct_total=np.zeros(18,dtype=np.int64);mirror_direct_total=np.zeros(18,dtype=np.int64);native_stray_total=np.zeros(18,dtype=np.int64);mirror_stray_total=np.zeros(18,dtype=np.int64);native_face_total=np.zeros(18,dtype=np.int64);mirror_face_total=np.zeros(18,dtype=np.int64)
        photodiode_native_total=np.zeros(18,dtype=np.float64);photodiode_mirror_total=np.zeros(18,dtype=np.float64);incidence_native_sum_total=np.zeros(18,dtype=np.float64);incidence_mirror_sum_total=np.zeros(18,dtype=np.float64)
        legacy_native_total=np.zeros(18,dtype=np.int64);legacy_mirror_total=np.zeros(18,dtype=np.int64);legacy_sym_total=np.zeros(18,dtype=np.float64)
        batch_scores=[];hardware_batch_scores=[];photodiode_batch_scores=[];legacy_batch_scores=[];batch_rows=[];prev_cum=None;recent_l1=[]
        total_rays=0;valid_air_total=0;numerical_failure_total=0;entered_total=0;source_accept_total=0;source_direct_accept_total=0;source_reflected_accept_total=0;source_bore_interactions_total=0;source_bore_reflections_total=0;axial_total=0;top_exit_total=0;bottom_exit_total=0
        sum_ic=sum_gic=sum_wic=sum_wcoh=sum_wmorph=0.0;ballistic_total=0;ic_hist=np.zeros(1,dtype=np.int64)
        sum_pr=0.0;rays_entry_pr=rays_internal_pr=particle_absorbed_total=0;sum_cir=sum_cor=0.0;rays_ctir=0
        sum_top_ref=sum_bottom_inner=sum_bottom_outer=0.0;rays_top_tir=rays_bottom_tir=0
        # V24.29: retained V24.27 H170/event-history decomposition from GPU-emitted primitive
        # counters.  No extra CUDA outputs are added and no ray probability is changed.
        # Interaction-order and branch-identity partitions are explicitly checked for closure.
        path_keys=['any_particle','any_geometric','any_wave','any_coherent_wave','any_morphology_wave',
                   'any_entry_reflection','any_internal_reflection','ballistic','single_interaction',
                   'single_entry_reflection','multiple_interaction','interaction_order_2','interaction_order_3',
                   'interaction_order_4plus','exactly_one_geometric','exactly_one_wave','single_coherent_wave_escape',
                   'single_external_entry_reflection_escape','single_internal_reflection_escape',
                   'single_surface_transmission_escape','mixed_geometric_wave','geometric_only_multiple',
                   'wave_only_multiple','multiple_with_entry_reflection','multiple_with_internal_reflection']
        path_scores={k:np.zeros(18,dtype=np.float64) for k in path_keys}
        path_interaction_sums=np.zeros(18,dtype=np.float64)
        path_geometric_interaction_sums=np.zeros(18,dtype=np.float64)
        path_wave_interaction_sums=np.zeros(18,dtype=np.float64)
        path_coherent_wave_interaction_sums=np.zeros(18,dtype=np.float64)
        path_morphology_wave_interaction_sums=np.zeros(18,dtype=np.float64)
        path_entry_reflection_sums=np.zeros(18,dtype=np.float64)
        path_internal_reflection_sums=np.zeros(18,dtype=np.float64)
        # V24.29 retained V24.27 causal final-event/path-length diagnostics. Arrays are accumulated
        # online and never modify ray weights or transport decisions.
        causal_event_codes=tuple(sorted(k for k in V2427_PARTICLE_EVENT_TYPE_NAMES if k>0))
        causal_last_event_scores={code:np.zeros(18,dtype=np.float64) for code in causal_event_codes}
        causal_first_event_scores={code:np.zeros(18,dtype=np.float64) for code in causal_event_codes}
        causal_last_event_angle_sums={code:np.zeros(18,dtype=np.float64) for code in causal_event_codes}
        causal_last_event_angle_counts={code:np.zeros(18,dtype=np.float64) for code in causal_event_codes}
        causal_first_backscatter_scores=np.zeros(18,dtype=np.float64)
        causal_first_source_hemi_scores=np.zeros(18,dtype=np.float64)
        causal_first_angle_sums=np.zeros(18,dtype=np.float64);causal_first_angle_counts=np.zeros(18,dtype=np.float64)
        causal_last_angle_sums=np.zeros(18,dtype=np.float64);causal_last_angle_counts=np.zeros(18,dtype=np.float64)
        causal_internal_path_sums=np.zeros(18,dtype=np.float64);causal_internal_opl_sums=np.zeros(18,dtype=np.float64)
        causal_post_last_water_path_sums=np.zeros(18,dtype=np.float64);causal_path_counts=np.zeros(18,dtype=np.float64)
        causal_last_order_sums=np.zeros(18,dtype=np.float64)
        causal_h170_angle_edges=np.asarray([0.0,30.0,60.0,90.0,120.0,150.0,180.0001],dtype=np.float64)
        causal_h170_last_angle_hist=np.zeros((8,6),dtype=np.float64)
        causal_h170_first_angle_hist=np.zeros(6,dtype=np.float64)
        ray_chunks=[] if collect_rays else None;stop_reason='fixed_n_rays'
        production_failure_code_counts=np.zeros(max(PRODUCTION_FAILURE_NAMES)+1,dtype=np.int64);production_failure_subcode_counts={};production_failure_samples=[];production_failure_sample_limit=512;particle_proposal_total=0;particle_accessible_total=0;particle_null_total=0
        while total_rays<maxN:
            batchN=min(self.statistics_batch_rays,maxN-total_rays)
            batch_native=np.zeros(18,dtype=np.int64);batch_mirror=np.zeros(18,dtype=np.int64);batch_native_direct=np.zeros(18,dtype=np.int64);batch_mirror_direct=np.zeros(18,dtype=np.int64);batch_native_stray=np.zeros(18,dtype=np.int64);batch_mirror_stray=np.zeros(18,dtype=np.int64);batch_native_face=np.zeros(18,dtype=np.int64);batch_mirror_face=np.zeros(18,dtype=np.int64)
            batch_photodiode_native=np.zeros(18,dtype=np.float64);batch_photodiode_mirror=np.zeros(18,dtype=np.float64);batch_incidence_native_sum=np.zeros(18,dtype=np.float64);batch_incidence_mirror_sum=np.zeros(18,dtype=np.float64)
            batch_legacy_native=np.zeros(18,dtype=np.int64);batch_legacy_mirror=np.zeros(18,dtype=np.int64)
            batch_start=total_rays
            for local_start in range(0,batchN,self.chunk_size):
                local_end=min(batchN,local_start+self.chunk_size);n=local_end-local_start;global_start=batch_start+local_start
                outf=[cp.full(n,cp.nan,dtype=cp.float32) for _ in range(47)];outi=[cp.zeros(n,dtype=cp.int32) for _ in range(40)]
                blocks=(n+threads-1)//threads
                self.kernel((blocks,),(threads,),(
                    np.float32(self.max_iterations),dev['mu_channels'],np.float32(self.n_water),np.float32(self.n_acrylic),np.float32(self.n_air),np.float32(self.particle_wavelength),np.float32(self.particle_surface_beckmann_alpha),np.float32(self.wave_spheroid_aspect_ratio),
                    np.int32(self.SPHEROID_ORIENTATION_MODEL_CODES[self.wave_spheroid_orientation_model]),np.float32(self.wave_spheroid_orientation_kappa),
                    np.int32(self.max_internal_bounces),np.int32(self.max_cell_bounces),np.float32(self.rin),np.float32(self.rout),np.float32(self.zmin),np.float32(self.zmax),np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),np.float32(self.vortex_delta_h),np.float32(self.vortex_core_radius),np.float32(self.bottom_disc_thickness),np.float32(self.bottom_external_n),np.int32(self.max_axial_bounces),
                    np.float32(self.ring_in),np.float32(self.throat_out),np.float32(self.ring_out),np.float32(self.source_launch_radius),np.float32(self.throat_radius),np.float32(self.counterbore_radius),np.float32(self.beam_sigma),np.int32(vis),np.float32(self.alpha1),np.float32(self.alpha2),np.int32(self.SOURCE_ANGULAR_MODEL_CODES[self.source_angular_model]),np.int32(self.BORE_SURFACE_MODEL_CODES[self.source_bore_surface_model]),np.float32(self.bore_refractive_index),np.int32(self.max_bore_bounces),np.int32(n),
                    dev['gcdf'],dev['wcdf'],dev['phase_cdf'],dev['theta'],np.int32(ntheta),dev['n'],dev['k'],dev['radius'],np.int32(npart),dev['moment_a'],dev['moment_b'],dev['moment_mode'],
                    np.int32(self.SEDIMENT_TRANSPORT_MODEL_CODES[self.sediment_transport_model]),dev['trace_geom_bin'],dev['trace_wave_bin'],dev['sed_A'],dev['sed_B'],dev['sed_norm'],dev['sed_radial_log'],np.int32(self.sediment_radial_lookup_points),dev['sed_majorants'],
                    dev['sed_grid'],np.int32(dev['sed_grid_nr']),np.int32(dev['sed_grid_nz']),np.float32(dev['sed_grid_zmax']),
                    hxy,hzy,*outf,*outi,seed0,seed1,seed2,np.uint32(global_start)))
                # Detector geometry/scoring is CUDA-resident. Only counts/indices are copied back.
                hnat=cp.zeros(18,dtype=cp.int32);hmir=cp.zeros(18,dtype=cp.int32);hnd=cp.zeros(18,dtype=cp.int32);hmd=cp.zeros(18,dtype=cp.int32);hns=cp.zeros(18,dtype=cp.int32);hms=cp.zeros(18,dtype=cp.int32);hnf=cp.zeros(18,dtype=cp.int32);hmf=cp.zeros(18,dtype=cp.int32)
                pdnat=cp.zeros(18,dtype=cp.float32);pdmir=cp.zeros(18,dtype=cp.float32);iangn=cp.zeros(18,dtype=cp.float32);iangm=cp.zeros(18,dtype=cp.float32)
                lnat=cp.zeros(18,dtype=cp.int32);lmir=cp.zeros(18,dtype=cp.int32);nid=cp.empty(n,dtype=cp.int32);mid=cp.empty(n,dtype=cp.int32);nbc=cp.empty(n,dtype=cp.int32);mbc=cp.empty(n,dtype=cp.int32);nfc=cp.empty(n,dtype=cp.int32);mfc=cp.empty(n,dtype=cp.int32)
                detector_seed=np.uint32((int(seed)*2654435761+97531)&0x7fffffff or 17)
                self.kernels['score_detectors_kernel']((blocks,),(threads,),(outf[0],outf[1],outf[2],outf[3],outf[4],outf[5],outi[20],outi[0],np.int32(n),np.float32(self.rout),np.float32(self.ring_in),np.float32(self.throat_out),np.float32(self.ring_out),np.float32(self.throat_radius),np.float32(self.counterbore_radius),np.float32(self.legacy_acceptance_deg),np.int32(self.BORE_SURFACE_MODEL_CODES[self.detector_bore_surface_model]),np.float32(self.n_air),np.float32(self.bore_refractive_index),np.int32(self.max_bore_bounces),np.int32(self.RING_FACE_SURFACE_MODEL_CODES[self.ring_inner_face_surface_model]),np.float32(self.ring_inner_face_refractive_index),np.float32(self.n_acrylic),np.int32(self.max_ring_face_bounces),np.int32(self.PHOTODIODE_RESPONSE_MODEL_CODES[self.photodiode_response_model]),detector_seed,np.uint32(global_start),hnat,hmir,hnd,hmd,hns,hms,hnf,hmf,pdnat,pdmir,iangn,iangm,lnat,lmir,nid,mid,nbc,mbc,nfc,mfc))
                cp.cuda.Stream.null.synchronize()
                batch_native+=cp.asnumpy(hnat).astype(np.int64);batch_mirror+=cp.asnumpy(hmir).astype(np.int64);batch_native_direct+=cp.asnumpy(hnd).astype(np.int64);batch_mirror_direct+=cp.asnumpy(hmd).astype(np.int64);batch_native_stray+=cp.asnumpy(hns).astype(np.int64);batch_mirror_stray+=cp.asnumpy(hms).astype(np.int64);batch_native_face+=cp.asnumpy(hnf).astype(np.int64);batch_mirror_face+=cp.asnumpy(hmf).astype(np.int64)
                batch_photodiode_native+=cp.asnumpy(pdnat).astype(np.float64);batch_photodiode_mirror+=cp.asnumpy(pdmir).astype(np.float64);batch_incidence_native_sum+=cp.asnumpy(iangn).astype(np.float64);batch_incidence_mirror_sum+=cp.asnumpy(iangm).astype(np.float64)
                batch_legacy_native+=cp.asnumpy(lnat).astype(np.int64);batch_legacy_mirror+=cp.asnumpy(lmir).astype(np.int64)
                af=[cp.asnumpy(a) for a in outf];ai=[cp.asnumpy(a) for a in outi]
                arrays=dict(zip(['air_x','air_y','air_z','air_vx','air_vy','air_vz','water_path','acrylic_path','first_particle_deflection_deg','last_particle_deflection_deg','internal_particle_path_total_m','internal_particle_opl_total_m','water_path_after_last_particle_event_m','failure_x','failure_y','failure_z','failure_vx','failure_vy','failure_vz','particle_diag_radius','particle_diag_rho','particle_diag_phi','particle_diag_cx','particle_diag_cy','particle_diag_cz','particle_diag_entry_x','particle_diag_entry_y','particle_diag_entry_z','particle_diag_exit_x','particle_diag_exit_y','particle_diag_exit_z','particle_diag_pre_x','particle_diag_pre_y','particle_diag_pre_z','particle_diag_pre_vx','particle_diag_pre_vy','particle_diag_pre_vz','particle_diag_post_x','particle_diag_post_y','particle_diag_post_z','particle_diag_post_vx','particle_diag_post_vy','particle_diag_post_vz','particle_diag_center_radial_clearance','particle_diag_finite_wall_clearance','particle_diag_bottom_clearance','particle_diag_top_clearance'],af));ints=dict(zip(['interaction_count','geometric_interactions','wave_interactions','coherent_wave_interactions','morphology_wave_interactions','particle_reflections','particle_entry_reflections','particle_internal_reflections','particle_absorptions','cell_inner_reflections','cell_outer_reflections','cell_tir','top_reflections','top_tir','bottom_inner_reflections','bottom_outer_reflections','bottom_tir','source_bore_interactions','source_bore_reflections','entered_water','status','production_failure_code','production_failure_subcode','production_failure_trace_iteration','production_failure_macro_bounces','production_failure_micro_order','production_failure_medium','production_failure_event_channel','particle_diag_bin','particle_diag_previous_event','particle_diag_overlap_class','particle_diag_proposal_accepted','particle_proposal_count','particle_accessible_count','particle_null_count','first_particle_event_type','last_particle_event_type','last_particle_event_order','first_particle_backscatter','first_particle_source_hemisphere'],ai))
                particle_proposal_total+=int(np.asarray(ints['particle_proposal_count'],dtype=np.int64).sum());particle_accessible_total+=int(np.asarray(ints['particle_accessible_count'],dtype=np.int64).sum());particle_null_total+=int(np.asarray(ints['particle_null_count'],dtype=np.int64).sum())
                nid_h=cp.asnumpy(nid);mid_h=cp.asnumpy(mid)
                def _det_hist(ids,mask,weights=None):
                    vv=(ids>=0)&(ids<18)&mask
                    if not np.any(vv): return np.zeros(18,dtype=np.float64)
                    ww=None if weights is None else np.asarray(weights,dtype=np.float64)[vv]
                    return np.bincount(ids[vv].astype(np.int64),weights=ww,minlength=18).astype(np.float64)
                ic_all=ints['interaction_count']; per_all=ints['particle_entry_reflections'];pir_all=ints['particle_internal_reflections']
                conds=_v2427_event_history_masks(ints)
                for pk,pmask in conds.items():
                    nn=_det_hist(nid_h,pmask);mm=_det_hist(mid_h,pmask)
                    path_scores[pk]+=0.5*(nn+mm) if self.symmetry_average_exact else nn
                anymask=np.ones(n,dtype=bool)
                for vals,accumulator in [(ic_all,path_interaction_sums),(ints['geometric_interactions'],path_geometric_interaction_sums),
                                    (ints['wave_interactions'],path_wave_interaction_sums),(ints['coherent_wave_interactions'],path_coherent_wave_interaction_sums),
                                    (ints['morphology_wave_interactions'],path_morphology_wave_interaction_sums),(per_all,path_entry_reflection_sums),
                                    (pir_all,path_internal_reflection_sums)]:
                    nn=_det_hist(nid_h,anymask,vals);mm=_det_hist(mid_h,anymask,vals)
                    accumulator+=0.5*(nn+mm) if self.symmetry_average_exact else nn
                # V24.29 retained V24.27 causal aggregation. Only successfully detected rays contribute;
                # native/mirror symmetry weighting is identical to the production estimator.
                first_type=ints['first_particle_event_type'];last_type=ints['last_particle_event_type']
                first_angle=arrays['first_particle_deflection_deg'];last_angle=arrays['last_particle_deflection_deg']
                valid_particle=last_type>0
                def _sym_hist(mask,weights=None):
                    nn=_det_hist(nid_h,mask,weights);mm=_det_hist(mid_h,mask,weights)
                    return 0.5*(nn+mm) if self.symmetry_average_exact else nn
                for code in causal_event_codes:
                    lm=last_type==code;fm=first_type==code
                    causal_last_event_scores[code]+=_sym_hist(lm)
                    causal_first_event_scores[code]+=_sym_hist(fm)
                    lav=lm&np.isfinite(last_angle)&(last_angle>=0.0)
                    causal_last_event_angle_sums[code]+=_sym_hist(lav,last_angle)
                    causal_last_event_angle_counts[code]+=_sym_hist(lav)
                causal_first_backscatter_scores+=_sym_hist(valid_particle&(ints['first_particle_backscatter']>0))
                causal_first_source_hemi_scores+=_sym_hist(valid_particle&(ints['first_particle_source_hemisphere']>0))
                fav=valid_particle&np.isfinite(first_angle)&(first_angle>=0.0);lav=valid_particle&np.isfinite(last_angle)&(last_angle>=0.0)
                causal_first_angle_sums+=_sym_hist(fav,first_angle);causal_first_angle_counts+=_sym_hist(fav)
                causal_last_angle_sums+=_sym_hist(lav,last_angle);causal_last_angle_counts+=_sym_hist(lav)
                path_valid=valid_particle&np.isfinite(arrays['internal_particle_path_total_m'])&np.isfinite(arrays['internal_particle_opl_total_m'])&np.isfinite(arrays['water_path_after_last_particle_event_m'])&(arrays['water_path_after_last_particle_event_m']>=0.0)
                causal_internal_path_sums+=_sym_hist(path_valid,arrays['internal_particle_path_total_m'])
                causal_internal_opl_sums+=_sym_hist(path_valid,arrays['internal_particle_opl_total_m'])
                causal_post_last_water_path_sums+=_sym_hist(path_valid,arrays['water_path_after_last_particle_event_m'])
                causal_last_order_sums+=_sym_hist(path_valid,ints['last_particle_event_order'])
                causal_path_counts+=_sym_hist(path_valid)
                # Compact H170 angle histograms by final-event type for causal interpretation.
                for b in range(len(causal_h170_angle_edges)-1):
                    lo=causal_h170_angle_edges[b];hi=causal_h170_angle_edges[b+1]
                    am=lav&(last_angle>=lo)&(last_angle<hi)
                    h=_sym_hist(am);causal_h170_last_angle_hist[0,b]+=h[17]
                    for code in causal_event_codes:
                        h=_sym_hist(am&(last_type==code));causal_h170_last_angle_hist[code,b]+=h[17]
                    fh=_sym_hist(fav&(first_angle>=lo)&(first_angle<hi));causal_h170_first_angle_hist[b]+=fh[17]
                pfcode=np.asarray(ints['production_failure_code'],dtype=np.int32);pfmask=pfcode!=0
                if np.any(pfmask):
                    bc=np.bincount(pfcode[pfmask].astype(np.int64),minlength=len(production_failure_code_counts));production_failure_code_counts[:len(bc)]+=bc[:len(production_failure_code_counts)]
                    for sub,count in zip(*np.unique(np.asarray(ints['production_failure_subcode'],dtype=np.int32)[pfmask],return_counts=True)):
                        production_failure_subcode_counts[int(sub)]=production_failure_subcode_counts.get(int(sub),0)+int(count)
                    room=production_failure_sample_limit-len(production_failure_samples)
                    if room>0:
                        loc=np.flatnonzero(pfmask)[:room]
                        for jj in loc:
                            code=int(pfcode[jj]);sub=int(ints['production_failure_subcode'][jj])
                            production_failure_samples.append({'ray_id':int(global_start+jj),'failure_code':code,'failure_name':PRODUCTION_FAILURE_NAMES.get(code,f'CODE_{code}'),'failure_subcode':sub,'failure_subcode_name':PRODUCTION_FAILURE_SUBCODE_NAMES.get(sub,f'SUBCODE_{sub}'),'trace_iteration':int(ints['production_failure_trace_iteration'][jj]),'macro_bounces':int(ints['production_failure_macro_bounces'][jj]),'micro_order':int(ints['production_failure_micro_order'][jj]),'medium_code':int(ints['production_failure_medium'][jj]),'medium_name':PRODUCTION_MEDIUM_NAMES.get(int(ints['production_failure_medium'][jj]),'unknown'),'event_channel_code':int(ints['production_failure_event_channel'][jj]),'event_channel_name':PRODUCTION_EVENT_CHANNEL_NAMES.get(int(ints['production_failure_event_channel'][jj]),'unknown'),'x':float(arrays['failure_x'][jj]),'y':float(arrays['failure_y'][jj]),'z':float(arrays['failure_z'][jj]),'vx':float(arrays['failure_vx'][jj]),'vy':float(arrays['failure_vy'][jj]),'vz':float(arrays['failure_vz'][jj]),'previous_event_type':int(ints['particle_diag_previous_event'][jj]),'previous_particle_event':bool(int(ints['particle_diag_bin'][jj])>=0),'particle_bin':int(ints['particle_diag_bin'][jj]),'particle_radius_m':float(arrays['particle_diag_radius'][jj]),'particle_diameter_m':2.0*float(arrays['particle_diag_radius'][jj]),'sampled_impact_parameter_rho':float(arrays['particle_diag_rho'][jj]),'sampled_phi_rad':float(arrays['particle_diag_phi'][jj]),'particle_center_x':float(arrays['particle_diag_cx'][jj]),'particle_center_y':float(arrays['particle_diag_cy'][jj]),'particle_center_z':float(arrays['particle_diag_cz'][jj]),'particle_entry_x':float(arrays['particle_diag_entry_x'][jj]),'particle_entry_y':float(arrays['particle_diag_entry_y'][jj]),'particle_entry_z':float(arrays['particle_diag_entry_z'][jj]),'particle_exit_x':float(arrays['particle_diag_exit_x'][jj]),'particle_exit_y':float(arrays['particle_diag_exit_y'][jj]),'particle_exit_z':float(arrays['particle_diag_exit_z'][jj]),'pre_particle_x':float(arrays['particle_diag_pre_x'][jj]),'pre_particle_y':float(arrays['particle_diag_pre_y'][jj]),'pre_particle_z':float(arrays['particle_diag_pre_z'][jj]),'pre_particle_vx':float(arrays['particle_diag_pre_vx'][jj]),'pre_particle_vy':float(arrays['particle_diag_pre_vy'][jj]),'pre_particle_vz':float(arrays['particle_diag_pre_vz'][jj]),'post_particle_x':float(arrays['particle_diag_post_x'][jj]),'post_particle_y':float(arrays['particle_diag_post_y'][jj]),'post_particle_z':float(arrays['particle_diag_post_z'][jj]),'post_particle_vx':float(arrays['particle_diag_post_vx'][jj]),'post_particle_vy':float(arrays['particle_diag_post_vy'][jj]),'post_particle_vz':float(arrays['particle_diag_post_vz'][jj]),'particle_center_radial_clearance_m':float(arrays['particle_diag_center_radial_clearance'][jj]),'particle_finite_wall_clearance_m':float(arrays['particle_diag_finite_wall_clearance'][jj]),'particle_bottom_clearance_m':float(arrays['particle_diag_bottom_clearance'][jj]),'particle_top_clearance_m':float(arrays['particle_diag_top_clearance'][jj]),'particle_overlap_class':int(ints['particle_diag_overlap_class'][jj]),'particle_proposal_accepted':int(ints['particle_diag_proposal_accepted'][jj])})
                generic_status4=int(np.count_nonzero(ints['status']==4));specific_failures=int(np.count_nonzero(pfmask))
                if generic_status4!=specific_failures: raise RuntimeError(f'V24.46 production failure-code closure mismatch in chunk: status4={generic_status4}, specific={specific_failures}')
                valid_air=(ints['status']==1)&np.isfinite(arrays['air_x'])&np.isfinite(arrays['air_vx']);entered=ints['entered_water']>0;ic=ints['interaction_count'][entered]
                valid_air_total+=int(valid_air.sum());numerical_failure_total+=specific_failures;entered_total+=int(entered.sum());srcok=(ints['status']!=3);source_accept_total+=int(np.count_nonzero(srcok));source_direct_accept_total+=int(np.count_nonzero(srcok & (ints['source_bore_reflections']==0)));source_reflected_accept_total+=int(np.count_nonzero(srcok & (ints['source_bore_reflections']>0)));source_bore_interactions_total+=int(ints['source_bore_interactions'].sum());source_bore_reflections_total+=int(ints['source_bore_reflections'].sum());top_exit_total+=int(np.count_nonzero(ints['status']==6));bottom_exit_total+=int(np.count_nonzero(ints['status']==7));axial_total+=int(np.count_nonzero((ints['status']==6)|(ints['status']==7)))
                if ic.size:
                    sum_ic+=float(ic.sum());sum_gic+=float(ints['geometric_interactions'][entered].sum());sum_wic+=float(ints['wave_interactions'][entered].sum());sum_wcoh+=float(ints['coherent_wave_interactions'][entered].sum());sum_wmorph+=float(ints['morphology_wave_interactions'][entered].sum());ballistic_total+=int(np.count_nonzero(ic==0))
                    bc=np.bincount(ic.astype(np.int64));
                    if bc.size>ic_hist.size: ic_hist=np.pad(ic_hist,(0,bc.size-ic_hist.size))
                    ic_hist[:bc.size]+=bc;pr=ints['particle_reflections'][entered];per=ints['particle_entry_reflections'][entered];pir=ints['particle_internal_reflections'][entered]
                    particle_absorbed_total+=int(np.count_nonzero(ints['particle_absorptions'][entered]>0));cir=ints['cell_inner_reflections'][entered];cor=ints['cell_outer_reflections'][entered];ct=ints['cell_tir'][entered]
                    sum_pr+=float(pr.sum());rays_entry_pr+=int(np.count_nonzero(per>0));rays_internal_pr+=int(np.count_nonzero(pir>0));sum_cir+=float(cir.sum());sum_cor+=float(cor.sum());rays_ctir+=int(np.count_nonzero(ct>0));sum_top_ref+=float(ints['top_reflections'][entered].sum());rays_top_tir+=int(np.count_nonzero(ints['top_tir'][entered]>0));sum_bottom_inner+=float(ints['bottom_inner_reflections'][entered].sum());sum_bottom_outer+=float(ints['bottom_outer_reflections'][entered].sum());rays_bottom_tir+=int(np.count_nonzero(ints['bottom_tir'][entered]>0))
                if collect_rays:
                    ray_chunks.append({**arrays,**ints,'native_detector_index':nid_h,'mirror_detector_index':mid_h,'native_detector_bore_reflections':cp.asnumpy(nbc),'mirror_detector_bore_reflections':cp.asnumpy(mbc),'native_ring_face_reflections':cp.asnumpy(nfc),'mirror_ring_face_reflections':cp.asnumpy(mfc)})
                del outf,outi,af,ai,hnat,hmir,hnd,hmd,hns,hms,hnf,hmf,pdnat,pdmir,iangn,iangm,lnat,lmir,nid,mid,nbc,mbc,nfc,mfc
            total_rays+=batchN;native_total+=batch_native;mirror_total+=batch_mirror;native_direct_total+=batch_native_direct;mirror_direct_total+=batch_mirror_direct;native_stray_total+=batch_native_stray;mirror_stray_total+=batch_mirror_stray;native_face_total+=batch_native_face;mirror_face_total+=batch_mirror_face
            photodiode_native_total+=batch_photodiode_native;photodiode_mirror_total+=batch_photodiode_mirror;incidence_native_sum_total+=batch_incidence_native_sum;incidence_mirror_sum_total+=batch_incidence_mirror_sum
            legacy_native_total+=batch_legacy_native;legacy_mirror_total+=batch_legacy_mirror
            batch_hardware_sym=0.5*(batch_native.astype(float)+batch_mirror.astype(float)) if self.symmetry_average_exact else batch_native.astype(float)
            batch_photodiode_sym=0.5*(batch_photodiode_native+batch_photodiode_mirror) if self.symmetry_average_exact else batch_photodiode_native.copy()
            batch_sym=batch_hardware_sym if self.photodiode_response_model=='ideal_hard_aperture' else batch_photodiode_sym
            batch_legacy_sym=0.5*(batch_legacy_native.astype(float)+batch_legacy_mirror.astype(float)) if self.symmetry_average_exact else batch_legacy_native.astype(float)
            sym_total+=batch_sym;legacy_sym_total+=batch_legacy_sym;batch_scores.append(batch_sym.copy());hardware_batch_scores.append(batch_hardware_sym.copy());photodiode_batch_scores.append(batch_photodiode_sym.copy());legacy_batch_scores.append(batch_legacy_sym.copy())
            den=float(sym_total.sum());cum=sym_total/den if den>0 else np.zeros_like(sym_total);l1=float(np.sum(np.abs(cum-prev_cum))) if prev_cum is not None else float('nan')
            if np.isfinite(l1):recent_l1.append(l1)
            prev_cum=cum.copy();rear_min=float(np.min(sym_total[15:18]));batch_rows.append({'batch':len(batch_rows)+1,'batch_rays':batchN,'cumulative_rays':total_rays,'batch_native_hits':int(batch_native.sum()),'batch_mirror_hits':int(batch_mirror.sum()),'batch_hardware_candidate_score':float(batch_hardware_sym.sum()),'batch_photodiode_weighted_score':float(batch_photodiode_sym.sum()),'batch_selected_score':float(batch_sym.sum()),'cumulative_selected_score':float(sym_total.sum()),'cumulative_H150_score':float(sym_total[15]),'cumulative_H160_score':float(sym_total[16]),'cumulative_H170_score':float(sym_total[17]),'cumulative_min_rear_score_H150_H170':rear_min,'batch_legacy_channel_score':float(batch_legacy_sym.sum()),'cumulative_legacy_channel_score':float(legacy_sym_total.sum()),'cumulative_l1_change':l1})
            if adaptive and total_rays>=minN and sym_total.sum()>=target:
                rear_ok=(rear_target is None) or (rear_min>=rear_target)
                h170_ok=(h170_target is None) or (float(sym_total[17])>=h170_target)
                stable=True
                if stability_l1_tolerance is not None:
                    wdw=max(1,int(stability_window));stable=len(recent_l1)>=wdw and all(q<=float(stability_l1_tolerance) for q in recent_l1[-wdw:])
                if rear_ok and h170_ok and stable:
                    bits=['target_detector_score']
                    if rear_target is not None:bits.append('rear_channels')
                    if h170_target is not None:bits.append('H170')
                    if stability_l1_tolerance is not None:bits.append('stability')
                    stop_reason='_and_'.join(bits);break
        if adaptive and total_rays>=maxN:
            rear_min=float(np.min(sym_total[15:18]))
            if sym_total.sum()<target:stop_reason='max_rays_before_target'
            elif rear_target is not None and rear_min<rear_target:stop_reason='max_rays_before_rear_channel_target'
            elif h170_target is not None and float(sym_total[17])<h170_target:stop_reason='max_rays_before_H170_target'
            elif stability_l1_tolerance is not None and stop_reason.startswith('fixed'):stop_reason='max_rays_before_stability'
            elif stop_reason.startswith('fixed'):stop_reason='max_rays'
        hardware_sym_total=0.5*(native_total.astype(float)+mirror_total.astype(float)) if self.symmetry_average_exact else native_total.astype(float)
        photodiode_sym_total=0.5*(photodiode_native_total+photodiode_mirror_total) if self.symmetry_average_exact else photodiode_native_total.copy()
        nsum=float(native_total.sum());msum=float(mirror_total.sum());hardware_ssum=float(hardware_sym_total.sum());nn=native_total.astype(float)/nsum if nsum>0 else np.zeros(18);nm=mirror_total.astype(float)/msum if msum>0 else np.zeros(18);nh=hardware_sym_total/hardware_ssum if hardware_ssum>0 else np.zeros(18)
        pnsum=float(photodiode_native_total.sum());pmsum=float(photodiode_mirror_total.sum());pssum=float(photodiode_sym_total.sum());pnn=photodiode_native_total/pnsum if pnsum>0 else np.zeros(18);pnm=photodiode_mirror_total/pmsum if pmsum>0 else np.zeros(18);pns=photodiode_sym_total/pssum if pssum>0 else np.zeros(18)
        selected_native_sum=nsum if self.photodiode_response_model=='ideal_hard_aperture' else pnsum
        selected_mirror_sum=msum if self.photodiode_response_model=='ideal_hard_aperture' else pmsum
        ssum=float(sym_total.sum());ns=sym_total/ssum if ssum>0 else np.zeros(18)
        direct_sym=0.5*(native_direct_total.astype(float)+mirror_direct_total.astype(float)) if self.symmetry_average_exact else native_direct_total.astype(float);stray_sym=0.5*(native_stray_total.astype(float)+mirror_stray_total.astype(float)) if self.symmetry_average_exact else native_stray_total.astype(float);det_stray_frac=np.divide(stray_sym,hardware_sym_total,out=np.zeros(18,dtype=float),where=hardware_sym_total>0);face_sym=0.5*(native_face_total.astype(float)+mirror_face_total.astype(float)) if self.symmetry_average_exact else native_face_total.astype(float);det_face_frac=np.divide(face_sym,hardware_sym_total,out=np.zeros(18,dtype=float),where=hardware_sym_total>0)
        photodiode_relative=np.divide(photodiode_sym_total,hardware_sym_total,out=np.zeros(18,dtype=float),where=hardware_sym_total>0)
        incidence_sym_sum=0.5*(incidence_native_sum_total+incidence_mirror_sum_total) if self.symmetry_average_exact else incidence_native_sum_total.copy()
        mean_incidence=np.divide(incidence_sym_sum,hardware_sym_total,out=np.full(18,np.nan,dtype=float),where=hardware_sym_total>0)
        lnsum=float(legacy_native_total.sum());lmsum=float(legacy_mirror_total.sum());lssum=float(legacy_sym_total.sum());lnn=legacy_native_total.astype(float)/lnsum if lnsum>0 else np.zeros(18);lnm=legacy_mirror_total.astype(float)/lmsum if lmsum>0 else np.zeros(18);lns=legacy_sym_total/lssum if lssum>0 else np.zeros(18)
        se=_jackknife_normalized_se(batch_scores);hardware_se=_jackknife_normalized_se(hardware_batch_scores);photodiode_se=_jackknife_normalized_se(photodiode_batch_scores);legacy_se=_jackknife_normalized_se(legacy_batch_scores);denom_enter=max(entered_total,1)
        # Detector-path mechanism fractions are normalized to the hard-aperture
        # symmetry score in each channel.  They diagnose the current particle
        # physics; they are not correction factors or fitted detector gains.
        path_diag={'hardware_symmetry_scores':hardware_sym_total.copy()}
        denp=np.asarray(hardware_sym_total,dtype=np.float64)
        for pk,pv in path_scores.items():
            path_diag[pk+'_scores']=pv.copy()
            path_diag[pk+'_fraction']=np.divide(pv,denp,out=np.zeros_like(pv),where=denp>0)
        path_diag['mean_particle_interactions_by_channel']=np.divide(path_interaction_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_geometric_interactions_by_channel']=np.divide(path_geometric_interaction_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_wave_interactions_by_channel']=np.divide(path_wave_interaction_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_coherent_wave_interactions_by_channel']=np.divide(path_coherent_wave_interaction_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_morphology_wave_interactions_by_channel']=np.divide(path_morphology_wave_interaction_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_entry_reflections_by_channel']=np.divide(path_entry_reflection_sums,denp,out=np.full(18,np.nan),where=denp>0)
        path_diag['mean_internal_reflections_by_channel']=np.divide(path_internal_reflection_sums,denp,out=np.full(18,np.nan),where=denp>0)
        # V24.29 retained V24.27 causal final-event and path-length outputs. Fractions use all detected
        # hardware candidates in each channel; conditional means use valid particle histories.
        for code in causal_event_codes:
            name=V2427_PARTICLE_EVENT_TYPE_NAMES[code]
            path_diag[f'last_event_{name}_scores']=causal_last_event_scores[code].copy()
            path_diag[f'last_event_{name}_fraction']=_v2427_safe_fraction(causal_last_event_scores[code],denp)
            path_diag[f'first_event_{name}_scores']=causal_first_event_scores[code].copy()
            path_diag[f'first_event_{name}_fraction']=_v2427_safe_fraction(causal_first_event_scores[code],denp)
            path_diag[f'last_event_{name}_mean_deflection_deg_by_channel']=np.divide(
                causal_last_event_angle_sums[code],causal_last_event_angle_counts[code],out=np.full(18,np.nan),where=causal_last_event_angle_counts[code]>0)
        path_diag['first_event_backscatter_scores']=causal_first_backscatter_scores.copy()
        path_diag['first_event_backscatter_fraction']=_v2427_safe_fraction(causal_first_backscatter_scores,denp)
        path_diag['first_event_source_hemisphere_scores']=causal_first_source_hemi_scores.copy()
        path_diag['first_event_source_hemisphere_fraction']=_v2427_safe_fraction(causal_first_source_hemi_scores,denp)
        path_diag['mean_first_particle_deflection_deg_by_channel']=np.divide(causal_first_angle_sums,causal_first_angle_counts,out=np.full(18,np.nan),where=causal_first_angle_counts>0)
        path_diag['mean_last_particle_deflection_deg_by_channel']=np.divide(causal_last_angle_sums,causal_last_angle_counts,out=np.full(18,np.nan),where=causal_last_angle_counts>0)
        path_diag['mean_total_internal_particle_path_m_by_channel']=np.divide(causal_internal_path_sums,causal_path_counts,out=np.full(18,np.nan),where=causal_path_counts>0)
        path_diag['mean_total_internal_particle_opl_m_by_channel']=np.divide(causal_internal_opl_sums,causal_path_counts,out=np.full(18,np.nan),where=causal_path_counts>0)
        path_diag['mean_water_path_after_last_particle_event_m_by_channel']=np.divide(causal_post_last_water_path_sums,causal_path_counts,out=np.full(18,np.nan),where=causal_path_counts>0)
        path_diag['mean_last_particle_event_order_by_channel']=np.divide(causal_last_order_sums,causal_path_counts,out=np.full(18,np.nan),where=causal_path_counts>0)
        path_diag['causal_last_event_partition_closure_error_scores']=path_scores['any_particle']-sum(causal_last_event_scores.values())
        path_diag['causal_first_event_partition_closure_error_scores']=path_scores['any_particle']-sum(causal_first_event_scores.values())
        path_diag['H170_last_event_angle_bin_edges_deg']=causal_h170_angle_edges.copy()
        path_diag['H170_last_event_angle_histogram_scores_by_event_code']=causal_h170_last_angle_hist.copy()
        path_diag['H170_first_event_angle_histogram_scores']=causal_h170_first_angle_hist.copy()
        # Exact partition-closure diagnostics.  Zero means the mutually exclusive bins
        # recover the corresponding parent score exactly.
        path_diag['interaction_order_partition_closure_error_scores']=(
            path_scores['any_particle']-(path_scores['single_interaction']+path_scores['interaction_order_2']+
            path_scores['interaction_order_3']+path_scores['interaction_order_4plus']))
        path_diag['multiple_branch_partition_closure_error_scores']=(
            path_scores['multiple_interaction']-(path_scores['mixed_geometric_wave']+path_scores['geometric_only_multiple']+
            path_scores['wave_only_multiple']))
        path_diag['single_branch_partition_closure_error_scores']=(
            path_scores['single_interaction']-(path_scores['single_external_entry_reflection_escape']+
            path_scores['single_internal_reflection_escape']+path_scores['single_surface_transmission_escape']+
            path_scores['single_coherent_wave_escape']))
        ray_data=None
        if collect_rays and ray_chunks:
            keys=ray_chunks[0].keys();ray_data={k:np.concatenate([q[k] for q in ray_chunks]) for k in keys};ni=ray_data['native_detector_index'];mi=ray_data['mirror_detector_index'];ray_data['native_detector_deg']=np.where(ni>=0,DETECTOR_ANGLES_DEG[np.clip(ni,0,17)],np.nan);ray_data['mirror_detector_deg']=np.where(mi>=0,DETECTOR_ANGLES_DEG[np.clip(mi,0,17)],np.nan)
        hmxy=cp.asnumpy(hxy).reshape(vis,vis) if vis>0 else None;hmzy=cp.asnumpy(hzy).reshape(vis,vis) if vis>0 else None
        vg=vortex_surface_geometry(self.free_surface_model,self.zmax,self.rin,self.vortex_delta_h,self.vortex_core_radius)
        pf_counts={PRODUCTION_FAILURE_NAMES.get(i,f'CODE_{i}'):int(v) for i,v in enumerate(production_failure_code_counts) if i!=0 and int(v)!=0}
        pf_sub_counts={str(k):int(v) for k,v in sorted(production_failure_subcode_counts.items())}
        production_watchdog_count=sum(int(production_failure_code_counts[i]) for i in PRODUCTION_WATCHDOG_CODES if i<len(production_failure_code_counts))
        production_geometry_failure_count=sum(int(production_failure_code_counts[i]) for i in PRODUCTION_GEOMETRY_FAILURE_CODES if i<len(production_failure_code_counts))
        production_nonfinite_count=sum(int(production_failure_code_counts[i]) for i in PRODUCTION_NONFINITE_CODES if i<len(production_failure_code_counts))
        production_failure_diagnostics={'code_counts':pf_counts,'subcode_counts':pf_sub_counts,'samples':production_failure_samples,'sample_limit':production_failure_sample_limit,'count_closure_ok':int(production_failure_code_counts[1:].sum())==int(numerical_failure_total),'particle_proposal_count':int(particle_proposal_total),'particle_accessible_count':int(particle_accessible_total),'particle_null_count':int(particle_null_total)}
        return ForwardResult(detector_angles_deg=DETECTOR_ANGLES_DEG.copy(),raw_hits=native_total.copy(),normalized_response=ns,native_exact_hits=native_total.copy(),mirror_exact_hits=mirror_total.copy(),symmetry_exact_scores=sym_total.copy(),normalized_native_exact_response=nn,normalized_mirror_exact_response=nm,normalized_symmetry_exact_response=ns,normalized_response_jackknife_se=se,
            hardware_native_hits=native_total.copy(),hardware_mirror_hits=mirror_total.copy(),hardware_symmetry_scores=hardware_sym_total.copy(),normalized_hardware_native_response=nn,normalized_hardware_mirror_response=nm,normalized_hardware_symmetry_response=nh,hardware_normalized_jackknife_se=hardware_se,
            legacy_native_channel_counts=legacy_native_total.copy(),legacy_mirror_channel_counts=legacy_mirror_total.copy(),legacy_symmetry_channel_scores=legacy_sym_total.copy(),normalized_legacy_native_response=lnn,normalized_legacy_mirror_response=lnm,normalized_legacy_symmetry_response=lns,legacy_normalized_jackknife_se=legacy_se,legacy_acceptance_deg=self.legacy_acceptance_deg,
            valid_exit_count=valid_air_total,numerical_failure_count=numerical_failure_total,numerical_failure_fraction=numerical_failure_total/max(total_rays,1),production_failure_count=numerical_failure_total,production_failure_code_counts=pf_counts,production_failure_subcode_counts=pf_sub_counts,production_watchdog_count=production_watchdog_count,production_geometry_failure_count=production_geometry_failure_count,production_nonfinite_count=production_nonfinite_count,production_failure_diagnostics=production_failure_diagnostics,n_rays=total_rays,requested_n_rays=requested,stopped_adaptively=bool(adaptive and total_rays<maxN),stop_reason=stop_reason,particle_event_model=self.particle_event_model,particle_surface_rms_slope_deg=self.particle_surface_rms_slope_deg,particle_surface_beckmann_alpha=self.particle_surface_beckmann_alpha,wave_surface_rms_height_nm=self.wave_surface_rms_height_m*1e9,wave_spheroid_aspect_ratio=self.wave_spheroid_aspect_ratio,wave_spheroid_orientation_model=self.wave_spheroid_orientation_model,wave_spheroid_orientation_kappa=self.wave_spheroid_orientation_kappa,aggregation_model=ph['aggregation_model'],aggregation_collision_exposure=ph['aggregation_collision_exposure'],aggregation_number_reduction_fraction=ph['aggregation_number_reduction_fraction'],aggregation_mass_conservation_error=ph['aggregation_mass_conservation_error'],aggregation_projected_area_ratio=ph['aggregation_projected_area_ratio'],aggregation_largest_bin_mass_fraction=ph['aggregation_largest_bin_mass_fraction'],aggregation_overflow_event_fraction=ph['aggregation_overflow_event_fraction'],aggregation_steps=ph['aggregation_steps'],population_number_reduction_fraction=ph['population_number_reduction_fraction'],fragmentation_breakup_exposure=ph['fragmentation_breakup_exposure'],fragmentation_reference_diameter_m=ph['fragmentation_reference_diameter_m'],fragmentation_size_exponent=ph['fragmentation_size_exponent'],fragmentation_underflow_event_fraction=ph['fragmentation_underflow_event_fraction'],aggregation_events_per_initial_particle=ph['aggregation_events_per_initial_particle'],fragmentation_events_per_initial_particle=ph['fragmentation_events_per_initial_particle'],wave_coherent_fraction_nominal_cuda=ph['wave_coherent_fraction_nominal_cuda'],mean_coherent_wave_interactions=sum_wcoh/denom_enter if entered_total else float('nan'),mean_morphology_wave_interactions=sum_wmorph/denom_enter if entered_total else float('nan'),morphology_fraction_of_wave_interactions=sum_wmorph/sum_wic if sum_wic>0 else 0.0,
            mu_geom_per_m=ph['mu_geom'],mu_wave_per_m=ph['mu_wave'],mu_event_per_m=ph['mu_event'],mu_sca_per_m=ph['mu_sca'],mu_ext_per_m=ph['mu_ext'],geometric_optical_depth_diameter=ph['mu_geom']*(2*self.rin),optical_depth_diameter=ph['mu_event']*(2*self.rin),
            mean_interactions=sum_ic/denom_enter if entered_total else float('nan'),mean_geometric_interactions=sum_gic/denom_enter if entered_total else float('nan'),mean_wave_interactions=sum_wic/denom_enter if entered_total else float('nan'),median_interactions=_histogram_median(ic_hist),ballistic_fraction=ballistic_total/denom_enter if entered_total else float('nan'),mean_fresnel_reflections=sum_pr/denom_enter if entered_total else float('nan'),entry_reflection_fraction=rays_entry_pr/denom_enter if entered_total else float('nan'),internal_reflection_fraction=rays_internal_pr/denom_enter if entered_total else float('nan'),particle_absorption_fraction=particle_absorbed_total/denom_enter if entered_total else float('nan'),particle_absorbed_ray_count=particle_absorbed_total,
            mean_cell_inner_reflections=sum_cir/denom_enter if entered_total else float('nan'),mean_cell_outer_reflections=sum_cor/denom_enter if entered_total else float('nan'),cell_tir_fraction=rays_ctir/denom_enter if entered_total else float('nan'),mean_top_surface_reflections=sum_top_ref/denom_enter if entered_total else float('nan'),top_surface_tir_fraction=rays_top_tir/denom_enter if entered_total else float('nan'),mean_top_surface_interactions=(sum_top_ref+top_exit_total)/denom_enter if entered_total else float('nan'),free_surface_model=self.free_surface_model,vortex_wall_center_height_difference_mm=self.vortex_delta_h*1e3,vortex_core_radius_mm=self.vortex_core_radius*1e3,vortex_asymptotic_height_parameter_mm=vg['h_inf']*1e3,vortex_area_mean_uncorrected_rise_mm=vg['h_bar']*1e3,vortex_equivalent_core_rpm=vg['equivalent_core_rpm'],vortex_equivalent_rigid_body_rpm=vg['equivalent_rigid_body_rpm'],vortex_center_surface_z_mm=vg['center_z']*1e3,vortex_wall_surface_z_mm=vg['wall_z']*1e3,sediment_transport_model=self.sediment_transport_model,sediment_eddy_diffusivity_m2_s=self.sediment_eddy_diffusivity,sediment_majorant_geom_per_m=ph['sediment_majorant_geom_per_m'],sediment_majorant_wave_per_m=ph['sediment_majorant_wave_per_m'],sediment_majorant_total_per_m=ph['sediment_majorant_total_per_m'],sediment_max_bin_density_scale=float(np.max(ph['sediment_max_density_scale'])) if len(ph['sediment_max_density_scale']) else 1.0,meridional_circulation_ratio=self.meridional_circulation_ratio,meridional_peak_speed_m_s=self.meridional_peak_speed,meridional_grid_nr=int(ph['meridional_grid_nr']),meridional_grid_nz=int(ph['meridional_grid_nz']),meridional_solver_iterations=int(ph['meridional_solver_iterations']),meridional_solver_max_residual=float(ph['meridional_solver_max_residual']),mean_bottom_inner_reflections=sum_bottom_inner/denom_enter if entered_total else float('nan'),mean_bottom_outer_reflections=sum_bottom_outer/denom_enter if entered_total else float('nan'),bottom_tir_fraction=rays_bottom_tir/denom_enter if entered_total else float('nan'),top_exit_fraction=top_exit_total/max(total_rays,1),bottom_exit_fraction=bottom_exit_total/max(total_rays,1),source_bore_surface_model=self.source_bore_surface_model,detector_bore_surface_model=self.detector_bore_surface_model,bore_refractive_index=self.bore_refractive_index,ring_inner_face_surface_model=self.ring_inner_face_surface_model,ring_inner_face_refractive_index=self.ring_inner_face_refractive_index,max_ring_face_bounces=self.max_ring_face_bounces,source_collimator_acceptance_fraction=source_accept_total/max(total_rays,1),source_direct_acceptance_fraction=source_direct_accept_total/max(total_rays,1),source_reflected_acceptance_fraction=source_reflected_accept_total/max(total_rays,1),source_bore_interactions_per_ray=source_bore_interactions_total/max(total_rays,1),source_bore_reflections_per_ray=source_bore_reflections_total/max(total_rays,1),hardware_native_direct_hits=native_direct_total.copy(),hardware_mirror_direct_hits=mirror_direct_total.copy(),hardware_symmetry_direct_scores=direct_sym.copy(),hardware_native_stray_hits=native_stray_total.copy(),hardware_mirror_stray_hits=mirror_stray_total.copy(),hardware_symmetry_stray_scores=stray_sym.copy(),detector_stray_fraction_by_channel=det_stray_frac.copy(),detector_stray_detection_fraction=float(stray_sym.sum())/max(total_rays,1),hardware_native_ring_face_stray_hits=native_face_total.copy(),hardware_mirror_ring_face_stray_hits=mirror_face_total.copy(),hardware_symmetry_ring_face_stray_scores=face_sym.copy(),detector_ring_face_stray_fraction_by_channel=det_face_frac.copy(),detector_ring_face_stray_detection_fraction=float(face_sym.sum())/max(total_rays,1),photodiode_response_model=self.photodiode_response_model,sfh213_half_angle_deg=10.0,sfh213_directional_fit_exponent=3.169925001442312,photodiode_native_weighted_scores=photodiode_native_total.copy(),photodiode_mirror_weighted_scores=photodiode_mirror_total.copy(),photodiode_symmetry_weighted_scores=photodiode_sym_total.copy(),normalized_photodiode_native_response=pnn,normalized_photodiode_mirror_response=pnm,normalized_photodiode_symmetry_response=pns,photodiode_relative_response_by_channel=photodiode_relative.copy(),detector_mean_incidence_angle_deg_by_channel=mean_incidence.copy(),hardware_candidate_detection_fraction=hardware_ssum/max(total_rays,1),entered_water_fraction=entered_total/max(total_rays,1),air_exit_fraction=valid_air_total/max(total_rays,1),axial_loss_fraction=axial_total/max(total_rays,1),detector_detection_fraction=ssum/max(total_rays,1),native_detector_detection_fraction=selected_native_sum/max(total_rays,1),mirror_detector_detection_fraction=selected_mirror_sum/max(total_rays,1),legacy_channel_score_per_ray=lssum/max(total_rays,1),symmetry_variance_reduction=self.symmetry_average_exact,batch_diagnostics=batch_rows,particle_diagnostics=ph,detector_particle_path_diagnostics=path_diag,ray_data=ray_data,heatmap_xy=hmxy,heatmap_zy=hmzy)

