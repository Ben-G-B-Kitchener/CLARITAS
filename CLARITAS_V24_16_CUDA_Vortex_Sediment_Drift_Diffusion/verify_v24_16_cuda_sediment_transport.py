#!/usr/bin/env python3
"""Verification for CLARITAS V24.16 vortex sediment drift-diffusion.

Host-only checks validate the closed-form drift coefficients, size monotonicity,
and independent vessel-volume mass normalization. CUDA checks additionally
compare GPU transport coefficients/field samples to the host reference, verify
the exact uniform-suspension control, and trace a non-uniform case.
"""
from __future__ import annotations
import argparse, math
import numpy as np
from claritas_tardiis_core_v24_16 import (
    TardiisForwardModel, LOESS_DIAMETER_M, sediment_transport_coefficients_host,
    vortex_surface_geometry,
)

R=0.0465
ZMEAN=0.142-0.093
ZMIN=-0.093
A=0.020
D=0.050


def surface_z(r):
    vg=vortex_surface_geometry('localized_scully_vortex',ZMEAN,R,D,A)
    return ZMEAN-vg['h_bar']+vg['h_inf']*r*r/(A*A+r*r)


def host_scale(coeff,idx,r,z):
    f=r*r/(A*A+r*r);fR=R*R/(A*A+R*R)
    return math.exp(coeff['radial_log_coefficient'][idx]*(f-fR)-coeff['vertical_inverse_length_m'][idx]*max(z-ZMIN,0.0))/coeff['normalization_scaled'][idx]


def host_checks():
    vg=vortex_surface_geometry('localized_scully_vortex',ZMEAN,R,D,A)
    print(f'frozen geometry: a=20 mm, D=50 mm, centre={vg["center_z"]*1e3:.3f} mm, wall={vg["wall_z"]*1e3:.3f} mm relative sensor')
    if not math.isclose(vg['wall_z']-vg['center_z'],D,abs_tol=2e-14):
        raise SystemExit('FAIL: vortex wall-centre depth')
    for Dt in (2.5e-4,5e-4,1e-3,2e-3):
        q=sediment_transport_coefficients_host(LOESS_DIAMETER_M,ZMEAN,ZMIN,R,D,A,Dt,quadrature_points=16384)
        for key in ('response_time_s','settling_velocity_m_s','radial_log_coefficient','vertical_inverse_length_m'):
            x=np.asarray(q[key])
            if np.any(np.diff(x)<-1e-14): raise SystemExit(f'FAIL: {key} is not monotone with particle size')
        if np.any(np.asarray(q['normalization_scaled'])<=0) or np.any(np.asarray(q['max_density_scale'])<1):
            raise SystemExit('FAIL: invalid transport normalization/majorant scale')
        # The peak-scaled construction must make the exact per-bin Woodcock
        # density bound occur at r=R,z=ZMIN. Check that bound independently at
        # deterministic interior points for fine/mid/coarse bins.
        for idx in (0,len(LOESS_DIAMETER_M)//2,len(LOESS_DIAMETER_M)-1):
            mx=float(q['max_density_scale'][idx])
            at_peak=host_scale(q,idx,R,ZMIN)
            if not math.isclose(at_peak,mx,rel_tol=2e-13,abs_tol=2e-13):
                raise SystemExit(f'FAIL: peak density scale mismatch Dt={Dt:g}, bin={idx}')
            for rf,zf in ((0.0,0.0),(0.25,0.1),(0.5,0.5),(0.95,0.9),(0.999,0.01)):
                r=rf*R; zs=surface_z(r); z=ZMIN+zf*(zs-ZMIN)
                sc=host_scale(q,idx,r,z)
                if not (0.0<sc<=mx*(1.0+2e-13)):
                    raise SystemExit(f'FAIL: local density exceeds majorant Dt={Dt:g}, bin={idx}, scale={sc}, max={mx}')
        # Independent 2-D midpoint integration of the already-normalized field for
        # representative fine/mid/coarse bins. This does not reuse the analytic z integral.
        nr,nz=600,300
        dr=R/nr
        for idx in (0,len(LOESS_DIAMETER_M)//2,len(LOESS_DIAMETER_M)-1):
            integ=0.0
            for j in range(nr):
                r=(j+0.5)*dr;H=surface_z(r)-ZMIN;dz=H/nz
                vals=0.0
                for k in range(nz):
                    z=ZMIN+(k+0.5)*dz
                    vals+=host_scale(q,idx,r,z)*dz
                integ+=r*vals*dr
            mean=(2.0/(R*R*(ZMEAN-ZMIN)))*integ
            if abs(mean-1.0)>1.2e-3:
                raise SystemExit(f'FAIL: mass normalization Dt={Dt:g}, bin={idx}, mean={mean:.9g}')
        print(f'Dt={Dt:g} m^2/s: max density scale={np.max(q["max_density_scale"]):.4g}; Amax={np.max(q["radial_log_coefficient"]):.4g}; Bmax={np.max(q["vertical_inverse_length_m"]):.4g}/m')
    # Mixing should weaken both radial and vertical segregation as Dt increases.
    qlo=sediment_transport_coefficients_host(LOESS_DIAMETER_M,ZMEAN,ZMIN,R,D,A,2.5e-4)
    qhi=sediment_transport_coefficients_host(LOESS_DIAMETER_M,ZMEAN,ZMIN,R,D,A,2e-3)
    if not (np.max(qlo['radial_log_coefficient'])>np.max(qhi['radial_log_coefficient']) and np.max(qlo['vertical_inverse_length_m'])>np.max(qhi['vertical_inverse_length_m'])):
        raise SystemExit('FAIL: eddy diffusivity does not weaken segregation')
    print('PASS: host drift laws, monotonicity and independent per-bin mass conservation')


def make(transport='stokes_drift_diffusion',Dt=5e-4):
    return TardiisForwardModel(
        particle_event_model='hybrid_mie_excess',particle_surface_rms_slope_deg=25.0,
        wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.0,
        wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
        free_surface_model='localized_scully_vortex',vortex_wall_center_height_difference_m=D,
        vortex_core_radius_m=A,sediment_transport_model=transport,sediment_eddy_diffusivity_m2_s=Dt,
        water_density_kg_per_m3=997.0,water_dynamic_viscosity_pa_s=8.9e-4,gravity_m_s2=9.81,
        sediment_transport_quadrature_points=4096,aggregation_model='none')


def cuda_checks(n_rays):
    uniform=make('uniform',5e-4)
    hu=uniform.particle_physics_diagnostics('loess',0.5)
    for key in ('sediment_normalization_scaled','sediment_max_density_scale','sediment_scale_sensor_center','sediment_scale_bottom_wall_r0p95R'):
        if not np.array_equal(np.asarray(hu[key]),np.ones_like(np.asarray(hu[key]))):
            raise SystemExit(f'FAIL: uniform control {key} is not exactly one')
    if abs(hu['sediment_majorant_total_per_m']-(hu['mu_trace_geometric']+hu['mu_trace_wave']))>1e-11:
        raise SystemExit('FAIL: uniform majorant does not equal historical event rate')
    print('PASS: exact uniform-suspension transport precompute control')

    m=make('stokes_drift_diffusion',5e-4)
    prep=m._prepare_gpu_particle_physics('loess',0.5)
    hg=prep['host'];dev=prep['device'];cp=m.cp
    ref=sediment_transport_coefficients_host(LOESS_DIAMETER_M,m.zmax,m.zmin,m.rin,m.vortex_delta_h,m.vortex_core_radius,m.sediment_eddy_diffusivity,
                                              particle_density_kg_m3=m.density,water_density_kg_m3=m.water_density,
                                              water_dynamic_viscosity_pa_s=m.water_dynamic_viscosity,gravity_m_s2=m.gravity,quadrature_points=m.sediment_transport_quadrature_points)
    checks=[
        ('sediment_response_time_s','response_time_s',2e-12,2e-12),
        ('sediment_settling_velocity_m_s','settling_velocity_m_s',2e-12,2e-12),
        ('sediment_radial_log_coefficient_A','radial_log_coefficient',2e-11,2e-11),
        ('sediment_vertical_inverse_length_B_per_m','vertical_inverse_length_m',2e-11,2e-11),
        ('sediment_normalization_scaled','normalization_scaled',2e-9,2e-9),
    ]
    for gk,rk,rt,at in checks:
        if not np.allclose(np.asarray(hg[gk]),np.asarray(ref[rk]),rtol=rt,atol=at):
            raise SystemExit(f'FAIL: GPU/host coefficient mismatch {gk}')
    print('PASS: CUDA transport coefficients and curved-volume normalization match host reference')

    # Sample a middle-size bin at several interior locations using the CUDA field helper.
    idx=len(LOESS_DIAMETER_M)//2
    pts=np.array([[0,0,0],[0.95*R,0,0],[0,0,ZMIN+1e-5],[0.95*R,0,ZMIN+1e-5],[0.5*R,0,0.5*(ZMIN+surface_z(0.5*R))]],dtype=np.float32)
    dp=cp.asarray(pts.ravel());out=cp.empty(len(pts),dtype=cp.float32)
    k=m.kernels['sediment_field_test_kernel']
    k(((len(pts)+63)//64,),(64,),(dp,np.int32(len(pts)),np.int32(idx),np.int32(m.SEDIMENT_TRANSPORT_MODEL_CODES[m.sediment_transport_model]),dev['sed_A'],dev['sed_B'],dev['sed_norm'],np.float32(m.rin),np.float32(m.zmin),np.float32(m.vortex_core_radius),out))
    cp.cuda.Stream.null.synchronize();got=cp.asnumpy(out)
    want=np.array([host_scale(ref,idx,float(np.hypot(p[0],p[1])),float(p[2])) for p in pts])
    if not np.allclose(got,want,rtol=2e-5,atol=2e-6):
        raise SystemExit(f'FAIL: CUDA local field samples mismatch\nGPU {got}\nREF {want}')
    print('PASS: CUDA local sediment field samples')

    ru=uniform.simulate('loess',0.5,n_rays=n_rays,seed=314159,heatmap_size=0)
    rr=m.simulate('loess',0.5,n_rays=n_rays,seed=271828,heatmap_size=0)
    for label,r in [('uniform',ru),('transport',rr)]:
        if not np.all(np.isfinite(r.normalized_response)) or abs(float(np.sum(r.normalized_response))-1.0)>1e-10:
            raise SystemExit(f'FAIL: {label} ray trace invalid detector normalization')
    if not (rr.sediment_majorant_total_per_m>=rr.mu_event_per_m and rr.sediment_max_bin_density_scale>=1.0):
        raise SystemExit('FAIL: invalid active-transport majorant diagnostics')
    print(f'PASS: {n_rays:,}-ray active transport trace finite; majorant={rr.sediment_majorant_total_per_m:.6g}/m; max-scale={rr.sediment_max_bin_density_scale:.6g}')
    print('PASS: V24.16 CUDA sediment-transport verification complete')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host-only',action='store_true');ap.add_argument('--rays',type=int,default=100000)
    a=ap.parse_args();host_checks()
    if not a.host_only: cuda_checks(a.rays)

if __name__=='__main__': main()
