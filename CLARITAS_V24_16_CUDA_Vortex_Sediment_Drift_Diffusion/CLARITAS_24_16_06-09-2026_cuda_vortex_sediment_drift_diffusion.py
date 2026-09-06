#!/usr/bin/env python3
"""CLARITAS V24.16: CUDA-authoritative vortex sediment drift-diffusion diagnostic.

V24.16 freezes the best non-destructive V24.15 localized free-surface geometry
(a=20 mm, wall-centre depth=50 mm) and adds size-resolved sediment transport.
For each PSD bin, Stokes centrifugal drift and gravitational settling are
balanced against a single physical eddy diffusivity D_t representing the
continuous stirring/recirculation.  Each bin is volume-normalised over the
curved vessel, so total sediment mass and the input bulk PSD are conserved.

The spatially varying optical event field is traced with CUDA Woodcock/delta
tracking and local size-bin sampling.  No concentration-specific or
material-specific scattering multiplier is used.  A uniform transport mode is
retained as the exact V24.15 regression control.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from claritas_tardiis_core_v24_16 import (
 TardiisForwardModel,DETECTOR_ANGLES_DEG,get_material_psd,get_material_optical_constants,
 vortex_surface_geometry
)

DEFAULTS=dict(
 n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,particle_wavelength_m=622e-9,
 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
 water_height_m=0.142,sensor_height_above_bottom_m=0.093,
 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,
 counterbore_depth_m=0.008,through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,
 source_launch_radius_m=0.0655,legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1e-5,
 alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d',density_kg_per_m3=2600.0,bottom_disc_thickness_m=0.003,bottom_external_refractive_index=1.0,max_axial_bounces=64,max_internal_bounces=64,max_cell_bounces=64,
 chunk_size=250000,statistics_batch_rays=250000,symmetry_average_exact=True,particle_event_model='hybrid_mie_excess',hybrid_wave_scale=1.0,mie_phase_grid_size=2049,particle_surface_rms_slope_deg=25.0,wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.0,wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,free_surface_model='localized_scully_vortex',vortex_wall_center_height_difference_m=0.050,vortex_core_radius_m=0.020,sediment_transport_model='stokes_drift_diffusion',sediment_eddy_diffusivity_m2_s=5e-4,water_density_kg_per_m3=997.0,water_dynamic_viscosity_pa_s=8.9e-4,gravity_m_s2=9.81,sediment_transport_quadrature_points=4096,aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,fragmentation_reference_diameter_m=100e-6,fragmentation_size_exponent=1.0,aggregation_step_safety=0.2,aggregation_max_steps=200000)


def surface_geometry(model):
 return vortex_surface_geometry(model.free_surface_model,model.zmax,model.rin,
                                model.vortex_delta_h,model.vortex_core_radius)

def make_model(args):
 d=DEFAULTS.copy()
 if args.config:
  cfg=json.loads(Path(args.config).read_text());d.update(cfg.get('forward_model',cfg))
 d['particle_event_model']=args.event_model
 d['hybrid_wave_scale']=args.wave_scale
 d['mie_phase_grid_size']=args.phase_grid_size
 d['particle_surface_rms_slope_deg']=args.surface_rms_slope_deg
 d['wave_surface_rms_height_m']=args.wave_rms_height_nm*1e-9
 d['wave_spheroid_aspect_ratio']=1.0
 d['wave_spheroid_orientation_model']='isotropic'
 d['wave_spheroid_orientation_kappa']=0.0
 d['free_surface_model']=args.free_surface_model
 d['vortex_wall_center_height_difference_m']=args.vortex_delta_h_mm*1e-3
 d['vortex_core_radius_m']=args.vortex_core_radius_mm*1e-3
 d['sediment_transport_model']=args.sediment_transport_model
 d['sediment_eddy_diffusivity_m2_s']=args.eddy_diffusivity_m2_s
 d['water_density_kg_per_m3']=args.water_density_kg_m3
 d['water_dynamic_viscosity_pa_s']=args.water_viscosity_mpa_s*1e-3
 d['gravity_m_s2']=args.gravity_m_s2
 d['sediment_transport_quadrature_points']=args.transport_quadrature_points
 # V24.16 freezes population evolution off while testing spatial transport.
 d['aggregation_model']='none'
 d['aggregation_collision_exposure']=0.0
 d['fragmentation_breakup_exposure']=0.0
 if args.source_launch_radius_mm is not None: d['source_launch_radius_m']=args.source_launch_radius_mm*1e-3
 if args.alpha1 is not None: d['alpha1']=args.alpha1
 if args.alpha2 is not None: d['alpha2']=args.alpha2
 d['source_angular_model']=args.source_angular_model
 if args.water_height_mm is not None: d['water_height_m']=args.water_height_mm*1e-3
 if args.sensor_height_mm is not None: d['sensor_height_above_bottom_m']=args.sensor_height_mm*1e-3
 if args.bottom_external_n is not None: d['bottom_external_refractive_index']=args.bottom_external_n

 # Build an explicit profile override only for the requested A/B decomposition mode.
 # The source n(d),k(d) arrays are taken from the configured profile when present,
 # otherwise from the built-in V24.5 per-bin profiles.
 if args.uniform_optics:
  d['particle_optical_constants_by_material']={
   'loess': {'n_real':[1.59]*38,'k_imag':[0.0]*38},
   'kaolin': {'n_real':[1.59]*37,'k_imag':[0.0]*37},
  }
 elif args.real_index_only or args.k_only:
  current=d.get('particle_optical_constants_by_material')
  override={}
  for mat in ('loess','kaolin'):
   n0,k0=get_material_optical_constants(mat,current,d.get('n_particle',1.59))
   if args.real_index_only:
    override[mat]={'n_real':n0.tolist(),'k_imag':[0.0]*len(k0)}
   else:
    override[mat]={'n_real':[1.59]*len(n0),'k_imag':k0.tolist()}
  d['particle_optical_constants_by_material']=override
 return TardiisForwardModel(**d),d

def main():
 ap=argparse.ArgumentParser(description='CLARITAS V24.16 CUDA-authoritative vortex sediment drift-diffusion diagnostic')
 ap.add_argument('--material',choices=['loess','kaolin'],default='loess');ap.add_argument('--concentration',type=float,default=0.5)
 ap.add_argument('--n-rays',type=int,default=1000000,help='fixed ray count unless --target-detector-score is used')
 ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--output-dir');ap.add_argument('--config');ap.add_argument('--dgb-only',action='store_true')
 ap.add_argument('--source-launch-radius-mm',type=float,help='override LED luminous-plane launch radius; DGB audit range is 57.5..65.5 mm')
 ap.add_argument('--alpha1',type=float,help='override CUDA source Beta alpha1; DGB-calibrated V24.10.2 value is 0.45')
 ap.add_argument('--alpha2',type=float,help='override CUDA source Beta alpha2; 622-nm thesis prior is 5.0')
 ap.add_argument('--source-angular-model',choices=['legacy_polar_pdf','beta_radiance_3d','thesis_planar_beta'],default='beta_radiance_3d',help='CUDA source law. beta_radiance_3d preserves the thesis Beta angular-intensity profile under the 3-D solid-angle measure; legacy_polar_pdf reproduces V24.10.1; thesis_planar_beta reproduces the old 2-D horizontal law.')
 ap.add_argument('--water-height-mm',type=float,help='override water height; steady-state 622-nm experiment used 142 mm')
 ap.add_argument('--sensor-height-mm',type=float,help='override sensor-ring height above bottom; steady-state 622-nm experiment used 93 mm')
 ap.add_argument('--bottom-external-n',type=float,help='refractive index below the 3-mm acrylic bottom disc; default 1.0 diagnostic')
 ap.add_argument('--heatmap-size',type=int,default=0,help='0 is recommended for high-statistics detector runs')
 ap.add_argument('--event-model',choices=['geometric','mie_qsca','mie_qext','hybrid_mie_excess'],default='hybrid_mie_excess',help='hybrid_mie_excess retains the V24.7 split event-rate model while V24.8 roughens only geometric particle interfaces')
 ap.add_argument('--wave-scale',type=float,default=1.0,help='scale applied to positive Mie-excess wave cross section; default 1.0')
 ap.add_argument('--phase-grid-size',type=int,default=2049,help='nonuniform Mie scattering-angle grid; >=257')
 ap.add_argument('--surface-rms-slope-deg',type=float,default=25.0,help='Beckmann RMS microfacet slope angle in degrees; 0 reproduces V24.7 smooth particle surfaces')
 ap.add_argument('--wave-rms-height-nm',type=float,default=100.0,help='RMS particle surface height used by the CUDA roughness-coherence model for supplemental wave events; 0 exactly recovers V24.8.1 coherent Mie wave scattering')
 ap.add_argument('--free-surface-model',choices=['flat','parabolic_vortex','localized_scully_vortex'],default='localized_scully_vortex',help='CUDA water/air free-surface geometry; V24.16 freezes the validated localized Scully geometry by default')
 ap.add_argument('--vortex-delta-h-mm',type=float,default=50.0,help='wall-to-centre free-surface height difference in mm; V24.16 default freezes the V24.15 best non-destructive case')
 ap.add_argument('--vortex-core-radius-mm',type=float,default=20.0,help='Scully vortex core radius a in mm; V24.16 default freezes the V24.15 a=20 mm case')
 ap.add_argument('--sediment-transport-model',choices=['uniform','stokes_drift_diffusion'],default='stokes_drift_diffusion',help='uniform is the exact V24.15 control; stokes_drift_diffusion enables the V24.16 size-resolved field')
 ap.add_argument('--eddy-diffusivity-m2-s',type=float,default=5e-4,help='effective sediment eddy diffusivity D_t in m^2/s; common sweep range supplied is 2.5e-4..2e-3')
 ap.add_argument('--water-density-kg-m3',type=float,default=997.0,help='water density used by Stokes settling')
 ap.add_argument('--water-viscosity-mpa-s',type=float,default=0.89,help='dynamic viscosity of water in mPa.s')
 ap.add_argument('--gravity-m-s2',type=float,default=9.81)
 ap.add_argument('--transport-quadrature-points',type=int,default=4096,help='CUDA radial midpoint quadrature used to mass-normalise every PSD bin')
 optics=ap.add_mutually_exclusive_group()
 optics.add_argument('--uniform-optics',action='store_true',help='A/B regression: use n=1.59, k=0 for every particle bin')
 optics.add_argument('--real-index-only',action='store_true',help='Use the V24.5 per-bin n(d) profile with k=0 in every bin')
 optics.add_argument('--k-only',action='store_true',help='Use n=1.59 in every bin with the V24.5 per-bin k(d) profile')
 ap.add_argument('--save-rays',action='store_true');ap.add_argument('--target-detector-score',type=float,help='adaptive selected-hardware score target')
 ap.add_argument('--min-rays',type=int);ap.add_argument('--max-rays',type=int);ap.add_argument('--stability-l1-tolerance',type=float)
 args=ap.parse_args();conc=0.0 if args.dgb_only else args.concentration;tag='DGB_water_only' if args.dgb_only else f'{args.material}_{conc:g}gL'
 out=Path(args.output_dir or f'claritas_v24_16_results/{tag}');out.mkdir(parents=True,exist_ok=True);model,geom=make_model(args)
 throat_outer=model.throat_out
 half_angle=math.degrees(math.atan2(model.throat_radius,model.source_launch_radius-model.ring_in))
 print('\n=========== CLARITAS V24.16 — CUDA VORTEX SEDIMENT DRIFT-DIFFUSION ===========')
 print(f'material={args.material}; concentration={conc:g} g/L; requested rays={args.n_rays:,}')
 print('Particle optics: per-bin n(d)+i*k(d); real-n Snell direction + complex Fresnel + internal Beer-Lambert absorption')
 print(f'Particle wavelength={model.particle_wavelength*1e9:.1f} nm; event model={model.particle_event_model}')
 if args.uniform_optics:
  optical_label='uniform 1.59+0i A/B regression'
 elif args.real_index_only:
  optical_label='size-dependent n(d) with k=0'
 elif args.k_only:
  optical_label='uniform n=1.59 with size-dependent k(d)'
 else:
  optical_label='explicit size-dependent complex profile'
 print('Optical profile:', optical_label)
 print('CUDA authority: Mie coefficients/efficiencies, phase CDFs, coherence partition, free-surface geometry, size-resolved sediment drift-diffusion normalisation, Woodcock transport/event sampling, particle optics, source sampling, and detector scoring are GPU-resident.')
 print(f'Geometric events retain V23/V24.5 sphere optics + V24.8 Beckmann roughness; RMS slope={model.particle_surface_rms_slope_deg:g} deg (alpha={model.particle_surface_beckmann_alpha:.6g})')
 print(f'Wave morphology frozen: RMS surface height={model.wave_surface_rms_height_m*1e9:g} nm; q={model.wave_spheroid_aspect_ratio:g}; orientation=isotropic')
 vg=surface_geometry(model)
 print(f'Free surface: model={model.free_surface_model}; wall-centre delta_h={model.vortex_delta_h*1e3:g} mm; core radius a={model.vortex_core_radius*1e3:g} mm')
 if model.free_surface_model=='localized_scully_vortex' and model.vortex_delta_h>0:
  print(f'Scully diagnostic: Hinf={vg["h_inf"]*1e3:.3f} mm; area-mean uncorrected rise hbar={vg["h_bar"]*1e3:.3f} mm; equivalent core rotation={vg["equivalent_core_rpm"]:.2f} rpm (diagnostic only)')
 print(f'Sediment transport: {model.sediment_transport_model}; D_t={model.sediment_eddy_diffusivity:.6g} m^2/s; water rho={model.water_density:g} kg/m^3; mu={model.water_dynamic_viscosity*1e3:g} mPa.s')
 print('Population evolution remains OFF: no aggregation or fragmentation. V24.16 redistributes each existing PSD bin spatially and conserves its vessel-integrated mass.')
 print('Cell optics: cylindrical side wall plus CUDA flat/parabolic/localized water-air Fresnel/TIR free surface and finite 3-mm acrylic bottom-disc slab.')
 print(f'Ring: ID={2*model.ring_in*1e3:.1f} mm; OD={2*model.ring_out*1e3:.1f} mm')
 print(f'Bore: {2*model.throat_radius*1e3:.1f} mm dia through, narrow length={(throat_outer-model.ring_in)*1e3:.1f} mm; counterbore {2*model.counterbore_radius*1e3:.1f} mm dia x {model.counterbore_depth*1e3:.1f} mm')
 print(f'Source launch radius={model.source_launch_radius*1e3:.1f} mm (DGB-calibrated mechanical uncertainty; 57.5..65.5 mm audit range)')
 print(f'622-nm steady-state geometry: mean water height={model.water_height*1e3:.1f} mm; sensor height={model.sensor_height*1e3:.1f} mm; zbottom={model.zmin*1e3:.1f} mm; mean ztop={model.zmax*1e3:.1f} mm')
 print(f'Vortex surface centre z={vg["center_z"]*1e3:.1f} mm; wall z={vg["wall_z"]*1e3:.1f} mm; beam plane z=0 mm')
 print(f'CUDA source Beta prior: alpha1={model.alpha1:g}, alpha2={model.alpha2:g}; angular model={model.source_angular_model}')
 print(f'Bottom disc: {model.bottom_disc_thickness*1e3:.1f} mm acrylic; external n={model.bottom_external_n:g}')
 print(f'Legacy comparison: +/-{model.legacy_acceptance_deg:g} deg original-CLARITAS exit-position scorer')
 print(f'Centered source-to-inner-throat geometric half-angle ~{half_angle:.3f} deg (diagnostic only)')
 print('Selected detector response: reconstructed hard mechanical bore + x-reflection symmetry average. No KDE.')
 print('Stirring interpretation: continuous-on steady ensemble; settling is balanced by vortex-driven eddy mixing rather than a post-stir settling clock.')
 res=model.simulate(args.material,conc,n_rays=args.n_rays,seed=args.seed,collect_rays=args.save_rays,heatmap_size=args.heatmap_size,
  target_detector_score=args.target_detector_score,min_rays=args.min_rays,max_rays=args.max_rays,stability_l1_tolerance=args.stability_l1_tolerance)
 df=pd.DataFrame({
  'Detector_deg':res.detector_angles_deg.astype(int),
  'normalized_response':res.normalized_response,
  'normalized_hardware_symmetry_exact':res.normalized_hardware_symmetry_response,
  'normalized_hardware_native_exact':res.normalized_hardware_native_response,
  'normalized_hardware_mirror_exact':res.normalized_hardware_mirror_response,
  'hardware_jackknife_se_normalized':res.hardware_normalized_jackknife_se,
  'hardware_symmetry_physical_equivalent_score':res.hardware_symmetry_scores,
  'hardware_native_hits':res.hardware_native_hits,'hardware_mirror_hits':res.hardware_mirror_hits,
  'normalized_legacy_6p5_symmetry':res.normalized_legacy_symmetry_response,
  'normalized_legacy_6p5_native':res.normalized_legacy_native_response,
  'normalized_legacy_6p5_mirror':res.normalized_legacy_mirror_response,
  'legacy_6p5_jackknife_se_normalized':res.legacy_normalized_jackknife_se,
  'legacy_6p5_symmetry_channel_score':res.legacy_symmetry_channel_scores,
  'legacy_6p5_native_channel_count':res.legacy_native_channel_counts,
  'legacy_6p5_mirror_channel_count':res.legacy_mirror_channel_counts,
 })
 df.to_csv(out/'detector_response_normalized.csv',index=False)
 pd.DataFrame({
  'Detector_deg':res.detector_angles_deg.astype(int),
  'hardware_native_hits':res.hardware_native_hits,'hardware_mirror_hits':res.hardware_mirror_hits,
  'hardware_symmetry_physical_equivalent_score':res.hardware_symmetry_scores,
  'legacy_6p5_native_channel_count':res.legacy_native_channel_counts,
  'legacy_6p5_mirror_channel_count':res.legacy_mirror_channel_counts,
  'legacy_6p5_symmetry_channel_score':res.legacy_symmetry_channel_scores,
 }).to_csv(out/'detector_hits_exact.csv',index=False)
 pd.DataFrame(res.batch_diagnostics or []).to_csv(out/'batch_convergence.csv',index=False)
 (out/'diagnostics.json').write_text(json.dumps(res.to_dict(),indent=2))
 tr=res.particle_diagnostics or {}
 sed_mass_weights=np.asarray(tr.get('source_weight',[]),float)
 def mass_scale(key):
  a=np.asarray(tr.get(key,[]),float)
  return float(np.sum(sed_mass_weights*a)) if a.size and sed_mass_weights.size==a.size else float('nan')
 sediment_diag={
  'sediment_transport_model':res.sediment_transport_model,
  'eddy_diffusivity_m2_s':res.sediment_eddy_diffusivity_m2_s,
  'water_density_kg_m3':model.water_density,
  'water_dynamic_viscosity_pa_s':model.water_dynamic_viscosity,
  'particle_density_kg_m3':model.density,
  'gravity_m_s2':model.gravity,
  'transport_quadrature_points':model.sediment_transport_quadrature_points,
  'radial_drift_law':'w_r=tau_p*v_theta(r)^2/r',
  'settling_law':'w_s=(rho_p-rho_w)*g*d^2/(18*mu)',
  'stokes_response_time':'tau_p=rho_p*d^2/(18*mu)',
  'steady_field':'n_i(r,z)=nbar_i*exp[A_i*(f(r)-f(R))-B_i*(z-z_bottom)]/normalization_scaled_i',
  'f_r':'r^2/(a^2+r^2)',
  'A_i':'tau_i*Omega_c^2*a^2/(2*D_t)',
  'B_i':'w_s,i/D_t',
  'mass_conservation':'each PSD bin is independently volume-normalised over the actual curved cell',
  'optical_transport':'CUDA Woodcock/delta tracking with exact per-bin majorant; accepted event bin sampled from local mu_i',
  'majorant_geom_per_m':res.sediment_majorant_geom_per_m,
  'majorant_wave_per_m':res.sediment_majorant_wave_per_m,
  'majorant_total_per_m':res.sediment_majorant_total_per_m,
  'max_bin_density_scale':res.sediment_max_bin_density_scale,
  'bulk_mass_concentration_scale_sensor_center':mass_scale('sediment_scale_sensor_center'),
  'bulk_mass_concentration_scale_sensor_wall_r0p95R':mass_scale('sediment_scale_sensor_wall_r0p95R'),
  'bulk_mass_concentration_scale_bottom_center':mass_scale('sediment_scale_bottom_center'),
  'bulk_mass_concentration_scale_bottom_wall_r0p95R':mass_scale('sediment_scale_bottom_wall_r0p95R'),
  'scope_limit':'Stokes slip and scalar eddy diffusivity are a first process-based transport closure; no CFD or finite-Re drag correction is claimed in V24.16'
 }
 (out/'sediment_transport_diagnostics.json').write_text(json.dumps(sediment_diag,indent=2))
 vortex_diag={
  'free_surface_model':res.free_surface_model,
  'vortex_wall_center_height_difference_mm':res.vortex_wall_center_height_difference_mm,
  'vortex_core_radius_mm':res.vortex_core_radius_mm,
  'vortex_asymptotic_height_parameter_Hinf_mm':res.vortex_asymptotic_height_parameter_mm,
  'vortex_area_mean_uncorrected_rise_hbar_mm':res.vortex_area_mean_uncorrected_rise_mm,
  'vortex_equivalent_core_rpm_diagnostic':res.vortex_equivalent_core_rpm,
  'vortex_equivalent_rigid_body_rpm_legacy_parabolic_only':res.vortex_equivalent_rigid_body_rpm,
  'vortex_center_surface_z_mm_relative_sensor':res.vortex_center_surface_z_mm,
  'vortex_wall_surface_z_mm_relative_sensor':res.vortex_wall_surface_z_mm,
  'mean_top_surface_interactions_per_entered_ray':res.mean_top_surface_interactions,
  'mean_top_surface_reflections_per_entered_ray':res.mean_top_surface_reflections,
  'top_surface_tir_fraction_per_entered_ray':res.top_surface_tir_fraction,
  'top_exit_fraction_per_launched_ray':res.top_exit_fraction,
  'surface_equation':'z(r)=Zmean-hbar+Hinf*r^2/(a^2+r^2)',
  'velocity_ansatz':'v_theta(r)=Omega_c*r/(1+(r/a)^2)',
  'Hinf_relation':'Hinf=delta_h*(a^2+R^2)/R^2',
  'hbar_relation':'hbar=Hinf*[1-(a^2/R^2)*ln(1+R^2/a^2)]',
  'volume_preserving':True,
  'headspace_treatment':'transmitted rays are counted as top exits; reflected/TIR water-side paths remain fully traced and may re-hit the curved surface',
  'particle_baseline':'q=1.0, isotropic, slope=25 deg, wave height=100 nm; no population evolution; size bins may redistribute spatially under V24.16 transport'
 }
 (out/'vortex_diagnostics.json').write_text(json.dumps(vortex_diag,indent=2))
 population_diag={
  'population_model':res.aggregation_model,
  'aggregation_model':res.aggregation_model,
  'aggregation_collision_exposure_EA':res.aggregation_collision_exposure,
  'collision_exposure_E':res.aggregation_collision_exposure,
  'fragmentation_breakup_exposure_EB':res.fragmentation_breakup_exposure,
  'fragmentation_reference_diameter_um':res.fragmentation_reference_diameter_m*1e6,
  'fragmentation_size_exponent_m':res.fragmentation_size_exponent,
  'net_number_reduction_fraction':res.population_number_reduction_fraction,
  'number_reduction_fraction':res.population_number_reduction_fraction,
  'mass_conservation_error_fraction':res.aggregation_mass_conservation_error,
  'projected_area_ratio_final_over_initial':res.aggregation_projected_area_ratio,
  'largest_bin_mass_fraction':res.aggregation_largest_bin_mass_fraction,
  'aggregation_overflow_event_fraction':res.aggregation_overflow_event_fraction,
  'overflow_event_fraction':res.aggregation_overflow_event_fraction,
  'fragmentation_underflow_event_fraction':res.fragmentation_underflow_event_fraction,
  'aggregation_events_per_initial_particle':res.aggregation_events_per_initial_particle,
  'fragmentation_events_per_initial_particle':res.fragmentation_events_per_initial_particle,
  'integration_steps':res.aggregation_steps
 }
 (out/'population_balance_diagnostics.json').write_text(json.dumps(population_diag,indent=2))
 # Compatibility filename retained so V24.11-era analysis scripts fail gracefully rather than losing diagnostics.
 (out/'aggregation_diagnostics.json').write_text(json.dumps(population_diag,indent=2))
 if args.dgb_only:
  # Thesis Figure 4-3 (622-nm water-only calibrated radiant-intensity chart) was
  # exported as vector graphics.  The 0/10-degree values below are recovered
  # directly from the plotted coordinates; only their ratio is used as a DGB
  # source/collimator constraint.  The remaining channels lie on the zero line
  # of the linear chart and are therefore not treated as precise zero targets.
  ref_i0=3.513809804411889
  ref_i10=0.5966680711444741
  target_ratio=ref_i10/ref_i0
  model_ratio=float(res.normalized_response[1]/res.normalized_response[0]) if res.normalized_response[0]>0 else float('nan')
  dgb_cmp={
   'reference':'Ben Kitchener thesis Figure 4-3, 622 nm water-only baseline (vector-chart extraction)',
   'reference_I0_mW_sr_approx':ref_i0,
   'reference_I10_mW_sr_approx':ref_i10,
   'reference_H10_over_H0':target_ratio,
   'model_H10_over_H0':model_ratio,
   'log10_ratio_error':abs(math.log10(model_ratio/target_ratio)) if model_ratio>0 else float('inf'),
   'model_H0':float(res.normalized_response[0]),
   'model_H10':float(res.normalized_response[1]),
   'model_sum_20_170':float(np.sum(res.normalized_response[2:])),
   'model_H160':float(res.normalized_response[16]),
   'model_H170':float(res.normalized_response[17]),
   'source_launch_radius_mm':model.source_launch_radius*1e3,
   'alpha1':model.alpha1,'alpha2':model.alpha2,'source_angular_model':model.source_angular_model,
   'water_height_mm':model.water_height*1e3,'sensor_height_mm':model.sensor_height*1e3,
   'bottom_disc_thickness_mm':model.bottom_disc_thickness*1e3,
   'bottom_external_n':model.bottom_external_n
  }
  (out/'dgb_622_reference_comparison.json').write_text(json.dumps(dgb_cmp,indent=2))
 geom.update({
  'stirrer_state_for_interpretation':'continuous_on',
  'forward_model_authority':'CUDA: Mie/phase/coherence/event/source/transport/particle/detector physics; Python: orchestration/diagnostics/I-O only',
  'detector_estimator':'reconstructed_hard_bore_symmetry_average_exact',
  'through_bore_outer_radius_m':model.throat_out,
  'through_bore_length_m':model.throat_out-model.ring_in,
  'source_launch_radius_m':model.source_launch_radius,
  'water_height_m':model.water_height,
  'sensor_height_above_bottom_m':model.sensor_height,
  'water_free_surface_mean_z_relative_sensor_m':model.zmax,
  'free_surface_model':model.free_surface_model,
  'vortex_wall_center_height_difference_m':model.vortex_delta_h,
  'vortex_core_radius_m':model.vortex_core_radius,
  'vortex_center_surface_z_relative_sensor_m':res.vortex_center_surface_z_mm*1e-3,
  'vortex_wall_surface_z_relative_sensor_m':res.vortex_wall_surface_z_mm*1e-3,
  'vortex_asymptotic_height_parameter_m':res.vortex_asymptotic_height_parameter_mm*1e-3,
  'vortex_area_mean_uncorrected_rise_m':res.vortex_area_mean_uncorrected_rise_mm*1e-3,
  'vortex_equivalent_core_rpm_diagnostic':res.vortex_equivalent_core_rpm,
  'vortex_equivalent_rigid_body_rpm_legacy_parabolic_only':res.vortex_equivalent_rigid_body_rpm,
  'bottom_water_acrylic_z_relative_sensor_m':model.zmin,
  'bottom_disc_thickness_m':model.bottom_disc_thickness,
  'bottom_external_refractive_index':model.bottom_external_n,
  'source_beta_alpha1':model.alpha1,
  'source_beta_alpha2':model.alpha2,
  'source_angular_model':model.source_angular_model,
  'source_angular_measure_note':'beta_radiance_3d treats the historical Beta law as angular radiance I(theta); CUDA samples p(theta) proportional to BetaPDF(theta/(pi/2))*sin(theta) with uniform azimuth.',
  'centered_source_to_inner_throat_half_angle_deg_diagnostic':half_angle,
  'legacy_comparison':'original CLARITAS +/-6.5 deg boundary-exit-position channel scorer; overlapping channels; not selected',
  'paper_dimension_choice':'Fig.3/Fig.4 machining values: 4 mm through bore, 8.7 mm counterbore x 8 mm deep; section 5.3 prose states 3 mm/10 mm and is treated as conflicting text.',
  'particle_optics_release':'V24.16 CUDA-authoritative vortex sediment drift-diffusion diagnostic built on frozen V24.10.2 source/apparatus and V24.15 surface state',
  'particle_wavelength_m':model.particle_wavelength,
  'particle_event_model':model.particle_event_model,
  'particle_event_cross_section':'hybrid: geometric pi*r^2 plus wave_scale*max(Qsca-1,0)*pi*r^2; legacy V24.6 event models retained for A/B',
  'mie_phase_function':'CUDA-precomputed homogeneous-sphere Mie CDF used for coherent supplemental wave events; reference q=1 morphology remains frozen',
  'geometric_particle_interface':'CUDA V23/V24.5 sphere geometry with V24.8 Beckmann-distributed local optical microfacet normals',
  'particle_surface_rms_slope_deg':model.particle_surface_rms_slope_deg,
  'particle_surface_beckmann_alpha':model.particle_surface_beckmann_alpha,
  'wave_surface_rms_height_nm':model.wave_surface_rms_height_m*1e9,
  'wave_coherence_law':'C=exp[-(4*pi*n_water*sigma_h/lambda0)^2], evaluated in CUDA; C branch=Mie, 1-C branch=direction-only explicit spheroid Fresnel/Snell',
  'wave_spheroid_aspect_ratio':model.wave_spheroid_aspect_ratio,
  'wave_spheroid_geometry':'volume-preserving semi-axes a=r*q^(2/3), b=r*q^(-1/3); q=a/b; projected-silhouette interception sampled in CUDA',
  'wave_spheroid_orientation_model':'isotropic',
  'wave_spheroid_orientation_kappa':0.0,
  'wave_spheroid_orientation_law':'frozen isotropic reference',
  'aggregation_model':model.aggregation_model,
  'aggregation_collision_exposure_EA':model.aggregation_collision_exposure,
  'fragmentation_breakup_exposure_EB':model.fragmentation_breakup_exposure,
  'fragmentation_reference_diameter_m':model.fragmentation_reference_diameter_m,
  'fragmentation_size_exponent_m':model.fragmentation_size_exponent,
  'aggregation_process':'dormant inherited V24.12 capability; disabled in this release diagnostic',
  'fragmentation_process':'dormant inherited V24.12 capability; disabled in this release diagnostic',
  'population_balance_interpretation':'aggregation/fragmentation disabled; V24.16 conserves the bulk PSD while redistributing each size bin spatially by drift-diffusion',
  'sediment_transport_model':model.sediment_transport_model,
  'sediment_eddy_diffusivity_m2_s':model.sediment_eddy_diffusivity,
  'sediment_water_density_kg_m3':model.water_density,
  'sediment_water_dynamic_viscosity_pa_s':model.water_dynamic_viscosity,
  'sediment_gravity_m_s2':model.gravity,
  'sediment_transport_quadrature_points':model.sediment_transport_quadrature_points,
  'sediment_transport_law':'size-resolved Stokes centrifugal drift + Stokes settling balanced by scalar eddy diffusion; each bin independently volume-normalised',
  'sediment_event_transport':'CUDA Woodcock/delta tracking with per-bin density majorants and local bin sampling',
  'hybrid_wave_scale':model.hybrid_wave_scale,
  'mie_phase_grid_size':model.mie_phase_grid_size,
  'absorption_law':'I/I0=exp(-4*pi*k*L/lambda) inside particle',
  'complex_fresnel':'unpolarised amplitude Fresnel coefficients evaluated with complex particle index; ray direction uses real n'})
 (out/'apparatus_geometry.json').write_text(json.dumps(geom,indent=2))
 d,w=get_material_psd(args.material)
 nbin,kbin=get_material_optical_constants(args.material,model.particle_optical_constants_by_material,model.n_particle)
 tr=res.particle_diagnostics or {}
 alpha_abs=4.0*math.pi*kbin/model.particle_wavelength
 abs_len=np.full_like(alpha_abs,np.inf,dtype=float);np.divide(1.0,alpha_abs,out=abs_len,where=alpha_abs>0)
 pd.DataFrame({
  'diameter_um':np.asarray(tr['diameter_m'])*1e6,'input_source_weight':tr['input_source_weight'],'effective_population_weight':tr['effective_population_weight'],'effective_aggregation_weight_compat':tr['effective_aggregation_weight'],'source_weight':tr['source_weight'],'event_weight':tr['particle_event_weights'],
  'geometric_event_weight':tr['geometric_event_weights'],'wave_event_weight':tr['wave_event_weights'],
  'n_real':tr['n_real'],'k_imag':tr['k_imag'],'size_parameter_water':tr['size_parameter'],
  'Qext':tr['qext'],'Qsca':tr['qsca'],'Qabs':tr['qabs'],'Qback':tr['qback'],
  'single_scattering_albedo':tr['single_scattering_albedo'],'mie_g':tr['asymmetry_g'],
  'geometric_cross_section_m2':tr['geometric_cross_section_m2'],'scattering_cross_section_m2':tr['scattering_cross_section_m2'],
  'extinction_cross_section_m2':tr['extinction_cross_section_m2'],'event_cross_section_m2':tr['event_cross_section_m2'],'wave_cross_section_m2':tr['wave_cross_section_m2'],
  'mu_geom_by_bin_per_m':tr['mu_geom_by_bin'],'mu_sca_by_bin_per_m':tr['mu_sca_by_bin'],'mu_ext_by_bin_per_m':tr['mu_ext_by_bin'],
  'mu_wave_by_bin_per_m':tr['mu_wave_by_bin'],'mu_event_by_bin_per_m':tr['mu_event_by_bin'],
  'sediment_response_time_ms':np.asarray(tr['sediment_response_time_s'])*1e3,'sediment_settling_velocity_mm_s':np.asarray(tr['sediment_settling_velocity_m_s'])*1e3,
  'sediment_radial_log_coefficient_A':tr['sediment_radial_log_coefficient_A'],'sediment_vertical_inverse_length_B_per_m':tr['sediment_vertical_inverse_length_B_per_m'],
  'sediment_normalization_scaled':tr['sediment_normalization_scaled'],'sediment_max_density_scale':tr['sediment_max_density_scale'],
  'sediment_scale_sensor_center':tr['sediment_scale_sensor_center'],'sediment_scale_sensor_wall_r0p95R':tr['sediment_scale_sensor_wall_r0p95R'],
  'sediment_scale_bottom_center':tr['sediment_scale_bottom_center'],'sediment_scale_bottom_wall_r0p95R':tr['sediment_scale_bottom_wall_r0p95R'],
  'absorption_coefficient_per_m':alpha_abs,'intensity_absorption_length_um':abs_len*1e6
 }).to_csv(out/'particle_interaction_diagnostics.csv',index=False)
 pd.DataFrame({
  'diameter_um':np.asarray(tr['diameter_m'])*1e6,
  'bulk_mass_fraction':tr['source_weight'],
  'response_time_ms':np.asarray(tr['sediment_response_time_s'])*1e3,
  'settling_velocity_mm_s':np.asarray(tr['sediment_settling_velocity_m_s'])*1e3,
  'radial_log_coefficient_A':tr['sediment_radial_log_coefficient_A'],
  'vertical_inverse_length_B_per_m':tr['sediment_vertical_inverse_length_B_per_m'],
  'normalization_scaled':tr['sediment_normalization_scaled'],
  'max_local_number_density_over_bin_mean':tr['sediment_max_density_scale'],
  'scale_sensor_center':tr['sediment_scale_sensor_center'],
  'scale_sensor_wall_r0p95R':tr['sediment_scale_sensor_wall_r0p95R'],
  'scale_bottom_center':tr['sediment_scale_bottom_center'],
  'scale_bottom_wall_r0p95R':tr['sediment_scale_bottom_wall_r0p95R'],
 }).to_csv(out/'sediment_transport_by_size.csv',index=False)
 pd.DataFrame({'diameter_um':d*1e6,'n_real':nbin,'k_imag':kbin,'absorption_coefficient_per_m':alpha_abs,'intensity_absorption_length_um':abs_len*1e6}).to_csv(out/'particle_optical_constants.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,5));ax.errorbar(res.detector_angles_deg,res.normalized_hardware_symmetry_response,yerr=res.hardware_normalized_jackknife_se,fmt='o-',capsize=2,label=f'V24.16 {model.sediment_transport_model}, Dt={model.sediment_eddy_diffusivity:.2g} m2/s')
 ax.plot(res.detector_angles_deg,res.normalized_legacy_symmetry_response,'--',label=f'legacy +/-{res.legacy_acceptance_deg:g} deg comparison')
 ax.set_xticks(DETECTOR_ANGLES_DEG);ax.set_xlabel('Detector angle (deg)');ax.set_ylabel('Normalized response');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'detector_response_normalized.png',dpi=200);plt.close(fig)
 if args.save_rays and res.ray_data is not None: pd.DataFrame(res.ray_data).to_csv(out/'ray_states.csv',index=False)
 print(f'Saved V24.16 results to {out}')
 print(f'actual rays={res.n_rays:,}; stop={res.stop_reason}; hardware native={res.hardware_native_hits.sum():,}; mirror={res.hardware_mirror_hits.sum():,}; equivalent={res.hardware_symmetry_scores.sum():.1f}')
 print(f'hardware detection fraction selected={res.detector_detection_fraction:.6g}; legacy channel-score/ray={res.legacy_channel_score_per_ray:.6g}')
 print(f'particle absorbed-ray fraction among entered-water rays={res.particle_absorption_fraction:.6g} ({res.particle_absorbed_ray_count:,} rays)')
 print(f'particle surface RMS slope={res.particle_surface_rms_slope_deg:g} deg; Beckmann alpha={res.particle_surface_beckmann_alpha:.6g}')
 print(f'event model={res.particle_event_model}; mu_geom={res.mu_geom_per_m:.6g}/m; mu_wave={res.mu_wave_per_m:.6g}/m; mu_event={res.mu_event_per_m:.6g}/m; tau_event(diameter)={res.optical_depth_diameter:.6g}')
 print(f'sediment transport={res.sediment_transport_model}; D_t={res.sediment_eddy_diffusivity_m2_s:.6g} m^2/s; Woodcock majorant={res.sediment_majorant_total_per_m:.6g}/m; max per-bin density scale={res.sediment_max_bin_density_scale:.6g}')
 print(f'axial optics: top-reflections/ray-entered={res.mean_top_surface_reflections:.6g}; top-TIR fraction={res.top_surface_tir_fraction:.6g}; bottom-inner-reflections/ray-entered={res.mean_bottom_inner_reflections:.6g}; bottom-outer-reflections/ray-entered={res.mean_bottom_outer_reflections:.6g}; bottom-TIR fraction={res.bottom_tir_fraction:.6g}; top-exit={res.top_exit_fraction:.6g}; bottom-exit={res.bottom_exit_fraction:.6g}')
 if args.dgb_only and res.normalized_response[0]>0:
  print(f'DGB 622 target H10/H0~0.169807; model H10/H0={res.normalized_response[1]/res.normalized_response[0]:.6g}; sum20-170={np.sum(res.normalized_response[2:]):.6g}')
 print(f'mean interactions={res.mean_interactions:.6g}; geometric={res.mean_geometric_interactions:.6g}; wave={res.mean_wave_interactions:.6g}; ballistic={res.ballistic_fraction:.6g}')
 print(f'free surface: model={res.free_surface_model}; delta_h={res.vortex_wall_center_height_difference_mm:g} mm; core a={res.vortex_core_radius_mm:g} mm; centre z={res.vortex_center_surface_z_mm:.3f} mm; wall z={res.vortex_wall_surface_z_mm:.3f} mm')
 if res.free_surface_model=='localized_scully_vortex' and res.vortex_wall_center_height_difference_mm>0:
  print(f'localized vortex diagnostics: Hinf={res.vortex_asymptotic_height_parameter_mm:.3f} mm; hbar={res.vortex_area_mean_uncorrected_rise_mm:.3f} mm; equivalent core rpm={res.vortex_equivalent_core_rpm:.3f} (not measured stirrer speed)')
 print(f'top-surface interactions/ray-entered={res.mean_top_surface_interactions:.6g}; population evolution={res.aggregation_model}')
 print(f'wave coherent fraction nominal CUDA={res.wave_coherent_fraction_nominal_cuda:.6g}; spheroid aspect ratio={res.wave_spheroid_aspect_ratio:g}; mean coherent-wave={res.mean_coherent_wave_interactions:.6g}; mean morphology-wave={res.mean_morphology_wave_interactions:.6g}; realized morphology fraction={res.morphology_fraction_of_wave_interactions:.6g}')
if __name__=='__main__':main()
