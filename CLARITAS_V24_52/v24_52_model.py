from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from v24_52_release import HERE,CONFIG_NAME,RELEASE
import claritas_tardiis_core_v24_52 as corrected
import v24_52_legacy_core as legacy

def load_config(path=None):
 p=HERE/CONFIG_NAME if path is None else Path(path);cfg=json.loads(p.read_text())
 if cfg.get('version')!=RELEASE: raise RuntimeError('release/config mismatch')
 b=cfg['boundary_process_v24_52'];f=cfg['forward_model']
 if abs(float(f.get('tube_length_m',-1.0))-0.491)>1e-12: raise RuntimeError('V24.52 canonical physical tube length must be 0.491 m')
 if float(f['particle_surface_rms_slope_deg'])!=25.0: raise RuntimeError('25-degree roughness must remain frozen')
 if any(np.any(np.asarray(v['k_imag'],float)!=0) for v in f['particle_optical_constants_by_material'].values()): raise RuntimeError('k=0 required')
 if f['particle_event_model']!='hybrid_mie_exact_rate': raise RuntimeError('V24.29 event-rate model required')
 if int(f['max_internal_bounces'])+1!=int(b['watchdog_internal_reflections']): raise RuntimeError('corrected macro-watchdog semantic mismatch')
 if int(b['microsurface_scattering_watchdog'])!=4096: raise RuntimeError('V24.52 microsurface watchdog must remain fail-only 4096')
 if b.get('mode')!='smith_beckmann_dielectric_signed_vndf_microsurface_random_walk': raise RuntimeError('V24.52 boundary-process mismatch')
 q=cfg.get('production_qualification_v24_52',{})
 if not (q.get('uses_real_production_trace') and q.get('material')=='loess' and float(q.get('concentration_g_per_L',-1))==0.5 and int(q.get('n_rays',-1))==1_000_000 and int(q.get('seed',-1))==2_441_000):
  raise RuntimeError('V24.52 compulsory production-qualification configuration mismatch')
 if not all(bool(q.get(k,False)) for k in ('zero_failures_required','zero_watchdogs_required','zero_nonfinite_required','zero_geometry_failures_required','zero_water_no_boundary_required','zero_headspace_failures_required','authorization_receipt_written_only_after_pass')):
  raise RuntimeError('V24.52 production qualification must remain fail-closed')
 if q.get('h170_used_for_design_or_acceptance') is not False or q.get('scientific_result_finalized') is not False:
  raise RuntimeError('V24.52 production qualification must remain engineering-only')
 g=cfg.get('geometry_validation_v24_52',{})
 if not (g.get('uses_shared_production_geometry_helpers') and int(g.get('randomized_connected_solid_samples',0))>=500000):
  raise RuntimeError('V24.52 connected-solid geometry validation configuration mismatch')
 required_geometry_flags=(
  'zero_topology_mismatches_required','zero_near_boundary_mismatches_required','sub_eps_forward_hits_required',
  'connected_acrylic_union_required','no_inner_interface_below_zmin_required','progress_watchdog_required',
  'particle_actual_center_required','particle_rng_single_sample_required','raw_cuda_state_c_contiguous_required',
  'fortran_layout_regression_required','particle_reference_finite_footprint_required',
  'particle_reference_tolerance_parity_required','terminal_state_semantics_required',
  'local_replay_release_gate_required','terminal_comparator_semantics_separate_from_global_orbit_match',
  'stale_process_decision_regression_required','cuda_scalar_input_parity_required',
  'case_1650_input_parity_regression_required','reference_math_high_precision_required',
  'tolerance_relaxation_forbidden','reference_uses_cuda_effective_float32_scalars',
  'geometry_audit_stops_after_local_replay_closure','complete_apparatus_geometry_required',
  'canonical_geometry_manifest_required','full_region_topology_required','surface_interface_manifest_required',
  'v24_50_failure_state_regression_required','headspace_transport_required',
  'wall_top_free_surface_separation_required','finite_outer_wall_support_required',
  'finite_inner_wall_support_required','open_top_rim_required','detector_source_geometry_consistency_required',
  'stop_geometry_audit_after_complete_gate','historical_local_replay_regression_required',
  'complete_apparatus_geometry_gate_is_final_geometry_gate','sediment_particle_geometry_consistency_required',
  'all_stored_v24_50_failure_samples_required')
 if not all(bool(g.get(k,False)) for k in required_geometry_flags):
  missing=[k for k in required_geometry_flags if not bool(g.get(k,False))]
  raise RuntimeError('V24.52 geometry validation must remain fail-closed; missing/false: '+','.join(missing))
 if g.get('old_global_orbit_comparison_release_gate') is not False or g.get('global_orbit_comparison_diagnostic_only') is not True:
  raise RuntimeError('V24.52 global-orbit comparison must remain diagnostic-only')
 if int(g.get('random_region_classification_samples',0))<250000:
  raise RuntimeError('V24.52 whole-apparatus random region classification must use at least 250000 states')
 if float(g.get('local_replay_distance_tolerance_m',-1))!=5e-7: raise RuntimeError('V24.52 local-replay distance tolerance must remain 5e-7 m')
 if g.get('h170_used_for_design_or_acceptance') is not False:
  raise RuntimeError('V24.52 geometry validation must remain engineering-only')
 return cfg

def particle_optics_override(coremod):
 out={}
 for mat in ('loess','kaolin'):
  n,k=coremod.get_material_optical_constants(mat,None,1.59)
  out[mat]={'n_real':np.asarray(n,float).tolist(),'k_imag':np.zeros_like(np.asarray(k,float)).tolist()}
 return out

def model_kwargs(cfg,legacy_mode=False):
 d=dict(cfg['forward_model']);coremod=legacy if legacy_mode else corrected
 d['particle_optical_constants_by_material']=particle_optics_override(coremod)
 d.update(source_bore_surface_model='absorbing',detector_bore_surface_model='absorbing',ring_inner_face_surface_model='absorbing',
          photodiode_response_model='ideal_hard_aperture',sediment_transport_model='uniform',meridional_circulation_ratio=0.0,
          aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
          particle_event_model='hybrid_mie_exact_rate',particle_surface_rms_slope_deg=25.0,wave_surface_rms_height_m=100e-9,
          wave_spheroid_aspect_ratio=1.0,wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
          source_launch_radius_m=0.0655,alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d')
 d['max_internal_bounces']=int(cfg['boundary_process_v24_52']['legacy_max_internal_bounces'] if legacy_mode else cfg['forward_model']['max_internal_bounces'])
 if legacy_mode: d.pop('tube_length_m',None)
 return d

def make_model(cfg=None,legacy_mode=False):
 cfg=load_config() if cfg is None else cfg;mod=legacy if legacy_mode else corrected
 return mod.TardiisForwardModel(**model_kwargs(cfg,legacy_mode=legacy_mode))
