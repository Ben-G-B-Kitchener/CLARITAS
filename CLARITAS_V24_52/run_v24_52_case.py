#!/usr/bin/env python3
"""CLARITAS V24.52.0 corrected rough-boundary production case runner.

Controlled experiment: keep the complete V24.27 particle-event angular treatment and
causal diagnostics, but enforce sigma_event(d)=sigma_sca,Mie(d) bin-by-bin. For
Qsca>=1 the V24.27 geometric + Mie-excess split is unchanged. For Qsca<1 the
geometric branch is thinned to the exact scattering cross-section and the wave branch
is zero. Both loess and kaolin are mandatory. H150/H160 remain excluded from
sediment fit metrics because of the known experimental sensor problems.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
from claritas_tardiis_core_v24_52 import TardiisForwardModel, DETECTOR_ANGLES_DEG, get_material_optical_constants, get_material_psd, V2427_PARTICLE_EVENT_TYPE_NAMES, PRODUCTION_FAILURE_NAMES, PRODUCTION_FAILURE_SUBCODE_NAMES

HERE=Path(__file__).resolve().parent
EXCLUDED_SEDIMENT_DETECTORS_DEG=(150,160)
TRUSTED_MASK=~np.isin(DETECTOR_ANGLES_DEG,np.asarray(EXCLUDED_SEDIMENT_DETECTORS_DEG,float))

DEFAULTS=dict(
 n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,particle_wavelength_m=622e-9,
 tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,tube_length_m=0.491,water_height_m=0.142,sensor_height_above_bottom_m=0.093,
 sensor_ring_inner_radius_m=0.0505,sensor_ring_outer_radius_m=0.0655,counterbore_depth_m=0.008,
 through_bore_diameter_m=0.004,counterbore_diameter_m=0.0087,source_launch_radius_m=0.0655,
 legacy_detector_acceptance_deg=6.5,source_beam_sigma_m=1e-5,
 source_bore_surface_model='absorbing',detector_bore_surface_model='absorbing',bore_refractive_index=1.53,max_bore_bounces=16,
 ring_inner_face_surface_model='absorbing',ring_inner_face_refractive_index=1.53,max_ring_face_bounces=16,
 photodiode_response_model='ideal_hard_aperture',alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d',
 density_kg_per_m3=2600.0,bottom_disc_thickness_m=0.003,bottom_external_refractive_index=1.0,max_axial_bounces=64,
 max_internal_bounces=4095,max_cell_bounces=64,chunk_size=250000,statistics_batch_rays=250000,symmetry_average_exact=True,
 particle_event_model='hybrid_mie_exact_rate',hybrid_wave_scale=1.0,mie_phase_grid_size=2049,particle_surface_rms_slope_deg=25.0,
 wave_surface_rms_height_m=100e-9,wave_spheroid_aspect_ratio=1.0,wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
 free_surface_model='localized_scully_vortex',vortex_wall_center_height_difference_m=0.050,vortex_core_radius_m=0.020,
 sediment_transport_model='uniform',sediment_eddy_diffusivity_m2_s=1e-3,meridional_circulation_ratio=0.0,
 meridional_grid_nr=48,meridional_grid_nz=96,meridional_solver_max_iterations=8000,meridional_solver_check_interval=250,
 meridional_solver_tolerance=1e-6,meridional_solver_sor=1.75,water_density_kg_per_m3=997.0,
 water_dynamic_viscosity_pa_s=8.9e-4,gravity_m_s2=9.81,sediment_transport_quadrature_points=4096,sediment_radial_lookup_points=2049,
 aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
 fragmentation_reference_diameter_m=100e-6,fragmentation_size_exponent=1.0,aggregation_step_safety=0.2,aggregation_max_steps=200000,
)

def particle_optics_override():
 out={}
 for mat in ('loess','kaolin'):
  n,k=get_material_optical_constants(mat,None,1.59)
  out[mat]={'n_real':n.tolist(),'k_imag':np.zeros_like(k).tolist()}
 return out

def make_model(config_path=None):
 d=DEFAULTS.copy()
 if config_path:
  cfg=json.loads(Path(config_path).read_text());d.update(cfg.get('forward_model',cfg))
 # Production lock after config: V24.29 changes only particle-event rate for Qsca<1; V24.27 angular/apparatus state is frozen.
 d.update(source_bore_surface_model='absorbing',detector_bore_surface_model='absorbing',ring_inner_face_surface_model='absorbing',
          photodiode_response_model='ideal_hard_aperture',sediment_transport_model='uniform',meridional_circulation_ratio=0.0,
          aggregation_model='none',aggregation_collision_exposure=0.0,fragmentation_breakup_exposure=0.0,
          particle_event_model='hybrid_mie_exact_rate',particle_surface_rms_slope_deg=25.0,wave_surface_rms_height_m=100e-9,
          wave_spheroid_aspect_ratio=1.0,wave_spheroid_orientation_model='isotropic',wave_spheroid_orientation_kappa=0.0,
          source_launch_radius_m=0.0655,alpha1=0.45,alpha2=5.0,source_angular_model='beta_radiance_3d')
 d['particle_optical_constants_by_material']=particle_optics_override()
 return TardiisForwardModel(**d),d

def measured_curve(material,conc):
 df=pd.read_csv(HERE/'measured_detector_responses_v3_5.csv')
 q=df[(df.material.str.lower()==material.lower())&np.isclose(df.concentration_g_per_L,float(conc))].sort_values('detector_deg')
 if len(q)!=18: raise RuntimeError(f'expected 18 measured channels for {material} {conc} g/L, got {len(q)}')
 return q.measured_normalized.to_numpy(float)

def trusted_normalize(v):
 a=np.asarray(v,float);out=np.full_like(a,np.nan,dtype=float);den=float(a[TRUSTED_MASK].sum())
 if den>0: out[TRUSTED_MASK]=a[TRUSTED_MASK]/den
 return out

def fresnel_R_unpolarized(theta_i_rad,n1,n2):
 ci=float(math.cos(theta_i_rad));si=float(math.sin(theta_i_rad));st=n1/n2*si
 if st>=1: return 1.0
 ct=math.sqrt(max(0.0,1.0-st*st))
 rs=((n1*ci-n2*ct)/(n1*ci+n2*ct))**2
 rp=((n1*ct-n2*ci)/(n1*ct+n2*ci))**2
 return 0.5*(rs+rp)

def external_reflection_band_probability(n1,n2,lo_deg=165.0,hi_deg=175.0):
 # Smooth sphere, uniformly sampled projected disk. theta=pi-2*i.
 th=np.linspace(math.radians(lo_deg),math.radians(hi_deg),2001)
 vals=[]
 for q in th:
  inc=0.5*(math.pi-q); vals.append(0.5*math.sin(q)*fresnel_R_unpolarized(inc,n1,n2))
 vals=np.asarray(vals,dtype=float)
 if hasattr(np,'trapezoid'): return float(np.trapezoid(vals,th))
 # Compatibility fallback for older NumPy without depending on a removed NumPy integration alias.
 dx=np.diff(th); return float(np.sum(0.5*(vals[:-1]+vals[1:])*dx))

def save_csa_table(out,material,conc,res,cfg):
 ph=res.particle_diagnostics;d,w=get_material_psd(material);w=np.asarray(w,float)
 n=np.asarray(ph['n_real'],float);k=np.asarray(ph['k_imag'],float);mu=np.asarray(ph['mu_geom_by_bin'],float)
 qback=np.asarray(ph['qback'],float);qsca=np.asarray(ph['qsca'],float);qext=np.asarray(ph['qext'],float);g=np.asarray(ph['asymmetry_g'],float)
 nd=np.asarray(ph['number_density_by_bin'],float);sg=np.asarray(ph['geometric_cross_section_m2'],float);mus=np.asarray(ph['mu_sca_by_bin'],float);muw=np.asarray(ph['mu_wave_by_bin'],float)
 p165=np.asarray(ph.get('mie_phase_probability_165_175_by_bin',np.zeros_like(mu)),float)
 p169=np.asarray(ph.get('mie_phase_probability_169_171_by_bin',np.zeros_like(mu)),float)
 mue=np.asarray(ph.get('mu_event_by_bin',mus),float)
 mueffg=np.asarray(ph.get('mu_effective_geometric_event_by_bin',np.maximum(mue-muw,0.0)),float)
 legacy_mu=np.asarray(ph.get('legacy_hybrid_mu_event_by_bin',mue),float)
 coh=float(ph['wave_coherent_fraction_nominal_cuda']);nwater=float(cfg['n_water'])
 R0=np.asarray([fresnel_R_unpolarized(0.0,nwater,x) for x in n])
 R170=np.asarray([fresnel_R_unpolarized(math.radians(5.0),nwater,x) for x in n])
 band=np.asarray([external_reflection_band_probability(nwater,x) for x in n])
 csa_sum=float(mu.sum());frac=np.divide(mu,csa_sum,out=np.zeros_like(mu),where=csa_sum>0)
 omega165=2*math.pi*(math.cos(math.radians(165))-math.cos(math.radians(175)))
 smooth_diff170=mu*R170/(4*math.pi)
 smooth_band=mu*band
 mie_coherent_band=muw*coh*p165
 mie_coherent_diff170=muw*coh*p169/(2*math.pi*(math.cos(math.radians(169))-math.cos(math.radians(171))))
 tab=pd.DataFrame(dict(bin_index=np.arange(len(d)),diameter_um=d*1e6,mass_fraction=w,n_real=n,k_imag=k,
  number_density_m3=nd,particle_projected_csa_m2=sg,projected_csa_density_per_m=mu,projected_csa_fraction=frac,
  projected_csa_m2_per_L=mu*1e-3,qsca=qsca,qext=qext,qback=qback,asymmetry_g=g,scattering_coefficient_per_m=mus,
  mie_backscatter_efficiency_proxy_per_m=mu*qback,fresnel_R_normal=R0,fresnel_R_for_170deg_specular=R170,
  smooth_external_reflection_differential_170_per_m_sr=smooth_diff170,smooth_external_reflection_band_165_175_per_m=smooth_band,
  mie_phase_probability_165_175=p165,mie_phase_probability_169_171=p169,
  exact_mie_event_coefficient_by_bin_per_m=mue,effective_geometric_event_coefficient_by_bin_per_m=mueffg,
  legacy_v24_27_hybrid_event_coefficient_by_bin_per_m=legacy_mu,legacy_minus_exact_event_coefficient_by_bin_per_m=legacy_mu-mue,
  coherent_mie_wave_band_165_175_per_m=mie_coherent_band,coherent_mie_wave_differential_170_per_m_sr=mie_coherent_diff170))
 tab.to_csv(out/'particle_csa_backreflectance_by_bin.csv',index=False)
 return dict(projected_csa_density_per_m=csa_sum,projected_csa_m2_per_L=csa_sum*1e-3,
             geometric_optical_depth_across_inner_diameter=csa_sum*2*float(cfg['tube_inner_radius_m']),
             fresnel_weighted_csa_normal_per_m=float(np.sum(mu*R0)),fresnel_weighted_csa_170_per_m=float(np.sum(mu*R170)),
             smooth_external_reflection_differential_170_per_m_sr=float(smooth_diff170.sum()),
             smooth_external_reflection_band_165_175_per_m=float(smooth_band.sum()),
             mie_backscatter_efficiency_proxy_per_m=float(np.sum(mu*qback)),
             coherent_mie_wave_band_165_175_per_m=float(mie_coherent_band.sum()),
             coherent_mie_wave_differential_170_per_m_sr=float(mie_coherent_diff170.sum()),
             exact_mie_event_coefficient_per_m=float(mue.sum()),
             effective_geometric_event_coefficient_per_m=float(mueffg.sum()),
             wave_event_coefficient_per_m=float(muw.sum()),
             legacy_v24_27_hybrid_event_coefficient_per_m=float(legacy_mu.sum()),
             legacy_over_exact_event_rate_ratio=float(legacy_mu.sum()/mue.sum()) if mue.sum()>0 else float('nan'),
             qsca_lt1_projected_csa_fraction=float(np.sum(mu[qsca<1.0])/csa_sum) if csa_sum>0 else 0.0)

def _summary_values_equal(a,b):
 """Strict-enough scalar comparison used only for duplicate diagnostic-key protection."""
 try:
  af=float(a);bf=float(b)
  if math.isnan(af) and math.isnan(bf): return True
  return af==bf
 except (TypeError,ValueError):
  return a==b

def _merge_summary_fields(dst,src,source_name):
 """Merge without allowing a silently conflicting summary key.

 Identical duplicate values are accepted because a small number of named H170
 diagnostics are intentionally promoted explicitly and are also present in the
 generic per-channel diagnostic map. A mismatch is treated as an internal error.
 """
 for key,value in src.items():
  if key in dst:
   if not _summary_values_equal(dst[key],value):
    raise RuntimeError(f'case-summary key collision from {source_name}: {key!r} has conflicting values {dst[key]!r} and {value!r}')
   continue
  dst[key]=value
 return dst

def _build_case_summary(args,res,p,csa,meas,model_all,meas_tr,model_tr,score,rmse,corr):
 h=17
 summary=dict(release='V24.52.0',material=args.material,concentration_g_per_L=float(args.concentration),n_rays=int(res.n_rays),stop_reason=res.stop_reason,
  source_beta_alpha1=0.45,source_beta_alpha2=5.0,source_model='beta_radiance_3d',photodiode_model='ideal_hard_aperture',
  particle_optics='V24.18 real-index-only n(d), k=0',particle_event_model=res.particle_event_model,particle_surface_model='V24.52 direct-double-support full-domain signed Smith-Beckmann dielectric microsurface random walk; V24.41 stable Lambda/log-height and V24.29 exact-Mie event rate frozen',
  particle_surface_rms_slope_deg=float(res.particle_surface_rms_slope_deg),sediment_transport_model='uniform',
  excluded_sediment_detectors_deg=list(EXCLUDED_SEDIMENT_DETECTORS_DEG),trusted_detector_count=int(TRUSTED_MASK.sum()),
  trusted_RMSE=float(rmse),trusted_correlation=float(corr),measured_H170_all18_normalized=float(meas[h]),model_H170_all18_normalized=float(model_all[h]),
  measured_H170_trusted_renormalized=float(meas_tr[h]),model_H170_trusted_renormalized=float(model_tr[h]),H170_hardware_score=float(score[h]),
  projected_csa_density_per_m=float(csa['projected_csa_density_per_m']),
  exact_mie_event_coefficient_per_m=float(csa['exact_mie_event_coefficient_per_m']),
  effective_geometric_event_coefficient_per_m=float(csa['effective_geometric_event_coefficient_per_m']),
  wave_event_coefficient_per_m=float(csa['wave_event_coefficient_per_m']),
  legacy_v24_27_hybrid_event_coefficient_per_m=float(csa['legacy_v24_27_hybrid_event_coefficient_per_m']),
  legacy_over_exact_event_rate_ratio=float(csa['legacy_over_exact_event_rate_ratio']),
  qsca_lt1_projected_csa_fraction=float(csa['qsca_lt1_projected_csa_fraction']),
  H170_mean_particle_interactions=float(p.get('mean_particle_interactions_by_channel',np.full(18,np.nan))[h]),
  H170_mean_geometric_interactions=float(p.get('mean_geometric_interactions_by_channel',np.full(18,np.nan))[h]),
  H170_mean_wave_interactions=float(p.get('mean_wave_interactions_by_channel',np.full(18,np.nan))[h]),
  H170_mean_entry_reflections=float(p.get('mean_entry_reflections_by_channel',np.full(18,np.nan))[h]),
  H170_mean_internal_reflections=float(p.get('mean_internal_reflections_by_channel',np.full(18,np.nan))[h]),
  H170_interaction_order_partition_closure_error_score=float(p.get('interaction_order_partition_closure_error_scores',np.full(18,np.nan))[h]),
  H170_multiple_branch_partition_closure_error_score=float(p.get('multiple_branch_partition_closure_error_scores',np.full(18,np.nan))[h]),
  H170_single_branch_partition_closure_error_score=float(p.get('single_branch_partition_closure_error_scores',np.full(18,np.nan))[h]),
  H170_causal_last_event_partition_closure_error_score=float(p.get('causal_last_event_partition_closure_error_scores',np.full(18,np.nan))[h]),
  H170_causal_first_event_partition_closure_error_score=float(p.get('causal_first_event_partition_closure_error_scores',np.full(18,np.nan))[h]),
  H170_first_event_backscatter_fraction=float(p.get('first_event_backscatter_fraction',np.zeros(18))[h]),
  H170_first_event_source_hemisphere_fraction=float(p.get('first_event_source_hemisphere_fraction',np.zeros(18))[h]),
  H170_mean_first_particle_deflection_deg=float(p.get('mean_first_particle_deflection_deg_by_channel',np.full(18,np.nan))[h]),
  H170_mean_last_particle_deflection_deg=float(p.get('mean_last_particle_deflection_deg_by_channel',np.full(18,np.nan))[h]),
  H170_mean_total_internal_particle_path_mm=1000.0*float(p.get('mean_total_internal_particle_path_m_by_channel',np.full(18,np.nan))[h]),
  H170_mean_total_internal_particle_opl_mm=1000.0*float(p.get('mean_total_internal_particle_opl_m_by_channel',np.full(18,np.nan))[h]),
  H170_mean_water_path_after_last_particle_event_mm=1000.0*float(p.get('mean_water_path_after_last_particle_event_m_by_channel',np.full(18,np.nan))[h]),
  H170_mean_last_particle_event_order=float(p.get('mean_last_particle_event_order_by_channel',np.full(18,np.nan))[h]))
 dynamic={('H170_'+k):float(np.asarray(v,float)[h]) for k,v in p.items()
          if (k.endswith('_fraction') or k.endswith('_scores')) and np.asarray(v).shape==(18,)}
 _merge_summary_fields(summary,dynamic,'detector_particle_path_diagnostics')
 explicit_csa={'projected_csa_density_per_m','exact_mie_event_coefficient_per_m','effective_geometric_event_coefficient_per_m','wave_event_coefficient_per_m','legacy_v24_27_hybrid_event_coefficient_per_m','legacy_over_exact_event_rate_ratio','qsca_lt1_projected_csa_fraction'}
 _merge_summary_fields(summary,{k:v for k,v in csa.items() if k not in explicit_csa},'CSA diagnostics')
 return summary

def write_production_failure_diagnostics(out:Path,res,case_metadata=None,prefix=''):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);pre=(str(prefix).rstrip('_')+'_') if prefix else ''
 total=int(getattr(res,'n_rays',0));failed=int(getattr(res,'production_failure_count',getattr(res,'numerical_failure_count',0)))
 watchdogs=int(getattr(res,'production_watchdog_count',0));geom=int(getattr(res,'production_geometry_failure_count',0));nonfinite=int(getattr(res,'production_nonfinite_count',0))
 diag=getattr(res,'production_failure_diagnostics',{}) or {};closure=bool(diag.get('count_closure_ok',False if failed else True))
 summary={'release':'V24.52.0','n_rays':total,'production_failure_count':failed,'production_failure_fraction':failed/max(total,1),'watchdog_count':watchdogs,'geometry_failure_count':geom,'nonfinite_count':nonfinite,'failure_code_count_closure':closure,'case_valid':bool(failed==0 and closure),'particle_proposal_count':int(diag.get('particle_proposal_count',0)),'particle_accessible_count':int(diag.get('particle_accessible_count',0)),'particle_null_count':int(diag.get('particle_null_count',0))}
 if case_metadata: summary.update(case_metadata)
 pd.DataFrame([summary]).to_csv(out/f'{pre}production_qualification_summary.csv' if prefix else out/'production_failure_summary.csv',index=False)
 counts=dict(getattr(res,'production_failure_code_counts',{}) or {})
 rows=[]
 for code,name in sorted(PRODUCTION_FAILURE_NAMES.items()):
  if code==0: continue
  c=int(counts.get(name,0));rows.append({'release':'V24.52.0','failure_code':code,'failure_name':name,'count':c,'percentage':100.0*c/max(total,1)})
 pd.DataFrame(rows).to_csv(out/f'{pre}production_failure_codes.csv' if prefix else out/'production_failure_codes.csv',index=False)
 sub=dict(getattr(res,'production_failure_subcode_counts',{}) or {})
 srows=[{'release':'V24.52.0','failure_subcode':int(k),'failure_subcode_name':PRODUCTION_FAILURE_SUBCODE_NAMES.get(int(k),f'SUBCODE_{k}'),'count':int(v),'percentage':100.0*int(v)/max(total,1)} for k,v in sorted(sub.items(),key=lambda kv:int(kv[0]))]
 pd.DataFrame(srows,columns=['release','failure_subcode','failure_subcode_name','count','percentage']).to_csv(out/f'{pre}production_failure_subcodes.csv' if prefix else out/'production_failure_subcodes.csv',index=False)
 samples=list(diag.get('samples',[]) or [])
 pd.DataFrame(samples).to_csv(out/f'{pre}production_failure_samples.csv' if prefix else out/'production_failure_samples.csv',index=False)
 meta=dict(summary);meta['failure_code_counts']=counts;meta['failure_subcode_counts']=sub;meta['diagnostic_sample_count']=len(samples);meta['diagnostic_sample_limit']=int(diag.get('sample_limit',512))
 (out/(f'{pre}production_qualification_decision.json' if prefix else 'production_failure_decision.json')).write_text(json.dumps(meta,indent=2,allow_nan=True)+'\n')
 if prefix: pd.DataFrame([meta]).to_csv(out/f'{pre}production_qualification_decision.csv',index=False)
 return summary

def main():
 ap=argparse.ArgumentParser(description='CLARITAS V24.52 Smith-microsurface rough-boundary production case')
 ap.add_argument('--material',choices=['loess','kaolin'],required=True);ap.add_argument('--concentration',type=float,required=True)
 ap.add_argument('--output-dir',required=True);ap.add_argument('--config');ap.add_argument('--n-rays',type=int,default=1000000);ap.add_argument('--seed',type=int,default=2441001)
 ap.add_argument('--target-detector-score',type=float);ap.add_argument('--min-h170-score',type=float);ap.add_argument('--min-rays',type=int);ap.add_argument('--max-rays',type=int)
 ap.add_argument('--stability-l1-tolerance',type=float);ap.add_argument('--stability-window',type=int,default=5)
 ap.add_argument('--diagnostic-run',action='store_true',help='mark this sediment execution as engineering diagnostic')
 ap.add_argument('--production-authorized',action='store_true',help='record that prerequisite production qualification already authorized the sediment campaign')
 args=ap.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
 model,cfg=make_model(args.config)
 t0=time.perf_counter()
 res=model.simulate(args.material,args.concentration,n_rays=args.n_rays,seed=args.seed,target_detector_score=args.target_detector_score,
                    min_h170_score=args.min_h170_score,min_rays=args.min_rays,max_rays=args.max_rays,
                    stability_l1_tolerance=args.stability_l1_tolerance,stability_window=args.stability_window)
 elapsed_s=time.perf_counter()-t0
 run_classification=('DIAGNOSTIC / PRODUCTION-AUTHORIZED' if args.production_authorized else ('DIAGNOSTIC / NOT PRODUCTION-AUTHORIZED' if args.diagnostic_run else 'PRODUCTION CASE'))
 diag=write_production_failure_diagnostics(out,res,{'material':args.material,'concentration_g_per_L':float(args.concentration),'seed':int(args.seed),'elapsed_s':float(elapsed_s),'sediment_run_classification':run_classification,'production_authorized_at_execution':bool(args.production_authorized)})
 if int(diag['production_failure_count']) != 0 or not bool(diag['failure_code_count_closure']):
  print(f"FAIL transport integrity within case: {diag['production_failure_count']} / {diag['n_rays']} rays failed; diagnostics: {out}")
  if not args.diagnostic_run:
   print('No production-authorized sediment result was finalized.')
   raise SystemExit(2)
  print('Continuing because this is explicitly a DIAGNOSTIC sediment execution; outputs remain NOT production-authorized.')
 meas=measured_curve(args.material,args.concentration);score=np.asarray(res.hardware_symmetry_scores,float)
 model_all=np.asarray(res.normalized_hardware_symmetry_response,float);model_tr=trusted_normalize(score);meas_tr=trusted_normalize(meas)
 p=res.detector_particle_path_diagnostics or {};status=np.where(TRUSTED_MASK,'trusted','excluded_known_sediment_sensor_issue')
 det_cols={'detector_deg':DETECTOR_ANGLES_DEG.astype(int),'sediment_sensor_status':status,
           'model_normalized_all18':model_all,'measured_normalized_all18':meas,
           'model_normalized_trusted_set':model_tr,'measured_normalized_trusted_set':meas_tr,
           'hardware_symmetry_score':score}
 # Retain compatibility detector table with V24.26 history fractions plus V24.27 causal means/fractions.
 # Build the column mapping first so pandas receives one contiguous frame and cannot emit
 # the repeated-insert fragmentation warning seen in V24.27.0.
 for key,val in p.items():
  arr=np.asarray(val)
  if (key.endswith('_fraction') or key.startswith('mean_')) and arr.shape==(18,): det_cols[key]=arr.astype(float)
 pd.DataFrame(det_cols).to_csv(out/'detector_response_and_particle_paths.csv',index=False)
 # Full event-history table: score and fraction for every mechanism/order category.
 hist_cols={'detector_deg':DETECTOR_ANGLES_DEG.astype(int),'sediment_sensor_status':status,'hardware_symmetry_score':score}
 for key,val in p.items():
  arr=np.asarray(val)
  if arr.shape==(18,): hist_cols[key]=arr.astype(float)
 pd.DataFrame(hist_cols).to_csv(out/'detector_event_history_decomposition.csv',index=False)
 # V24.29 retained V24.27 causal H170 final-event table. Event codes are mutually exclusive for
 # particle histories, so this directly answers which final particle process feeds H170.
 h=17
 causal_rows=[]
 for code,name in sorted(V2427_PARTICLE_EVENT_TYPE_NAMES.items()):
  if code==0: continue
  causal_rows.append(dict(
   event_code=code,event_name=name,
   H170_score=float(np.asarray(p.get(f'last_event_{name}_scores',np.zeros(18)),float)[h]),
   H170_fraction_of_all_detected=float(np.asarray(p.get(f'last_event_{name}_fraction',np.zeros(18)),float)[h]),
   H170_mean_deflection_deg=float(np.asarray(p.get(f'last_event_{name}_mean_deflection_deg_by_channel',np.full(18,np.nan)),float)[h]),
   H170_first_event_score=float(np.asarray(p.get(f'first_event_{name}_scores',np.zeros(18)),float)[h]),
   H170_first_event_fraction_of_all_detected=float(np.asarray(p.get(f'first_event_{name}_fraction',np.zeros(18)),float)[h])))
 pd.DataFrame(causal_rows).to_csv(out/'H170_causal_last_event_summary.csv',index=False)
 edges=np.asarray(p.get('H170_last_event_angle_bin_edges_deg',[]),float)
 ah=np.asarray(p.get('H170_last_event_angle_histogram_scores_by_event_code',[]),float)
 if edges.size>=2 and ah.ndim==2 and ah.shape[1]==edges.size-1:
  ar=[]
  for code in range(ah.shape[0]):
   nm='all_particle_events' if code==0 else V2427_PARTICLE_EVENT_TYPE_NAMES.get(code,f'code_{code}')
   for b in range(edges.size-1): ar.append(dict(event_code=code,event_name=nm,angle_lo_deg=float(edges[b]),angle_hi_deg=float(edges[b+1]),H170_score=float(ah[code,b])))
  pd.DataFrame(ar).to_csv(out/'H170_last_event_deflection_histogram.csv',index=False)
 fh=np.asarray(p.get('H170_first_event_angle_histogram_scores',[]),float)
 if edges.size>=2 and fh.shape==(edges.size-1,):
  pd.DataFrame(dict(angle_lo_deg=edges[:-1],angle_hi_deg=edges[1:],H170_score=fh)).to_csv(out/'H170_first_event_deflection_histogram.csv',index=False)
 csa=save_csa_table(out,args.material,args.concentration,res,cfg)
 trusted_diff=model_tr[TRUSTED_MASK]-meas_tr[TRUSTED_MASK]
 rmse=float(np.sqrt(np.mean(trusted_diff**2)));corr=float(np.corrcoef(model_tr[TRUSTED_MASK],meas_tr[TRUSTED_MASK])[0,1])
 summary=_build_case_summary(args,res,p,csa,meas,model_all,meas_tr,model_tr,score,rmse,corr)
 summary.update(elapsed_s=float(elapsed_s),sediment_run_classification=run_classification,production_authorized_at_execution=bool(args.production_authorized),diagnostic_execution=bool(args.diagnostic_run or not args.production_authorized),geometry_wall_top_m=float(getattr(model,'zwall_top',float('nan'))),geometry_mean_free_surface_m=float(getattr(model,'zmean',float('nan'))))
 (out/'case_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=True)+'\n')
 print(json.dumps(summary,indent=2,allow_nan=True))

if __name__=='__main__': main()
