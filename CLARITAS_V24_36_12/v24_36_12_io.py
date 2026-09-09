from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from v24_36_12_release import RELEASE,PREFIX,AUDIT_SUBDIR

SCHEMAS={

'complete_kernel_reciprocity_pairs':['release','material','bin_index','qsca','event_rate_weight','pair_kind','state_a','state_b','forward_total_count','reverse_total_count','forward_accepted_rough_count','forward_fallback_count','reverse_accepted_rough_count','reverse_fallback_count','forward_weighted_flux','reverse_weighted_flux','absolute_residual','relative_residual','standard_error','z_score','engineering_relative_floor','z_limit','passed'],
'complete_kernel_reciprocity_summary':['release','material','pair_kind','state_a','state_b','event_weighted_forward_flux','event_weighted_reverse_flux','absolute_residual','relative_residual','combined_standard_error','z_score','passed'],
'fallback_flux_decomposition':['release','material','bin_index','qsca','event_rate_weight','transition','total_count','accepted_rough_count','fallback_count','accepted_rough_fraction','fallback_fraction'],
'macro_tir_reciprocal_flux':['release','material','event_weighted_outside_to_tir_flux','event_weighted_tir_to_outside_flux','relative_residual','z_score','accepted_forward_fraction','fallback_forward_fraction','accepted_reverse_fraction','fallback_reverse_fraction','passed'],
'macro_tir_transition_matrix':['release','material','bin_index','qsca','event_rate_weight','mechanism','from_state','to_state','count','fraction_of_from_state'],
'macro_tir_flux_by_bin':['release','material','bin_index','diameter_m','qsca','event_rate_weight','projected_csa_weight','samples','outside_to_tir','tir_to_outside','tir_to_escape','tir_to_tir','outside_to_outside','outside_to_escape','tir_boundary_count','outside_boundary_count','fallback_boundaries','accepted_rough_boundaries','candidate_escape_states','candidate_escape_rejected','candidate_escape_rejected_to_geom_tir','candidate_escape_probability_mass','rejected_escape_probability_mass','rejected_to_geom_tir_probability_mass','production_cap_hits','other_production_failure_count','diagnostic_censored','max_prefix_direction_l2','max_prefix_position_l2','prefix_rng_mismatch_fraction','prefix_counter_mismatch_fraction'],
'macro_tir_flux_aggregate':['release','material','event_weighted_outside_to_tir_per_boundary','event_weighted_tir_to_outside_per_tir_boundary','event_weighted_tir_to_escape_per_tir_boundary','event_weighted_tir_to_tir_per_tir_boundary','event_weighted_candidate_escape_rejection_fraction','event_weighted_rejected_escape_probability_mass_fraction','event_weighted_rejected_to_geom_tir_mass_fraction','event_weighted_fallback_fraction','event_weighted_production_cap_fraction','event_weighted_diagnostic_censored_fraction'],
'fallback_transition_classes':['release','material','bin_index','qsca','reflection_order','class_name','count','fraction_of_fallbacks_at_order'],
'candidate_escape_rejection':['release','material','bin_index','qsca','reflection_order','macro_tir_state','rough_non_tir_count','rough_non_tir_compatible_count','rough_non_tir_incompatible_count','candidate_escape_probability_mass','rejected_escape_probability_mass','rejected_to_geom_tir_probability_mass','rejection_fraction_count','rejection_fraction_probability_mass'],
'reciprocity_pairs':['release','material','bin_index','qsca','sample_id','reflection_order','category','macro_tir_before','rough_tir','fallback','geom_tir','rough_compatible','n1','n2','ix','iy','iz','gnx','gny','gnz','mnx','mny','mnz','fox','foy','foz','final_ox','final_oy','final_oz','rough_fresnel_R','geom_fresnel_R','rough_escape_probability','compatibility_metric'],
'reciprocity_residuals':['release','material','bin_index','sample_id','category','fixed_facet_reverse_direction_l2','fresnel_reverse_abs','reflection_bsdf_reciprocity_rel','transmission_basic_radiance_reciprocity_rel','current_sampler_forward_density','current_sampler_reverse_density','current_sampler_density_ratio','current_kernel_detailed_balance_ratio','current_kernel_detailed_balance_relative_error','physical_vndf_forward_density','physical_vndf_reverse_density','note','passed_local_optics'],
'fixed_microfacet_reversibility':['release','case_name','mode','n1','n2','incident_cos_macro','microfacet_cos_macro','forward_ok','reverse_ok','forward_reverse_direction_l2','fresnel_reverse_abs','bsdf_reciprocity_relative_error','expected_relation','passed'],
'stochastic_microfacet_reciprocity':['release','incident_cos_macro','samples','raw_facing_fraction','eight_try_exhaustion_estimate','current_conditioned_mean_microfacet_angle_deg','vndf_weighted_mean_microfacet_angle_deg','vndf_normalization','current_vs_vndf_l1_histogram','sampler_classification'],
'microfacet_visibility_audit':['release','incident_cos_macro','samples','raw_facing_count','raw_backfacing_count','raw_facing_fraction','smith_G1','vndf_normalization_estimate','current_conditioned_mean_abs_v_dot_m','vndf_weighted_mean_abs_v_dot_m','current_conditioned_mean_m_dot_n','vndf_weighted_mean_m_dot_n','eight_try_exhaustion_estimate','classification'],
'microfacet_visibility_histogram':['release','incident_cos_macro','theta_lo_deg','theta_hi_deg','raw_facing_count','current_conditioned_fraction','vndf_weighted_fraction','absolute_fraction_difference'],
'trapped_basin_survival':['release','material','bin_index','qsca','reflection_order','reached_count','macro_tir_count','macro_tir_fraction_of_reached','fallback_count','fallback_fraction_of_reached'],
'trapped_basin_escape_hazard':['release','material','bin_index','qsca','reflection_order','macro_tir_boundary_count','escape_count','outside_next_count','tir_next_count','escape_hazard','leave_basin_hazard','fallback_count','fallback_fraction'],
'probability_conservation':['release','material','bin_index','qsca','samples','entry_reflection_count','escaped_count','absorbed_count','numerical_geometry_failure_count','diagnostic_censored_count','resolved_or_censored_count','count_closure_error','production_cap_hits','production_cap_loss_fraction'],
'rough_smooth_control':['release','material','bin_index','diameter_m','qsca','event_rate_weight','rough_production_cap_fraction','smooth_production_cap_fraction','rough_mean_reflections','smooth_mean_reflections','rough_macro_tir_boundary_fraction','smooth_macro_tir_boundary_fraction','rough_fallback_fraction','smooth_fallback_fraction'],
}

def finalize_metadata(meta:dict|None=None)->dict:
    out=dict(meta or {})
    supplied=out.get('release',RELEASE)
    if supplied!=RELEASE:raise RuntimeError(f'stale release metadata rejected: {supplied} != {RELEASE}')
    out['release']=RELEASE;out['diagnostic_only']=True;out['production_ray_campaign_authorized']=False;out['process_correction_authorized']=False;out['sediment_transport_run']=False
    return out

def audit_dir(root:str|Path)->Path:
    d=Path(root)/AUDIT_SUBDIR;d.mkdir(parents=True,exist_ok=True);return d

def _write_df(path:Path,rows:list[dict],columns:list[str])->Path:
    frame=pd.DataFrame(rows,columns=columns)
    if len(frame) and set(frame['release'].astype(str))!={RELEASE}:raise RuntimeError(f'{path.name}: stale/mixed release metadata')
    path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False);return path

def write_acquisition_outputs(root:str|Path,tables:dict[str,list[dict]],meta:dict)->dict[str,Path]:
    d=audit_dir(root);out={}
    for key,cols in SCHEMAS.items():out[key]=_write_df(d/f'{PREFIX}_{key}.csv',tables.get(key,[]),cols)
    m=finalize_metadata(meta);m['table_rows']={k:int(len(tables.get(k,[]))) for k in SCHEMAS}
    p=d/f'{PREFIX}_acquisition_summary.json';p.write_text(json.dumps(m,indent=2)+'\n');out['meta']=p
    return out
