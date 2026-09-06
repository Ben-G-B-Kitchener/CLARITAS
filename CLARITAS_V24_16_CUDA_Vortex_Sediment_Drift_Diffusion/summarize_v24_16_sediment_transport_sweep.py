#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd

MEAS={
0.5:np.array([0.572525,0.202847,0.033292,0.012736,0.008691,0.007422,0.006639,0.006316,0.006484,0.006615,0.0071,0.007423,0.007962,0.010363,0.005626,0.004187,0.018959,0.074813]),
2.0:np.array([0.119113,0.103634,0.063749,0.04004,0.03314,0.029325,0.027061,0.025631,0.02671,0.02716,0.0294,0.031389,0.031777,0.040875,0.022572,0.015979,0.068737,0.263707]),
4.0:np.array([0.039599,0.048221,0.045777,0.040045,0.037967,0.035255,0.03221,0.030837,0.033189,0.03412,0.03642,0.037951,0.03559,0.047269,0.027809,0.020113,0.084777,0.33285])}


def parse_c(run:Path)->float:
    return float(run.name.replace('loess_','').replace('gL','').replace('p','.'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',nargs='?',default='../claritas_v24_16_sediment_transport_sweep')
    a=ap.parse_args();root=Path(a.root);rows=[]
    for sp in sorted(root.rglob('sediment_transport_diagnostics.json')):
        run=sp.parent;dj=run/'diagnostics.json';vp=run/'vortex_diagnostics.json'
        if not dj.exists() or not vp.exists():continue
        d=json.loads(dj.read_text());s=json.loads(sp.read_text());v=json.loads(vp.read_text());C=parse_c(run);m=MEAS[C];y=np.asarray(d['normalized_response'],float)
        se=np.asarray(d.get('hardware_normalized_jackknife_se',[np.nan]*18),float)
        rows.append(dict(
            sediment_transport_model=s['sediment_transport_model'],eddy_diffusivity_m2_s=float(s['eddy_diffusivity_m2_s']),
            core_radius_mm=float(v.get('vortex_core_radius_mm',np.nan)),delta_h_mm=float(v['vortex_wall_center_height_difference_mm']),
            center_z_relative_sensor_mm=float(v.get('vortex_center_surface_z_mm_relative_sensor',np.nan)),concentration_g_per_L=C,
            bulk_scale_sensor_center=float(s.get('bulk_mass_concentration_scale_sensor_center',np.nan)),
            bulk_scale_sensor_wall=float(s.get('bulk_mass_concentration_scale_sensor_wall_r0p95R',np.nan)),
            bulk_scale_bottom_center=float(s.get('bulk_mass_concentration_scale_bottom_center',np.nan)),
            bulk_scale_bottom_wall=float(s.get('bulk_mass_concentration_scale_bottom_wall_r0p95R',np.nan)),
            max_bin_density_scale=float(s.get('max_bin_density_scale',np.nan)),majorant_total_per_m=float(s.get('majorant_total_per_m',np.nan)),
            rmse=float(np.sqrt(np.mean((y-m)**2))),corr=float(np.corrcoef(y,m)[0,1]),
            model_H0=y[0],model_H10=y[1],model_H20=y[2],model_H30=y[3],model_H140=y[14],model_H150=y[15],model_H160=y[16],model_H170=y[17],
            measured_H170=m[17],H170_abs_error=float(abs(y[17]-m[17])),
            model_sum_0_30=float(y[:4].sum()),measured_sum_0_30=float(m[:4].sum()),
            model_sum_40_130=float(y[4:14].sum()),measured_sum_40_130=float(m[4:14].sum()),
            model_sum_140_170=float(y[14:].sum()),measured_sum_140_170=float(m[14:].sum()),
            model_H170_H160=float(y[17]/y[16]) if y[16]>0 else float('nan'),measured_H170_H160=float(m[17]/m[16]),
            H170_H160_abs_error=float(abs(y[17]/y[16]-m[17]/m[16])) if y[16]>0 else float('nan'),
            mean_top_surface_interactions=float(v.get('mean_top_surface_interactions_per_entered_ray',np.nan)),
            top_surface_tir_fraction=float(v.get('top_surface_tir_fraction_per_entered_ray',np.nan)),
            mean_channel_jackknife_se=float(np.nanmean(se)),max_channel_jackknife_se=float(np.nanmax(se)),
            equivalent_detector_score=float(np.sum(d['hardware_symmetry_scores'])),n_rays=int(d['n_rays']),stop_reason=d['stop_reason'],run_dir=str(run)))
    if not rows:raise SystemExit(f'No V24.16 sediment-transport results found under {root}')
    df=pd.DataFrame(rows).sort_values(['sediment_transport_model','eddy_diffusivity_m2_s','concentration_g_per_L'])
    df.to_csv(root/'v24_16_case_summary.csv',index=False)
    rank=[]
    for (tm,dt),g in df.groupby(['sediment_transport_model','eddy_diffusivity_m2_s'],dropna=False):
        if set(np.round(g.concentration_g_per_L,6))!={0.5,2.0,4.0}:continue
        gg=g.set_index('concentration_g_per_L')
        rank.append(dict(sediment_transport_model=tm,eddy_diffusivity_m2_s=float(dt),core_radius_mm=float(g.core_radius_mm.iloc[0]),delta_h_mm=float(g.delta_h_mm.iloc[0]),
            mean_rmse=float(g.rmse.mean()),max_rmse=float(g.rmse.max()),mean_corr=float(g['corr'].mean()),
            mean_forward_abs_error=float(np.mean(np.abs(g.model_sum_0_30-g.measured_sum_0_30))),mean_mid_abs_error=float(np.mean(np.abs(g.model_sum_40_130-g.measured_sum_40_130))),mean_rear_abs_error=float(np.mean(np.abs(g.model_sum_140_170-g.measured_sum_140_170))),
            mean_H170_abs_error=float(g.H170_abs_error.mean()),mean_ratio_abs_error=float(np.nanmean(g.H170_H160_abs_error)),
            mean_bulk_scale_sensor_center=float(g.bulk_scale_sensor_center.mean()),mean_bulk_scale_sensor_wall=float(g.bulk_scale_sensor_wall.mean()),
            max_bin_density_scale=float(g.max_bin_density_scale.max()),mean_majorant_total_per_m=float(g.majorant_total_per_m.mean()),
            min_score=float(g.equivalent_detector_score.min()),total_rays=int(g.n_rays.sum()),
            rmse_0p5=float(gg.loc[0.5,'rmse']),rmse_2=float(gg.loc[2.0,'rmse']),rmse_4=float(gg.loc[4.0,'rmse']),
            H170_0p5=float(gg.loc[0.5,'model_H170']),H170_2=float(gg.loc[2.0,'model_H170']),H170_4=float(gg.loc[4.0,'model_H170']),
            ratio_0p5=float(gg.loc[0.5,'model_H170_H160']),ratio_2=float(gg.loc[2.0,'model_H170_H160']),ratio_4=float(gg.loc[4.0,'model_H170_H160'])))
    rd=pd.DataFrame(rank)
    if rd.empty:raise SystemExit('No complete 3-concentration transport groups found')
    base=rd.loc[rd.sediment_transport_model.eq('uniform'),'mean_rmse']
    baseline=float(base.iloc[0]) if len(base) else np.nan
    rd['delta_mean_rmse_vs_uniform']=rd.mean_rmse-baseline
    rd=rd.sort_values(['mean_rmse','max_rmse']).reset_index(drop=True);rd.insert(0,'rank',np.arange(1,len(rd)+1))
    rd.to_csv(root/'v24_16_joint_ranking.csv',index=False)
    print(rd.to_string(index=False))

if __name__=='__main__':main()
