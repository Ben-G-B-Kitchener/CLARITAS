#!/usr/bin/env python3
"""CLARITAS V24.3: exact TARDIIS optics with streaming symmetry-averaged detector statistics.

Optical physics is unchanged from V24.  The only estimator change is an unbiased
left/right reflection-symmetry average of the SAME hard physical two-plane
collimator test. Native one-sided exact hits are always retained.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from claritas_tardiis_core_v24_3 import TardiisForwardModel,DETECTOR_ANGLES_DEG,get_material_psd,build_geometric_transport

DEFAULTS=dict(n_water=1.33,n_particle=1.59,n_acrylic=1.4906,n_air=1.0,tube_inner_radius_m=0.0465,tube_outer_radius_m=0.0500,
 water_height_m=0.426,sensor_height_above_bottom_m=0.102,collimator_inner_plane_radius_m=0.0505,collimator_outer_plane_radius_m=0.0595,
 collimator_diameter_m=0.004,source_beam_sigma_m=1e-5,alpha1=1.0,alpha2=100.0,density_kg_per_m3=2600.0,max_internal_bounces=64,
 max_cell_bounces=64,chunk_size=250000,statistics_batch_rays=250000,symmetry_average_exact=True)

def make_model(args):
 d=DEFAULTS.copy()
 if args.config:
  cfg=json.loads(Path(args.config).read_text());d.update(cfg.get('forward_model',cfg))
 return TardiisForwardModel(**d),d

def main():
 ap=argparse.ArgumentParser(description='CLARITAS V24.3 exact TARDIIS transport + unbiased symmetry-averaged hard-aperture scoring')
 ap.add_argument('--material',choices=['loess','kaolin'],default='loess');ap.add_argument('--concentration',type=float,default=0.5)
 ap.add_argument('--n-rays',type=int,default=1000000,help='fixed ray count unless --target-detector-score is used')
 ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--output-dir');ap.add_argument('--config');ap.add_argument('--dgb-only',action='store_true')
 ap.add_argument('--heatmap-size',type=int,default=0,help='0 is recommended for high-statistics detector runs')
 ap.add_argument('--save-rays',action='store_true');ap.add_argument('--target-detector-score',type=float,help='adaptive exact-score target; omitted = fixed n-rays')
 ap.add_argument('--min-rays',type=int);ap.add_argument('--max-rays',type=int);ap.add_argument('--stability-l1-tolerance',type=float)
 args=ap.parse_args();conc=0.0 if args.dgb_only else args.concentration;tag='DGB_water_only' if args.dgb_only else f'{args.material}_{conc:g}gL'
 out=Path(args.output_dir or f'claritas_v24_3_results/{tag}');out.mkdir(parents=True,exist_ok=True);model,geom=make_model(args)
 print('\n=========== CLARITAS V24.3 — EXACT TARDIIS STATISTICS ===========')
 print(f'material={args.material}; concentration={conc:g} g/L; requested rays={args.n_rays:,}')
 print('Optics: unchanged V24 cylindrical water/acrylic/air + V23.2 particle Snell/Fresnel')
 print('Detector: exact physical two-plane 4-mm collimator test only (NO KDE / widening / extrapolation)')
 print('Variance reduction: hard-aperture native ray + x-mirrored ray, averaged 1/2 under exact model symmetry')
 print('Stirring interpretation: continuous-on ensemble; no settling clock.')
 res=model.simulate(args.material,conc,n_rays=args.n_rays,seed=args.seed,collect_rays=args.save_rays,heatmap_size=args.heatmap_size,
  target_detector_score=args.target_detector_score,min_rays=args.min_rays,max_rays=args.max_rays,stability_l1_tolerance=args.stability_l1_tolerance)
 df=pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'normalized_response':res.normalized_response,
  'normalized_symmetry_exact':res.normalized_symmetry_exact_response,'normalized_native_exact':res.normalized_native_exact_response,
  'normalized_mirror_exact':res.normalized_mirror_exact_response,'jackknife_se_normalized':res.normalized_response_jackknife_se,
  'symmetry_exact_physical_equivalent_score':res.symmetry_exact_scores,'native_exact_hits':res.native_exact_hits,'mirror_exact_hits':res.mirror_exact_hits})
 df.to_csv(out/'detector_response_normalized.csv',index=False)
 pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'native_exact_hits':res.native_exact_hits,'mirror_exact_hits':res.mirror_exact_hits,
  'symmetry_exact_physical_equivalent_score':res.symmetry_exact_scores}).to_csv(out/'detector_hits_exact.csv',index=False)
 pd.DataFrame(res.batch_diagnostics or []).to_csv(out/'batch_convergence.csv',index=False)
 (out/'diagnostics.json').write_text(json.dumps(res.to_dict(),indent=2));geom['stirrer_state_for_interpretation']='continuous_on';geom['detector_estimator']='hard_aperture_symmetry_average_exact'
 (out/'apparatus_geometry.json').write_text(json.dumps(geom,indent=2))
 d,w=get_material_psd(args.material);tr=build_geometric_transport(d,w,conc,model.density)
 pd.DataFrame({'diameter_um':d*1e6,'source_weight':w,'event_weight':tr['particle_event_weights'],'mu_geom_by_bin_per_m':tr['mu_geom_by_bin']}).to_csv(out/'particle_interaction_diagnostics.csv',index=False)
 fig,ax=plt.subplots(figsize=(9,5));ax.errorbar(res.detector_angles_deg,res.normalized_response,yerr=res.normalized_response_jackknife_se,fmt='o-',capsize=2,label='symmetry-averaged exact')
 ax.plot(res.detector_angles_deg,res.normalized_native_exact_response,'--',label='native exact');ax.set_xticks(DETECTOR_ANGLES_DEG);ax.set_xlabel('Detector angle (deg)');ax.set_ylabel('Normalized response');ax.grid(True,alpha=.25);ax.legend();fig.tight_layout();fig.savefig(out/'detector_response_normalized.png',dpi=200);plt.close(fig)
 if args.save_rays and res.ray_data is not None: pd.DataFrame(res.ray_data).to_csv(out/'ray_states.csv',index=False)
 print(f'Saved V24.3 results to {out}');print(f'actual rays={res.n_rays:,}; stop={res.stop_reason}; native hits={res.native_exact_hits.sum():,}; mirror hits={res.mirror_exact_hits.sum():,}; symmetry-equivalent={res.symmetry_exact_scores.sum():.1f}')
 print(f'detection fraction selected={res.detector_detection_fraction:.6g}')
if __name__=='__main__':main()
