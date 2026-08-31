#!/usr/bin/env python3
"""Fast CLARITAS V24.3 exact forward interface for PSD Inference V3.3."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from claritas_tardiis_core_v24_3 import TardiisForwardModel,ForwardResult,DETECTOR_ANGLES_DEG,get_material_psd,build_geometric_transport
ClaritasForwardModel=TardiisForwardModel

def main():
 ap=argparse.ArgumentParser(description='Fast CLARITAS V24.3 exact TARDIIS forward evaluation');ap.add_argument('--material',choices=['kaolin','loess'],required=True);ap.add_argument('--concentration',type=float,required=True);ap.add_argument('--n-rays',type=int,default=1000000);ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--config',default='psd_inference_v3_3_config.json');ap.add_argument('--output',default='forward_v24_3_response.csv');ap.add_argument('--diagnostics-json',default='forward_v24_3_diagnostics.json');args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {};model=ClaritasForwardModel(**cfg.get('forward_model',{}));res=model.simulate(args.material,args.concentration,n_rays=args.n_rays,seed=args.seed)
 pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'normalized_response':res.normalized_response,'normalized_native_exact':res.normalized_native_exact_response,'normalized_mirror_exact':res.normalized_mirror_exact_response,'symmetry_exact_score':res.symmetry_exact_scores,'native_exact_hits':res.native_exact_hits,'mirror_exact_hits':res.mirror_exact_hits,'jackknife_se':res.normalized_response_jackknife_se}).to_csv(args.output,index=False)
 Path(args.diagnostics_json).write_text(json.dumps(res.to_dict(),indent=2));print(f'Saved {args.output}');print(f'mu_geom={res.mu_geom_per_m:.6g}; native={res.native_exact_hits.sum()}; mirror={res.mirror_exact_hits.sum()}; selected_equivalent={res.symmetry_exact_scores.sum():.1f}')
if __name__=='__main__':main()
