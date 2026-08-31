#!/usr/bin/env python3
"""CLARITAS V24.4 reconstructed-hardware forward interface for PSD Inference V3.4."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from claritas_tardiis_core_v24_4 import TardiisForwardModel,ForwardResult,DETECTOR_ANGLES_DEG,get_material_psd,build_geometric_transport
ClaritasForwardModel=TardiisForwardModel

def main():
 ap=argparse.ArgumentParser(description='CLARITAS V24.4 reconstructed-hardware forward evaluation');ap.add_argument('--material',choices=['kaolin','loess'],required=True);ap.add_argument('--concentration',type=float,required=True);ap.add_argument('--n-rays',type=int,default=1000000);ap.add_argument('--seed',type=int,default=24681357);ap.add_argument('--config',default='psd_inference_v3_4_config.json');ap.add_argument('--output',default='forward_v24_4_response.csv');ap.add_argument('--diagnostics-json',default='forward_v24_4_diagnostics.json');args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {};model=ClaritasForwardModel(**cfg.get('forward_model',{}));res=model.simulate(args.material,args.concentration,n_rays=args.n_rays,seed=args.seed)
 pd.DataFrame({'Detector_deg':res.detector_angles_deg.astype(int),'normalized_response':res.normalized_response,'normalized_hardware_symmetry_exact':res.normalized_hardware_symmetry_response,'normalized_hardware_native_exact':res.normalized_hardware_native_response,'normalized_hardware_mirror_exact':res.normalized_hardware_mirror_response,'hardware_symmetry_score':res.hardware_symmetry_scores,'hardware_native_hits':res.hardware_native_hits,'hardware_mirror_hits':res.hardware_mirror_hits,'normalized_legacy_6p5_symmetry':res.normalized_legacy_symmetry_response,'legacy_6p5_symmetry_channel_score':res.legacy_symmetry_channel_scores,'hardware_jackknife_se':res.hardware_normalized_jackknife_se}).to_csv(args.output,index=False)
 Path(args.diagnostics_json).write_text(json.dumps(res.to_dict(),indent=2));print(f'Saved {args.output}');print(f'mu_geom={res.mu_geom_per_m:.6g}; hardware native={res.hardware_native_hits.sum()}; mirror={res.hardware_mirror_hits.sum()}; selected_equivalent={res.hardware_symmetry_scores.sum():.1f}')
if __name__=='__main__':main()
