#!/usr/bin/env python3
from __future__ import annotations
import sys as _sys;_sys.dont_write_bytecode=True
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from v24_36_12_release import RELEASE,PREFIX,AUDIT_SUBDIR
from v24_36_12_io import SCHEMAS,finalize_metadata
import v24_36_12_model as vm
EXPECTED_BINS={'loess':38,'kaolin':28}

def _read(root,key):
    p=Path(root)/AUDIT_SUBDIR/f'{PREFIX}_{key}.csv'
    if not p.is_file():raise RuntimeError(f'missing audit table: {p}')
    d=pd.read_csv(p);exp=SCHEMAS[key]
    if list(d.columns)!=exp:raise RuntimeError(f'{p.name}: schema mismatch')
    if len(d) and set(d.release.astype(str))!={RELEASE}:raise RuntimeError(f'{p.name}: stale release metadata')
    return d

def summarize(root:Path):
    cfg=vm.load_config();tol=cfg['reciprocity_fallback_audit_v24_36_12']['invariant_tolerances']
    mp=root/AUDIT_SUBDIR/f'{PREFIX}_acquisition_summary.json';meta=json.loads(mp.read_text())
    tables={k:_read(root,k) for k in SCHEMAS};host=bool(meta.get('host_smoke',False));checks=[]
    def add(name,value,limit,passed,note=''):checks.append({'release':RELEASE,'check':name,'value':value,'limit':limit,'passed':bool(passed),'note':note})
    by=tables['macro_tir_flux_by_bin'];pc=tables['probability_conservation'];fix=tables['fixed_microfacet_reversibility'];kp=tables['complete_kernel_reciprocity_pairs'];ks=tables['complete_kernel_reciprocity_summary'];mf=tables['macro_tir_reciprocal_flux']
    if host:
        add('host_real_writer_complete_kernel_rows',len(kp),'>0',len(kp)>0)
        add('host_probability_closure',float(np.abs(pc.count_closure_error).max()) if len(pc) else -1,0.0,len(pc)>0 and float(np.abs(pc.count_closure_error).max())==0.0)
        add('host_fixed_microfacet',len(fix),'>=4',len(fix)>=4 and fix.passed.astype(str).str.lower().isin(['true','1']).all())
        add('host_macro_tir_summary_both_materials',','.join(sorted(set(mf.material.astype(str)))),'kaolin,loess',set(mf.material.astype(str))=={'kaolin','loess'})
        c=pd.DataFrame(checks);c.to_csv(root/f'{PREFIX}_process_decision.csv',index=False);ok=bool(c.passed.all())
        dec=finalize_metadata({'gpu_acquisition_complete':False,'host_smoke':True,'diagnostic_integrity_passed':ok,'scientific_disposition':'INSUFFICIENT_EVIDENCE','decision':'host_smoke_structural_only','production_65_reflection_numerical_sink_still_present':True})
        (root/f'{PREFIX}_process_decision.json').write_text(json.dumps(dec,indent=2)+'\n')
        if not ok:raise RuntimeError('host-smoke structural checks failed')
        return {'decision':dec,'checks':c}
    if not meta.get('gpu_acquisition_complete'):raise RuntimeError('GPU acquisition metadata incomplete')
    for mat,n in EXPECTED_BINS.items():
        g=by[(by.material==mat)&(by.qsca>=1.0)];add(f'{mat}_qsca_ge1_bin_coverage',int(g.bin_index.nunique()),n,int(g.bin_index.nunique())==n)
        kg=kp[(kp.material==mat)&(kp.qsca>=1.0)];add(f'{mat}_complete_kernel_bin_coverage',int(kg.bin_index.nunique()),n,int(kg.bin_index.nunique())==n)
    add('probability_count_closure_abs_max',float(np.abs(pc.count_closure_error).max()),0.0,float(np.abs(pc.count_closure_error).max())==0.0,'entry reflections are already included in escaped_count and are not double-counted')
    fixed_pass=bool(len(fix)>=4 and fix.passed.astype(str).str.lower().isin(['true','1']).all());add('fixed_microfacet_reversibility_all',int(fixed_pass),1,fixed_pass)
    add('fixed_microfacet_fresnel_reverse_max',float(fix.fresnel_reverse_abs.max()),tol['fresnel_reciprocity_abs'],float(fix.fresnel_reverse_abs.max())<=tol['fresnel_reciprocity_abs'],'reflection uses same n1/n2 orientation; transmission swaps media')
    add('prefix_rng_mismatch_max_fraction',float(by.prefix_rng_mismatch_fraction.max()),0.0,float(by.prefix_rng_mismatch_fraction.max())==0.0)
    add('prefix_counter_mismatch_max_fraction',float(by.prefix_counter_mismatch_fraction.max()),0.0,float(by.prefix_counter_mismatch_fraction.max())==0.0)
    add('other_production_failure_total',int(by.other_production_failure_count.sum()),0,int(by.other_production_failure_count.sum())==0)
    add('frozen_production_cap_hits_detected',int(by.production_cap_hits.sum()),'>0',int(by.production_cap_hits.sum())>0)
    add('complete_kernel_pair_rows',len(kp),'>0',len(kp)>0)
    add('complete_kernel_summary_rows',len(ks),'>0',len(ks)>0)
    add('macro_tir_primary_material_rows',len(mf),2,len(mf)==2 and set(mf.material)=={'loess','kaolin'})
    c=pd.DataFrame(checks);c.to_csv(root/f'{PREFIX}_process_decision.csv',index=False);integrity=bool(c.passed.all())
    if integrity:
        p={str(r.material):bool(r.passed) for _,r in mf.iterrows()}
        if p.get('kaolin') is False and p.get('loess') is False: disposition='PROCESS_DEFECT_CONFIRMED'
        elif p.get('kaolin') is True and p.get('loess') is True: disposition='PROCESS_DEFECT_NOT_CONFIRMED'
        else: disposition='INSUFFICIENT_EVIDENCE'
    else:disposition='INSUFFICIENT_EVIDENCE'
    dec=finalize_metadata({'gpu_acquisition_complete':True,'diagnostic_integrity_passed':integrity,'scientific_disposition':disposition,'fixed_microfacet_local_reversibility_passed':fixed_pass,'complete_kernel_test':'equilibrium surface-flux Monte Carlo; internal state measure proportional to n_particle^2*mu*dOmega, cross-interface comparison includes n^2 weighting; accepted rough and fallback outcomes included in final transition counts','macro_tir_complete_kernel':mf.to_dict('records'),'process_correction_authorized':False,'production_65_reflection_numerical_sink_still_present':True,'numerical_photon_destruction_removed_from_production':False,'roughness_rms_slope_deg':25.0,'roughness_provenance_status':'inherited physically motivated but independently unvalidated morphology parameter','decision':'final_mixed_kernel_decision_test_complete'})
    (root/f'{PREFIX}_process_decision.json').write_text(json.dumps(dec,indent=2,allow_nan=False)+'\n')
    if not integrity:raise RuntimeError('GPU diagnostic integrity checks failed; see process_decision.csv')
    return {'decision':dec,'checks':c}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',nargs='?',default='results');a=ap.parse_args();r=summarize(Path(a.root));print(json.dumps(r['decision'],indent=2,allow_nan=False))
if __name__=='__main__':main()
