#!/usr/bin/env python3
from __future__ import annotations
import sys as _sys;_sys.dont_write_bytecode=True
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from v24_36_12_release import PREFIX,AUDIT_SUBDIR

def main():
    ap=argparse.ArgumentParser();ap.add_argument('root',nargs='?',default='results');a=ap.parse_args();d=Path(a.root)/AUDIT_SUBDIR
    p=pd.read_csv(d/f'{PREFIX}_complete_kernel_reciprocity_summary.csv')
    m=p[p.pair_kind=='macro_tir_primary'].copy()
    if len(m):
        x=range(len(m));plt.figure(figsize=(9,5));plt.errorbar(x,m.event_weighted_forward_flux,yerr=m.combined_standard_error,fmt='o',label='outside -> macro-TIR');plt.errorbar(x,m.event_weighted_reverse_flux,yerr=m.combined_standard_error,fmt='s',label='macro-TIR -> outside');plt.xticks(list(x),m.material);plt.ylabel('n²-weighted equilibrium transition flux');plt.title('Complete mixed-kernel reciprocal flux');plt.legend();plt.tight_layout();plt.savefig(d/f'{PREFIX}_complete_kernel_forward_reverse.png',dpi=160);plt.close()
    if len(p):
        q=p.copy();labels=[f'{r.material}:{r.pair_kind}:{r.state_a}->{r.state_b}' for _,r in q.iterrows()];plt.figure(figsize=(11,max(5,0.22*len(q))));plt.errorbar(q.relative_residual,range(len(q)),xerr=0,fmt='o');plt.yticks(range(len(q)),labels,fontsize=7);plt.axvline(0.01,linestyle='--');plt.xlabel('relative reciprocity residual');plt.title('Complete-kernel residuals (1% engineering floor shown)');plt.tight_layout();plt.savefig(d/f'{PREFIX}_complete_kernel_residuals.png',dpi=160);plt.close()
    f=pd.read_csv(d/f'{PREFIX}_fallback_flux_decomposition.csv');f=f[f.transition.isin(['outside_to_macro_tir','macro_tir_to_outside'])]
    if len(f):
        g=f.groupby(['material','transition'])[['accepted_rough_count','fallback_count']].sum().reset_index();labels=[f'{r.material}:{r.transition}' for _,r in g.iterrows()];plt.figure(figsize=(10,5));x=range(len(g));plt.bar(x,g.accepted_rough_count,label='accepted rough');plt.bar(x,g.fallback_count,bottom=g.accepted_rough_count,label='fallback');plt.xticks(list(x),labels,rotation=25,ha='right');plt.ylabel('transition count');plt.title('Accepted-rough vs fallback contribution');plt.legend();plt.tight_layout();plt.savefig(d/f'{PREFIX}_fallback_flux_decomposition.png',dpi=160);plt.close()
    print('PASS plots: V24.36.12 final decision plots written')
if __name__=='__main__':main()
