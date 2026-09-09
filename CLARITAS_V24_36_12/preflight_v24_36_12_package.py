#!/usr/bin/env python3
from __future__ import annotations
import sys as _v243611_sys
_v243611_sys.dont_write_bytecode=True
import hashlib,json,re,sys
from pathlib import Path
from v24_36_12_release import HERE,RELEASE,PACKAGE,OUTPUT_FILENAMES,AUDIT_SUBDIR
import v24_36_12_model as vm
import claritas_tardiis_core_v24_36 as core

MANIFEST=HERE/'PACKAGE_MANIFEST.json'
CORE_FILE_SHA='38ce70b515adba207484ed0ce66065cd12dfd5bc96067b61719c6999384a4057'
CORE_CUDA_SHA='a64339e0167644cae3abe86294aaae1001a9d398cd60ebc36e54eedadef98671'
def shab(b:bytes):return hashlib.sha256(b).hexdigest()
def sha_file(p:Path):return shab(p.read_bytes())

def main():
    if not MANIFEST.is_file():raise RuntimeError('PACKAGE_MANIFEST.json missing')
    m=json.loads(MANIFEST.read_text())
    if m.get('release')!=RELEASE or m.get('package')!=PACKAGE:raise RuntimeError('manifest release/package mismatch')
    files=m.get('files',{})
    missing=[];bad=[]
    for rel,h in files.items():
        p=HERE/rel
        if not p.is_file():missing.append(rel)
        elif sha_file(p)!=h:bad.append(rel)
    if missing or bad:raise RuntimeError(f'package manifest failure missing={missing} hash_mismatch={bad}')
    cfg=vm.load_config();old=json.loads((HERE/'references'/'claritas_v24_36_9_config.json').read_text())
    if cfg['forward_model']!=old['forward_model']:raise RuntimeError('forward_model differs from archived V24.36.9 diagnostic baseline')
    if sha_file(HERE/'claritas_tardiis_core_v24_36.py')!=CORE_FILE_SHA:raise RuntimeError('frozen core file hash mismatch')
    if shab(core.CUDA_SRC.encode())!=CORE_CUDA_SHA:raise RuntimeError('embedded production CUDA hash mismatch')
    if int(cfg['forward_model']['max_internal_bounces'])!=64:raise RuntimeError('production max_internal_bounces changed')
    if float(cfg['forward_model']['particle_surface_rms_slope_deg'])!=25.0:raise RuntimeError('25-degree roughness changed')
    if not all(max(map(abs,v['k_imag']))==0 for v in vm.particle_optics_override().values()):raise RuntimeError('audit particle k must be zero')
    # No stale executable predecessor files at package root.
    stale=[p.name for p in HERE.glob('*.py') if re.search(r'24_36_9|v24_36_9',p.name,re.I)]
    if stale:raise RuntimeError(f'stale V24.36.9 runtime files present: {stale}')
    # Windows default path budget, including final result filenames.
    root=r'F:\Google Drive Sync\CLARITAS_code\CLARITAS\CLARITAS_V24_36_12'
    decision={f'v24_36_12_process_decision.csv',f'v24_36_12_process_decision.json'}
    paths=[f'results\\{x}' if x in decision else f'results\\{AUDIT_SUBDIR}\\{x}' for x in OUTPUT_FILENAMES]
    paths += list(files)
    longest=max(len(root)+1+len(x.replace('/','\\')) for x in paths)
    if longest>=240:raise RuntimeError(f'Windows path budget exceeded: {longest}')
    print(f'PASS package preflight: {RELEASE} complete; {len(files)} critical files hashed; frozen core/config/roughness/cap/k=0 checks pass; longest default path {longest} chars (<240)')
if __name__=='__main__':main()
