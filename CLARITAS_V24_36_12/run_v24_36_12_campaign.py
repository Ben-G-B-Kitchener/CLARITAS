#!/usr/bin/env python3
from __future__ import annotations
import sys as _v243611_sys
_v243611_sys.dont_write_bytecode=True
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
from v24_36_12_release import HERE,RELEASE,PREFIX,AUDIT_SUBDIR,RECEIPT_NAME
import claritas_tardiis_core_v24_36 as core
import v24_36_12_complete_kernel as ck
import importlib.util

DRIVER=HERE/'CLARITAS_24_36_12_09-09-2026_cuda_reciprocity_fallback_audit.py'
_spec=importlib.util.spec_from_file_location('v243610_driver',DRIVER);drv=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(drv)

def sha(s:str)->str:return hashlib.sha256(s.encode()).hexdigest()

def finalize(root:Path,*,allow_host_smoke:bool=False,require_receipt:bool=True)->dict:
    mp=root/AUDIT_SUBDIR/f'{PREFIX}_acquisition_summary.json'
    if not mp.is_file():raise RuntimeError(f'missing acquisition metadata: {mp}')
    m=json.loads(mp.read_text())
    if m.get('release')!=RELEASE:raise RuntimeError('acquisition metadata stale/mismatched')
    host_smoke=bool(m.get('host_smoke',False))
    if host_smoke and not allow_host_smoke:raise RuntimeError('host-smoke acquisition cannot authorize canonical campaign finalization')
    if not host_smoke and not m.get('gpu_acquisition_complete'):raise RuntimeError('GPU acquisition metadata incomplete')
    core_sha=sha(core.CUDA_SRC);audit_sha=sha(drv.AUDIT_CUDA+ck.COMPLETE_KERNEL_CUDA)
    if m.get('production_core_cuda_sha256') not in (None,core_sha):raise RuntimeError('acquisition production CUDA hash mismatch')
    if m.get('audit_cuda_sha256') not in (None,audit_sha):raise RuntimeError('acquisition audit CUDA hash mismatch')
    if require_receipt:
        rp=HERE/RECEIPT_NAME
        if not rp.is_file():raise RuntimeError(f'missing {RECEIPT_NAME}; complete GPU verifier first')
        r=json.loads(rp.read_text())
        if r.get('release')!=RELEASE or not r.get('gpu_acquisition_complete'):raise RuntimeError('stale/incomplete GPU receipt')
        if r.get('production_core_cuda_sha256')!=core_sha or r.get('audit_cuda_sha256')!=audit_sha:raise RuntimeError('GPU receipt source hash mismatch')
    for script in ('summarize_v24_36_12_reciprocity_fallback_audit.py','plot_v24_36_12_reciprocity_fallback_audit.py'):
        q=subprocess.run([sys.executable,str(HERE/script),str(root)],cwd=HERE)
        if q.returncode:raise RuntimeError(f'{script} failed with exit {q.returncode}')
    dp=root/f'{PREFIX}_process_decision.json'
    if not dp.is_file():raise RuntimeError(f'missing process decision: {dp}')
    d=json.loads(dp.read_text())
    if d.get('release')!=RELEASE:raise RuntimeError('process-decision release mismatch')
    if d.get('process_correction_authorized') or d.get('production_ray_campaign_authorized') or d.get('sediment_transport_run'):
        raise RuntimeError('diagnostic-only release attempted to authorize a process correction or sediment campaign')
    if not host_smoke and not d.get('diagnostic_integrity_passed'):raise RuntimeError('diagnostic integrity failed; finalization blocked')
    return d

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output-dir',default='results');a=ap.parse_args();finalize(Path(a.output_dir),allow_host_smoke=False,require_receipt=True);print(f'PASS campaign: {RELEASE} diagnostic finalization complete; no sediment transport or process correction authorized')
if __name__=='__main__':main()
