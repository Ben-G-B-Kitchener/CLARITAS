#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,py_compile,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
RELEASE='V24.52.0'
REQUIRED=[
 'claritas_tardiis_core_v24_52.py','claritas_v24_52_config.json','v24_52_release.py','v24_52_model.py','v24_52_complete_geometry.py','v24_52_validation.py','v24_52_geometry_validation.py','v24_52_production_qualification.py','v24_52_integrity_matrix.py','run_v24_52_case.py','run_v24_52_campaign.py','verify_v24_52_complete_geometry.py','preflight_v24_52_package.py','v24_52_geometry_manifest.json','v24_52_region_surface_manifest.json','sediment_case_inventory.csv','V24_52_COMPLETE_GEOMETRY_ENGINEERING_NOTE.md','V24_52_VALIDATOR_AND_CAMPAIGN_ENGINEERING_NOTE.md','V24_52_HORIZONTAL_FREE_SURFACE_REGRESSION.csv','V24_52_V24_51_MISMATCH_RECONSTRUCTION.csv','V24_52_V24_51_MISMATCH_RECONSTRUCTION.json','V24_52_HOST_VALIDATION_REPORT.json','V24_52_PACKAGED_QUALIFICATION_STATUS.json','V24_52_SEDIMENT_TEST_REPORT_TEMPLATE.md','README.md','RELEASE_NOTES.txt','SHA256SUMS.txt','PACKAGE_MANIFEST.json']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 errs=[]
 if HERE.name!='CLARITAS_V24_52':errs.append(f'top directory must be CLARITAS_V24_52, got {HERE.name}')
 for n in REQUIRED:
  if not (HERE/n).is_file():errs.append(f'missing required file {n}')
 try:
  cfg=json.loads((HERE/'claritas_v24_52_config.json').read_text())
  if cfg.get('version')!=RELEASE:errs.append('config version mismatch')
  f=cfg['forward_model'];
  if abs(float(f.get('tube_length_m',-1))-.491)>1e-12:errs.append('tube_length_m must be 0.491')
 except Exception as e:errs.append('config parse: '+repr(e))
 # Active Python files must compile and use V24.52 naming.
 for p in HERE.glob('*.py'):
  try:py_compile.compile(str(p),doraise=True,cfile=str(Path('/tmp')/(p.stem+'.pyc')))
  except Exception as e:errs.append(f'compile {p.name}: {e}')
 # Reject transient artifacts.
 for p in HERE.rglob('*'):
  if p.is_dir() and p.name=='__pycache__':errs.append(f'packaged __pycache__: {p.relative_to(HERE)}')
  if p.is_file() and p.suffix=='.pyc':errs.append(f'packaged pyc: {p.relative_to(HERE)}')
  if p.is_file() and p.suffix.lower()=='.zip':errs.append(f'nested zip: {p.relative_to(HERE)}')
 # Verify manifest and SHA list if present.
 mp=HERE/'PACKAGE_MANIFEST.json'
 if mp.is_file():
  try:
   m=json.loads(mp.read_text());files=m.get('files',[])
   for r in files:
    p=HERE/r['path']
    if not p.is_file():errs.append('manifest missing '+r['path'])
    elif sha(p)!=r['sha256']:errs.append('manifest hash mismatch '+r['path'])
  except Exception as e:errs.append('manifest verify: '+repr(e))
 sp=HERE/'SHA256SUMS.txt'
 if sp.is_file():
  for line in sp.read_text().splitlines():
   if not line.strip():continue
   h,name=line.split(None,1);name=name.strip().lstrip('*');p=HERE/name
   if not p.is_file():errs.append('SHA256SUMS missing '+name)
   elif sha(p)!=h:errs.append('SHA256SUMS mismatch '+name)
 if errs:
  print('FAIL package preflight');[print('-',e) for e in errs];raise SystemExit(2)
 print(f'PASS package preflight: {RELEASE}; required files present; Python compiles; package hashes verify')
if __name__=='__main__':main()
