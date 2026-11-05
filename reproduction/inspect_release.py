#!/usr/bin/env python3
"""Inspect this release using the standard library; no scientific computation."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--resources',type=Path,help='Optional local resource-root JSON to inspect; never runs a study')
 p.add_argument('--skip-hashes',action='store_true',help='List metadata without rehashing supplied files')
 a=p.parse_args();m=json.loads((ROOT/'SOURCE_MANIFEST.json').read_text())
 missing=[];changed=[]
 for r in m['files']:
  path=ROOT/r['path']
  if not path.is_file():missing.append(r['path']);continue
  if not a.skip_hashes and (path.stat().st_size!=r['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=r['sha256']):changed.append(r['path'])
 result={'status':'PASS'if not missing and not changed else'FAIL','mode':'inspect_only','files':len(m['files']),
 'hashes_checked':not a.skip_hashes,'missing':missing,'changed':changed,'training_performed':False,
 'model_forward_performed':False,'AP_performed':False,'bootstrap_performed':False,'data_or_model_downloads':False,
 'scientific_reproduction_claimed':False,
 'entry_points':{'portable_contract':'portable/core.py','original_pair':'recorded/experiments/pair_conflict_material_projection_20261101/src/pair_projection.py',
 'CI':'recorded/new_module_evidence/ci/scripts/run_ci.py','ablation':'recorded/new_module_evidence/ablation/scripts/run_dataset.py',
 'full_image_runtime':'recorded/new_module_evidence/runtime/scripts/run_raw_benchmark.py'}}
 if a.resources:
  config=json.loads(a.resources.read_text());roots=config.get('roots',{})
  result['resources']={k:{'configured':True,'exists':Path(os.path.expandvars(os.path.expanduser(v))).exists()}for k,v in roots.items()}
  result['resource_check_scope']='Existence only; no provider data, checkpoint or native input identity validated'
 print(json.dumps(result,indent=2));return 0 if result['status']=='PASS'else 1
if __name__=='__main__':raise SystemExit(main())
