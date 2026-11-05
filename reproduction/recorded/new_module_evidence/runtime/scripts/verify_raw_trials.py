"""All formal raw runs must reproduce independent-golden input/state/output."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import json,sys,importlib.util,hashlib,datetime
import numpy as np
R=Path('/mnt/data/yxy/program/RAXO');C=R/'experiments/new_module_evidence_20261102/runtime';O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime');sys.path.insert(0,str(C.parent/'raw_connector'))
from features import build_features
s=importlib.util.spec_from_file_location('frozen_Gaussian',R/'experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py');G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
def read(p):return json.loads(Path(p).read_text())
def ref(p):p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert read(O/'gates/raw_compatibility_review.json')['status']=='PASS'
P=read(R/'experiments/pair_conflict_material_projection_20261101/config/protocol.json');c=P['cases']['CLCXray_test'];cf=read(c['C_feature_manifest']['path']);nf=read(c['N_feature_manifest']['path']);af=read(c['auxiliary_feature_manifest']['path']);cats=read(R/'outputs_rmdx/cache_multidataset/CLCXray_groundingDINO_100_0_full/category_meta.json');proto=read(R/'outputs_rmdx/cache_multidataset/CLCXray_groundingDINO_100_0_full/prototype_meta.json')
captures=read(O/'raw_trials/compatibility_pair/captures.json');expected=dict(strong=[],pair=[]);meta=[]
for e in captures:
 cap=e['capture'];entry=e['entry'];rows=cap['rows'];n=len(rows);V=np.asarray(cap['V'],np.float32).reshape(n,len(proto));B=np.asarray(cap['B'],np.float32).reshape(n,len(cats))
 d=build_features(V,B,B,rows,{entry['original_image_id']:(entry['width'],entry['height'])},proto,cats,native_support=cf['material_support_count'],original_support=nf['actual_material_support_counts'],aux_support=af['actual_material_support_counts']);tr=d['trace'];u=len(d['M']);q0=np.asarray(cap['q0'],np.float32).reshape(u,2);sel=np.asarray(cap['selected'],np.int64);assert np.all(sel>=0)
 rr=np.arange(u);slot=d['mapping'][rr,sel];cat=tr['branch_category_id'][rr,sel];pre=np.power(tr['raw_score'].astype(np.float64),.5)*np.power(np.clip(q0[rr,slot].astype(np.float64),1e-6,1),.5);post,rank=G.gaussian(pre,tr['bbox'],tr['image_id'],cat,tr['global_cache_index'])
 strong=[dict(image_id=int(tr['image_id'][i]),category_id=int(cat[i]),bbox=tr['bbox'][i].tolist(),score=float(post[i]))for i in rr]
 expected['strong'].extend(strong);expected['pair'].extend(cap['visual_records']);meta.append(dict(image_id=entry['original_image_id'],rows=rows))
checks=[]
for g in read(C/'config/raw_protocol.json')['groups']:
 d=O/'raw_trials'/g['id'];actual=read(d/'records.json');assert actual==expected[g['method']],('raw_final_prediction_mismatch',g)
 metadata=read(d/'metadata.json');assert len(metadata)==len(meta)
 for a,e in zip(metadata,meta):
  assert a['original_image_id']==e['image_id'];assert a['proposal_rows']==e['rows'],('raw_proposal_input_mismatch',g,e['image_id'])
 checks.append(dict(group=g,actual_records=len(actual),same_raw_proposal_rows=True,exact_final_prediction_equal_independent_golden=True,actual_records_ref=ref(d/'records.json')))
summary=dict(status='PASS',completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,independent_golden=ref(O/'gates/raw_compatibility_review.json'),verifier=ref(Path(__file__)),scope='Every192formalrawimagerequestoutputsandproposalinputsreproduceindependent32imagegolden; strongbaselinepostderivedfromindependentlyverifiedq0andoriginalGaussian;noAP')
(O/'gates/raw_all_trial_prediction_golden.json').write_text(json.dumps(summary,indent=2)+'\n');print('PASS_ALL_SIX_RAW_TRIALS')
