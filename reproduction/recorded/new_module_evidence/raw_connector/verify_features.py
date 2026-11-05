from pathlib import Path
import importlib.util,json,sys,hashlib,datetime
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path('/mnt/data/yxy/program/RAXO');OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/raw_connector');OUT.mkdir(exist_ok=True,parents=True)
spec=importlib.util.spec_from_file_location('dynamic_features',Path(__file__).with_name('features.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
import torch
torch.set_num_threads(8);torch.set_grad_enabled(False)
def read(p):return json.loads(Path(p).read_text())
def arr(r):
 p=Path(r['path'])
 return np.load(p,allow_pickle=False) if p.suffix=='.npy' else torch.load(p,map_location='cpu',weights_only=True).numpy()
def z(r):
 with np.load(r['path'],allow_pickle=False)as f:return{k:f[k].copy()for k in f.files}
P=read(ROOT/'experiments/pair_conflict_material_projection_20261101/config/protocol.json');c=P['cases']['CLCXray_test'];f=read(c['C_feature_manifest']['path']);a=f['identity']['assets'];nf=read(c['N_feature_manifest']['path']);af=read(c['auxiliary_feature_manifest']['path']);meta=[json.loads(x)for x in Path(a['proposal_meta']['path']).read_text().splitlines()]
ids=sorted({int(r['image_id'])for r in meta})[:4];sel=np.asarray([i for i,r in enumerate(meta)if int(r['image_id'])in ids]);rows=[meta[i]for i in sel]
V=arr(a['visual_similarity'])[sel];B=arr(a['material_class_similarity'])[sel];cats=read(a['category_meta']['path']);proto=read(a['prototype_meta']['path']);bank=read(a['material_bank']['path']);by={v['category_name']:v for v in bank['prototypes'].values()};native=[by[c['category_name']]['sample_count']for c in cats]
trace=z(f['outputs']['source_trace.npz']);loc={int(g):i for i,g in enumerate(sel)};gold=np.flatnonzero(np.isin(trace['global_cache_index'],sel));bf=np.zeros((len(sel),len(cats)),np.float32);ba=bf.copy();qv=np.ones(len(sel),bool)
N=arr(nf['outputs']['fresh_material_B.npy']);A=arr(af['outputs']['fresh_material_B.npy']);Q=arr(nf['outputs']['query_valid.npy'])
for i in gold:
 j=loc[int(trace['global_cache_index'][i])];bf[j]=N[i];ba[j]=A[i];qv[j]=Q[i]
states=[]
for label in ['V','F']:
 s=z(f['source_native_records'][label]['state']);idx=np.searchsorted(s['global_cache_index'],sel);assert np.array_equal(s['global_cache_index'][idx],sel);states.append({k:v[idx]for k,v in s.items()if len(v)==len(s['global_cache_index'])})
derived=m.native_states(V,B,rows,proto,cats)
checks={}
for bi,b in enumerate(['V','F']):
 for k in ['winner','category','negative','after_unc']:
  checks[b+'_'+k]=int(np.count_nonzero(derived[bi][k]!=states[bi][k]));assert checks[b+'_'+k]==0,(b,k,checks)
d=m.build_features(V,B,bf,rows,{int(r['image_id']):(r['width'],r['height'])for r in rows},proto,cats,native,aux_support=af['actual_material_support_counts'],original_support=nf['actual_material_support_counts'],Baux=ba,query_valid=qv)
assert np.array_equal(d['trace']['global_cache_index'],trace['global_cache_index'][gold])
for key,ref in [('C_X',f['outputs']['candidate_features.npy']),('T_X',nf['outputs']['candidate_features.npy']),('aux_X',af['outputs']['candidate_features.npy']),('M',f['outputs']['candidate_valid.npy']),('mapping',f['outputs']['branch_to_candidate.npy']),('ci',af['outputs']['candidate_class_order_index.npy'])]:
 expected=arr(ref)[gold];actual=d[key];assert expected.shape==actual.shape
 diff=int(np.count_nonzero(actual.view('uint8')!=expected.view('uint8')));checks[key+'_byte_differences']=diff;assert diff==0,(key,diff,float(np.max(np.abs(actual.astype(float)-expected.astype(float)))))
result=dict(status='PASS_DYNAMIC_FEATURES_AND_NATIVE_BRANCH_STATES_EXACT',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),case='CLCXray_test',images=ids,all_proposals=len(rows),union_proposals=len(gold),checks=checks,source_refs=m.source_refs(),GT_opened=False,scope='Exact current feature/native helper recipe on four saved source images; this does not certify historical raw image encoder reconstruction')
(OUT/'feature_golden.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
