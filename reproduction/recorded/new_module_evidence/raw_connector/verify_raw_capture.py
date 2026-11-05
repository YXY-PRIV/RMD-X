"""Independent original-model reference for new raw runtime capture (no AP).

Does not import the newly implemented deployment class or pair projection.
Uses original conditional raw_components instead of the runtime unrolled head.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
from pathlib import Path
import sys,json,hashlib,importlib.util,datetime,time
import numpy as np
import torch
from PIL import Image
sys.dont_write_bytecode=True
ROOT=Path('/mnt/data/yxy/program/RAXO');sys.path.insert(0,str(ROOT))
HERE=Path(__file__).resolve().parent
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime')
NIGHT=ROOT/'experiments/night_exploration_20261101';P=json.loads((ROOT/'experiments/pair_conflict_material_projection_20261101/config/protocol.json').read_text());c=P['cases']['CLCXray_test']
def mod(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
F=mod(HERE/'features.py','golden_dynamic_features')
TM=mod(ROOT/'experiments/material_reliability_20261101/src/quality_model.py','golden_original_T')
CM=mod(ROOT/'experiments/material_conditional_ranking_fix1_20261101/src/quality_model.py','golden_original_conditional')
EM=mod(NIGHT/'batch_e_affine_robustness/src/affine_model.py','golden_original_E')
GA=mod(ROOT/'experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py','golden_original_gaussian')
ND=F._N;GD=F._G
from rmdx import preargmax_fusion as producer
from rmdx import material

def models():
 out={}
 for seed in [17,29,43]:
  t=torch.load(P['sources']['T_checkpoints']['availability_control'][str(seed)]['path'],map_location='cpu',weights_only=True)
  st={k[len('anchor.'):]:v for k,v in t['state_dict'].items()if k.startswith('anchor.')}
  tm=TM.GatedResidual(st);tm.load_state_dict(t['state_dict']);tm.eval().requires_grad_(False).cuda()
  cc=torch.load(P['native_models']['material'][str(seed)]['path'],map_location='cpu',weights_only=True)
  cm=CM.ConditionalRanker(TM.GatedResidual(st));cm.load_state_dict(cc['state_dict']);cm.eval().requires_grad_(False).cuda()
  e=torch.load(Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_e_affine_robustness/models/visual_clean')/f'seed{seed}/model.pt',map_location='cpu',weights_only=True)
  em=EM.AffineQuality();em.load_state_dict(e['state_dict']);em.eval().requires_grad_(False).cuda()
  out[seed]=(tm,t['config'],cm,cc['config'],em,e['config'])
 return out

def mix(e,t,M):
 out=np.zeros(M.shape,np.float32)
 out[M]=(np.power(np.clip(e[M].astype(np.float64),1e-6,1),.5)*np.power(np.clip(t[M].astype(np.float64),1e-6,1),.5)).astype(np.float32)
 return out

def direct(d,models):
 M=d['M'];n=len(M);C=d['B'].shape[1]
 if n==0:return dict(q=np.zeros((0,2),np.float32),q0=np.zeros((0,2),np.float32),selected=np.zeros(0,np.int8),post=np.zeros(0,np.float64),records=[])
 support=np.broadcast_to(d['has'],(n,C))&d['qv'][:,None]
 cols,T=ND['supported_columns'](d['B'],support)
 full=np.zeros((n,C,13),np.float32);full[:,:,:8]=d['geometry'][:,None,:];full[:,:,8]=np.where(support,d['B'],np.float32(0));full[:,:,9:12]=cols;full[:,:,12]=support.astype(np.float32)
 context,ct=GD['gate_context'](full,support,d['qv']);assert np.array_equal(ct,T)
 tv=[];pv=[];ev=[]
 for seed,(tm,tc,cm,cc,em,ec)in models.items():
  # Original installed scientific producer gives independent GPU operations.
  inp=TM.prepared_inputs(d['T_X'][M],'availability_control',tc['residual_normalization']['constant_coordinates'],tc['anchor_normalization'],'cuda:0')
  original=TM.components_cpu(tm,inp)
  anchor=np.zeros(M.shape,np.float32);cor=anchor.copy();fullold=anchor.copy();avail=anchor.copy();kq=anchor.copy()
  for target,key in [(anchor,'anchor_logits'),(cor,'correction_logits'),(fullold,'full_logits'),(avail,'availability')]:target[M]=original[key].numpy()
  kq[M]=torch.sigmoid(original['anchor_logits']).numpy()
  tq=torch.sigmoid(torch.from_numpy(anchor)+torch.from_numpy(cor)).numpy();fallback=(cor==0)|(avail==0)|~M;tq[fallback]=kq[fallback];tv.append(tq.astype(np.float64))
  ex=EM.unit(d['C_X'][M],'visual_clean',ec['normalization']['constant_coordinates']);eq=np.zeros(M.shape,np.float32);eq[M]=EM.pred(em,torch.from_numpy(ex).cuda(),16384);ev.append(eq.astype(np.float64))
  with torch.no_grad():raw,gate,bounded=cm.raw_components(torch.from_numpy(full).cuda(),torch.from_numpy(context).cuda(),torch.from_numpy(support).cuda(),cc['guards'])
  raw=raw.cpu().numpy();counts=support.sum(1);mean=np.divide(raw[:,d['has']].astype(np.float64).sum(1),counts,out=np.zeros(n,np.float64),where=counts>0)
  delta=((raw.astype(np.float64)-mean[:,None])*.5).astype(np.float32);delta[~(support&(counts[:,None]>1))]=0.
  gathered=np.zeros(M.shape,np.float32)
  for i,j in np.argwhere(M):gathered[i,j]=delta[i,d['ci'][i,j]]
  # Independently apply supported-conflict active mask and exact pair transform.
  tr=d['trace'];active=tr['branch_kept'].all(1)&(tr['branch_category_id'][:,0]!=tr['branch_category_id'][:,1])&d['qv']&(T>1)
  pair=np.zeros(M.shape,np.float32)
  for i in np.flatnonzero(active):
   v,f=map(int,d['mapping'][i]);assert v!=f and M[i,v]and M[i,f]
   if d['aux_X'][i,v,12]!=1. or d['aux_X'][i,f,12]!=1.:active[i]=False;continue
   D=np.float32((np.float64(gathered[i,f])-np.float64(gathered[i,v]))/2.)
   pair[i,v]=-D;pair[i,f]=D
  pq=torch.sigmoid(torch.from_numpy(fullold)+torch.from_numpy(pair)).numpy();same=(pair==0)|~active[:,None]|~M;pq[same]=tq[same];pv.append(pq.astype(np.float64))
 base=np.mean(np.stack(ev),axis=0).astype(np.float32);q0=mix(base,np.mean(np.stack(tv),axis=0).astype(np.float32),M);q=mix(base,np.mean(np.stack(pv),axis=0).astype(np.float32),M)
 tr=d['trace'];selected=np.full(n,-1,np.int8)
 for i in range(n):
  kv,kf=tr['branch_kept'][i];v,f=map(int,d['mapping'][i]);selected[i]=0 if kv else 1
  if kf and (not kv or np.float64(q0[i,f])-np.float64(q0[i,v])>0):selected[i]=1
 rr=np.arange(n);slots=d['mapping'][rr,selected];cid=tr['branch_category_id'][rr,selected]
 scores=np.power(tr['raw_score'].astype(np.float64),.5)*np.power(np.clip(q[rr,slots].astype(np.float64),1e-6,1),.5)
 post,rank=GA.gaussian(scores,tr['bbox'],tr['image_id'],cid,tr['global_cache_index'])
 records=[dict(image_id=int(tr['image_id'][i]),category_id=int(cid[i]),bbox=tr['bbox'][i].tolist(),score=float(post[i]))for i in rr]
 return dict(q=q,q0=q0,selected=selected,post=post,records=records)

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 source=OUT/'raw_trials/compatibility_pair/captures.json';captures=read(source);protocol=read(ROOT/'experiments/new_module_evidence_20261102/runtime/config/raw_protocol.json');assert len(captures)==32 and [x['entry']['original_image_id']for x in captures]==[x['original_image_id']for x in protocol['measured']];allmodels=models();cf=read(c['C_feature_manifest']['path']);nf=read(c['N_feature_manifest']['path']);af=read(c['auxiliary_feature_manifest']['path']);assets=cf['identity']['assets'];proto=read(assets['prototype_meta']['path']);cats=read(assets['category_meta']['path']);bank=read(assets['material_bank']['path']);names=[v['category_name']for v in cats];lookup=producer.build_category_material_matrix(names,names,bank)
 visual_bank_path=Path('/mnt/data/yxy-codex/home/outputs/RAXO/P16_P25_revision_20261024/validation/banks/visual_prototypes.pt')
 vb=torch.load(visual_bank_path,map_location='cpu',weights_only=True)
 vk=[k for k,v in vb.items()if v['supercategory']in names or k=='negative'];assert vk==[v['prototype_key']for v in proto]
 visual_bank=torch.stack([vb[k]['prototype']for k in vk]).float().cuda()
 results=[]
 for entry in captures:
  e,cap=entry['entry'],entry['capture'];rows=cap['rows'];n=len(rows);V=np.asarray(cap['V'],np.float32).reshape(n,len(proto));B=np.asarray(cap['B'],np.float32).reshape(n,len(cats));assert V.shape==(n,len(proto))and B.shape==(n,len(cats))
  embedding=np.asarray(cap['embedding'],np.float32).reshape(n,768)
  recovered=[]
  with torch.no_grad():
   for lo in range(0,n,64): recovered.append(torch.nn.functional.cosine_similarity(torch.from_numpy(embedding[lo:lo+64]).cuda()[:,None,:],visual_bank[None,:,:],dim=-1).cpu().numpy())
  recovered=np.concatenate(recovered,axis=0)if n else np.empty((0,len(proto)),np.float32)
  assert np.array_equal(recovered.view(np.uint32),V.view(np.uint32)),('cosine_visual_bank_embedding',e['original_image_id'])
  rgb=np.asarray(Image.open(e['image']['path']).convert('RGB'),np.uint8)
  raw=np.asarray([material.extract_colour_feature_from_array(rgb,r['bbox'])for r in rows],np.float64).reshape(n,22)
  assert np.array_equal(raw.view(np.uint64),np.asarray(cap['exact22'],np.float64).reshape(n,22).view(np.uint64))
  # Public unmodified producer separately decodes and calculates class cosine.
  bref=producer.compute_material_similarity_for_proposals(rows,Path(e['image']['path']).parent,lookup,device='cpu').numpy()
  assert np.array_equal(bref.view(np.uint32),B.view(np.uint32)),('B_current12class_lookup',e['original_image_id'])
  d=F.build_features(V,B,B,rows,{e['original_image_id']:(e['width'],e['height'])},proto,cats,native_support=cf['material_support_count'],original_support=nf['actual_material_support_counts'],aux_support=af['actual_material_support_counts'])
  check=direct(d,allmodels);checks={}
  for key,dtype in [('q',np.float32),('q0',np.float32),('selected',np.int8),('post',np.float64)]:
   expected=check[key];actual=np.asarray(cap[key],dtype).reshape(expected.shape);assert actual.shape==expected.shape
   diff=int(np.count_nonzero(np.ascontiguousarray(actual).view(np.uint8)!=np.ascontiguousarray(expected).view(np.uint8)));checks[key+'_byte_diff']=diff;assert diff==0,(e['original_image_id'],key,diff,float(np.max(np.abs(actual-expected)))if actual.size else 0)
  assert cap['visual_records']==check['records']
  results.append(dict(image_id=e['original_image_id'],all_proposals=n,union_proposals=len(d['M']),current22B_exact=True,visual_cosine_from_captured_embedding_and_bound_bank_exact=True,checks=checks))
  print('RAW_INDEPENDENT_GOLDEN',e['original_image_id'],'PASS',flush=True)
 summary=dict(status='PASS',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='32 fresh raw benchmark images; original model component classes/raw_components and independent pair/routing assembled from captured descriptors; not historical accuracy identity',images=len(results),results=results,captures=ref(source),raw_protocol=ref(ROOT/'experiments/new_module_evidence_20261102/runtime/config/raw_protocol.json'),verifier=ref(Path(__file__)),feature_builder=ref(HERE/'features.py'),visual_bank=ref(visual_bank_path),source_feature_golden=ref(OUT.parent/'raw_connector/feature_golden.json'),GPU=1,new_AP=0)
 compat_summary=OUT/'raw_trials/compatibility_pair/summary.json'
 for _ in range(120):
  if compat_summary.exists():break
  time.sleep(.5)
 assert compat_summary.exists(),'Compatibility actor has not finished; do not authorize formal trials'
 done=read(compat_summary);assert done['actual_worker_exit']==0 and done['status']=='PASS_FRESH_PROCESS_RAW_TO_FINAL_INFERENCE' and done['compatibility_only'] is True
 summary['compatibility_actor_summary']=ref(compat_summary)
 (OUT/'gates').mkdir(exist_ok=True);(OUT/'gates/raw_compatibility_review.json').write_text(json.dumps(summary,indent=2)+'\n')
 print(json.dumps(dict(status='PASS',images=len(results))))
if __name__=='__main__':main()
