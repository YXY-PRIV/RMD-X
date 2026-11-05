"""Actual complete P5 quality ensemble, no saved logits/quality used at runtime.

Input boundary: native proposal/candidate features. GPU T+full-catalogue conditional
and E forwards, exact CPU deployment sigmoid/fallback/mix/router/Gaussian. The
same callable accepts freshly constructed raw-image-native rows. No GT/AP reads.
"""
from __future__ import annotations
import os,sys,json,hashlib,importlib.util,time
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
ROOT=Path('/mnt/data/yxy/program/RAXO')
PAIR=ROOT/'experiments/pair_conflict_material_projection_20261101'
P=json.loads((PAIR/'config/protocol.json').read_text())
NIGHT=ROOT/'experiments/night_exploration_20261101'
SEEDS=(17,29,43)
def mod(path,name):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
TM=mod(ROOT/'experiments/material_reliability_20261101/src/quality_model.py','runtime_T')
CM=mod(ROOT/'experiments/material_conditional_ranking_fix1_20261101/src/quality_model.py','runtime_C')
EM=mod(NIGHT/'batch_e_affine_robustness/src/affine_model.py','runtime_E')
ND=mod(NIGHT/'batch_n_supported_material_quality/src/supported_data.py','runtime_N')
GD=mod(ROOT/'experiments/material_conditional_ranking_fix1_20261101/src/group_data.py','runtime_groups')
PP=mod(PAIR/'src/pair_projection.py','runtime_pairproject')
GA=mod(ROOT/'experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py','runtime_gaussian')
def read(p):return json.loads(Path(p).read_text())
def loadz(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def mix(base,head,M):
 q=np.zeros(M.shape,np.float32);q[M]=(np.power(np.clip(base[M].astype(np.float64),1e-6,1.),.5)*np.power(np.clip(head[M].astype(np.float64),1e-6,1.),.5)).astype(np.float32);return q

def case_data(case):
 c=P['cases'][case];cf=read(c['C_feature_manifest']['path']);nf=read(c['auxiliary_feature_manifest']['path']);old=read(read(P['sources']['T_protocol']['path'])['cases'][case]['N_feature_manifest']['path'])
 def arr(f,n):return np.load(f['outputs'][n+'.npy']['path'],allow_pickle=False)
 d=dict(case=case,C_X=arr(cf,'candidate_features'),T_X=arr(old,'candidate_features'),aux_X=arr(nf,'candidate_features'),M=arr(cf,'candidate_valid'),ci=arr(nf,'candidate_class_order_index'),cats=arr(cf,'candidate_category_id'),mapping=arr(cf,'branch_to_candidate'),B=arr(nf,'fresh_material_B'),has=arr(nf,'actual_supported_class_mask'),qv=arr(nf,'query_valid'),T=arr(nf,'supported_reference_T'),trace=loadz(cf['outputs']['source_trace.npz']['path']))
 d['geometry']=d['aux_X'][:,0,:8].copy()
 d['input_refs']=[c['C_feature_manifest'],c['auxiliary_feature_manifest'],read(P['sources']['T_protocol']['path'])['cases'][case]['N_feature_manifest']]
 return d

class Deployment:
 def __init__(self,device='cuda:0',method='pair'):
  self.dev=torch.device(device);self.models={};self.config={};self.refs=[]
  for seed in SEEDS:
   refs=dict(T=P['sources']['T_checkpoints']['availability_control'][str(seed)],C=P['native_models']['material'][str(seed)],E=dict(path=str(Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_e_affine_robustness/models/visual_clean')/f'seed{seed}/model.pt')))
   ckT=torch.load(refs['T']['path'],map_location='cpu',weights_only=True);cT=ckT['config'];state={k[len('anchor.'):]:v for k,v in ckT['state_dict'].items()if k.startswith('anchor.')}
   tm=TM.GatedResidual(state);tm.load_state_dict(ckT['state_dict']);tm.requires_grad_(False);tm.eval().to(self.dev)
   if method=='pair':
    ckC=torch.load(refs['C']['path'],map_location='cpu',weights_only=True);cm=CM.ConditionalRanker(TM.GatedResidual(state));cm.load_state_dict(ckC['state_dict']);cm.requires_grad_(False);cm.eval().to(self.dev)
    assert CM.state_digest(cm.anchor.state_dict())==CM.state_digest(tm.state_dict())
    cm.anchor=tm  # One immutable shared T anchor; no duplicate resident tensors.
   else:
    cm=None;ckC=dict(config=None)
   ckE=torch.load(refs['E']['path'],map_location='cpu',weights_only=True);em=EM.AffineQuality();em.load_state_dict(ckE['state_dict']);em.requires_grad_(False);em.eval().to(self.dev)
   self.models[seed]=(tm,cm,em);self.config[seed]=(cT,ckC['config'],ckE['config'])
   self.refs.extend(dict(role=r,seed=seed,path=x['path'],sha256=sha(x['path']))for r,x in refs.items()if method=='pair'or r!='C')
  self.parameter_count=sum(sum(p.numel()for p in self.models[s][0].parameters())+(sum(p.numel()for n,p in self.models[s][1].named_parameters()if not n.startswith('anchor.'))if self.models[s][1]is not None else 0)+sum(p.numel()for p in self.models[s][2].parameters())for s in SEEDS)
  assert self.parameter_count==(8322 if method=='pair'else 7236)
  self.unique_resident_parameters=len({p.data_ptr()for s in SEEDS for m in self.models[s]if m is not None for p in m.parameters()})
  self.unique_resident_parameter_count=sum(p.numel()for p in {p.data_ptr():p for s in SEEDS for m in self.models[s]if m is not None for p in m.parameters()}.values())
  assert self.unique_resident_parameter_count==self.parameter_count
  self.last={}
 def sync(self):torch.cuda.synchronize(self.dev)
 def run(self,d,method='pair',capture=False):
  assert method in ('strong','pair')
  M=d['M'];n=len(M);C=d['B'].shape[1];stages={};details={};seedT=[];seedPair=[];seedE=[];started=time.perf_counter_ns()
  def timed(name,fn):
   self.sync();t=time.perf_counter_ns();v=fn();self.sync();stages[name]=stages.get(name,0)+time.perf_counter_ns()-t;return v
  def fullcols():
   material=np.zeros((n,C,5),np.float32);context=np.zeros((n,13),np.float32)
   for lo in range(0,n,8192):
    hi=min(lo+8192,n);support=np.broadcast_to(d['has'],(hi-lo,C))&d['qv'][lo:hi,None];cols,count=ND.supported_columns(d['B'][lo:hi],support);assert np.array_equal(count,d['T'][lo:hi])
    full=np.zeros((hi-lo,C,13),np.float32);full[:,:,:8]=d['geometry'][lo:hi,None,:];full[:,:,8]=np.where(support,d['B'][lo:hi],np.float32(0.));full[:,:,9:12]=cols;full[:,:,12]=support.astype(np.float32)
    material[lo:hi]=full[:,:,8:];context[lo:hi]=GD.gate_context(full,support,d['qv'][lo:hi])[0]
   return material,context
  if method=='pair':material,context=timed('full_catalogue_feature_columns_and_context',fullcols)
  for seed in SEEDS:
   tm,cm,em=self.models[seed];ct,cc,ce=self.config[seed]
   def anchor_forward():
    out={k:np.zeros(M.shape,np.float32)for k in ['anchor_logits','correction_logits','full_logits','availability']};kq=np.zeros(M.shape,np.float32)
    for lo in range(0,n,8192):
     hi=min(lo+8192,n);m=M[lo:hi];raw=d['T_X'][lo:hi][m]
     tensors=TM.prepared_inputs(raw,'availability_control',ct['residual_normalization']['constant_coordinates'],ct['anchor_normalization'],self.dev)
     vals=TM.components_cpu(tm,tensors)
     for key in out:out[key][lo:hi][m]=vals[key].numpy().astype(np.float32)
     # Identical K geometry input/state and identical compressed execution layout.
     kq[lo:hi][m]=torch.sigmoid(vals['anchor_logits']).numpy().astype(np.float32)
    t=torch.sigmoid(torch.from_numpy(out['anchor_logits'])+torch.from_numpy(out['correction_logits'])).numpy().astype(np.float32)
    fallback=(out['correction_logits']==0.)|(out['availability']==0.)|~M;t[fallback]=kq[fallback]
    return out,t,kq
   tv,tq,kq=timed('three_T_availability_models_GPU_and_CPU_sigmoid',anchor_forward);seedT.append(tq.astype(np.float64))
   def visual_forward():
    out=np.zeros(M.shape,np.float32)
    for lo in range(0,n,8192):
     hi=min(lo+8192,n);m=M[lo:hi];x=EM.unit(d['C_X'][lo:hi][m],'visual_clean',ce['normalization']['constant_coordinates']);xx=torch.from_numpy(x).to(self.dev)
     out[lo:hi][m]=EM.pred(em,xx,16384)
    return out
   eq=timed('three_E_visual_clean_models_GPU',visual_forward);seedE.append(eq.astype(np.float64))
   if method=='pair':
    def conditional_forward():
     cg=np.zeros(M.shape,np.float32);rawg=np.zeros(M.shape,np.float32)
     for lo in range(0,n,8192):
      hi=min(lo+8192,n);b=hi-lo;support=np.broadcast_to(d['has'],(b,C))&d['qv'][lo:hi,None]
      full=np.empty((b,C,13),np.float32);full[:,:,:8]=d['geometry'][lo:hi,None,:];full[:,:,8:]=material[lo:hi]
      cx=CM.normalized(torch.from_numpy(context[lo:hi]).to(self.dev),cc['guards']['context_constant_coordinates'])
      with torch.no_grad():gg=torch.sigmoid(cm.gate(cx).squeeze(-1)).cpu().numpy().astype(np.float32)
      flat=full.reshape(-1,13);gflat=np.repeat(gg,C);sflat=support.reshape(-1);rawblock=np.empty(b*C,np.float32)
      for start in range(0,b*C,8192):
       stop=min(start+8192,b*C);xx=CM.normalized(torch.from_numpy(flat[start:stop]).to(self.dev),cc['guards']['residual_constant_coordinates'])
       with torch.no_grad():bounded=2.*torch.tanh(cm.residual(xx).squeeze(-1));vv=torch.from_numpy(sflat[start:stop].astype(np.float32)).to(self.dev)*torch.from_numpy(gflat[start:stop]).to(self.dev)*bounded
       rawblock[start:stop]=vv.cpu().numpy()
      rawblock=rawblock.reshape(b,C);idx=np.flatnonzero(d['has']);num=rawblock[:,idx].astype(np.float64).sum(1);mu=np.divide(num,d['T'][lo:hi],out=np.zeros(b,np.float64),where=d['T'][lo:hi]>0)
      centered=((rawblock.astype(np.float64)-mu[:,None])*.5).astype(np.float32);centered[~(support&(d['T'][lo:hi,None]>1))]=0.
      for slot in (0,1):
       local=np.flatnonzero(M[lo:hi,slot]);rows=local+lo;columns=d['ci'][rows,slot];cg[rows,slot]=centered[local,columns];rawg[rows,slot]=rawblock[local,columns]
     avail=d['aux_X'][:,:,12].copy();delta,active,D,stats=PP.project(cg,M,avail,d['T'],d['ci'],d['cats'],d['mapping'],d['trace']['branch_kept'],d['trace']['branch_category_id'])
     pq=torch.sigmoid(torch.from_numpy(tv['full_logits'])+torch.from_numpy(delta)).numpy().astype(np.float32);exact=(delta==0.)|~active[:,None]|~M;pq[exact]=tq[exact]
     return pq,cg,rawg,delta,active
    pq,cg,rawg,delta,active=timed('three_conditional_full_catalogue_models_GPU_projection_CPU',conditional_forward);seedPair.append(pq.astype(np.float64))
   if capture:
    details[seed]=dict(T=tv,Tq=tq,Kq=kq,Eq=eq)
    if method=='pair':details[seed].update(centered=cg,rawg=rawg,delta=delta,active=active,Pairq=pq)
  def finish():
   base=np.mean(np.stack(seedE),axis=0).astype(np.float32);head0=np.mean(np.stack(seedT),axis=0).astype(np.float32);q0=mix(base,head0,M)
   q=mix(base,np.mean(np.stack(seedPair),axis=0).astype(np.float32),M)if method=='pair'else q0.copy()
   tr=d['trace'];eff=np.zeros(M.shape,np.float32)
   for branch in (0,1):
    r=np.flatnonzero(tr['branch_kept'][:,branch]);eff[r,branch]=q0[r,d['mapping'][r,branch]]
   selected=np.full(n,-1,np.int8);selected[tr['branch_kept'][:,0]]=0;f=tr['branch_kept'][:,1]&((eff[:,1].astype(np.float64)-eff[:,0].astype(np.float64))>0);selected[f]=1
   rows=np.flatnonzero(selected>=0);branch=selected[rows].astype(np.int64);slot=d['mapping'][rows,branch];cat=tr['branch_category_id'][rows,branch];pre=np.power(tr['raw_score'][rows].astype(np.float64),.5)*np.power(np.clip(q[rows,slot].astype(np.float64),1e-6,1.),.5)
   post,rank=GA.gaussian(pre,tr['bbox'][rows],tr['image_id'][rows],cat,tr['global_cache_index'][rows])
   records=[dict(image_id=int(tr['image_id'][i]),category_id=int(c),bbox=tr['bbox'][i].tolist(),score=float(s))for i,c,s in zip(rows,cat,post)]
   return dict(q=q,q0=q0,selected=selected,post=post,rows=rows,records=records)
  result=timed('ensemble_geometric_mix_fixed_router_sqrt_score_Gaussian_serialization',finish)
  stages['total']=time.perf_counter_ns()-started;result['stages_ns']=stages
  if capture:result['details']=details
  self.last=result;return result
