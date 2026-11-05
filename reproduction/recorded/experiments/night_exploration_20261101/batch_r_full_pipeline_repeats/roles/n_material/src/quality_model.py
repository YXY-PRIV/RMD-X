"""K source-balanced geometry/material utility heads; no detector/test GT access."""
from __future__ import annotations
import datetime,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True;os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_r_full_pipeline_repeats/roles/n_material')
PSHA='1f113a0d7a1dffea0331fbe2e9aaff5fdaf2e6e45156c9df59ffce43a149f418'
GPU='GPU-cfd4e982-ea9c-6090-3f22-88ea28c7360d';VARIANTS=['material']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
 p=Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],str(p);return p
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def loadz(p):
 with np.load(p,allow_pickle=False)as a:return{k:a[k].copy()for k in a.files}
def protocol(p):
 assert sha(p)==PSHA,'R N role protocol SHA differs';d=read(p);assert d['status']=='FROZEN_R_N_MATERIAL_ROLE_BEFORE_ANY_NEW_TRAINING_OR_INFERENCE'and d['training']['variants']==VARIANTS and d['training']['architecture']==[13,32,16,1];verify(d['parent_protocol']);verify(d['inference_clarification']);return d
class Quality(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(13,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def unit(raw,active,constant=()):
 x=np.array(raw,dtype=np.float32,copy=True)
 if x.ndim!=2 or x.shape[1]!=13 or not np.isfinite(x).all():raise ValueError('K head needs finite13D features')
 x[:,[j for j in range(13)if j not in active]]=0.
 if constant:x[:,list(constant)]=0.
 return x

def populations(d):
 s,img,g,c,cid=d['source_index'],d['image_index'],d['global_cache_index'],d['class_count'],d['category_id'];N=len(s)
 assert set(np.unique(s))=={0,1,2}and np.all(c>=2)
 keys=np.stack([s,img,g],1);pk,pi,pc=np.unique(keys,axis=0,return_inverse=True,return_counts=True);assert np.array_equal(pc[pi],c),'Every proposal must contain all C classes'
 assert len(np.unique(np.stack([s,img,g,cid],1),axis=0))==N,'Duplicate class tuple'
 ik,ii,ic=np.unique(np.stack([s,img],1),axis=0,return_inverse=True,return_counts=True);assert np.array_equal(ic[ii]%c,np.zeros(N,np.int64))
 nprops=ic[ii]//c;w=1./(nprops.astype(np.float64)*c.astype(np.float64));valw=np.empty(N,np.float64);source={};groups=[]
 for sid in range(3):
  rows=np.flatnonzero(s==sid);classes=np.unique(c[rows]);assert len(classes)==1;images=np.unique(img[rows]);localw=w[rows].copy();groups.append((rows,torch.from_numpy(localw)))
  valw[rows]=1./(3.*len(images)*ic[ii[rows]])
  source[str(sid)]=dict(valid_tuples=len(rows),effective_images=len(images),proposals=int(np.sum(pk[:,0]==sid)),classes=int(classes[0]),sampler_weights_f64_sha256=hashlib.sha256(localw.tobytes()).hexdigest())
 assert abs(valw.sum()-1.)<1e-12
 return groups,valw,source

def dataset(path,P):
 r=read(P['training']['source_input']['actual_ref_from']);assert r['status']=='COMPLETE'and r['protocol']==P['source_N_protocol']and ref(path)==r['training_npz']==P['source_N_training_npz']
 a=loadz(path);names=a['feature_names'].astype(str).tolist();meta=json.loads(str(a['metadata_json'].item()));assert names==P['feature_schema']['names']and meta['protocol_sha256']==P['source_N_protocol']['sha256'];data={}
 for role in ['train','val']:
  d={k:a[k+'_'+role]for k in ['X','y','source_index','image_index','class_count','category_id','global_cache_index','query_valid']};N=len(d['y']);assert d['X'].dtype==np.float32 and d['X'].shape==(N,13)and d['y'].dtype==np.float32 and d['y'].shape==(N,)and np.isfinite(d['X']).all()and np.isfinite(d['y']).all()and np.all((d['y']>=0)&(d['y']<=1))
  for k in ['source_index','image_index','class_count','category_id','global_cache_index']:assert d[k].dtype==np.int64 and d[k].shape==(N,)
  assert d['query_valid'].dtype==bool and d['query_valid'].shape==(N,)and not np.any(d['X'][~d['query_valid'],8:]!=0),'Empty ROI rows must remain finite with zero material'
  groups,vw,stats=populations(d);d.update(groups=groups,balance_weights=vw,statistics=stats);data[role]=d
 assert not set(zip(data['train']['source_index'],data['train']['image_index']))&set(zip(data['val']['source_index'],data['val']['image_index']))
 return data,names,meta,r

def cuda(dev):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==GPU,'R N material only GPU2 allowed';d=torch.device(dev);assert d.type=='cuda'and torch.cuda.is_available();torch.cuda.set_device(d);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);return d

def logits_cpu(model,x,batch=8192):
 model.eval()
 with torch.no_grad():return torch.cat([model(b)for b in x.split(batch)]).cpu()
def monitor(model,x,y,C,weights):
 logits=logits_cpu(model,x);yy=torch.from_numpy(y);pos=torch.from_numpy((C-1).astype(np.float32));weighted=nn.functional.binary_cross_entropy_with_logits(logits,yy,pos_weight=pos,reduction='none').numpy().astype(np.float64);plain=nn.functional.binary_cross_entropy_with_logits(logits,yy,reduction='none').numpy().astype(np.float64);q=torch.sigmoid(logits).numpy().astype(np.float32);m=dict(n=len(y),balanced_weighted_BCE=float(np.dot(weights,weighted)),balanced_unweighted_BCE=float(np.dot(weights,plain)),balanced_MSE=float(np.dot(weights,np.square(q.astype(np.float64)-y.astype(np.float64)))),pooled_unweighted_BCE=float(plain.mean()),prediction_mean=float(np.dot(weights,q)),target_mean=float(np.dot(weights,y)))
 return q,m

def checkpoint(path,d):
 p=Path(path);q=p.with_name(p.name+'.tmp');torch.save(d,q);q.replace(p)

def train(a):
 P=protocol(a.protocol);v,seed=a.variant,a.seed;t=P['training'];assert v in VARIANTS and seed in P['seeds'];out=Path(a.output_dir).resolve();assert out.is_relative_to(OUT/'models')and not(out.exists()and any(out.iterdir()));out.mkdir(parents=True,exist_ok=True)
 data,names,meta,receipt=dataset(a.input,P);dev=cuda(a.device);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator(device='cpu').manual_seed(seed);active=P['feature_schema']['active_coordinates'][v];reference=unit(data['train']['X'],active);const=np.flatnonzero(np.ptp(reference,axis=0)==0).tolist();norm=dict(kind='natural_fixed13_no_fitted_mean_std',active_coordinates=active,constant_coordinates=const,constant_fit='Unique training tuples only after inactive mask',mean_or_std_fitted=False)
 xs={r:torch.from_numpy(unit(data[r]['X'],active,const)).to(dev)for r in ['train','val']};yt=torch.from_numpy(data['train']['y']).to(dev);ct=torch.from_numpy((data['train']['class_count']-1).astype(np.float32)).to(dev);model=Quality().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=t['learning_rate'],weight_decay=t['weight_decay']);config=dict(protocol=ref(a.protocol),parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],role=P['role'],input=ref(a.input),data_receipt=ref(P['training']['source_input']['actual_ref_from']),variant=v,seed=seed,feature_names=names,architecture=[13,32,16,1],normalization=norm,training=t,source_statistics={r:data[r]['statistics']for r in ['train','val']},teacher_metadata=meta,validation_aggregation='Float32 CPU BCE_with_logits then Float64 dot with equal-source/image/tuple weights',view_count=2,updates_per_epoch=29,draws_per_epoch=29370,sigmoid_location='CPUFloat32 on GPU-produced logits; no CPU model fallback')
 env=dict(PID=os.getpid(),PPID=os.getppid(),argv=sys.argv,started_utc=now(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_name=torch.cuda.get_device_name(dev),device=str(dev),torch_version=str(torch.__version__),deterministic_algorithms=True,TF32_enabled=False);atomic(out/'run_config.json',dict(config=config,environment=env));atomic(out/'normalization.json',norm);atomic(out/'run_status.json',dict(status='RUNNING_CUDA',variant=v,seed=seed,PID=os.getpid()));hist=[];best=float('inf');be=-1;stale=0;state=None;start=time.perf_counter();torch.cuda.reset_peak_memory_stats(dev)
 for ep in range(t['max_epochs']):
  if stale>=t['patience']:break
  chunks=[torch.from_numpy(rows)[torch.multinomial(w,9790,replacement=True,generator=gen)]for rows,w in data['train']['groups']];draw=torch.cat(chunks);draw=draw[torch.randperm(29370,generator=gen)];assert len(draw)==29370;model.train();steps=0
  for b in draw.split(1024):
   ii=b.to(dev);xx=torch.cat([xs['train'][ii],xs['train'][ii]]);yy=torch.cat([yt[ii],yt[ii]]);ww=torch.cat([ct[ii],ct[ii]]);opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy,pos_weight=ww)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weighted BCE')
   loss.backward();opt.step();steps+=1
  assert steps==29;model.eval();_,vm=monitor(model,xs['val'],data['val']['y'],data['val']['class_count'],data['val']['balance_weights']);vl=vm['balanced_weighted_BCE']
  if vl<best-t['min_delta']:
   best,be,stale=vl,ep,0;state={k:w.detach().cpu().clone()for k,w in model.state_dict().items()};checkpoint(out/'model.pt',dict(format='R_N_supported_reference_repeat_v1',state_dict=state,config=config,best_epoch=ep,best_val_BCE=best))
  else:stale+=1
  hh=dict(epoch=ep,val_BCE=vl,val_metrics=vm,best_epoch=be,stale_epochs=stale,optimizer_steps=29,draw_sha256=hashlib.sha256(draw.numpy().tobytes()).hexdigest(),draws_per_source={str(s):int((data['train']['source_index'][draw.numpy()]==s).sum())for s in range(3)},elapsed_seconds=time.perf_counter()-start);assert set(hh['draws_per_source'].values())=={9790};hist.append(hh);atomic(out/'train_history.json',hist);checkpoint(out/'last.pt',dict(config=config,epoch=ep,state_dict=model.state_dict(),optimizer_state=opt.state_dict(),best_state=state,best_loss=best,best_epoch=be,stale=stale,history=hist,draw_generator_state=gen.get_state()));print(json.dumps(hh),flush=True)
 assert state is not None;model.load_state_dict(state);metrics={}
 for r in ['train','val']:
  q,m=monitor(model,xs[r],data[r]['y'],data[r]['class_count'],data[r]['balance_weights']);np.savez_compressed(out/f'{r}_natural_quality.npz',quality=q);metrics[r]=m
 torch.cuda.synchronize(dev);result=dict(status='COMPLETE',variant=v,seed=seed,actual_CUDA=True,environment=env,checkpoint=ref(out/'model.pt'),best_epoch=be,best_val_BCE=best,epochs_completed=len(hist),optimizer_updates=29*len(hist),draws_per_epoch=29370,source_draws_per_epoch=9790,view_count=2,metrics=metrics,elapsed_seconds=time.perf_counter()-start,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(dev),test_labels_read=False,AP_computed=False,threshold_selected=False);atomic(out/'metrics.json',result);atomic(out/'run_status.json',{k:result[k]for k in ['status','variant','seed','actual_CUDA','best_epoch','epochs_completed','optimizer_updates','test_labels_read','AP_computed']});return result

def infer(a):
 P=protocol(a.protocol);fp=Path(a.feature_manifest).resolve();assert ref(fp)==P['cases'][a.case]['source_N_feature_manifest'];F=read(fp);assert F['protocol_sha256']==P['source_N_protocol']['sha256']and F['source_C_feature_manifest']==P['cases'][a.case]['feature_manifest']and F['candidate_row_key_sha256']==P['cases'][a.case]['candidate_row_key_sha256'];X=np.load(verify(F['outputs']['candidate_features.npy']),mmap_mode='r',allow_pickle=False);M=np.load(verify(F['outputs']['candidate_valid.npy']),allow_pickle=False);assert X.shape[1:]==(2,13)and M.shape==X.shape[:2];dev=cuda(a.device);ck=torch.load(a.checkpoint,map_location=dev,weights_only=True);c=ck['config'];assert ck['format']=='R_N_supported_reference_repeat_v1'and c['protocol']['sha256']==PSHA and c['feature_names']==F['feature_names'];model=Quality().to(dev);model.load_state_dict(ck['state_dict']);out=Path(a.output_dir).resolve();assert out.is_relative_to(OUT/'inference')and not(out.exists()and any(out.iterdir()));out.mkdir(parents=True,exist_ok=True);q=np.zeros(M.shape,np.float32);start=time.perf_counter()
 for lo in range(0,len(X),8192):
  hi=min(lo+8192,len(X));mask=M[lo:hi];x=unit(X[lo:hi][mask],c['normalization']['active_coordinates'],c['normalization']['constant_coordinates']);xx=torch.from_numpy(x).to(dev);q[lo:hi][mask]=torch.sigmoid(logits_cpu(model,xx)).numpy().astype(np.float32)
 assert np.isfinite(q).all()and np.all((q>=0)&(q<=1))and not np.any(q[~M]);qp=out/'quality_scores.npz';np.savez_compressed(qp,quality=q);torch.cuda.synchronize(dev);r=dict(status='COMPLETE_LABEL_FREE_SHARED_UNION_INFERENCE',completed_utc=now(),PID=os.getpid(),argv=sys.argv,case=a.case,variant=c['variant'],seed=c['seed'],protocol=ref(a.protocol),parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],role=P['role'],protocol_sha256=PSHA,feature_manifest=ref(fp),candidate_row_key_sha256=F['candidate_row_key_sha256'],checkpoint=ref(a.checkpoint),output=ref(qp),output_sha256=sha(qp),shape=list(q.shape),dtype='float32',invalid_quality_exact0=True,sigmoid_location='CPUFloat32 on GPU-produced logits; no CPU model fallback',CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(dev),GPU_name=torch.cuda.get_device_name(dev),elapsed_seconds=time.perf_counter()-start,GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False);atomic(out/'inference.json',r);return r
