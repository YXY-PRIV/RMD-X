"""Relative material features on frozen native-union class utility tuples."""
from __future__ import annotations
import datetime,hashlib,json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_i_relative_material')
PSHA='4c6710ffc5a8c9652477b9992db546870db0a9ac86a61a642d403ba5ba4de776'
CSHA='285b2fd5fcdbc974c762332cf33b3962a6876f75bd8e8a6c0e53e8235d394cd7'
GPU='GPU-1397025e-282b-4fa6-4abf-af166feb9117'
VARIANTS=['rank_agreement','agreement_only']
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
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');t.replace(p)
def loadz(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def protocol(path):
 if sha(path)!=PSHA:raise ValueError('I protocol differs from frozen SHA')
 p=read(path)
 if p['status']!='FROZEN_BEFORE_ANY_I_NUMERICAL_RUN':raise ValueError('I protocol not frozen')
 if p['training']['variants']!=VARIANTS or p['training']['architecture']!=[15,32,16,1]:raise ValueError('Model schema differs')
 return p
class RelativeQuality(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def unit(raw,variant,constants=()):
 x=np.array(raw,dtype=np.float32,copy=True)
 if x.ndim!=2 or x.shape[1]!=15 or not np.isfinite(x).all()or not np.isin(x[:,13],[0.,1.]).all()or variant not in VARIANTS:raise ValueError('Malformed relative quality input')
 x[x[:,13]==0.,9:]=0.
 x[:,[2,5,7,11]]/=np.float32(2)
 x[:,[0,3,8,9,11,12,13]]=0.
 if variant=='agreement_only':x[:,10]=0.
 if constants:x[:,list(constants)]=0.
 if not np.isfinite(x).all():raise ValueError('Nonfinite transformed input')
 return x

def dataset(path,p):
 if ref(path)!=p['sources']['C_training_npz']:raise ValueError('C teacher NPZ differs')
 d=loadz(path);meta=json.loads(str(d['metadata_json'].item()))
 if meta['protocol_sha256']!=CSHA or d['feature_names'].astype(str).tolist()!=p['feature_schema']['names']:raise ValueError('Source schema differs')
 data={}
 for role,n,validn,imgn,pos in [('train',1921,2013,152,497),('val',436,452,38,115)]:
  X,Y,M=d['X_'+role],d['y_'+role],d['candidate_valid_'+role]
  if X.shape!=(n,2,15)or X.dtype!=np.float32 or Y.shape!=(n,2)or Y.dtype!=np.float32 or M.shape!=(n,2)or M.dtype!=bool or M.sum()!=validn:raise ValueError('Population differs')
  if np.any(X[~M]!=0)or np.any(Y[~M]!=0)or not np.isfinite(Y).all()or np.any((Y<0)|(Y>1))or int((Y>0).sum())!=pos:raise ValueError('Invalid teachers/padding')
  if len(np.unique(d['image_ids_'+role]))!=imgn or len(np.unique(d['global_cache_index_'+role]))!=n:raise ValueError('Identity differs')
  data[role]=dict(X=X[M].copy(),Y=Y[M].copy(),image_ids=d['image_ids_'+role].copy())
 if set(data['train']['image_ids'])&set(data['val']['image_ids']):raise ValueError('Image split leakage')
 return data,d['feature_names'].astype(str).tolist(),meta

def cuda(device):
 if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU:raise ValueError('Only allocated physical GPU1 allowed')
 d=torch.device(device)
 if d.type!='cuda'or not torch.cuda.is_available():raise RuntimeError('Real CUDA required')
 torch.cuda.set_device(d);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);return d

def pred(model,x,batch=8192):
 model.eval()
 with torch.no_grad():return torch.cat([model(b).sigmoid()for b in x.split(batch)]).cpu().numpy().astype(np.float32)
def metrics(q,y):
 a=np.asarray(q,np.float64);b=np.asarray(y,np.float64);c=np.clip(a,1e-7,1-1e-7);return dict(n=len(a),soft_BCE=float(-(b*np.log(c)+(1-b)*np.log(1-c)).mean()),MSE=float(np.square(a-b).mean()),target_mean=float(b.mean()),prediction_mean=float(a.mean()))
def checkpoint(p,d):
 p=Path(p);t=p.with_name(p.name+'.tmp');torch.save(d,t);t.replace(p)
def train(a):
 p=protocol(a.protocol);v,s=a.variant,a.seed
 if v not in VARIANTS or s not in [17,29,43]:raise ValueError('Undeclared model')
 out=Path(a.output_dir).resolve()
 if not out.is_relative_to(OUT/'models')or(out.exists()and any(out.iterdir())):raise FileExistsError('Fresh I model output required')
 out.mkdir(parents=True,exist_ok=True);d,names,teacher=dataset(a.input,p);device=cuda(a.device);t=p['training'];random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
 drawgen=torch.Generator(device='cpu').manual_seed(s)
 baseline=unit(d['train']['X'],v);const=np.flatnonzero(np.ptp(baseline,axis=0)==0).tolist();normal=dict(kind='fixed_natural_units',divide_by_two=[2,5,7,11],always_zero=[0,3,8,9,11,12,13],variant=v,constant_coordinates=const,unique_train_tuples=2013,constant_fit_role='Natural unique valid training tuples only after fixed/variant masks; no padding/val/test',mean_or_std_fitted=False,full_always_zero13=True,visual_material_zero=False,active_material_coordinates=[10,14]if v=='rank_agreement'else[14])
 xs={r:torch.from_numpy(unit(d[r]['X'],v,const)).to(device)for r in ['train','val']};ys={r:torch.from_numpy(d[r]['Y']).to(device)for r in ['train','val']}
 model=RelativeQuality().to(device);opt=torch.optim.AdamW(model.parameters(),lr=t['lr'],weight_decay=t['weight_decay'])
 config=dict(protocol=ref(a.protocol),source_C_protocol=p['sources']['C_protocol'],input=ref(a.input),variant=v,seed=s,feature_names=names,normalization=normal,architecture=[15,32,16,1],training=t,teacher_metadata=teacher,draws_per_epoch=29370,updates_per_epoch=29,view_count=2,validation_view='natural only')
 env=dict(PID=os.getpid(),PPID=os.getppid(),argv=sys.argv,started_utc=now(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_name=torch.cuda.get_device_name(device),device=str(device),torch_version=str(torch.__version__),deterministic_algorithms=True,TF32_enabled=False)
 atomic(out/'run_config.json',dict(config=config,environment=env));atomic(out/'normalization.json',normal);atomic(out/'run_status.json',dict(status='RUNNING_CUDA',variant=v,seed=s,PID=os.getpid()))
 best=float('inf');bestep=-1;stale=0;weights=None;hist=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
 for ep in range(t['max_epochs']):
  if stale>=t['early_stop_patience']:break
  model.train();draw=torch.cat([torch.randperm(2013,generator=drawgen)for _ in range(15)])[:29370];steps=0
  for batch in draw.split(1024):
   ii=batch.to(device);first=xs['train'][ii]
   second=first
   xx=torch.cat([first,second]);yy=torch.cat([ys['train'][ii],ys['train'][ii]]);opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite paired-view BCE')
   loss.backward();opt.step();steps+=1
  if steps!=29:raise AssertionError('Update budget differs')
  model.eval()
  with torch.no_grad():losses={r:float(nn.functional.binary_cross_entropy_with_logits(model(xs[r]),ys[r]).item())for r in ['train','val']}
  vl=losses['val']
  if vl<best-t['early_stop_min_delta']:
   best,bestep,stale=vl,ep,0;weights={k:w.detach().cpu().clone()for k,w in model.state_dict().items()};checkpoint(out/'model.pt',dict(format='relative_material_union_quality_v1',state_dict=weights,config=config,best_epoch=ep,best_val_BCE=best))
  else:stale+=1
  h=dict(epoch=ep,train_natural_BCE=losses['train'],val_BCE=vl,best_epoch=bestep,stale_epochs=stale,optimizer_steps=29,draw_sha256=hashlib.sha256(draw.numpy().tobytes()).hexdigest(),elapsed_seconds=time.perf_counter()-start);hist.append(h);atomic(out/'train_history.json',hist);checkpoint(out/'last.pt',dict(config=config,epoch=ep,state_dict=model.state_dict(),optimizer_state=opt.state_dict(),best_state=weights,best_loss=best,best_epoch=bestep,stale=stale,history=hist,draw_generator_state=drawgen.get_state()));print(json.dumps(h),flush=True)
 if weights is None:raise RuntimeError('No checkpoint')
 model.load_state_dict(weights);values={}
 for role in ['train','val']:
  q=pred(model,xs[role]);np.savez_compressed(out/f'{role}_natural_quality.npz',quality=q);values[role]=metrics(q,d[role]['Y'])
 torch.cuda.synchronize(device);result=dict(status='COMPLETE',variant=v,seed=s,actual_CUDA=True,environment=env,checkpoint=ref(out/'model.pt'),best_epoch=bestep,best_val_BCE=best,epochs_completed=len(hist),optimizer_updates=29*len(hist),draws_per_epoch=29370,unique_train_tuples=2013,view_count=2,validation_view='natural only',metrics=values,elapsed_seconds=time.perf_counter()-start,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device),test_labels_read=False,new_GT_annotations_read=False,AP_computed=False,threshold_selected=False)
 atomic(out/'metrics.json',result);atomic(out/'run_status.json',{k:result[k]for k in ['status','variant','seed','actual_CUDA','epochs_completed','best_epoch','optimizer_updates','test_labels_read','AP_computed']});return result

def infer(a):
 p=protocol(a.protocol);fp=verify(p['cases'][a.case]['feature_manifest']);fm=read(fp)
 if ref(a.feature_manifest)!=p['cases'][a.case]['feature_manifest']or fm['protocol_sha256']!=CSHA:raise ValueError('I inference source C manifest differs')
 for k in ['candidate_features.npy','candidate_valid.npy','feature_names.json']:verify(fm['outputs'][k])
 X=np.load(fm['outputs']['candidate_features.npy']['path'],allow_pickle=False,mmap_mode='r');valid=np.load(fm['outputs']['candidate_valid.npy']['path'],allow_pickle=False);d=cuda(a.device);ck=torch.load(a.checkpoint,map_location=d,weights_only=True);c=ck['config'];names=read(fm['outputs']['feature_names.json']['path'])
 if isinstance(names,dict):names=names.get('names',names.get('feature_names'))
 if ck['format']!='relative_material_union_quality_v1'or c['protocol']['sha256']!=PSHA or c['feature_names']!=names or X.shape[1:]!=(2,15)or valid.shape!=X.shape[:2]:raise ValueError('Checkpoint or feature schema differs')
 out=Path(a.output_dir).resolve()
 if not out.is_relative_to(OUT/'inference')or(out.exists()and any(out.iterdir())):raise FileExistsError('Fresh I inference output required')
 out.mkdir(parents=True,exist_ok=True);model=RelativeQuality().to(d);model.load_state_dict(ck['state_dict']);q=np.zeros(valid.shape,np.float32);start=time.perf_counter()
 for lo in range(0,len(X),8192):
  hi=min(lo+8192,len(X));m=valid[lo:hi];raw=X[lo:hi][m];flat=unit(raw,c['variant'],c['normalization']['constant_coordinates']);q[lo:hi][m]=pred(model,torch.from_numpy(flat).to(d),16384)
 if not np.isfinite(q).all()or np.any((q<0)|(q>1))or np.any(q[~valid]!=0):raise ValueError('Output range/padding differs')
 qp=out/'quality_scores.npz';np.savez_compressed(qp,quality=q);torch.cuda.synchronize(d)
 receipt=dict(status='COMPLETE_LABEL_FREE_SHARED_UNION_INFERENCE',completed_utc=now(),PID=os.getpid(),argv=sys.argv,case=a.case,variant=c['variant'],seed=c['seed'],protocol=ref(a.protocol),protocol_sha256=PSHA,source_C_protocol_sha256=CSHA,feature_manifest=ref(fp),candidate_row_key_sha256=fm['candidate_row_key_sha256'],checkpoint=ref(a.checkpoint),output=ref(qp),output_sha256=sha(qp),shape=list(q.shape),dtype='float32',invalid_quality_exact0=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),device=str(d),GPU_name=torch.cuda.get_device_name(d),elapsed_seconds=time.perf_counter()-start,GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False)
 atomic(out/'inference.json',receipt);return receipt
