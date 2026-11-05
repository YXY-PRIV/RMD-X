"""Population-aligned robust shared candidate quality; frozen batch-B only."""
from __future__ import annotations
import hashlib,json,os,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True
import numpy as np
import torch
from torch import nn
PROTOCOL_SHA='a27a2cc680f96844388b762745147db12ee97a95bc0332913377f7ee1b6bf651'
SOURCE=Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_candidate_refinement_20261031')
PILOT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_reliability_pilot_20261031')
DATA_SHA='479efb3825a05415295806a0e6d6a83227baf0ed0fa4fb2b4cc0c415cf66b6a1'
CSV_SHA='99e2ab3e1e330ef41bbb24380735097ccac0aa2bfbffc9e0b539b0dbef4ac603'
GPU_CHOICES=['GPU-7d53f26e-0c06-c368-43bf-de34b45f6a52','GPU-cfd4e982-ea9c-6090-3f22-88ea28c7360d']
UNIT_SCALES=[1.,1.,2.,1.,1.,2.,1.,2.,1.,1.,1.,2.,1.,1.,1.]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
 p=Path(r['path']);assert sha(p)==r['sha256']and('bytes'not in r or p.stat().st_size==r['bytes']),str(p);return p
def write(p,j):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n');t.replace(p)
def save_model(p,j):
 p=Path(p);t=p.with_name(p.name+'.tmp');torch.save(j,t);t.replace(p)
def read_protocol(p):
 p=Path(p).resolve()
 if PROTOCOL_SHA=='PENDING'or sha(p)!=PROTOCOL_SHA:raise ValueError('Batch-B scientific protocol not frozen/mismatch')
 j=read(p)
 if not j['status'].startswith('FROZEN')or len(j['feature_schema']['names'])!=15:raise ValueError('Wrong frozen B protocol/schema')
 if j['sources']['training_npz']['sha256']!=DATA_SHA or j['sources']['native_calfit_csv']['sha256']!=CSV_SHA:raise ValueError('B source identities differ')
 verify(j['sources']['training_npz']);verify(j['sources']['native_calfit_csv']);verify(j['sources']['image_split'])
 if j['normalization']['unit_scales']!=UNIT_SCALES or j['training']['inactive_features']!=[8,12]:raise ValueError('B fixed input policy differs')
 for k,v in {'lr':.001,'weight_decay':.0001,'batch_size':1024,'max_epochs':100,'early_stop_patience':15,'early_stop_min_delta':1e-6}.items():
  if j['training'][k]!=v:raise ValueError('B frozen training constant differs:'+k)
 return j
def recipe_for(P,recipe_id):
 entries=P['recipes'];entries=list(entries.values())if isinstance(entries,dict)else entries
 matches=[r for r in entries if str(r.get('id',r.get('recipe_id')))==str(recipe_id)]
 if len(matches)!=1:raise ValueError('Unknown/ambiguous recipe:'+str(recipe_id))
 r=matches[0]
 if r['population']not in ['all','kept']or r['normalization']not in ['zclip3','unit']or not isinstance(r['confidence_q_input'],bool):raise ValueError('Unsupported frozen B recipe')
 return r
def allowed_gpus(P):
 gs=P['resources']['allowed_gpu_uuids']
 if not gs or any(g not in GPU_CHOICES for g in gs):raise ValueError('B may only use authorized GPU0/GPU2')
 return gs
def cuda(device,P):
 if os.environ.get('CUDA_VISIBLE_DEVICES')not in allowed_gpus(P):raise ValueError('A single authorized physical B GPU must be visible')
 d=torch.device(device)
 if d.type!='cuda'or not torch.cuda.is_available():raise RuntimeError('Actual CUDA required; CPU fallback disabled')
 torch.cuda.set_device(d);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);return d
class RobustMLP(nn.Module):
 def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def raw_view(raw,input_conf_factor=1.,entropy_shift=0.):
 x=np.array(raw,np.float32,copy=True)
 if not np.isfinite(x).all()or not np.isin(x[...,13],[0.,1.]).all():raise ValueError('Bad raw candidate values')
 x[...,0]=(x[...,0].astype(np.float64)*float(input_conf_factor)).astype(np.float32);x[...,3]=np.clip(x[...,3].astype(np.float64)+float(entropy_shift),0.,1.).astype(np.float32)
 x[x[...,13]==0.,9:]=0.
 return x
def standardize(raw,normalizer,inactive,input_conf_factor=1.,entropy_shift=0.):
 x=raw_view(raw,input_conf_factor,entropy_shift).astype(np.float64);x=(x-np.asarray(normalizer['mean'],np.float64))/np.asarray(normalizer['std'],np.float64);x=x.astype(np.float32)
 if normalizer['kind']=='zclip3':np.clip(x,-3.,3.,out=x)
 x[...,sorted(set(inactive)|set(normalizer['zero_std_coordinates']))]=0.
 if not np.isfinite(x).all():raise ValueError('Nonfinite robust-normalized input')
 return x
def load_population(path,P,recipe,variant):
 import csv
 path=Path(path).resolve()
 if sha(path)!=DATA_SHA:raise ValueError('Immutable candidate source changed')
 csvpath=PILOT/'training/cal_fit_target_rows.csv'
 if sha(csvpath)!=CSV_SHA:raise ValueError('Immutable native-kept calibration labels changed')
 source_rows={int(r['global_cache_index']):r for r in csv.DictReader(csvpath.open())};data={};names=None;foldinfo={}
 with np.load(path,allow_pickle=False)as z:
  names=z['feature_names'].astype(str).tolist()
  if names!=P['feature_schema']['names']:raise ValueError('Feature order differs')
  for s,n in [('train',5874),('val',1408)]:
   x=z['X_'+s];y=z['y_'+s];ids=z['image_ids_'+s];rows=z['global_cache_index_'+s]
   if x.dtype!=np.float32 or x.shape!=(n,5,15)or y.dtype!=np.float32 or y.shape!=(n,5):raise ValueError('Source shape differs')
   if not np.isfinite(x).all()or not np.isfinite(y).all()or np.any((y<0)|(y>1)):raise ValueError('Malformed source values')
   kept=np.asarray([source_rows[int(i)]['visual_kept']=='True'for i in rows],bool);select=kept if s=='val'or recipe['population']=='kept'else np.ones(n,bool)
   if int(kept.sum())!=(1140 if s=='train'else 264):raise ValueError('Native-kept mask count differs')
   full=raw_view(x[select]);missing=full.copy();missing[...,9:]=0.
   data['X_full_'+s]=full.reshape(-1,15);data['X_missing_'+s]=missing.reshape(-1,15);data['y_'+s]=y[select].reshape(-1).copy();data['image_ids_'+s]=ids[select].copy();data['global_cache_index_'+s]=rows[select].copy()
   foldinfo[s]=dict(assigned_images=len(np.unique(ids)),effective_images=len(np.unique(ids[select])),proposals=int(select.sum()),unique_candidates=int(select.sum()*5),nativeV_kept_proposals=int(kept.sum()),teacher_positive_candidates=int((y[select]>0).sum()),teacher_quality_mean=float(y[select].mean()))
 if set(data['image_ids_train'])&set(data['image_ids_val']):raise ValueError('Train/validation image overlap')
 full=data['X_full_train'].astype(np.float64);missing=data['X_missing_train'].astype(np.float64);joint=np.concatenate([full,missing]);actual_mean,actual_std=joint.mean(0),joint.std(0,ddof=0);zero=np.flatnonzero(actual_std==0.).tolist()
 if recipe['normalization']=='zclip3':
  mean,std=actual_mean.copy(),actual_std.copy();mean[:9]=full.mean(0)[:9];std[:9]=full.std(0,ddof=0)[:9];std[std==0.]=1.
 else:mean,std=np.zeros(15,np.float64),np.asarray(UNIT_SCALES,np.float64)
 inactive={8,12}
 if variant=='visual':inactive.update(range(9,15))
 elif variant!='full_missing':raise ValueError('Unknown B variant')
 if not recipe['confidence_q_input']:inactive.add(0)
 normalizer=dict(kind=recipe['normalization'],fit_role='Unique selected train candidates and their material-missing views, never repeated draw stream',unique_train_candidates=len(full),paired_train_views=len(joint),mean=mean.tolist(),std=std.tolist(),zero_std_coordinates=zero,inactive_features=sorted(inactive),clip=3. if recipe['normalization']=='zclip3'else None,fixed_unit_scales=UNIT_SCALES if recipe['normalization']=='unit'else None,raw_training_mean_audit=actual_mean.tolist(),raw_training_std_audit=actual_std.tolist(),unit_interpretation='Natural bounded score scale; no physical material quantity')
 info=dict(source_input=ref(path),native_kept_csv=ref(csvpath),feature_names=names,folds=foldinfo,source_targets_unchanged=True,new_GT_annotations_read=False,validation_population_always_nativeV_kept=True,base_draws_per_epoch=29370,base_batch_size=1024,optimizer_steps_per_epoch=29,kept_sampling='Concatenate seeded independent randperms of5700 unique candidates until29370 draws, then truncate; stats never fit draw stream',paired_forward_views=2,variant=variant,recipe=recipe)
 return data,normalizer,info
def epoch_order(n,generator):
 remain=29370;parts=[]
 while remain:
  p=torch.randperm(n,generator=generator);take=min(n,remain);parts.append(p[:take]);remain-=take
 return torch.cat(parts)
@torch.no_grad()
def predict_tensor(model,x,batch=8192):
 model.eval();return torch.cat([model(t).sigmoid()for t in x.split(batch)]).cpu().numpy().astype(np.float32)if len(x)else np.empty(0,np.float32)
def train(args):
 pp=Path(args.protocol).resolve();P=read_protocol(pp);r=recipe_for(P,args.recipe_id);tr=P['training'];o=Path(args.output_dir).resolve()
 if args.variant not in ['visual','full_missing']or args.seed not in [17,29,43]:raise ValueError('Unknown variant/seed')
 if o.exists()and any(o.iterdir())and not args.resume:raise FileExistsError('New B model output required')
 o.mkdir(parents=True,exist_ok=True);data,norm,info=load_population(args.input,P,r,args.variant);d=cuda(args.device,P);random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
 xs={s:{v:torch.from_numpy(standardize(data['X_'+v+'_'+s],norm,norm['inactive_features'])).to(d)for v in ['full','missing']}for s in ['train','val']};ys={s:torch.from_numpy(data['y_'+s]).to(d)for s in ['train','val']}
 if args.variant=='visual'and any(not torch.equal(xs[s]['full'],xs[s]['missing'])for s in ['train','val']):raise ValueError('Visual paired views should be identical')
 model=RobustMLP().to(d);opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay']);gen=torch.Generator(device='cpu').manual_seed(args.seed)
 config=dict(protocol=ref(pp),input=ref(args.input),recipe_id=args.recipe_id,recipe=r,variant=args.variant,seed=args.seed,feature_names=info['feature_names'],normalizer=norm,training=tr,architecture=[15,32,16,1],loss='Paired full/missing BCE mean; one update per original-size base batch',input_stress_applied_during_training=False)
 best,bestloss,bestepoch,stale,hist,start_epoch,prior=None,float('inf'),-1,0,[],0,0.
 if args.resume:
  old=torch.load(args.resume,map_location=d,weights_only=True)
  if old['config']!=config:raise ValueError('Resume config differs')
  model.load_state_dict(old['state_dict']);opt.load_state_dict(old['optimizer_state']);gen.set_state(old['shuffle_state'].cpu());best,bestloss,bestepoch,stale,hist,start_epoch,prior=old['best_state'],old['best_loss'],old['best_epoch'],old['stale'],old['history'],old['epoch']+1,old['elapsed_seconds']
 env=dict(PID=os.getpid(),PPID=os.getppid(),argv=sys.argv,started_utc=now(),device=str(d),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_name=torch.cuda.get_device_name(d),torch_version=str(torch.__version__),CUDA_version=torch.version.cuda,TF32_enabled=False,deterministic_algorithms=True)
 write(o/'run_config.json',dict(config=config,data=info,environment=env));write(o/'normalizer.json',norm);write(o/'data_receipt.json',info);write(o/'run_status.json',dict(status='RUNNING_CUDA',recipe_id=args.recipe_id,variant=args.variant,seed=args.seed,PID=os.getpid()))
 print(json.dumps(dict(status='ACTUAL_CUDA_B_TRAINING',recipe_id=args.recipe_id,variant=args.variant,seed=args.seed,PID=os.getpid(),GPU_UUID=os.environ.get('CUDA_VISIBLE_DEVICES'),unique_train_candidates=len(ys['train']),validation_candidates=len(ys['val']),draws_per_epoch=29370,optimizer_steps_per_epoch=29)),flush=True);torch.cuda.reset_peak_memory_stats(d);torch.cuda.synchronize(d);start=time.perf_counter()
 for epoch in range(start_epoch,tr['max_epochs']):
  if stale>=tr['early_stop_patience']:break
  model.train();order=epoch_order(len(ys['train']),gen).to(d);steps=0
  for ii in order.split(1024):
   opt.zero_grad(set_to_none=True);xx=torch.cat([xs['train']['full'][ii],xs['train']['missing'][ii]]);yy=torch.cat([ys['train'][ii],ys['train'][ii]]);loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy)
   if not torch.isfinite(loss):raise FloatingPointError('Nonfinite B train loss')
   loss.backward();opt.step();steps+=1
  if steps!=29:raise AssertionError('B update budget differs')
  model.eval();losses={}
  with torch.no_grad():
   for s in ['train','val']:
    ll=[nn.functional.binary_cross_entropy_with_logits(model(xs[s][v]),ys[s])for v in ['full','missing']];losses[s]=[float(v.item())for v in ll];losses[s+'_mean']=float(torch.stack(ll).mean().item())
  vl=losses['val_mean']
  if not np.isfinite(vl):raise FloatingPointError('Nonfinite B validation loss')
  if vl<bestloss-tr['early_stop_min_delta']:
   bestloss,bestepoch,stale=vl,epoch,0;best={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};save_model(o/'model.pt',dict(format='night_B_robust_quality_v1',state_dict=best,config=config,best_epoch=epoch,best_val_paired_BCE=vl))
  else:stale+=1
  torch.cuda.synchronize(d);elapsed=prior+time.perf_counter()-start;hist.append(dict(epoch=epoch,train_full_BCE=losses['train'][0],train_missing_BCE=losses['train'][1],train_paired_BCE=losses['train_mean'],val_full_BCE=losses['val'][0],val_missing_BCE=losses['val'][1],val_paired_BCE=vl,best_epoch=bestepoch,stale_epochs=stale,draws=29370,optimizer_steps=steps,elapsed_seconds=elapsed));write(o/'train_history.json',hist);save_model(o/'last.pt',dict(config=config,epoch=epoch,state_dict=model.state_dict(),optimizer_state=opt.state_dict(),shuffle_state=gen.get_state(),best_state=best,best_loss=bestloss,best_epoch=bestepoch,stale=stale,history=hist,elapsed_seconds=elapsed));print(json.dumps(hist[-1]),flush=True)
 if best is None:raise RuntimeError('No B checkpoint')
 model.load_state_dict(best);pred_metrics={}
 for s in ['train','val']:
  pred_metrics[s]={}
  for v in ['full','missing']:
   q=predict_tensor(model,xs[s][v]);np.savez_compressed(o/f'{s}_{v}_quality.npz',quality=q);truth=data['y_'+s].astype(np.float64);qq=np.clip(q.astype(np.float64),1e-7,1-1e-7);pred_metrics[s][v]=dict(rows=len(q),target_mean=float(truth.mean()),prediction_mean=float(q.mean()),soft_BCE=float(-(truth*np.log(qq)+(1-truth)*np.log(1-qq)).mean()))
 torch.cuda.synchronize(d);result=dict(status='COMPLETE',actual_CUDA=True,recipe_id=args.recipe_id,variant=args.variant,seed=args.seed,best_epoch=bestepoch,epochs_completed=len(hist),optimizer_steps_completed=sum(x['optimizer_steps']for x in hist),base_draws_completed=len(hist)*29370,forward_view_draws_completed=len(hist)*58740,best_val_paired_BCE=bestloss,checkpoint=ref(o/'model.pt'),elapsed_seconds=prior+time.perf_counter()-start,environment=env,metrics=pred_metrics,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(d),peak_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(d),AP_computed=False,test_labels_read=False,recipe_selected=False)
 write(o/'metrics.json',result);write(o/'run_status.json',{k:result[k]for k in ['status','actual_CUDA','recipe_id','variant','seed','best_epoch','epochs_completed','optimizer_steps_completed','AP_computed','test_labels_read','recipe_selected']});return result
def predict_checkpoint(cp,features,names,device,batch,P,input_conf_factor=1.,entropy_shift=0.):
 d=cuda(device,P);ck=torch.load(cp,map_location=d,weights_only=True);conf=ck['config']
 if ck.get('format')!='night_B_robust_quality_v1'or conf['protocol']['sha256']!=PROTOCOL_SHA or conf['feature_names']!=names or features.shape[1:]!=(5,15):raise ValueError('Wrong B model/input identity')
 model=RobustMLP().to(d);model.load_state_dict(ck['state_dict']);out=[]
 for a in range(0,len(features),batch):
  b=min(a+batch,len(features));x=standardize(features[a:b],conf['normalizer'],conf['normalizer']['inactive_features'],input_conf_factor,entropy_shift);out.append(predict_tensor(model,torch.from_numpy(x.reshape(-1,15)).to(d),batch*5).reshape(b-a,5))
 return np.concatenate(out)if out else np.empty((0,5),np.float32),conf
