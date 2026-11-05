"""One class-conditional scalar quality function on valid native-union tuples."""
from __future__ import annotations
import json,os,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.dont_write_bytecode=True
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from .union_data import PSHA,read,ref,sha,verify,atomic,now
GPU_UUID='GPU-1397025e-282b-4fa6-4abf-af166feb9117'
VARIANTS=['full','visual','missing']
class SharedQuality(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x):return self.net(x).squeeze(-1)
def cuda(device):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU_UUID:raise ValueError('Batch C exclusively uses physical GPU1')
    d=torch.device(device)
    if d.type!='cuda'or not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    torch.cuda.set_device(d);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);return d
def unit(raw,variant,missing=False,constants=()):
    x=np.array(raw,dtype=np.float32,copy=True)
    if x.shape[-1]!=15 or not np.isfinite(x).all()or not np.isin(x[...,13],[0.,1.]).all():raise ValueError('Malformed raw union features')
    x[x[...,13]==0.,9:]=0.
    if missing:x[...,9:]=0.
    x[...,[2,5,7,11]]/=np.float32(2)
    x[...,[0,8,12]]=0.
    if variant=='visual':x[...,9:]=0.
    elif variant=='full':x[...,13]=0.
    elif variant!='missing':raise ValueError('Invalid union model variant')
    if constants:x[...,list(constants)]=0.
    if not np.isfinite(x).all():raise ValueError('Nonfinite natural-unit input')
    return x
def load_data(path,protocol):
    with np.load(path,allow_pickle=False)as z:
        meta=json.loads(str(z['metadata_json'].item()));names=z['feature_names'].astype(str).tolist();data={}
        if meta['protocol_sha256']!=PSHA or names!=protocol['feature_schema']['names']:raise ValueError('Union data schema/protocol differs')
        for split,proposals,valid_count,images in [('train',1921,2013,152),('val',436,452,38)]:
            X,Y,M=z['X_'+split],z['y_'+split],z['candidate_valid_'+split]
            if X.dtype!=np.float32 or X.shape!=(proposals,2,15)or Y.dtype!=np.float32 or Y.shape!=(proposals,2)or M.dtype!=bool or M.shape!=(proposals,2)or M.sum()!=valid_count:raise ValueError('Union valid training population differs')
            if np.any(X[~M]!=0)or np.any(Y[~M]!=0)or not np.isfinite(Y).all()or np.any((Y<0)|(Y>1)):raise ValueError('Padding/targets invalid')
            ids=z['image_ids_'+split];g=z['global_cache_index_'+split]
            if len(np.unique(ids))!=images or len(np.unique(g))!=proposals:raise ValueError('Effective image/global row identity differs')
            data[split]=dict(X=X[M].copy(),Y=Y[M].copy(),original_image_ids=ids.copy(),global_cache_index=g.copy(),proposals=proposals)
        if set(data['train']['original_image_ids'])&set(data['val']['original_image_ids']):raise ValueError('Train/val image leakage')
    return data,names,meta
def predict(model,x,batch=8192):
    model.eval()
    with torch.no_grad():return torch.cat([model(t).sigmoid()for t in x.split(batch)]).cpu().numpy().astype(np.float32)
def metrics(q,y):
    q=np.asarray(q,np.float64);y=np.asarray(y,np.float64);p=np.clip(q,1e-7,1-1e-7);return dict(n=len(q),soft_BCE=float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()),MSE=float(np.square(q-y).mean()),target_mean=float(y.mean()),prediction_mean=float(q.mean()))
def checkpoint(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
def train(args):
    pp=Path(args.protocol).resolve();P=read(pp)
    if PSHA=='PENDING'or sha(pp)!=PSHA:raise ValueError('Union protocol not frozen')
    variant,seed=args.variant,args.seed
    if variant not in VARIANTS or seed not in [17,29,43]:raise ValueError('Invalid model configuration')
    out=Path(args.output_dir).resolve()
    if out.exists()and any(out.iterdir()):raise FileExistsError('Fresh model output required')
    out.mkdir(parents=True,exist_ok=True);raw,names,meta=load_data(args.input,P);tr=P['training'];device=cuda(args.device)
    if tr['batch_size']!=1024:raise ValueError('Matched optimizer base batch budget differs')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);gen=torch.Generator(device='cpu').manual_seed(seed)
    reference=unit(raw['train']['X'],variant)
    if variant=='missing':reference=np.concatenate([reference,unit(raw['train']['X'],variant,True)],axis=0)
    constants=np.flatnonzero(np.ptp(reference,axis=0)==0.).tolist();normalization=dict(kind='fixed_natural_units',divide_by_two=[2,5,7,11],always_zero=[0,8,12],variant=variant,full_always_zero13=variant=='full',visual_material_zero=variant=='visual',constant_coordinates=constants,constant_fit_role='All unique valid train class tuples only; both views for missing variant',unique_train_tuples=2013,constant_fit_rows=len(reference),mean_or_std_fitted=False,raw_missing_policy='raw13==0 => raw9..14 exact0 before unit transform')
    xs={};ys={}
    for split in ['train','val']:
        xs[split]={'full':torch.from_numpy(unit(raw[split]['X'],variant,False,constants)).to(device),'second':torch.from_numpy(unit(raw[split]['X'],variant,variant=='missing',constants)).to(device)}
        ys[split]=torch.from_numpy(raw[split]['Y']).to(device)
    model=SharedQuality().to(device);opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay']);config=dict(protocol=ref(pp),input=ref(args.input),variant=variant,seed=seed,feature_names=names,normalization=normalization,architecture=[15,32,16,1],training=tr,teacher_metadata=meta,draw_sampling='Concatenate fresh seeded randperm2013 cycles and truncate to29370; same per-seed base draws across variants',draws_per_epoch=29370,updates_per_epoch=29,view_count=2,second_view='raw material zero'if variant=='missing'else'duplicate first view')
    env=dict(PID=os.getpid(),PPID=os.getppid(),argv=sys.argv,started_utc=now(),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),GPU_name=torch.cuda.get_device_name(device),device=str(device),torch_version=str(torch.__version__),deterministic_algorithms=True,TF32_enabled=False)
    atomic(out/'run_config.json',dict(config=config,environment=env));atomic(out/'normalization.json',normalization);atomic(out/'run_status.json',dict(status='RUNNING_CUDA',variant=variant,seed=seed,PID=os.getpid()));best=None;bestloss=float('inf');bestepoch=-1;stale=0;history=[];draw_hashes=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(tr['max_epochs']):
        if stale>=tr['early_stop_patience']:break
        model.train();draw=torch.cat([torch.randperm(len(ys['train']),generator=gen)for _ in range((29370+len(ys['train'])-1)//len(ys['train']))])[:29370];draw_hashes.append(__import__('hashlib').sha256(draw.numpy().tobytes()).hexdigest());steps=0
        for batch in draw.split(1024):
            ii=batch.to(device);xx=torch.cat([a[ii]for a in xs['train'].values()],dim=0);yy=torch.cat([ys['train'][ii]for _ in xs['train']],dim=0);opt.zero_grad(set_to_none=True);loss=nn.functional.binary_cross_entropy_with_logits(model(xx),yy)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite union training loss')
            loss.backward();opt.step();steps+=1
        if steps!=29:raise AssertionError('Optimizer update budget differs')
        model.eval();losses={}
        with torch.no_grad():
            for split in ['train','val']:
                vals=[nn.functional.binary_cross_entropy_with_logits(model(a),ys[split])for a in xs[split].values()];losses[split]=float(torch.stack(vals).mean().item())
        vl=losses['val']
        if vl<bestloss-tr['early_stop_min_delta']:
            bestloss,bestepoch,stale=vl,epoch,0;best={k:v.detach().cpu().clone()for k,v in model.state_dict().items()};checkpoint(out/'model.pt',dict(format='native_union_shared_quality_v1',state_dict=best,config=config,best_epoch=epoch,best_val_BCE=vl))
        else:stale+=1
        h=dict(epoch=epoch,train_BCE=losses['train'],val_BCE=vl,best_epoch=bestepoch,stale_epochs=stale,optimizer_steps=29,draw_sha256=draw_hashes[-1],elapsed_seconds=time.perf_counter()-start);history.append(h);atomic(out/'train_history.json',history);checkpoint(out/'last.pt',dict(config=config,epoch=epoch,state_dict=model.state_dict(),optimizer_state=opt.state_dict(),best_state=best,best_loss=bestloss,best_epoch=bestepoch,stale=stale,history=history,draw_generator_state=gen.get_state()));print(json.dumps(h),flush=True)
    if best is None:raise RuntimeError('No valid union checkpoint')
    model.load_state_dict(best);scores={}
    for split in ['train','val']:
        scores[split]={}
        for view,x in xs[split].items():
            q=predict(model,x);np.savez_compressed(out/f'{split}_{view}_quality.npz',quality=q);scores[split][view]=metrics(q,raw[split]['Y'])
    torch.cuda.synchronize(device);result=dict(status='COMPLETE',variant=variant,seed=seed,actual_CUDA=True,environment=env,checkpoint=ref(out/'model.pt'),best_epoch=bestepoch,best_val_BCE=bestloss,epochs_completed=len(history),optimizer_updates=29*len(history),draws_per_epoch=29370,unique_train_tuples=2013,view_count=config['view_count'],metrics=scores,elapsed_seconds=time.perf_counter()-start,peak_CUDA_allocated_bytes=torch.cuda.max_memory_allocated(device),test_labels_read=False,AP_computed=False,threshold_selected=False)
    atomic(out/'metrics.json',result);atomic(out/'run_status.json',{k:result[k]for k in ['status','variant','seed','actual_CUDA','epochs_completed','best_epoch','optimizer_updates','test_labels_read','AP_computed']});return result
def infer(cp,X,valid,names,device,batch):
    d=cuda(device);ck=torch.load(cp,map_location=d,weights_only=True);c=ck['config']
    if ck['format']!='native_union_shared_quality_v1'or c['protocol']['sha256']!=PSHA or c['feature_names']!=names or X.shape[1:]!=(2,15)or valid.shape!=X.shape[:2]:raise ValueError('Union checkpoint/input schema mismatch')
    model=SharedQuality().to(d);model.load_state_dict(ck['state_dict']);out=np.zeros(valid.shape,np.float32)
    for a in range(0,len(X),batch):
        b=min(a+batch,len(X));m=valid[a:b];flat=X[a:b][m];values=unit(flat,c['variant'],False,c['normalization']['constant_coordinates']);out[a:b][m]=predict(model,torch.from_numpy(values).to(d),batch*2)
    return out,c
