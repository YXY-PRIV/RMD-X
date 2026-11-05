#!/usr/bin/env python3
"""One fresh cached-feature cost trial, using immutable original pure head code."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,datetime,hashlib,importlib.util,json,resource,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_s_fixed_quality_cost');PSHA='589054d18fb8bc808204137eb608504bc14adbe5958ca0526ceda1eb7d338b57'
sys.path.insert(0,str(CODE/'src'));from composition import direct,compose
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(r):assert ref(r['path'])==r;return Path(r['path'])
def write(p,j):
 with Path(p).open('x')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()
def npzload(r):
 with np.load(check(r),allow_pickle=False)as z:return z['quality'].copy()
def module(name,r):
 spec=importlib.util.spec_from_file_location(name,check(r));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def rss():return int(next(s.split()[1]for s in Path('/proc/self/status').read_text().splitlines()if s.startswith('VmRSS:')))*1024
def compare(a,b,tol):
 assert a.dtype==b.dtype==np.float32 and a.shape==b.shape
 diff=float(np.abs(a.astype(np.float64)-b.astype(np.float64)).max(initial=0.));return dict(passed=diff<=tol,max_abs=diff,atol=tol,rtol=0.,bitwise_equal=a.tobytes()==b.tobytes(),fresh_sha256=digest(a),reference_sha256=digest(b))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--variant',required=True);ap.add_argument('--kind',choices=['warmup','measured'],required=True);ap.add_argument('--trial',type=int,required=True);ap.add_argument('--output-dir',required=True);a=ap.parse_args();pp=CODE/'config/protocol.json';assert sha(pp)==PSHA;P=read(pp);F=read(OUT/'automation/code_freeze.json')
 for r in F['sources']:check(r)
 assert F['protocol']==ref(pp)and a.case in P['cases']and a.variant in P['variants'];keys=P['variants'][a.variant];c=P['cases'][a.case];out=Path(a.output_dir).resolve();assert out.is_relative_to(OUT/'trials')and not out.exists();out.mkdir(parents=True)
 report=dict(status='RUNNING_FRESH_FORWARD',created_utc=now(),PID=os.getpid(),protocol=ref(pp),code_freeze=ref(OUT/'automation/code_freeze.json'),case=a.case,variant=a.variant,kind=a.kind,trial=a.trial,components=keys,seeds=P['seeds'],stages={},cached_q_used_in_measured_computation=False,new_AP_GT_training_or_features=False,attempted_model_cohort_forwards=0,completed_model_cohort_forwards=0,completed_three_seed_role_means=0)
 stop=threading.Event();observed=[rss()];baseline=observed[0]
 def sampler():
  while not stop.wait(.02):observed.append(rss())
 monitor=threading.Thread(target=sampler,daemon=True);monitor.start();dev=None
 try:
  assert os.environ['CUDA_VISIBLE_DEVICES']==P['resources']['GPU_UUID'];mods={key:module('_S_original_'+key,P['components'][key]['source'])for key in keys}
  for key in keys:
   check(P['components'][key]['protocol'])
   for r in P['components'][key]['checkpoints']:check(r)
   check(c['features'][key]['manifest']);check(c['features'][key]['array'])
  check(c['valid']);integrity_end=time.perf_counter();dev=torch.device('cuda:0');torch.cuda.set_device(dev);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True);torch.empty(1,device=dev);torch.cuda.synchronize(dev);report['CUDA_setup_seconds']=time.perf_counter()-integrity_end;torch.cuda.reset_peak_memory_stats(dev);report['GPU_name']=torch.cuda.get_device_name(dev);report['CUDA_VISIBLE_DEVICES']=os.environ['CUDA_VISIBLE_DEVICES'];report['TF32_enabled']=False;report['actual_CUDA']=True
  def timed(name,fn):
   torch.cuda.synchronize(dev);t=time.perf_counter();value=fn();torch.cuda.synchronize(dev);report['stages'][name]=time.perf_counter()-t;return value
  def features():
   valid=np.load(c['valid']['path'],allow_pickle=False);xs={key:np.load(c['features'][key]['array']['path'],allow_pickle=False)for key in keys};assert valid.dtype==bool
   for key,x in xs.items():assert x.dtype==np.float32 and x.shape[:2]==valid.shape and x.shape[2]==(15 if key=='E'else 13)
   return valid,xs
  valid,xs=timed('feature_read',features);report['shape']=list(valid.shape);report['valid_candidates']=int(valid.sum());report['feature_input_bytes']=sum(c['features'][key]['array']['bytes']for key in keys)+c['valid']['bytes']
  def weights():
   models={}
   for key in keys:
    models[key]=[];m=mods[key]
    for seed,r in zip(P['seeds'],P['components'][key]['checkpoints']):
     ck=torch.load(r['path'],map_location=dev,weights_only=True);cfg=ck['config'];assert cfg['protocol']==P['components'][key]['protocol']and cfg['seed']==seed and cfg['variant']==P['components'][key]['variant'];model=(m.AffineQuality()if key=='E'else m.Quality()).to(dev);model.load_state_dict(ck['state_dict']);model.eval();models[key].append((model,cfg))
   return models
  models=timed('weights_load_to_GPU',weights)
  def forwards():
   result={}
   for key in keys:
    m=mods[key];X=xs[key];values=[]
    for model,cfg in models[key]:
     report['attempted_model_cohort_forwards']+=1
     q=np.zeros(valid.shape,np.float32)
     for lo in range(0,len(X),8192):
      hi=min(lo+8192,len(X));mask=valid[lo:hi];raw=X[lo:hi][mask]
      if key=='E':flat=m.unit(raw,cfg['variant'],cfg['normalization']['constant_coordinates']);q[lo:hi][mask]=m.pred(model,torch.from_numpy(flat).to(dev),16384)
      else:flat=m.unit(raw,cfg['normalization']['active_coordinates'],cfg['normalization']['constant_coordinates']);xx=torch.from_numpy(flat).to(dev);q[lo:hi][mask]=torch.sigmoid(m.logits_cpu(model,xx)).numpy().astype(np.float32)
     assert np.isfinite(q).all()and not np.any(q[~valid]);values.append(q);report['completed_model_cohort_forwards']+=1
    result[key]=values
   return result
  fresh=timed('fresh_all_models_forward',forwards)
  def combine():
   means={}
   for key in keys:
    means[key]=np.mean(np.stack([v.astype(np.float64)for v in fresh[key]]),axis=0,dtype=np.float64).astype(np.float32);report['completed_three_seed_role_means']+=1
   if a.variant=='E3':q=means['E'].copy()
   elif a.variant=='N6':q=direct(means['E'],means['N'],valid)
   else:
    available=(xs['N'][:,:,12]>0)&valid;assert np.array_equal(available,(xs['K'][:,:,12]>0)&valid);q,_=compose(means['E'],means['K'],means['N'],available,valid,.5)
   assert np.isfinite(q).all()and not np.any(q[~valid]);return means,q
  means,q=timed('three_seed_means_and_fixed_combination',combine);report['primary_cached_feature_head_combination_seconds']=sum(report['stages'].values());report['GPU_peak_allocated_bytes']=torch.cuda.max_memory_allocated(dev);report['GPU_peak_reserved_bytes']=torch.cuda.max_memory_reserved(dev);report['CPU_primary_sampled_peak_RSS_bytes']=max(observed+[rss()]);report['CPU_primary_kernel_peak_RSS_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
  t=time.perf_counter();validation={};references={}
  for key in keys:
   checks=[compare(v,npzload(r),P['correctness']['seed_quality_atol'])for v,r in zip(fresh[key],c['reference_seed_scores'][key])];reference=npzload(c['reference_ensembles'][key]);meancheck=compare(means[key],reference,P['correctness']['ensemble_quality_atol']);validation[key]=dict(seeds=checks,mean=meancheck);references[key]=reference;assert all(z['passed']for z in checks)and meancheck['passed']
  if a.variant=='E3':reference=references['E'].copy()
  elif a.variant=='N6':reference=direct(references['E'],references['N'],valid)
  else:reference,_=compose(references['E'],references['K'],references['N'],(xs['N'][:,:,12]>0)&valid,valid,.5)
  if a.variant!='E3':assert digest(reference)==c['reference_combined_hash'][a.variant],'Original combination reference hash differs'
  validation['combined']=compare(q,reference,P['correctness']['combined_quality_atol']);assert validation['combined']['passed'];report['stages']['validation_against_frozen_q']=time.perf_counter()-t;report['validation']=validation
  t=time.perf_counter();qp=out/'fresh_quality.npz'
  with qp.open('xb')as f:np.savez_compressed(f,quality=q)
  report['stages']['output_persistence']=time.perf_counter()-t;report['output']=ref(qp);report['status']='PASS_FRESH_CUDA_FORWARD_AND_FIXED_COMBINATION';report['source_feature_manifests']={k:c['features'][k]['manifest']for k in keys};report['source_checkpoints']={k:P['components'][k]['checkpoints']for k in keys};report['candidate_row_key_sha256']=c['candidate_row_key_sha256'];report['scope']=P['scope'];report['hash_verification_prewarms_OS_cache_before_feature_read']=True
 except BaseException as e:report.update(status='FAILED_RETAINED_NO_RETRY',error=repr(e),traceback=traceback.format_exc())
 finally:
  stop.set();monitor.join();observed.append(rss());report.update(completed_utc=now(),CPU_baseline_RSS_bytes=baseline,CPU_sampled_peak_RSS_bytes=max(observed),CPU_kernel_process_peak_RSS_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),CPU_RSS_sample_count=len(observed),CPU_RSS_sample_period_seconds=.02);write(out/'receipt.json',report)
 print(json.dumps(dict(status=report['status'],receipt=ref(out/'receipt.json'))),flush=True)
 if report['status']!='PASS_FRESH_CUDA_FORWARD_AND_FIXED_COMBINATION':raise SystemExit(1)
if __name__=='__main__':main()
