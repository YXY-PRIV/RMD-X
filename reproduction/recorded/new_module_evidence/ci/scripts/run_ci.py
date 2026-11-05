#!/usr/bin/env python3
"""N-specific post-hoc gates and replay; original D has strict formal priority."""
from __future__ import annotations
import os
for _k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[_k]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,concurrent.futures,contextlib,datetime,fcntl,gc,hashlib,json,math,multiprocessing,pickle,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ci')
PSHA=os.environ.get('CI_PROTOCOL_SHA256')
if PSHA:os.sched_setaffinity(0,set(range(96,120))&os.sched_getaffinity(0))
sys.dont_write_bytecode=True;sys.path.insert(0,str(CODE/'vendor'))
import native_core as nc
from vector_core import replay_fast
NAMES=['AP','AP50','AP75','AP_small','AP_medium','AP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def check(r):
 p=Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],str(p);return p
def scalar(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):scalar(v)for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [scalar(v)for v in x]
 return x
def save(p,d):
 p=Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(scalar(d),indent=2,allow_nan=False)+'\n');t.replace(p)
def fresh(p,d):
 p=Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(scalar(d),f,indent=2,allow_nan=False);f.write('\n')
def npz(p,**d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.npz');assert not p.exists();np.savez_compressed(t,**d);t.replace(p)
def event(message,**kw):print(json.dumps(dict(utc=now(),message=message,**scalar(kw))),flush=True)
def available_gib():return int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))/1024**2
def frozen(full=False,case=None):
 pp=CODE/'config/protocol.json';assert PSHA and sha(pp)==PSHA;P=read(pp);assert P['status']=='FROZEN_BEFORE_NEW_MATCHING_OR_BOOTSTRAP'
 freeze=read(OUT/'automation/code_freeze.json');assert freeze['protocol']==ref(pp)
 for r in freeze['sources']:check(r)
 for r in P['native_libraries'].values():check(r)
 for r in P['source_refs']:check(r)
 if full:
  c=P['cases'][case]
  for key in ['source_metric','source_export','prediction','GT','draws','gate_draws']:check(c[key])
 return P,ref(pp),ref(OUT/'automation/code_freeze.json')
@contextlib.contextmanager
def native_slot(P):
 directory=Path(P['resources']['shared_native_slot_directory']);directory.mkdir(parents=True,exist_ok=True);f=None;slot=None
 try:
  while f is None:
   for i,name in enumerate(P['resources']['native_slot_names']):
    s=(directory/name).open('a')
    try:fcntl.flock(s,fcntl.LOCK_EX|fcntl.LOCK_NB);f=s;slot=i;break
    except BlockingIOError:s.close()
   if f is None:time.sleep(.2)
  yield slot
 finally:
  if f is not None:fcntl.flock(f,fcntl.LOCK_UN);f.close()
def native_eval(P,case,plan,gt,pred,log):
 with native_slot(P)as slot:
  event('ACTUAL_NATIVE_MATCHING_STARTED',case=case,plan=plan,PID=os.getpid(),shared_slot=slot)
  with Path(log).open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):e=nc.native(gt,pred)
 return e
def gate(case):
 P,pr,fr=frozen(True,case);c=P['cases'][case];directory=OUT/'gates'/case;gp=directory/'gate.json'
 if gp.exists():
  g=read(gp);assert g['status']=='PASS'and g['protocol']==pr and g['code_freeze']==fr;check(g['cache']);check(g['point']);return g
 directory.mkdir(parents=True,exist_ok=True);start=time.perf_counter();gt=read(c['GT']['path']);pred=read(c['prediction']['path']);ids=sorted(int(r['id'])for r in gt['images'])
 assert ids==c['params']['imgIds']and sorted(int(r['id'])for r in gt['categories'])==c['params']['catIds']
 anno=[int(a['id'])for a in gt['annotations']];assert anno and min(anno)>0 and len(set(anno))==len(anno)
 allowed_images=set(ids)
 assert len(pred)==c['prediction_count']and all(int(r['image_id'])in allowed_images for r in pred)
 point=native_eval(P,case,'fresh_point',gt,pred,directory/'point_native.log');point_compare=nc.compare(point.stats,c['expected_point_stats']);assert point_compare['passed']and nc.params(point.params)==c['params']
 pa=directory/'point_arrays.npz';npz(pa,**{k:point.eval[k]for k in ['precision','recall','scores']},stats=point.stats)
 point_report=dict(status='PASS_FRESH_CANDIDATE_NATIVE_POINT',protocol=pr,code_freeze=fr,source_metric=c['source_metric'],source_export=c['source_export'],prediction=c['prediction'],GT=c['GT'],params=nc.params(point.params),stats=point.stats,source_stats_comparison=point_compare,arrays=ref(pa),canonical_prediction=nc.canonical(pred),actual_new_native_matching=True)
 pp=directory/'point.json';fresh(pp,point_report);cache=nc.make_cache(point,gt,pred,ids);cp=directory/'matching_cache.pickle'
 with cp.open('xb')as f:pickle.dump(cache,f,protocol=5)
 draw=np.load(c['draws']['path'],mmap_mode='r',allow_pickle=False);assert draw.shape==(2000,len(ids))and draw.dtype==np.int64
 plans={'identity':np.arange(len(ids),dtype=np.int64),'bootstrap_0000':draw[0],'bootstrap_0001':draw[1]}
 with np.load(c['gate_draws']['path'],allow_pickle=False)as z:
  assert set(z.files)==set(plans)
  for name,a in plans.items():assert np.array_equal(a,z[name])
 reports=[]
 for name,a in plans.items():
  t=time.perf_counter();gg,dd=nc.clone(gt,pred,a,ids);reference=native_eval(P,case,name,gg,dd,directory/(name+'_native.log'));nt=time.perf_counter()-t;t=time.perf_counter()
  with(directory/(name+'_replay.log')).open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):replay=replay_fast(cache,a)
  ft=time.perf_counter()-t;checks={k:nc.compare(reference.eval[k],replay.eval[k])for k in ['precision','recall','scores']};checks['stats']=nc.compare(reference.stats,replay.stats);checks['params']=dict(passed=nc.params(reference.params)==nc.params(replay.params));checks['records']=nc.compare_records(reference,replay)
  if name=='identity':checks['identity_point_stats']=nc.compare(reference.stats,point.stats)
  arrays=directory/(name+'_native_vs_replay.npz');npz(arrays,**{prefix+'_'+k:e.eval[k]for prefix,e in [('native',reference),('replay',replay)]for k in ['precision','recall','scores']},native_stats=reference.stats,replay_stats=replay.stats)
  record=dict(plan=name,passed=all(v['passed']for v in checks.values()),checks=checks,arrays=ref(arrays),native_seconds=nt,replay_seconds=ft,unique_original_images=len(np.unique(a)),sampled_occurrences=len(a),valid_GT_categories=int(nc.valid_mask(reference).sum()),positive_GT_IDs=all(x['id']>0 for x in gg['annotations']))
  fresh(directory/(name+'_check.json'),record);reports.append(record);assert record['passed'],case+'/'+name
  del gg,dd,reference,replay;gc.collect()
 g=dict(status='PASS',case=case,method=c['method'],protocol=pr,code_freeze=fr,input_binding={k:c[k]for k in ['source_metric','source_export','prediction','GT','draws','gate_draws']},cache=ref(cp),point=ref(pp),reports=reports,actual_fresh_native_matching_calls=4,gate_vector_replays=3,wall_seconds=time.perf_counter()-start,gate_peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,formal_engine_authorized='gated-vector-replay-not-fresh-matching')
 fresh(gp,g);event('GATE_PASS',case=case,seconds=g['wall_seconds'],cache=g['cache']);return g
def probe(case):
 P,pr,fr=frozen();g=read(OUT/'gates'/case/'gate.json');assert g['status']=='PASS'and g['protocol']==pr;rp=OUT/'resources'/case/'probe.json'
 if rp.exists():r=read(rp);assert r['cache']['sha256']==g['cache']['sha256'];return r
 with check(g['cache']).open('rb')as f:cache=pickle.load(f)
 draw=np.load(P['cases'][case]['draws']['path'],mmap_mode='r',allow_pickle=False);t=time.perf_counter()
 with open(os.devnull,'w')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):e=replay_fast(cache,draw[0])
 assert np.isfinite(e.stats).all();r=dict(status='RESOURCE_PROBE_COMPLETE_NOT_FORMAL_STAT',protocol=pr,code_freeze=fr,case=case,cache=g['cache'],draws=P['cases'][case]['draws'],draw_id=0,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,seconds=time.perf_counter()-t,plan_chosen_by_resource_only=True,AP_values_not_used_for_resource_choice=True)
 fresh(rp,r);event('RESOURCE_PROBE_COMPLETE',case=case,peak_RSS_MiB=r['peak_RSS_MiB'],seconds=r['seconds']);return r
_CACHE=None;_DRAWS=None;_CASE=None
def initialize_worker(cache,draws,case):
 global _CACHE,_DRAWS,_CASE
 with Path(cache).open('rb')as f:_CACHE=pickle.load(f)
 _DRAWS=np.load(draws,mmap_mode='r',allow_pickle=False);_CASE=case
def shard(job):
 lo,hi,binding=job;directory=OUT/'results'/_CASE/'shards';pp=directory/f'{lo:04d}_{hi:04d}.npz';jp=pp.with_suffix('.json')
 if jp.exists():r=read(jp);assert r['binding']==binding;check(r['arrays']);return r
 if pp.exists():raise FileExistsError('Preserve unsealed shard before explicit recovery: '+str(pp))
 stats=np.empty((hi-lo,12),np.float64);valid=np.empty((hi-lo,_CACHE['K']),bool);mult=np.empty(valid.shape,np.uint32);unique=np.empty(hi-lo,np.int32);start=time.perf_counter()
 with open(os.devnull,'w')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
  for j,bid in enumerate(range(lo,hi)):
   draw=_DRAWS[bid];e=replay_fast(_CACHE,draw);stats[j]=e.stats;valid[j]=nc.valid_mask(e);mult[j]=_CACHE['incidence'][:,draw].sum(axis=1);unique[j]=len(np.unique(draw));assert valid[j].any();del e
 npz(pp,bootstrap_id=np.arange(lo,hi,dtype=np.int64),stats=stats,valid_category_mask=valid,category_image_multiplicity=mult,unique_original_images=unique)
 r=dict(case=_CASE,start=lo,stop=hi,binding=binding,arrays=ref(pp),seconds=time.perf_counter()-start,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,engine='gated-vector-replay')
 fresh(jp,r);return r
def intervals(stats,first=1000):
 result={}
 for i,name in enumerate(NAMES):
  allv=stats[:,i];v=allv[allv>=0];head=allv[:first];head=head[head>=0]
  full=np.quantile(v,[.025,.975],method='linear').tolist()if len(v)else None;initial=np.quantile(head,[.025,.975],method='linear').tolist()if len(head)else None
  result[name]=dict(valid_replicates=len(v),missing_GT_sentinel_replicates=int(np.sum(allv<0)),full_B2000_percentile=full,first1000_percentile=initial,endpoint_shift_full_minus_first=(np.asarray(full)-initial).tolist()if full is not None and initial is not None else None)
 return result
def formal(case,update):
 P,pr,fr=frozen(True,case);c=P['cases'][case];gp=OUT/'gates'/case/'gate.json';g=read(gp);assert g['status']=='PASS'and g['protocol']==pr and g['code_freeze']==fr;check(g['cache']);check(g['point']);probe_ref=ref(OUT/'resources'/case/'probe.json');probe_data=read(probe_ref['path']);directory=OUT/'results'/case;completion=directory/'completion.json'
 if completion.exists():r=read(completion);check(r['arrays']);return r
 lockpath=Path(P['resources']['shared_formal_lock']);lockpath.parent.mkdir(parents=True,exist_ok=True)
 with lockpath.open('a')as global_lock:
  while True:
   try:fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
   except BlockingIOError:update(status='WAITING_NIGHT_FORMAL_LOCK',case=case);time.sleep(30)
  try:
   plan_path=OUT/'resources'/case/'formal_plan.json';peak=max(.1,probe_data['peak_RSS_MiB']/1024)
   while True:
    available=available_gib();limit=int((available-64)/(1.25*peak))
    if plan_path.exists():plan=read(plan_path);workers=plan['workers'];ready=limit>=workers
    else:workers=min(24,max(0,limit));ready=workers>=1
    if ready:break
    update(status='WAITING_RESERVED_RAM',case=case,available_RAM_GiB=available,required_reserve_GiB=64);time.sleep(30)
   if not plan_path.exists():fresh(plan_path,dict(status='FROZEN_BEFORE_FORMAL_SHARDS',created_utc=now(),protocol=pr,case=case,workers=workers,threads_each=1,available_RAM_GiB=available,reserve_GiB=64,probe_peak_RSS_GiB=peak,guard_factor=1.25,probe=probe_ref,choice_independent_of_AP_sign=True))
   binding=dict(protocol_sha256=pr['sha256'],code_freeze_sha256=fr['sha256'],cache_sha256=g['cache']['sha256'],draws_sha256=c['draws']['sha256'],gate_sha256=sha(gp),resource_plan_sha256=sha(plan_path),engine='gated-vector-replay',case=case)
   start=time.perf_counter();done=set();min_available=available;update(status='FORMAL_GATED_VECTOR_REPLAY',case=case,workers=workers,completed_replicates=0,total_replicates=2000)
   jobs=[(lo,min(2000,lo+5),binding)for lo in range(0,2000,5)]
   with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('fork'),initializer=initialize_worker,initargs=(g['cache']['path'],c['draws']['path'],case))as pool:
    pending={pool.submit(shard,job)for job in jobs}
    while pending:
     finished,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
     for f in finished:
      r=f.result();done.update(range(r['start'],r['stop']))
     av=available_gib();min_available=min(min_available,av);update(status='FORMAL_GATED_VECTOR_REPLAY',case=case,workers=workers,completed_replicates=len(done),total_replicates=2000,elapsed_seconds=time.perf_counter()-start,available_RAM_GiB=av,min_observed_available_GiB=min_available)
   stats=np.empty((2000,12),np.float64);valid=np.empty((2000,c['categories']),bool);mult=np.empty(valid.shape,np.uint32);unique=np.empty(2000,np.int32);seen=np.zeros(2000,bool)
   for sp in sorted((directory/'shards').glob('*.json')):
    r=read(sp);assert r['binding']==binding
    with np.load(check(r['arrays']),allow_pickle=False)as z:
     ids=z['bootstrap_id'];assert not seen[ids].any();seen[ids]=True;stats[ids]=z['stats'];valid[ids]=z['valid_category_mask'];mult[ids]=z['category_image_multiplicity'];unique[ids]=z['unique_original_images']
   assert seen.all()and np.isfinite(stats).all();frozen(True,case);check(g['cache']);check(g['point']);ap=directory/'all_replicates.npz';extra={}
   if c['cohort']=='DET':extra['fixed307_zero_allarea_AP_AP50_AP75']=stats[:,:3]*(valid.sum(axis=1)/307.)[:,None]
   npz(ap,bootstrap_id=np.arange(2000,dtype=np.int64),stats=stats,valid_category_mask=valid,category_image_multiplicity=mult,unique_original_images=unique,stat_names=np.asarray(NAMES),**extra)
   summary=dict(status='COMPLETE_ENDPOINT_B2000_GATED_REPLAY',case=case,method=c['method'],B=2000,source_point_stats=c['expected_point_stats'],intervals=intervals(stats),valid_GT_categories_range=[int(valid.sum(1).min()),int(valid.sum(1).max())],valid_GT_category_counts=valid.sum(1),full_sampled_image_occurrences=c['images'],unique_original_images_range=[int(unique.min()),int(unique.max())],engine='gated-vector-replay-not2000freshmatching',actual_native_matching_calls=4,gate_vector_replays=3,resource_probe_replays=1,workers=workers,threads_per_worker=1,min_observed_available_RAM_GiB=min_available,planned_reserve_GiB=64,formal_seconds=time.perf_counter()-start,paired_baseline_included=False)
   if c['cohort']=='DET':
    fixed=extra['fixed307_zero_allarea_AP_AP50_AP75'];summary['fixed307_zero_sensitivity']={name:dict(full_B2000_percentile=np.quantile(fixed[:,i],[.025,.975],method='linear'),first1000_percentile=np.quantile(fixed[:1000,i],[.025,.975],method='linear'))for i,name in enumerate(NAMES[:3])}
   sj=directory/'summary.json';fresh(sj,summary);r=dict(status='COMPLETE_ENDPOINT_B2000',case=case,B=2000,arrays=ref(ap),summary=ref(sj),binding=binding,gate=ref(gp),point=g['point'],resource_plan=ref(plan_path),all_unique_ids_0_1999=True,seconds=time.perf_counter()-start,engine='gated-vector-replay')
   fresh(completion,r);event('METHOD_ENDPOINT_COMPLETE',case=case,seconds=r['seconds'],B=2000);return r
  finally:fcntl.flock(global_lock,fcntl.LOCK_UN)
def run():
 P,pr,fr=frozen();OUT.mkdir(parents=True,exist_ok=True);(OUT/'automation').mkdir(parents=True,exist_ok=True);fd=(OUT/'automation/controller.lock').open('a');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(status='STARTING_FROZEN_CI',PID=os.getpid(),started_utc=now(),protocol=pr,completed_methods=[],B=2000,total_methods=len(P['cases']),CPU_affinity=sorted(os.sched_getaffinity(0)),active=None)
 def update(**kw):
  state.update(kw,updated_utc=now());save(OUT/'automation/status.json',state);event('CI_STATUS',**{k:state[k]for k in ['status','case','completed_replicates','workers','elapsed_seconds','completed_methods']if k in state})
 def child(case,stage):
  log=OUT/'logs'/f'{case}_{stage}_controller.log';argv=[sys.executable,'-B',str(Path(__file__).resolve()),stage,'--case',case]
  with log.open('a')as stream:
   proc=subprocess.Popen(argv,stdout=stream,stderr=subprocess.STDOUT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'),cwd=CODE.parents[2])
   event('CHILD_START',case=case,stage=stage,PID=proc.pid,log=str(log))
   while proc.poll()is None:time.sleep(2)
  if proc.returncode:raise RuntimeError(f'{case}/{stage} exit{proc.returncode}: {log}')
  event('CHILD_COMPLETE',case=case,stage=stage)
 def prepare(case):
  if not(OUT/'gates'/case/'gate.json').exists():child(case,'gate')
  if not(OUT/'resources'/case/'probe.json').exists():child(case,'probe')
  return case
 try:
  update(status='NATIVE_GATES_AND_RESOURCE_PROBES')
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   for case in pool.map(prepare,P['method_order']):
    event('GATE_AND_PROBE_READY',case=case)
  update(status='ALL18_GATES_READY')
  save(OUT/'automation/all_gates_ready.json',dict(status='ALL18_GATES_AND_PROBES_PASS',protocol=pr,code_freeze=fr,utc=now(),methods=P['method_order']))
  completed=[]
  for case in P['method_order']:
   quiet=Path(P['resources']['runtime_pause_request'])
   while quiet.exists()and read(quiet).get('requested',False):update(status='WAITING_RUNTIME_QUIET_WINDOW',case=case);time.sleep(10)
   result=formal(case,update);completed.append(result);state['completed_methods']=list(x['case']for x in completed);save(OUT/'run_index.json',dict(status='IN_PROGRESS',protocol=pr,code_freeze=fr,methods=completed))
   subprocess.run([sys.executable,'-B',str(CODE/'scripts/summarize_ci.py'),'--partial'],check=True,env=os.environ)
  index=dict(status='COMPLETE18_ENDPOINTS_B2000',completed_utc=now(),protocol=pr,code_freeze=fr,methods=completed);save(OUT/'run_index.json',index)
  subprocess.run([sys.executable,'-B',str(CODE/'scripts/summarize_ci.py')],check=True,env=os.environ)
  update(status='COMPLETE_PAIRED_SIX_TWO_COMPARISONS_B2000',completed_utc=now(),active=None,paired_summary=ref(OUT/'paired_summary.json'))
 except BaseException as e:update(status='FAILED',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['run','gate','probe']);parser.add_argument('--case');args=parser.parse_args()
 if args.stage=='run':run()
 elif args.stage=='gate':gate(args.case)
 else:probe(args.case)
