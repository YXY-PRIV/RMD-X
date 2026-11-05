#!/usr/bin/env python3
"""GPU0/2 batch-B screen, then frozen-recipe three-seed confirmation."""
import argparse,concurrent.futures,contextlib,fcntl,json,os,subprocess,sys,threading,time
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from robust_model import PROTOCOL_SHA,SOURCE,GPU_CHOICES,read_protocol,recipe_for,allowed_gpus,load_population,read,ref,sha,verify,write,now
CODE=Path(__file__).resolve().parents[1]
NIGHT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101')
OUT=NIGHT/'batch_b_robust_quality';VARIANTS=['visual','full_missing'];SEEDS=[17,29,43]
def gpu_jobs():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);rows=[]
 for line in raw.splitlines():
  x=[z.strip()for z in line.split(',')]
  if len(x)==2:rows.append(dict(gpu_uuid=x[0],PID=int(x[1])))
 return rows
@contextlib.contextmanager
def gpu_lease(P):
 area=NIGHT/'resources/gpu_slots';area.mkdir(parents=True,exist_ok=True);chosen=None;which=None;snapshot=None
 try:
  while chosen is None:
   for uuid in allowed_gpus(P):
    f=(area/(uuid+'.lock')).open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:f.close();continue
    snap=gpu_jobs()
    if any(r['gpu_uuid']==uuid for r in snap):fcntl.flock(f,fcntl.LOCK_UN);f.close();continue
    chosen,which,snapshot=f,uuid,snap;break
   if chosen is None:time.sleep(3)
  yield which,snapshot
 finally:
  if chosen is not None:fcntl.flock(chosen,fcntl.LOCK_UN);chosen.close()
def profiles_for(P):
 profiles=P['stress_profiles']
 if not profiles or len({x['id']for x in profiles})!=len(profiles):raise ValueError('Bad B screen profiles')
 for p in profiles:
  if not np.isfinite(p['input_conf_factor'])or p['input_conf_factor']<0 or not np.isfinite(p['entropy_shift']):raise ValueError('Bad B input stress')
 clean=[p for p in profiles if p['input_conf_factor']==1. and p['entropy_shift']==0.]
 if len(clean)!=1:raise ValueError('Exactly one clean B profile required')
 return profiles,clean[0]
def main():
 pa=argparse.ArgumentParser();pa.add_argument('--protocol',default=str(CODE/'config/protocol.json'));pa.add_argument('--resume',action='store_true');a=pa.parse_args();pp=Path(a.protocol).resolve();P=read_protocol(pp);profiles,clean=profiles_for(P);source=SOURCE/'training/candidate_train_val.npz';OUT.mkdir(parents=True,exist_ok=True)
 for d in ['automation','models','training','screen/inference','screen/logs','confirm/inference','confirm/ensembles','inference','ensembles','logs']:(OUT/d).mkdir(parents=True,exist_ok=True)
 owner=(OUT/'automation/model_worker.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB);mu=threading.RLock()
 sources={n:ref(CODE/n)for n in ['src/robust_model.py','scripts/train_robust.py','scripts/infer_robust.py','scripts/run_models.py']};freeze=OUT/'automation/model_code_freeze.json'
 if freeze.exists():
  if not a.resume or read(freeze)['sources']!=sources:raise ValueError('B source changed or explicit resume absent')
 else:write(freeze,dict(status='FROZEN_BEFORE_ANY_B_TRAINING_OR_INFERENCE',created_utc=now(),protocol=ref(pp),sources=sources))
 state=dict(status='PREPARING_SCREEN',PID=os.getpid(),protocol=ref(pp),started_utc=now(),active={},completed_trainings=0,completed_inferences=0,completed_ensembles=0)
 def update(**kw):
  with mu:state.update(kw,updated_utc=now());write(OUT/'automation/model_worker_status.json',state)
 def check_code():
  if sources!={n:ref(CODE/n)for n in sources}:raise ValueError('B frozen model source changed')
 def run(label,argv):
  check_code()
  with gpu_lease(P)as(uuid,snapshot):
   env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8');log=OUT/'logs'/(label+('_resume_'+str(os.getpid())if a.resume else '')+'.log')
   with log.open('x')as stream:
    p=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=CODE.parents[2]);launch=dict(PID=p.pid,worker_PID=os.getpid(),argv=argv,started_utc=now(),log=str(log),CUDA_VISIBLE_DEVICES=uuid,GPU_prelaunch_compute_snapshot=snapshot,authorized_GPU_was_compute_idle=not any(x['gpu_uuid']==uuid for x in snapshot),protocol=ref(pp),code_freeze=ref(freeze));write(OUT/'automation'/('launch_'+label+('_resume_'+str(os.getpid())if a.resume else '')+'.json'),launch)
    with mu:state['active'][label]=launch
    update();rc=p.wait()
   with mu:state['active'].pop(label,None)
   update()
   if rc:raise RuntimeError(f'{label} exit{rc}: {log}')
   return dict(launch,completed_utc=now(),exit_code=0)
 def batches(jobs,fn,consume):
  with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
   for f in concurrent.futures.as_completed([pool.submit(fn,*j)for j in jobs]):consume(f.result())
 def model(recipe_id,variant,seed,stage):
  d=OUT/'models'/recipe_id/variant/f'seed{seed}';mp=d/'metrics.json';argv=[sys.executable,'-B',str(CODE/'scripts/train_robust.py'),'--input',str(source),'--protocol',str(pp),'--recipe-id',recipe_id,'--variant',variant,'--seed',str(seed),'--output-dir',str(d),'--device','cuda:0']
  if mp.exists():
   if not a.resume:raise FileExistsError(mp)
   metric=read(mp);verify(metric['checkpoint']);job=dict(resume_existing_completed=True,actual_source=metric['environment'])
  else:
   if a.resume and(d/'last.pt').exists():argv+=['--resume',str(d/'last.pt')]
   job=run(f'train_{recipe_id}_{variant}_{seed}',argv);metric=read(mp)
  if metric['status']!='COMPLETE'or metric['recipe_id']!=recipe_id or metric['variant']!=variant or metric['seed']!=seed or not metric['actual_CUDA']or metric['environment']['CUDA_VISIBLE_DEVICES']not in allowed_gpus(P):raise ValueError('B training receipt identity mismatch')
  return dict(job,stage=stage,recipe_id=recipe_id,variant=variant,seed=seed,checkpoint=ref(d/'model.pt'),metrics=ref(mp),run_config=ref(d/'run_config.json'),normalizer=ref(d/'normalizer.json'),data_receipt=ref(d/'data_receipt.json'),history=ref(d/'train_history.json'),actual_CUDA=True)
 def infer(recipe_id,variant,seed,case,profile,stage):
  fmp=SOURCE/'features'/case/'feature_manifest.json';fm=read(fmp);cp=OUT/'models'/recipe_id/variant/f'seed{seed}'/'model.pt'
  d=(OUT/'screen/inference'/recipe_id/variant/profile['id']/f'seed{seed}')if stage=='screen'else(OUT/'confirm/inference'/case/variant/profile['id']/f'seed{seed}')
  rp=d/'inference.json';argv=[sys.executable,'-B',str(CODE/'scripts/infer_robust.py'),'--input',fm['outputs']['candidate_features.npy']['path'],'--feature-manifest',str(fmp),'--checkpoint',str(cp),'--protocol',str(pp),'--output-dir',str(d),'--device','cuda:0','--batch-size','8192','--profile',profile['id'],'--input-conf-factor',str(profile['input_conf_factor']),'--entropy-shift',str(profile['entropy_shift'])]
  if rp.exists():
   if not a.resume:raise FileExistsError(rp)
   r=read(rp);verify(r['output']);job=dict(resume_existing_completed=True)
  else:
   if a.resume and d.exists()and any(d.iterdir()):
    import shutil
    archive=OUT/'round_attempts'/f'partial_infer_{recipe_id}_{variant}_{seed}_{case}_{profile["id"]}_{os.getpid()}';archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(d,archive)
   job=run(f'infer_{stage}_{recipe_id}_{variant}_{seed}_{case}_{profile["id"]}',argv);r=read(rp)
  if r['status']!='COMPLETE_LABEL_FREE_B_INFERENCE'or r['protocol_sha256']!=PROTOCOL_SHA or(r['recipe_id'],r['variant'],r['seed'],r['case'],r['profile'])!=(recipe_id,variant,seed,case,profile['id'])or r['checkpoint']!=ref(cp)or r['feature_manifest']!=ref(fmp)or r['candidate_row_key_sha256']!=fm['candidate_row_key_sha256']or r['input_conf_factor']!=profile['input_conf_factor']or r['entropy_shift']!=profile['entropy_shift']or r['CUDA_VISIBLE_DEVICES']not in allowed_gpus(P):raise ValueError('B inference receipt identity mismatch')
  verify(r['output']);return dict(job,stage=stage,recipe_id=recipe_id,variant=variant,seed=seed,case=case,profile=profile['id'],input_conf_factor=profile['input_conf_factor'],entropy_shift=profile['entropy_shift'],scores=r['output'],inference_receipt=ref(rp),checkpoint=r['checkpoint'],feature_manifest=r['feature_manifest'],candidate_row_key_sha256=r['candidate_row_key_sha256'],shape=r['shape'],actual_CUDA=True)
 def ensemble(case,variant,profile,records,selection_ref):
  ordered=[next(r for r in records if r['seed']==s)for s in SEEDS];values=[]
  for r in ordered:
   verify(r['inference_receipt'])
   with np.load(verify(r['scores']),allow_pickle=False)as z:values.append(z['quality'].astype(np.float64))
  q=np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32);d=(OUT/'ensembles'/case/variant/'all3')if profile['id']==clean['id']else(OUT/'confirm/ensembles'/case/variant/profile['id']/'all3');d.mkdir(parents=True,exist_ok=True);sp=d/'quality_scores.npz';rp=d/'ensemble.json'
  if rp.exists():
   if not a.resume:raise FileExistsError(rp)
   old=read(rp);verify(old['scores'])
   with np.load(sp,allow_pickle=False)as z:
    if not np.array_equal(q,z['quality']):raise ValueError('B resumed ensemble arithmetic differs')
  else:
   if sp.exists():raise FileExistsError('Unsealed B ensemble '+str(sp))
   with sp.open('xb')as f:np.savez_compressed(f,quality=q)
   e=dict(status='COMPLETE_SAVED_B_ENSEMBLE',created_utc=now(),protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,case=case,profile=profile['id'],input_conf_factor=profile['input_conf_factor'],entropy_shift=profile['entropy_shift'],recipe_id=selected_id,variant=variant,seeds=SEEDS,inputs=ordered,input_count=3,feature_manifest=ordered[0]['feature_manifest'],candidate_row_key_sha256=ordered[0]['candidate_row_key_sha256'],shape=list(q.shape),dtype='float32',scores=ref(sp),output=ref(sp),output_sha256=sha(sp),selection=selection_ref,generator=ref(__file__),GT_opened=False,AP_computed=False,recipe_selected_by_model_worker=False,rule='F64 arithmetic mean of all3 seed qualities then oneF32 cast');write(rp,e)
  r=dict(case=case,profile=profile['id'],recipe_id=selected_id,variant=variant,seeds=SEEDS,scores=ref(sp),inference_receipt=ref(rp),candidate_row_key_sha256=ordered[0]['candidate_row_key_sha256'],shape=list(q.shape))
  if profile['id']==clean['id']:write(OUT/'ensembles'/case/variant/'status.json',dict(status='COMPLETE',protocol=ref(pp),case=case,variant=variant,ensemble=r))
  return r
 try:
  update(status='SCREEN_TRAINING');recipes=P['recipes'];recipes=list(recipes.values())if isinstance(recipes,dict)else recipes;ids=[str(r.get('id',r.get('recipe_id')))for r in recipes]
  if len(ids)!=5:raise ValueError('BatchB requires5 paired recipes')
  for rid in ids:recipe_for(P,rid)
  contracts=[]
  for rid in ids:
   for variant in VARIANTS:
    data,norm,info=load_population(source,P,recipe_for(P,rid),variant)
    if info['folds']['val']['proposals']!=264 or info['folds']['val']['effective_images']!=37:raise ValueError('Common B validation scope differs')
    contracts.append(dict(recipe_id=rid,variant=variant,normalizer=norm,data=info));del data
  write(OUT/'training/input_contracts.json',dict(status='PASS_FROZEN_SOURCE_POPULATION_NORMALIZATION_PREFLIGHT',created_utc=now(),protocol=ref(pp),code_freeze=ref(freeze),records=contracts,GT_annotations_opened=False,GPU_started=False))
  train_index=dict(status='RUNNING_SCREEN',protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,runs=[])
  def collect_train(r):train_index['runs'].append(r);write(OUT/'training/summary.json',train_index);update(completed_trainings=len(train_index['runs']))
  batches([(rid,v,17,'screen')for rid in ids for v in VARIANTS],model,collect_train)
  screen=dict(status='IN_PROGRESS',protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,profiles=profiles,expected_records=10*len(profiles),records=[])
  update(status='SCREEN_TUNE_INFERENCE')
  def collect_screen(r):screen['records'].append(r);write(OUT/'screen/inference_index.json',screen);update(completed_inferences=len(screen['records']))
  batches([(rid,v,17,'CLC_tune',p,'screen')for rid in ids for v in VARIANTS for p in profiles],infer,collect_screen)
  screen.update(status='COMPLETE',completed_utc=now());write(OUT/'screen/inference_index.json',screen);train_index['screen_complete_utc']=now();write(OUT/'training/summary.json',train_index);update(status='WAITING_FROZEN_RECIPE_SELECTION')
  selpath=OUT/'parameters/selected_recipe.json';deadline=time.monotonic()+7200
  while True:
   if selpath.exists():
    try:sel=read(selpath)
    except json.JSONDecodeError:sel={}
    if sel.get('status','').startswith(('FROZEN','COMPLETE')):break
   if time.monotonic()>deadline:raise TimeoutError('Root selected B recipe')
   time.sleep(2)
  selected_id=str(sel['selected_recipe_id']);recipe_for(P,selected_id);selection_ref=ref(selpath);update(status='CONFIRM_TRAINING',selected_recipe_id=selected_id);train_index['selection']=selection_ref
  batches([(selected_id,v,s,'confirm')for v in VARIANTS for s in [29,43]],model,collect_train)
  train_index.update(status='COMPLETE',completed_utc=now());write(OUT/'training/summary.json',train_index)
  confirmation=dict(status='IN_PROGRESS',protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,selection=selection_ref,records=[],exact_reused_screen_tune=[])
  all_inferences=list(screen['records']);groups={};update(status='CONFIRM_TUNE_AND_TEST_INFERENCE')
  for v in VARIANTS:
   for p in profiles:
    r=next(r for r in screen['records']if(r['recipe_id'],r['variant'],r['seed'],r['profile'])==(selected_id,v,17,p['id']));confirmation['exact_reused_screen_tune'].append(r);groups['CLC_tune',v,p['id']]=[r]
  jobs=[(selected_id,v,s,'CLC_tune',p,'confirm')for v in VARIANTS for s in [29,43]for p in profiles]+[(selected_id,v,s,case,clean,'confirm')for case in P['test_cases']for v in VARIANTS for s in SEEDS]
  ensemble_index=dict(status='IN_PROGRESS',protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,selection=selection_ref,records=[]);profile_ensembles=[]
  def collect_confirm(r):
   if ref(selpath)!=selection_ref:raise ValueError('Frozen B recipe selection changed')
   confirmation['records'].append(r);all_inferences.append(r);key=(r['case'],r['variant'],r['profile']);groups.setdefault(key,[]).append(r);write(OUT/'confirm/inference_index.json',confirmation)
   if len(groups[key])==3:
    p=next(p for p in profiles if p['id']==r['profile']);e=ensemble(r['case'],r['variant'],p,groups[key],selection_ref);profile_ensembles.append(e)
    if p['id']==clean['id']:ensemble_index['records'].append(e);write(OUT/'ensemble_index.json',ensemble_index)
    write(OUT/'confirm/profile_ensemble_index.json',dict(status='IN_PROGRESS',protocol=ref(pp),selection=selection_ref,records=profile_ensembles))
   update(completed_inferences=len(all_inferences),completed_ensembles=len(profile_ensembles))
  batches(jobs,infer,collect_confirm)
  if len(train_index['runs'])!=14 or len(all_inferences)!=14*len(profiles)+36 or len(ensemble_index['records'])!=14 or len(profile_ensembles)!=12+2*len(profiles):raise ValueError('B training/inference/ensemble totals incomplete')
  confirmation.update(status='COMPLETE',completed_utc=now());write(OUT/'confirm/inference_index.json',confirmation);ensemble_index.update(status='COMPLETE',completed_utc=now());write(OUT/'ensemble_index.json',ensemble_index);write(OUT/'confirm/profile_ensemble_index.json',dict(status='COMPLETE',protocol=ref(pp),selection=selection_ref,records=profile_ensembles));write(OUT/'inference_summary.json',dict(status='COMPLETE',protocol=ref(pp),protocol_sha256=PROTOCOL_SHA,jobs=all_inferences,new_actual_GPU_inferences=len(all_inferences),exact_reused_screen_tune=len(confirmation['exact_reused_screen_tune']),screen_inference_index=ref(OUT/'screen/inference_index.json'),confirmation_inference_index=ref(OUT/'confirm/inference_index.json')));check_code();update(status='COMPLETE',active={},completed_utc=now(),new_trainings=14,new_actual_GPU_inferences=len(all_inferences),exact_reused_screen_tune=len(confirmation['exact_reused_screen_tune']),main_ensembles=14,all_ensembles=len(profile_ensembles))
  print(json.dumps(dict(status='COMPLETE',new_CUDA_trainings=14,new_actual_GPU_inferences=len(all_inferences),exact_reused_screen_tune=len(confirmation['exact_reused_screen_tune']),new_ensembles=len(profile_ensembles))),flush=True)
 except BaseException as e:update(status='FAILED',error=repr(e),completion_claimed=False);raise
if __name__=='__main__':main()
