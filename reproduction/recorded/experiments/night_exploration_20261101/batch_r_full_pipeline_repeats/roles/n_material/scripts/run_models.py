#!/usr/bin/env python3
"""Twelve material-role replications on GPU2; four prescribed three-seed groups."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,ast,fcntl,json,subprocess,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model import CODE,OUT,PSHA,GPU,read,ref,verify,atomic,now,protocol,dataset,unit
def main():
 a=argparse.ArgumentParser();a.add_argument('--protocol',default=str(CODE/'config/protocol.json'));a.add_argument('--freeze-only',action='store_true');args=a.parse_args();pp=Path(args.protocol).resolve();P=protocol(pp);pr=ref(pp)
 for n in ['automation','training','models','inference','ensembles','logs','verification']:(OUT/n).mkdir(parents=True,exist_ok=True)
 lock=(OUT/'automation/model_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 names=['src/quality_model.py','scripts/train_quality.py','scripts/infer_quality.py','scripts/run_models.py'];sources={n:ref(CODE/n)for n in names};generator=ref(CODE/'scripts/freeze_protocol.py')
 for n in names:ast.parse((CODE/n).read_text())
 freeze=OUT/'automation/model_code_freeze.json'
 if freeze.exists():
  old=read(freeze);assert old['sources']==sources and old['protocol']==pr and old['parent_protocol']==P['parent_protocol']and old['protocol_generator']==generator and not(OUT/'automation/model_worker_status.json').exists(),'Preserve previous numerical R role attempt; never silently retry'
 else:atomic(freeze,dict(status='FROZEN_BEFORE_FIRST_R_N_MATERIAL_NUMERICAL_RUN',created_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],role=P['role'],sources=sources,protocol_generator=generator))
 if args.freeze_only:print(json.dumps(dict(status='FROZEN_NO_NUMERICAL_RUN',model_code_freeze=ref(freeze),protocol=pr)),flush=True);return
 state=dict(status='FROZEN_INPUT_PREFLIGHT',PID=os.getpid(),started_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],active=None,completed_trainings=0,completed_inferences=0,completed_ensembles=0)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_worker_status.json',state)
 def apps():
  text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
  return [dict(GPU_UUID=r.split(',')[0].strip(),PID=int(r.split(',')[1].strip()))for r in text.splitlines()if r.strip()]
 def available():return int(next(s.split()[1]for s in Path('/proc/meminfo').read_text().splitlines()if s.startswith('MemAvailable:')))/1024**2
 def gpu_empty():
  while True:
   used=[r for r in apps()if r['GPU_UUID']==GPU];ram=available()
   if not used and ram>=64:return dict(checked_utc=now(),GPU_UUID=GPU,compute_processes_on_GPU2=[],available_RAM_GiB=ram,minimum_RAM_GiB=64,other_GPUs_untouched=True)
   update(status='WAITING_GPU2_EMPTY_AND_RESERVED_RAM',GPU2_processes=used,available_RAM_GiB=ram);time.sleep(2)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def run(name,argv):
  assert sources=={n:ref(CODE/n)for n in names}and generator==ref(CODE/'scripts/freeze_protocol.py'),'R frozen source changed'
  preflight=gpu_empty();log=OUT/'logs'/(name+'.log');launchpath=OUT/'automation'/('launch_'+name+'.json');registration=None
  with log.open('x')as stream:
   child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd='/mnt/data/yxy/program/RAXO');launch=dict(PID=child.pid,worker_PID=os.getpid(),argv=argv,started_utc=now(),log=str(log),protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],CUDA_VISIBLE_DEVICES=GPU,GPU_preflight=preflight);atomic(launchpath,launch);update(active=launch)
   try:
    while child.poll()is None:
     records=[r for r in apps()if r['PID']==child.pid]
     if records:
      assert records==[dict(GPU_UUID=GPU,PID=child.pid)],'Child CUDA PID registered on unexpected GPU'
      registration=dict(status='PASS_ACTUAL_CHILD_PID_REGISTERED_ON_ASSIGNED_GPU2',observed_utc=now(),GPU_UUID=GPU,PID=child.pid,records=records);break
     time.sleep(.25)
    rc=child.wait()
   except BaseException:
    if child.poll()is None:child.terminate();child.wait()
    raise
  assert rc==0,f'{name} failed exit{rc}: {log}'
  assert registration is not None,'No actual GPU PID registration observed; preserve attempt'
  launch.update(exit_code=0,completed_utc=now(),GPU_registration=registration);atomic(launchpath,launch);update(active=None);return launch
 try:
  update(status='VERIFYING_ORIGINAL_N_FROZEN_INPUTS');inputpath=verify(P['source_N_training_npz']);data,names15,meta,receipt=dataset(inputpath,P)
  x=data['train']['X'];active=P['feature_schema']['active_coordinates']['material'];const=np.flatnonzero(np.ptp(unit(x,active),axis=0)==0).tolist()
  atomic(OUT/'verification/input_contracts.json',dict(status='PASS_ORIGINAL_N_DATA_NO_REBUILD',created_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],source_N_protocol=P['source_N_protocol'],training_NPZ=ref(inputpath),source_N_data_receipt=P['source_N_data_receipt'],train_rows=len(x),val_rows=len(data['val']['X']),source_statistics={r:data[r]['statistics']for r in ['train','val']},active_coordinates=active,unique_train_constant_coordinates=const,new_GT_or_features=False))
  training=dict(status='RUNNING',protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],runs=[],expected_runs=12)
  for seed in P['seeds']:
   update(status='RUNNING_ACTUAL_CUDA_TRAINING');md=OUT/'models/material'/f'seed{seed}';job=run(f'train_material_{seed}',[sys.executable,'-B',str(CODE/'scripts/train_quality.py'),'--input',str(inputpath),'--protocol',str(pp),'--variant','material','--seed',str(seed),'--output-dir',str(md),'--device','cuda:0']);m=read(md/'metrics.json');c=read(md/'run_config.json')['config']
   assert m['status']=='COMPLETE'and m['actual_CUDA']and m['environment']['CUDA_VISIBLE_DEVICES']==GPU and c['protocol']==pr and c['parent_protocol']==P['parent_protocol']and c['seed']==seed and c['input']==ref(inputpath)
   training['runs'].append(dict(job,variant='material',seed=seed,checkpoint=ref(md/'model.pt'),metrics=ref(md/'metrics.json'),run_config=ref(md/'run_config.json'),history=ref(md/'train_history.json'),normalization=ref(md/'normalization.json'),actual_CUDA=True));atomic(OUT/'training/summary.json',training);update(completed_trainings=len(training['runs']))
  training.update(status='COMPLETE',completed_utc=now());atomic(OUT/'training/summary.json',training)
  inference=dict(status='RUNNING',protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],expected_jobs=72,jobs=[]);index=dict(status='IN_PROGRESS',protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],groups={g['id']:dict(seeds=g['seeds'],cases={})for g in P['groups']})
  for group in P['groups']:
   gid=group['id'];seeds=group['seeds']
   for case in P['inference_cases']:
    fmref=P['cases'][case]['source_N_feature_manifest'];fp=verify(fmref);fm=read(fp);valid=np.load(verify(fm['outputs']['candidate_valid.npy']),allow_pickle=False);values=[];inputs=[];models=[]
    for seed in seeds:
     update(status='RUNNING_ACTUAL_CUDA_INFERENCE',group=gid,case=case);cp=OUT/'models/material'/f'seed{seed}'/'model.pt';rd=OUT/'inference'/case/'material'/f'seed{seed}';job=run(f'infer_{case}_material_{seed}',[sys.executable,'-B',str(CODE/'scripts/infer_quality.py'),'--case',case,'--feature-manifest',str(fp),'--checkpoint',str(cp),'--protocol',str(pp),'--output-dir',str(rd),'--device','cuda:0']);rp=rd/'inference.json';r=read(rp)
     assert r['protocol']==pr and r['parent_protocol']==P['parent_protocol']and r['role']==P['role']and r['case']==case and r['seed']==seed and r['feature_manifest']==fmref and r['checkpoint']==ref(cp)and r['candidate_row_key_sha256']==fm['candidate_row_key_sha256']and r['CUDA_VISIBLE_DEVICES']==GPU
     with np.load(verify(r['output']),allow_pickle=False)as z:q=z['quality'].copy()
     assert q.dtype==np.float32 and q.shape==valid.shape and np.isfinite(q).all()and not np.any(q[~valid])
     inference['jobs'].append(dict(job,group=gid,case=case,variant='material',seed=seed,receipt=ref(rp),scores=r['output'],actual_CUDA=True));atomic(OUT/'inference/summary.json',inference);values.append(q.astype(np.float64));models.append(ref(cp));inputs.append(dict(seed=seed,scores=r['output'],inference_receipt=ref(rp),checkpoint=ref(cp)));update(completed_inferences=len(inference['jobs']))
    q=np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32);ed=OUT/'ensembles'/gid/case/'all3';ed.mkdir(parents=True,exist_ok=True);sp=ed/'quality_scores.npz';rp=ed/'ensemble.json'
    with sp.open('xb')as stream:np.savez_compressed(stream,quality=q)
    r=dict(status='COMPLETE_FIXED_THREE_SEED_GROUP',created_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],role=P['role'],group=gid,seeds=seeds,case=case,variant='material',source_N_protocol=P['source_N_protocol'],feature_manifest=fmref,candidate_row_key_sha256=fm['candidate_row_key_sha256'],inputs=inputs,model_refs=models,shape=list(q.shape),dtype='float32',scores=ref(sp),rule='Own prescribed3 arithmetic F64 mean followed by one F32 cast; no mean12 or bestgroup',invalid_quality_exact0=True,AP_computed=False,new_GT_features=False);atomic(rp,r)
    index['groups'][gid]['cases'][case]=dict(scores=ref(sp),receipt=ref(rp),model_refs=models,feature_manifest=fmref,candidate_row_key_sha256=fm['candidate_row_key_sha256']);atomic(OUT/'ensemble_index.json',index);update(completed_ensembles=sum(len(g['cases'])for g in index['groups'].values()))
  assert len(training['runs'])==12 and len(inference['jobs'])==72 and sum(len(g['cases'])for g in index['groups'].values())==24
  inference.update(status='COMPLETE',completed_utc=now());index.update(status='COMPLETE',completed_utc=now());atomic(OUT/'inference/summary.json',inference);atomic(OUT/'ensemble_index.json',index);completion=dict(status='COMPLETE_ROLE_READY_FOR_INDEPENDENT_REVIEW',completed_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],role=P['role'],model_code_freeze=ref(freeze),training=ref(OUT/'training/summary.json'),inference=ref(OUT/'inference/summary.json'),ensemble_index=ref(OUT/'ensemble_index.json'),new_trainings=12,new_GPU_inferences=72,new_three_seed_groups=24,new_AP_GT_features=0,independent_acceptance=False);atomic(OUT/'actor_completion.json',completion);update(status='COMPLETE',active=None,completed_utc=now());print(json.dumps(completion),flush=True)
 except BaseException as e:update(status='FAILED',error=repr(e),completion_claimed=False);raise
if __name__=='__main__':main()
