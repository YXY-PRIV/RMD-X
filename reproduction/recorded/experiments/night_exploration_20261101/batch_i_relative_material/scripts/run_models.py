#!/usr/bin/env python3
"""Only physical GPU1,6 fresh relative-feature models,42 inference jobs,14 means."""
import argparse,fcntl,json,os,subprocess,sys,time,ast
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.relative_model import CODE,OUT,PSHA,CSHA,GPU,VARIANTS,read,sha,ref,verify,atomic,now,protocol,dataset,unit

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',default=str(CODE/'config/protocol.json'));a=ap.parse_args();pp=Path(a.protocol).resolve();P=protocol(pp);pr=ref(pp)
 for name in ['automation','models','training','inference','ensembles','logs','verification']:(OUT/name).mkdir(parents=True,exist_ok=True)
 lock=(OUT/'automation/model_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 paths=['src/relative_model.py','scripts/train_relative.py','scripts/infer_relative.py','scripts/run_models.py'];sources={n:ref(CODE/n)for n in paths}
 for name in paths:ast.parse((CODE/name).read_text())
 fp=OUT/'automation/model_code_freeze.json'
 if fp.exists():raise FileExistsError('Preserve previous I model worker attempt; no implicit repair')
 atomic(fp,dict(status='FROZEN_BEFORE_FIRST_I_NUMERICAL_RUN',created_utc=now(),protocol=pr,sources=sources,scope='Exactly four model-owned files; independent root endpoint/controller files excluded'))
 state=dict(status='FROZEN_INPUT_PREFLIGHT',PID=os.getpid(),started_utc=now(),protocol=pr,active=None,completed_trainings=0,completed_inferences=0,completed_ensembles=0)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_worker_status.json',state)
 def free_gpu1():
  text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);other=[x.strip()for x in text.splitlines()if x.split(',')[0].strip()==GPU]
  if other:raise RuntimeError('GPU1 has a compute process; refusing to touch it: '+str(other))
  return dict(checked_utc=now(),GPU_UUID=GPU,compute_processes_on_GPU1=[],other_GPU_processes_untouched=True)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def run(name,argv):
  if sources!={n:ref(CODE/n)for n in paths}:raise ValueError('Frozen I model source changed')
  availability=free_gpu1();log=OUT/'logs'/(name+'.log')
  with log.open('x')as stream:
   proc=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=Path('/mnt/data/yxy/program/RAXO'));launch=dict(PID=proc.pid,worker_PID=os.getpid(),argv=argv,started_utc=now(),log=str(log),CUDA_VISIBLE_DEVICES=GPU,protocol=pr,GPU_preflight=availability);atomic(OUT/'automation'/('launch_'+name+'.json'),launch);update(active=launch);rc=proc.wait()
  update(active=None)
  if rc:raise RuntimeError(f'{name} exit{rc}: {log}')
  return dict(launch,exit_code=0,completed_utc=now())
 try:
  update();inputpath=verify(P['sources']['C_training_npz']);d,names,meta=dataset(inputpath,P)
  # Numeric read-only input contract after source freeze; no model/GPU/GT/AP.
  pre=[]
  for v in VARIANTS:
   raw=d['train']['X'];before=raw.copy();const=np.flatnonzero(np.ptp(unit(raw,v),axis=0)==0).tolist();natural=unit(raw,v,const)
   inactive=[0,3,8,9,11,12,13]+([10]if v=='agreement_only'else[])
   if not np.array_equal(raw,before)or np.any(natural[:,inactive]!=0):raise ValueError('Raw mutation or inactive coordinate leak')
   absent=raw[:8].copy();absent[:,13]=0.;av=unit(absent,v,const)
   if np.any(av[:,9:]):raise ValueError('Absent material invented')
   pre.append(dict(variant=v,unique_train=len(raw),unique_val=len(d['val']['X']),constant_coordinates=const,inactive=inactive,active_material_coordinates=[10,14]if v=='rank_agreement'else[14],raw_input_unchanged=True,actual_missing_not_fabricated=True))
  atomic(OUT/'verification/input_contracts.json',dict(status='PASS_FROZEN_I_RELATIVE_INPUT_CONTRACTS',created_utc=now(),protocol=pr,input=ref(inputpath),variants=pre,GT_annotations_read=False,AP_or_GPU_run=False))
  ts=dict(status='RUNNING',protocol=pr,input=ref(inputpath),runs=[],new_GT_annotations_read=False,threshold_selected=False)
  for v in VARIANTS:
   for seed in [17,29,43]:
    update(status='RUNNING_ACTUAL_CUDA_TRAINING');md=OUT/'models'/v/f'seed{seed}';job=run(f'train_{v}_{seed}',[sys.executable,'-B',str(CODE/'scripts/train_relative.py'),'--input',str(inputpath),'--protocol',str(pp),'--variant',v,'--seed',str(seed),'--output-dir',str(md),'--device','cuda:0']);m=read(md/'metrics.json');c=read(md/'run_config.json')['config']
    if m['status']!='COMPLETE'or not m['actual_CUDA']or m['environment']['CUDA_VISIBLE_DEVICES']!=GPU or c['input']!=ref(inputpath)or c['protocol']!=pr or(c['variant'],c['seed'])!=(v,seed):raise ValueError('Fresh training identity differs')
    ts['runs'].append(dict(job,variant=v,seed=seed,checkpoint=ref(md/'model.pt'),metrics=ref(md/'metrics.json'),run_config=ref(md/'run_config.json'),history=ref(md/'train_history.json'),normalization=ref(md/'normalization.json'),actual_CUDA=True));atomic(OUT/'training/summary.json',ts);update(completed_trainings=len(ts['runs']))
  ts.update(status='COMPLETE',completed_utc=now());atomic(OUT/'training/summary.json',ts);ins=dict(status='RUNNING',protocol=pr,jobs=[],expected_jobs=42,training_summary=ref(OUT/'training/summary.json'),GT_annotations_read=False,AP_computed=False);ens=dict(status='IN_PROGRESS',protocol=pr,records=[],expected_records=14)
  for case in P['inference_cases']:
   fmp=verify(P['cases'][case]['feature_manifest']);fm=read(fmp);valid=np.load(verify(fm['outputs']['candidate_valid.npy']),allow_pickle=False)
   if fm['protocol_sha256']!=CSHA or fm['candidate_row_key_sha256']!=P['cases'][case]['candidate_row_key_sha256']:raise ValueError('Frozen source C features changed')
   for v in VARIANTS:
    values=[];inputs=[]
    for seed in [17,29,43]:
     update(status='RUNNING_ACTUAL_CUDA_INFERENCE');rd=OUT/'inference'/case/v/f'seed{seed}';cp=OUT/'models'/v/f'seed{seed}'/'model.pt';job=run(f'infer_{case}_{v}_{seed}',[sys.executable,'-B',str(CODE/'scripts/infer_relative.py'),'--case',case,'--feature-manifest',str(fmp),'--checkpoint',str(cp),'--protocol',str(pp),'--output-dir',str(rd),'--device','cuda:0']);rp=rd/'inference.json';r=read(rp)
     if r['protocol_sha256']!=PSHA or(r['case'],r['variant'],r['seed'])!=(case,v,seed)or r['checkpoint']!=ref(cp)or r['feature_manifest']!=ref(fmp)or r['candidate_row_key_sha256']!=fm['candidate_row_key_sha256']or r['CUDA_VISIBLE_DEVICES']!=GPU:raise ValueError('Inference identity mismatch')
     with np.load(verify(r['output']),allow_pickle=False)as z:q=z['quality'].copy()
     if q.dtype!=np.float32 or q.shape!=valid.shape or np.any(q[~valid]!=0)or not np.isfinite(q).all():raise ValueError('Invalid saved quality')
     ins['jobs'].append(dict(job,case=case,variant=v,seed=seed,receipt=ref(rp),scores=r['output'],actual_CUDA=True));atomic(OUT/'inference/summary.json',ins);values.append(q.astype(np.float64));inputs.append(dict(seed=seed,scores=r['output'],inference_receipt=ref(rp),checkpoint=r['checkpoint']));update(completed_inferences=len(ins['jobs']))
    q=np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32);ed=OUT/'ensembles'/case/v/'all3';ed.mkdir(parents=True,exist_ok=True);sp=ed/'quality_scores.npz';rp=ed/'ensemble.json'
    with sp.open('xb')as f:np.savez_compressed(f,quality=q)
    receipt=dict(status='COMPLETE_SAVED_SHARED_UNION_ENSEMBLE',created_utc=now(),protocol=pr,protocol_sha256=PSHA,source_C_protocol_sha256=CSHA,case=case,variant=v,seeds=[17,29,43],inputs=inputs,feature_manifest=ref(fmp),candidate_row_key_sha256=fm['candidate_row_key_sha256'],shape=list(q.shape),dtype='float32',scores=ref(sp),output_sha256=sha(sp),invalid_quality_exact0=True,rule='F64 arithmetic mean of all3 seeds followed by one F32 cast',GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False);atomic(rp,receipt);record=dict(case=case,variant=v,group='all3',scores=ref(sp),inference_receipt=ref(rp),candidate_row_key_sha256=fm['candidate_row_key_sha256']);ens['records'].append(record);atomic(OUT/'ensemble_index.json',ens);atomic(OUT/'ensembles'/case/v/'status.json',dict(status='COMPLETE',case=case,variant=v,protocol=pr,ensemble=record));update(completed_ensembles=len(ens['records']))
  if len(ins['jobs'])!=42 or len(ens['records'])!=14:raise ValueError('Expected6train42infer14mean')
  ins.update(status='COMPLETE',completed_utc=now());ens.update(status='COMPLETE',completed_utc=now());atomic(OUT/'inference/summary.json',ins);atomic(OUT/'ensemble_index.json',ens);update(status='COMPLETE',active=None,completed_utc=now());print(json.dumps(dict(status='COMPLETE',trainings=6,inferences=42,ensembles=14)),flush=True)
 except BaseException as e:update(status='FAILED',error=repr(e),completion_claimed=False);raise
if __name__=='__main__':main()
