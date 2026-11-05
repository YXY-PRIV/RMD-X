#!/usr/bin/env python3
"""Finish24 incomplete forwards after one diagnostic-only guard failure.

Six trained models and18 completed forwards are immutable. The failed original
controller stays an exit1 event; this controller never trains or repeats a
completed logical job. All43 actual inference attempts are disclosed.
"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model_continued import CODE,OUT,VARIANTS,SEEDS,now,ref,read,verify,atomic,protocol,actor_gate,continuation_gate,prepare_case


def new(path,value):
 p=Path(path).resolve();assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def stamp(path):
 s=Path(path).stat();return(s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)

def ram():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/2**20
 raise AssertionError('MemAvailable absent')

def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(PID=pid,starttime_ticks=int(s[19]),argv=[v.decode()for v in(p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])


def main():
 pp=CODE/'config/protocol.json';P,pr=protocol(pp,os.environ['RAXO_PROTOCOL_SHA256']);actor_gate(pr)
 am,amref,qaref=continuation_gate(pr)
 lock=(OUT/'automation/model_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 retained=read(verify(am['retained_inference_index']));trained=read(verify(am['retained_training_complete']))
 assert retained['protocol']==pr and len(retained['records'])==18
 assert trained['status']=='COMPLETE_SIX_CONDITIONAL_CUDA_TRAININGS'and trained['protocol']==pr and trained['new_trainings']==6
 assert len(trained['runs'])==6 and all(r['job']['exit_code']==0 for r in trained['runs'])
 current=ref(OUT/'inference_index.json')
 assert all(current[k]==am['retained_inference_index'][k]for k in ['bytes','sha256']), 'Operational index must still equal the retained pre-continuation snapshot'
 failure_ref=ref(OUT/'automation/model_failure.json');failure=read(failure_ref);assert failure['status']=='FAILED'
 original_exit_ref=ref(OUT/'automation/models_service_exit.json');original_exit=read(original_exit_ref)
 assert original_exit['observed']['EXIT_CODE']=='exited'and original_exit['observed']['EXIT_STATUS']=='1'
 assert original_exit['observed']['SERVICE_RESULT']=='exit-code'
 immutable=[amref,qaref,am['retained_inference_index'],am['retained_training_complete'],am['partial_archive_manifest'],failure_ref,original_exit_ref,
  ref(OUT/'automation/model_code_freeze.json')]
 for row in trained['runs']:
  immutable.extend([row['checkpoint'],row['metrics']]);assert(row['variant'],row['seed'])in{(v,s)for v in VARIANTS for s in SEEDS}
 records=list(retained['records']);prior_keys={(r['case'],r['variant'],r['seed'])for r in records}
 assert len(prior_keys)==18
 for row in records:
  r=read(verify(row['receipt']));assert r['protocol']==pr and r['status']=='COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE'
  assert r['output']==row['components']and row['job']['exit_code']==0
  immutable.extend([row['receipt'],row['components'],row['full_raw_correction'],row['full_centered_correction'],row['permutation_index'],r['checkpoint'],r['feature_manifest'],r['full_anchor_logits']])
 for value in immutable:verify(value)
 stamps={r['path']:stamp(r['path'])for r in immutable}
 jobs=[(case,v,s)for case in P['inference_cases']for v in VARIANTS for s in SEEDS if(case,v,s)not in prior_keys]
 assert len(jobs)==24 and len({*prior_keys,*jobs})==42
 provenance=dict(verification_amendment=amref,independent_continuation_preflight=qaref,retained_inference_index=am['retained_inference_index'],
  retained_training_complete=am['retained_training_complete'],partial_archive_manifest=am['partial_archive_manifest'],
  original_model_failure=failure_ref,original_model_service_exit=original_exit_ref,original_controller_exit_code=1,
  prior_completed_GPU_inferences=18,prior_partial_failed_GPU_inference_attempts=1,new_continued_GPU_inferences_expected=24,
  actual_total_GPU_inference_attempts_expected=43,scientific_completed_inferences_expected=42,rerun_completed_inferences=0,new_trainings=0)
 state=dict(status='CONTINUING_DIAGNOSTIC_ONLY_GUARD',protocol=pr,PID=os.getpid(),started_utc=now(),active=None,
  completed_trainings=6,completed_new_GPU_inferences=18,completed_continued_GPU_inferences=0,continuation=provenance)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_status.json',state)
 new(OUT/'automation/model_continuation_started.json',dict(**state,argv=sys.argv,controller_source=ref(__file__)))
 update();continued=[];feature_refs={}
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=P['resources']['GPU_UUID'],CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',
  OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def guard():
  actor_gate(pr);assert continuation_gate(pr)[1:]==(amref,qaref)
  assert all(stamp(path)==st for path,st in stamps.items())
  assert ram()>=P['resources']['min_MemAvailable_GiB']
 for case,variant,seed in jobs:
  guard()
  fm_path=OUT/'features'/case/'feature_manifest.json'
  if case not in feature_refs:
   if fm_path.exists():
    fm=read(fm_path);assert fm['status']=='COMPLETE_CONDITIONAL_FULL_C_FEATURES'and fm['protocol']==pr
    assert fm['source_auxiliary_feature_manifest']==P['cases'][case]['auxiliary_feature_manifest']
    assert fm['source_C_feature_manifest']==P['cases'][case]['C_feature_manifest']
    assert read(verify(fm['gather_proof']))['all13_candidate_feature_gather_bits_exact']is True
    for value in fm['outputs'].values():verify(value)
   else:prepare_case(P,pr,case)
   feature_refs[case]=ref(fm_path)
  directory=OUT/'inference'/case/variant/f'seed{seed}'
  assert not directory.exists()or not any(directory.iterdir()),'Root must archive an incomplete partial; controller cannot overwrite it'
  while True:
   output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
   busy=[s for s in output.splitlines()if s.split(',')[0].strip()==P['resources']['GPU_UUID']]
   if not busy:break
   update(status='WAITING_RESERVED_GPU_FREE',busy=busy);time.sleep(10)
  name=f'continue_infer_{case}_{variant}_{seed}'
  argv=[sys.executable,'-B',str(CODE/'scripts/infer_continued.py'),'--variant',variant,'--case',case,
   '--checkpoint',str(OUT/'models'/variant/f'seed{seed}'/'model.pt'),'--output-dir',str(directory),
   '--protocol',str(pp),'--protocol-sha256',pr['sha256'],'--device','cuda:0']
  log=OUT/'logs'/(name+'.log')
  with log.open('x')as stream:
   child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT)
   launch=dict(identity=identity(child.pid),argv=argv,log=str(log),protocol=pr,verification_amendment=amref,started_utc=now(),
    GPU_UUID=P['resources']['GPU_UUID'],available_RAM_GiB=ram())
   new(OUT/'automation'/('launch_'+name+'.json'),launch);update(status='ACTUAL_CONTINUED_LABEL_FREE_CUDA_INFERENCE',active=launch)
   code=child.wait()
  assert code==0,(name,code,str(log));update(active=None)
  job=dict(**launch,exit_code=0,exited_utc=now())
  r=read(directory/'inference.json');assert r['status']=='COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE'and r['protocol']==pr
  assert(r['case'],r['variant'],r['seed'])==(case,variant,seed)and r['verification_amendment']==amref
  assert r['continued_model_preflight_review']==qaref and r['continued_inference_after_diagnostic_guard_only']is True
  record=dict(case=case,variant=variant,seed=seed,receipt=ref(directory/'inference.json'),components=r['output'],job=job,
   full_raw_correction=r['full_raw_correction'],full_centered_correction=r['full_centered_correction'],permutation_index=r['permutation_index'],
   actual_new_GPU=True,continued_after_diagnostic_guard=True,verification_amendment=amref)
  continued.append(record);records.append(record)
  update(completed_new_GPU_inferences=len(records),completed_continued_GPU_inferences=len(continued))
  atomic(OUT/'inference_index.json',dict(status='IN_PROGRESS_CONTINUED',protocol=pr,records=records,continuation=provenance,
   completed_continued_GPU_inferences=len(continued),actual_inference_attempts_to_date=19+len(continued)))
 guard();assert len(records)==42 and len(continued)==24
 assert{(r['case'],r['variant'],r['seed'])for r in records}=={(c,v,s)for c in P['inference_cases']for v in VARIANTS for s in SEEDS}
 all_features=[ref(OUT/'features'/c/'feature_manifest.json')for c in P['inference_cases']]
 complete=dict(status='COMPLETE_CONDITIONAL_MODELS_AND_FULL_C_INFERENCE',protocol=pr,completed_utc=now(),new_trainings=6,new_GPU_inferences=42,
  logical_inference_records=42,records=records,actual_new_GPU_records=records,feature_manifests=all_features,
  actual_model_input_tuples=sum(read(r['receipt']['path'])['actual_model_input_tuples']for r in records),
  actual_GPU_model_forward_minibatches=sum(read(r['receipt']['path'])['actual_GPU_model_forward_minibatches']for r in records),
  continuation=provenance,continued_GPU_inferences=24,retained_completed_GPU_inferences=18,partial_failed_GPU_inference_attempts=1,
  actual_total_inference_attempts=43,scientific_completed_GPU_inferences=42,original_controller_exit_code=1,continuation_completed=True,
  original_controller_succeeded=False,continued_controller_child_jobs_all_exit0=True)
 atomic(OUT/'inference_index.json',complete);new(OUT/'automation/inference_complete.json',complete)
 new(OUT/'continuation/anchor_diagnostic_guard/continuation_complete.json',complete)
 update(status='COMPLETE_CONDITIONAL_MODELS_AND_FULL_C_INFERENCE',active=None,completed_utc=now(),actual_total_inference_attempts=43)

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  failure=dict(status='FAILED_CONTINUATION',failed_utc=now(),error=repr(exc),automatic_retry_allowed=False,new_trainings=0)
  if(OUT/'automation/model_status.json').exists():
   state=read(OUT/'automation/model_status.json');state.update(failure);atomic(OUT/'automation/model_status.json',state)
  atomic(OUT/'automation/model_continuation_failure.json',failure)
  raise
