#!/usr/bin/env python3
"""Six conditional trainings and42 full-C GPU forwards after independent gates."""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model import CODE,OUT,VARIANTS,SEEDS,now,ref,read,verify,atomic,protocol,prepare_case


def new(path,value):
 p=Path(path).resolve();assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def ram():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/2**20
 raise AssertionError('Missing MemAvailable')

def identity(pid):
 d=Path('/proc')/str(pid);st=(d/'stat').read_text().rsplit(')',1)[1].split()
 return dict(PID=pid,starttime_ticks=int(st[19]),argv=[b.decode()for b in(d/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])


def main():
 pp=CODE/'config/protocol.json';P,pr=protocol(pp,os.environ['RAXO_PROTOCOL_SHA256'])
 for name in ['automation','models','inference','features','logs','verification']:(OUT/name).mkdir(parents=True,exist_ok=True)
 lock=(OUT/'automation/model_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(status='WAITING_INDEPENDENT_CONDITIONAL_DATA_AND_MODEL_PREFLIGHT',protocol=pr,PID=os.getpid(),started_utc=now(),
  active=None,completed_trainings=0,completed_new_GPU_inferences=0)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_status.json',state)
 update();start=time.monotonic()
 gates=[OUT/'verification/data_review.json',OUT/'verification/model_preflight_review.json']
 while not all(p.exists()for p in gates):
  assert time.monotonic()-start<4*3600 and ref(pp)==pr;time.sleep(2)
 for path in gates:
  g=read(path);assert g['status']=='PASS'and g['protocol_sha256']==pr['sha256']
 gate_refs=[ref(p)for p in gates]
 receipt_path=Path(P['training']['source_input']['actual_ref_from']);receipt=read(receipt_path)
 assert receipt['status']=='COMPLETE_GROUPED_SOURCE_DATA'and receipt['protocol']==pr;verify(receipt['training_npz'])
 names=['src/quality_model.py','scripts/train_models.py','scripts/run_models.py','config/model_design.json','src/group_data.py','config/protocol.json']
 sources=[ref(CODE/name)for name in names]
 readonly=[P['sources'][k]for k in ['T_model_source','T_protocol','N_supported_source','aligned_training','aligned_training_receipt']]
 readonly.extend([ref(receipt_path),receipt['training_npz'],ref(OUT/'data_completion.json')])
 for seed in SEEDS:readonly.append(P['sources']['T_checkpoints']['availability_control'][str(seed)])
 new(OUT/'automation/model_code_freeze.json',dict(status='FROZEN_BEFORE_SIX_CONDITIONAL_TRAININGS_AND42_FULL_C_FORWARDS',created_utc=now(),
  protocol=pr,sources=sources,readonly_sources=readonly,independent_reviews=gate_refs,training_npz=receipt['training_npz'],training_receipt=ref(receipt_path)))
 def guard():
  assert all(ref(r['path'])==r for r in sources+readonly+gate_refs)
  assert ram()>=P['resources']['min_MemAvailable_GiB']
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=P['resources']['GPU_UUID'],CUBLAS_WORKSPACE_CONFIG=':4096:8',
  PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def actual(name,args):
  guard()
  while True:
   lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
   busy=[s for s in lines.splitlines()if s.split(',')[0].strip()==P['resources']['GPU_UUID']]
   if not busy:break
   update(status='WAITING_RESERVED_GPU_FREE',busy=busy);time.sleep(10)
  argv=[sys.executable,'-B',str(CODE/'scripts/train_models.py'),*args,'--protocol',str(pp),'--protocol-sha256',pr['sha256'],'--device','cuda:0']
  log=OUT/'logs'/(name+'.log')
  with log.open('x')as stream:
   child=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT)
   launch=dict(identity=identity(child.pid),argv=argv,log=str(log),protocol=pr,started_utc=now(),GPU_UUID=P['resources']['GPU_UUID'],available_RAM_GiB=ram())
   new(OUT/'automation'/('launch_'+name+'.json'),launch);update(status='ACTUAL_CONDITIONAL_CUDA_'+args[0].upper(),active=launch)
   code=child.wait()
  assert code==0,(name,code,str(log));update(active=None)
  return dict(**launch,exit_code=0,exited_utc=now())
 trainings=[]
 for variant in VARIANTS:
  for seed in SEEDS:
   directory=OUT/'models'/variant/f'seed{seed}'
   job=actual(f'train_{variant}_{seed}',['train','--variant',variant,'--seed',str(seed),'--output-dir',str(directory)])
   result=read(directory/'metrics.json');assert result['status']=='COMPLETE'and result['actual_CUDA']is True
   assert result['protocol']==pr and(result['variant'],result['seed'])==(variant,seed)
   trainings.append(dict(variant=variant,seed=seed,checkpoint=ref(directory/'model.pt'),metrics=ref(directory/'metrics.json'),job=job))
   update(completed_trainings=len(trainings))
 new(OUT/'automation/training_complete.json',dict(status='COMPLETE_SIX_CONDITIONAL_CUDA_TRAININGS',protocol=pr,completed_utc=now(),runs=trainings,new_trainings=6))
 feature_refs=[];records=[]
 for case in P['inference_cases']:
  guard();manifest=prepare_case(P,pr,case);feature_refs.append(ref(OUT/'features'/case/'feature_manifest.json'))
  for variant in VARIANTS:
   for seed in SEEDS:
    directory=OUT/'inference'/case/variant/f'seed{seed}'
    job=actual(f'infer_{case}_{variant}_{seed}',['infer','--variant',variant,'--case',case,'--checkpoint',str(OUT/'models'/variant/f'seed{seed}'/'model.pt'),'--output-dir',str(directory)])
    result=read(directory/'inference.json');assert result['status']=='COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE'
    assert result['protocol']==pr and(result['case'],result['variant'],result['seed'])==(case,variant,seed)
    record=dict(case=case,variant=variant,seed=seed,receipt=ref(directory/'inference.json'),components=result['output'],job=job,
     full_raw_correction=result['full_raw_correction'],full_centered_correction=result['full_centered_correction'],permutation_index=result['permutation_index'],actual_new_GPU=True)
    records.append(record);update(completed_new_GPU_inferences=len(records))
    atomic(OUT/'inference_index.json',dict(status='IN_PROGRESS',protocol=pr,records=records))
 guard();assert len(trainings)==6 and len(records)==42 and len(feature_refs)==7
 complete=dict(status='COMPLETE_CONDITIONAL_MODELS_AND_FULL_C_INFERENCE',protocol=pr,completed_utc=now(),new_trainings=6,
  new_GPU_inferences=42,logical_inference_records=42,records=records,actual_new_GPU_records=records,feature_manifests=feature_refs,
  actual_model_input_tuples=sum(read(r['receipt']['path'])['actual_model_input_tuples']for r in records),
  actual_GPU_model_forward_minibatches=sum(read(r['receipt']['path'])['actual_GPU_model_forward_minibatches']for r in records))
 atomic(OUT/'inference_index.json',complete);new(OUT/'automation/inference_complete.json',complete)
 update(status='COMPLETE_CONDITIONAL_MODELS_AND_FULL_C_INFERENCE',active=None,completed_utc=now())

if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if(OUT/'automation').exists():
   failure=dict(status='FAILED',failed_utc=now(),error=repr(exc),automatic_retry_allowed=False)
   if(OUT/'automation/model_status.json').exists():
    state=read(OUT/'automation/model_status.json');state.update(failure);atomic(OUT/'automation/model_status.json',state)
   atomic(OUT/'automation/model_failure.json',failure)
  raise
