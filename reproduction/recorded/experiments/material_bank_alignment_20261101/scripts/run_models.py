#!/usr/bin/env python3
"""Exactly3 new trainings,24 new CUDA forwards,39 explicit old references."""
from pathlib import Path
import fcntl,json,os,subprocess,sys,time
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model import CODE,OUT,ROLES,SEEDS,read,ref,verify,atomic,protocol,inference_feature,now


def new(path,data):
 p=Path(path).resolve();assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')

def ram():
 for line in Path('/proc/meminfo').read_text().splitlines():
  if line.startswith('MemAvailable:'):return int(line.split()[1])/2**20
 raise AssertionError('MemAvailable missing')

def identity(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split()
 return dict(PID=pid,starttime_ticks=int(s[19]),argv=[x.decode()for x in (p/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')])


def main():
 pp=CODE/'config/protocol.json';P,pr=protocol(pp,os.environ['RAXO_PROTOCOL_SHA256'])
 assert P['variants']==list(ROLES)and len(P['inference_cases'])==7 and len(set(P['inference_cases']))==7
 for name in ['automation','models','inference','logs','verification']:(OUT/name).mkdir(parents=True,exist_ok=True)
 controller=(OUT/'automation/model_controller.lock').open('a');fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(status='WAITING_INDEPENDENT_ALIGNED_DATA_REVIEW',PID=os.getpid(),started_utc=now(),protocol=pr,
  active=None,completed_trainings=0,completed_new_GPU_inferences=0,referenced_inferences=0)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_status.json',state)
 update();started=time.monotonic();review_path=OUT/'verification/data_review.json'
 while not review_path.exists():
  assert time.monotonic()-started<4*3600;assert ref(pp)==pr;time.sleep(2)
 review=read(review_path);assert review['status']=='PASS'and review['protocol_sha256']==pr['sha256']
 bridge_path=OUT/'verification/control_effective_input_bridge.json';bridge=read(bridge_path)
 assert bridge['status']=='PASS'and bridge['protocol']==pr
 bridge_ref=ref(bridge_path);review_ref=ref(review_path)
 receipt_path=Path(P['training']['source_input']['actual_ref_from']);receipt=read(receipt_path)
 assert receipt['status']=='COMPLETE'and receipt['protocol']==pr
 npz_ref=receipt['training_npz'];verify(npz_ref)
 sources=[ref(CODE/n)for n in ['src/quality_model.py','scripts/train_models.py','scripts/run_models.py','config/model_design.json','config/protocol.json']]
 new(OUT/'automation/model_code_freeze.json',dict(status='FROZEN_BEFORE_THREE_ALIGNED_TRAININGS_AND_24_FORWARDS',created_utc=now(),protocol=pr,sources=sources,
  independent_data_review=review_ref,control_effective_input_bridge=bridge_ref,training_npz=npz_ref,training_receipt=ref(receipt_path)))
 def guard():
  assert all(ref(r['path'])==r for r in sources)
  assert ref(review_path)==review_ref and ref(bridge_path)==bridge_ref and ram()>=64
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=P['resources']['GPU_UUID'],CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def actual(name,args):
  guard()
  while True:
   text=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
   busy=[line for line in text.splitlines()if line.split(',')[0].strip()==P['resources']['GPU_UUID']]
   if not busy:break
   update(status='WAITING_GPU_FREE',busy=busy);time.sleep(10)
  argv=[sys.executable,'-B',str(CODE/'scripts/train_models.py'),*args,'--protocol',str(pp),'--protocol-sha256',pr['sha256'],'--device','cuda:0']
  log=OUT/'logs'/(name+'.log')
  with log.open('x')as f:
   child=subprocess.Popen(argv,env=env,stdout=f,stderr=subprocess.STDOUT)
   launch=dict(identity=identity(child.pid),argv=argv,log=str(log),started_utc=now(),protocol=pr,GPU_UUID=P['resources']['GPU_UUID'],available_RAM_GiB=ram())
   new(OUT/'automation'/('launch_'+name+'.json'),launch);update(status='ACTUAL_CUDA_'+args[0].upper(),active=launch);code=child.wait()
  assert code==0,(name,code,str(log));update(active=None)
  return dict(**launch,exit_code=0,exited_utc=now())
 trainings=[]
 for seed in SEEDS:
  directory=OUT/'models/material'/f'seed{seed}'
  job=actual(f'train_material_{seed}',['train','--variant','material','--seed',str(seed),'--output-dir',str(directory)])
  m=read(directory/'metrics.json');assert m['status']=='COMPLETE'and m['actual_CUDA']and m['variant']=='material'and m['seed']==seed
  trainings.append(dict(variant='material',seed=seed,checkpoint=ref(directory/'model.pt'),metrics=ref(directory/'metrics.json'),job=job))
  update(completed_trainings=len(trainings))
 new(OUT/'automation/training_complete.json',dict(status='COMPLETE_THREE_ALIGNED_MATERIAL_CUDA_TRAININGS',completed_utc=now(),protocol=pr,runs=trainings,new_trainings=3))
 records=[];actual_runs=[];references=[]
 for case in P['inference_cases']:
  fp,F=inference_feature(case,P,pr);feature_ref=ref(fp)
  for role in ROLES:
   for seed in SEEDS:
    guard();directory=OUT/'inference'/case/role/f'seed{seed}'
    if role=='material'or(role=='legacy_material'and case=='CLC_tune'):
     checkpoint=OUT/'models/material'/f'seed{seed}'/'model.pt'if role=='material'else verify(P['sources']['T_checkpoints']['material'][str(seed)])
     job=actual(f'infer_{case}_{role}_{seed}',['infer','--case',case,'--role',role,'--checkpoint',str(checkpoint),'--output-dir',str(directory)])
     r=read(directory/'inference.json');assert r['protocol']==pr and r['role']==role and r['case']==case and r['seed']==seed
     assert r['actual_CUDA']is True and r['new_GPU_inference']is True and r['reference_reused']is False
     actual_runs.append(dict(case=case,role=role,seed=seed,receipt=ref(directory/'inference.json'),components=r['output'],job=job))
    else:
     native='availability_control'if role=='availability_control'else'material'
     old=P['cases'][case]['T_inference'][native][str(seed)];source=read(verify(old['receipt']));verify(old['components'])
     assert source['protocol']==P['sources']['T_protocol']and(source['case'],source['variant'],source['seed'])==(case,native,seed)
     assert source['output']==old['components']and source['checkpoint']==P['sources']['T_checkpoints'][native][str(seed)]
     assert source['candidate_row_key_sha256']==F['candidate_row_key_sha256']
     if role=='legacy_material':assert case!='CLC_tune'and source['feature_manifest']==feature_ref
     else:assert role=='availability_control'and bridge['status']=='PASS'
     directory.mkdir(parents=True,exist_ok=False)
     r=dict(status='COMPLETE_STRICT_REFERENCE_REUSED_INFERENCE',created_utc=now(),protocol=pr,protocol_sha256=pr['sha256'],
      case=case,variant=role,role=role,native_variant=native,seed=seed,native_model_protocol=P['sources']['T_protocol'],
      feature_manifest=feature_ref,source_native_feature_manifest=source['feature_manifest'],
      candidate_row_key_sha256=F['candidate_row_key_sha256'],checkpoint=source['checkpoint'],anchor_checkpoint=source['anchor_checkpoint'],
      output=old['components'],output_sha256=old['components']['sha256'],shape=source['shape'],dtype='float32',
      source_native_receipt=old['receipt'],effective_input_bridge=bridge_ref,actual_CUDA=False,new_GPU_inference=False,reference_reused=True,actual_fresh_GPU=False,strict_reuse=True,
      reuse_reason='All active geometry/availability inputs, teachers, sampling and model training decisions are bitexact under independently reviewed bank alignment'if role=='availability_control'else'All six target test features and legacy material checkpoint are the exact original T inputs',
      new_training=0,GT_opened=False,labels_read=False,AP_computed=False,deployment_parameters_selected=False)
     new(directory/'inference.json',r);references.append(dict(case=case,role=role,seed=seed,receipt=ref(directory/'inference.json'),components=r['output'],source_native_receipt=old['receipt']))
    records.append(dict(case=case,variant=role,role=role,seed=seed,receipt=ref(directory/'inference.json'),components=r['output'],actual_new_GPU=r['new_GPU_inference'],strict_reference_reused=r['reference_reused']))
    update(completed_new_GPU_inferences=len(actual_runs),referenced_inferences=len(references))
    atomic(OUT/'inference_index.json',dict(status='IN_PROGRESS',protocol=pr,records=records,actual_new_GPU_records=actual_runs,reference_records=references))
 guard();assert len(trainings)==3 and len(actual_runs)==24 and len(references)==39 and len(records)==63
 complete=dict(status='COMPLETE_24_NEW_CUDA_AND_39_REFERENCE_INFERENCES',completed_utc=now(),protocol=pr,records=records,
  actual_new_GPU_records=actual_runs,reference_records=references,new_trainings=3,new_GPU_inferences=24,
  strict_old_inference_references=39,logical_role_cohort_seed_records=63,control_effective_input_bridge=bridge_ref)
 atomic(OUT/'inference_index.json',complete);new(OUT/'automation/inference_complete.json',complete)
 update(status='COMPLETE_ALIGNED_MODELS_AND_INFERENCE_REFERENCES',completed_utc=now(),active=None)


if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if(OUT/'automation').exists():
   failure=dict(status='FAILED',failed_utc=now(),error=repr(exc),automatic_retry_allowed=False)
   if(OUT/'automation/model_status.json').exists():
    prior=read(OUT/'automation/model_status.json');prior.update(failure);atomic(OUT/'automation/model_status.json',prior)
   atomic(OUT/'automation/model_failure.json',failure)
  raise
