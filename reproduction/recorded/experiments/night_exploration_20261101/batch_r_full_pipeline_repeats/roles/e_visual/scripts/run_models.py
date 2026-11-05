"""R_E: exact original E visual_clean,12 prescribed seeds,GPU0 only; no AP."""
import argparse,fcntl,json,os,subprocess,sys,time,ast
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.affine_model import CODE,OUT,PSHA,CSHA,GPU,VARIANTS,SEEDS,read,sha,ref,verify,atomic,now,protocol,dataset,unit
PATHS=['src/affine_model.py','scripts/train_affine.py','scripts/infer_affine.py','scripts/run_models.py']
GUARD=OUT.parents[1]/'verification/e_visual_static_guard.json'
def available_gib():return int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))/1024**2
def freeze_only(pp,P):
 for n in ['automation','models','training','inference','ensembles','logs','verification']:(OUT/n).mkdir(parents=True,exist_ok=True)
 p=OUT/'automation/model_code_freeze.json';assert not p.exists(),'Preserve an old role attempt explicitly; no implicit overwrite'
 for n in PATHS:ast.parse((CODE/n).read_text())
 for rr in P['original_model_sources']+[P['original_model_code_freeze'],P['protocol_creation_source']]:verify(rr)
 sources={n:ref(CODE/n)for n in PATHS};r=dict(status='FROZEN_BEFORE_ANY_R_E_NUMERICAL_WORK',created_utc=now(),role='e_visual',protocol=ref(pp),parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],sources=sources,original_E_protocol=P['original_E_protocol'],original_E_sources=P['original_model_sources'],child_protocol_creation_source=P['protocol_creation_source'],scope='Four model-owned sources including new runner. Old E/C/N and R parent immutable.',independent_static_guard_required_before_numeric=str(GUARD))
 atomic(p,r);print(json.dumps(dict(status=r['status'],model_code_freeze=ref(p),role_protocol=ref(pp),parent_protocol=P['parent_protocol'])),flush=True)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--protocol',default=str(CODE/'config/protocol.json'));ap.add_argument('--freeze-only',action='store_true');a=ap.parse_args();pp=Path(a.protocol).resolve();P=protocol(pp);pr=ref(pp)
 if a.freeze_only:freeze_only(pp,P);return
 lock=(OUT/'automation/model_worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);frp=OUT/'automation/model_code_freeze.json';fr=read(frp);frref=ref(frp)
 assert fr['protocol']==pr and fr['parent_protocol']==P['parent_protocol']and fr['inference_clarification']==P['inference_clarification']
 sources=fr['sources'];state=dict(status='WAITING_INDEPENDENT_STATIC_GUARD_BEFORE_NUMERICAL',role='e_visual',PID=os.getpid(),started_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],active=None,completed_trainings=0,completed_inferences=0,completed_ensembles=0)
 def update(**kw):state.update(kw,updated_utc=now());atomic(OUT/'automation/model_worker_status.json',state)
 def frozen():
  assert ref(frp)==frref and ref(pp)==pr and {n:ref(CODE/n)for n in PATHS}==sources
  verify(P['parent_protocol']);verify(P['inference_clarification'])
 update()
 while not GUARD.exists():time.sleep(2)
 g=read(GUARD);assert g['status'].startswith('PASS_')and g['role_protocol']==pr and g['model_code_freeze']==frref and g['parent_protocol']==P['parent_protocol'];frozen()
 registry=dict(role='e_visual',protocol=pr,parent_protocol=P['parent_protocol'],registered={});registered=registry['registered']
 def idle_gpu0():
  while True:
   output=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True);target=[];checks=[]
   for line in output.splitlines():
    x=[t.strip()for t in line.split(',')]
    if len(x)>=2 and x[0]==GPU:
     pid=int(x[1]);target.append(pid);known=registered.get(str(pid));ok=None
     if known:
      proc=Path('/proc')/str(pid)/'cmdline'
      if proc.exists():ok=[b for b in proc.read_bytes().decode().split('\0')if b]==known['argv']
     checks.append(dict(PID=pid,registered_by_this_role=known is not None,argv_identity_match=ok))
   ram=available_gib()
   if not target and ram>=64:return dict(checked_utc=now(),GPU_UUID=GPU,physical_GPU_index=0,compute_processes_on_GPU0=[],registered_PID_argv_checks=checks,other_GPU_processes_untouched=True,available_RAM_GiB=ram,reserved_RAM_GiB=64)
   update(status='WAITING_GPU0_IDLE_OR_RESERVED_RAM',GPU0_compute_PIDs=target,registered_PID_argv_checks=checks,available_RAM_GiB=ram);time.sleep(2)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8')
 def run(name,argv):
  frozen();pre=idle_gpu0();log=OUT/'logs'/(name+'.log')
  with log.open('x')as stream:
   proc=subprocess.Popen(argv,env=env,stdout=stream,stderr=subprocess.STDOUT,cwd=Path('/mnt/data/yxy/program/RAXO'));launch=dict(PID=proc.pid,worker_PID=os.getpid(),argv=argv,started_utc=now(),log=str(log),CUDA_VISIBLE_DEVICES=GPU,protocol=pr,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],GPU_preflight=pre)
   registered[str(proc.pid)]=dict(argv=argv,job=name,CUDA_VISIBLE_DEVICES=GPU,started_utc=launch['started_utc']);atomic(OUT/'automation/registered_pids.json',registry);atomic(OUT/'automation'/('launch_'+name+'.json'),launch);update(active=launch);rc=proc.wait()
  registered[str(proc.pid)].update(exit_code=rc,completed_utc=now());atomic(OUT/'automation/registered_pids.json',registry);update(active=None)
  if rc:raise RuntimeError(f'{name} exit{rc}; preserve attempt: {log}')
  return dict(launch,exit_code=0,completed_utc=now())
 try:
  update(status='NUMERICAL_INPUT_PREFLIGHT_AFTER_GUARD');inputpath=verify(P['sources']['C_training_npz']);d,names,meta=dataset(inputpath,P);raw=d['train']['X'];before=raw.copy();const=np.flatnonzero(np.ptp(unit(raw,'visual_clean'),axis=0)==0).tolist();natural=unit(raw,'visual_clean',const)
  assert np.array_equal(raw,before)and np.isfinite(natural).all()and not np.any(natural[:,[0,3,8,12]])and not np.any(natural[:,9:])
  atomic(OUT/'verification/input_contracts.json',dict(status='PASS_ACTOR_INPUT_PREFLIGHT_NOT_INDEPENDENT_REVIEW',created_utc=now(),protocol=pr,parent_protocol=P['parent_protocol'],independent_static_guard=ref(GUARD),input=ref(inputpath),unique_train=len(raw),unique_val=len(d['val']['X']),constant_coordinates=const,raw_unchanged=True,old_C_metadata_identity_preserved=True,GT_read=False,AP_or_GPU_run=False))
  ts=dict(status='RUNNING',role='e_visual',protocol=pr,parent_protocol=P['parent_protocol'],input=ref(inputpath),runs=[],new_GT_annotations_read=False,AP_computed=False)
  for seed in SEEDS:
   v='visual_clean';update(status='RUNNING_ACTUAL_CUDA_TRAINING');md=OUT/'models'/v/f'seed{seed}';job=run(f'train_{v}_{seed}',[sys.executable,'-B',str(CODE/'scripts/train_affine.py'),'--input',str(inputpath),'--protocol',str(pp),'--variant',v,'--seed',str(seed),'--output-dir',str(md),'--device','cuda:0']);m=read(md/'metrics.json');cfg=read(md/'run_config.json')['config']
   assert m['actual_CUDA']and m['environment']['CUDA_VISIBLE_DEVICES']==GPU and cfg['input']==ref(inputpath)and cfg['protocol']==pr and cfg['parent_protocol']==P['parent_protocol']and(cfg['variant'],cfg['seed'])==(v,seed)
   ts['runs'].append(dict(job,variant=v,seed=seed,checkpoint=ref(md/'model.pt'),metrics=ref(md/'metrics.json'),history=ref(md/'train_history.json'),run_config=ref(md/'run_config.json'),normalization=ref(md/'normalization.json'),actual_CUDA=True));atomic(OUT/'training/summary.json',ts);update(completed_trainings=len(ts['runs']))
  ts.update(status='COMPLETE',completed_utc=now());atomic(OUT/'training/summary.json',ts)
  ins=dict(status='RUNNING',role='e_visual',protocol=pr,parent_protocol=P['parent_protocol'],jobs=[],expected_jobs=72,training_summary=ref(OUT/'training/summary.json'),GT_annotations_read=False,AP_computed=False)
  ens=dict(status='IN_PROGRESS',role='e_visual',protocol=pr,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],groups={g['id']:dict(seeds=g['seeds'],cases={})for g in P['groups']},expected_group_case_means=24);nmean=0
  for group in P['groups']:
   gid=group['id'];seeds=group['seeds'];assert len(seeds)==3
   for case in P['inference_cases']:
    fmp=verify(P['cases'][case]['feature_manifest']);fm=read(fmp);valid=np.load(verify(fm['outputs']['candidate_valid.npy']),allow_pickle=False);assert fm['protocol_sha256']==CSHA and fm['candidate_row_key_sha256']==P['cases'][case]['candidate_row_key_sha256'];values=[];inputs=[];modelrefs=[]
    for seed in seeds:
     v='visual_clean';update(status='RUNNING_ACTUAL_CUDA_INFERENCE',group=gid,case=case);rd=OUT/'inference'/case/v/f'seed{seed}';cp=OUT/'models'/v/f'seed{seed}'/'model.pt';job=run(f'infer_{case}_{v}_{seed}',[sys.executable,'-B',str(CODE/'scripts/infer_affine.py'),'--case',case,'--feature-manifest',str(fmp),'--checkpoint',str(cp),'--protocol',str(pp),'--output-dir',str(rd),'--device','cuda:0']);rp=rd/'inference.json';r=read(rp)
     assert r['protocol_sha256']==PSHA and r['parent_protocol']==P['parent_protocol']and r['inference_clarification']==P['inference_clarification']and(r['case'],r['variant'],r['seed'])==(case,v,seed)and r['checkpoint']==ref(cp)and r['feature_manifest']==ref(fmp)and r['candidate_row_key_sha256']==fm['candidate_row_key_sha256']and r['CUDA_VISIBLE_DEVICES']==GPU
     with np.load(verify(r['output']),allow_pickle=False)as z:q=z['quality'].copy()
     assert q.dtype==np.float32 and q.shape==valid.shape and np.isfinite(q).all()and np.all((q>=0)&(q<=1))and np.all(q[~valid]==0)
     ins['jobs'].append(dict(job,case=case,variant=v,seed=seed,group=gid,receipt=ref(rp),scores=r['output'],actual_CUDA=True));atomic(OUT/'inference/summary.json',ins);values.append(q.astype(np.float64));inputs.append(dict(seed=seed,scores=r['output'],inference_receipt=ref(rp),checkpoint=r['checkpoint']));modelrefs.append(r['checkpoint']);update(completed_inferences=len(ins['jobs']))
    q=np.mean(np.stack(values),axis=0,dtype=np.float64).astype(np.float32);ed=OUT/'ensembles'/gid/case;ed.mkdir(parents=True,exist_ok=True);sp=ed/'quality_scores.npz';rp=ed/'ensemble.json'
    with sp.open('xb')as f:np.savez_compressed(f,quality=q)
    receipt=dict(status='COMPLETE_PRESCRIBED_TRIPLE_E_VISUAL_ENSEMBLE',created_utc=now(),role='e_visual',group=gid,protocol=pr,protocol_sha256=PSHA,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],original_E_protocol=P['original_E_protocol'],source_C_protocol_sha256=CSHA,case=case,variant='visual_clean',seeds=seeds,inputs=inputs,model_refs=modelrefs,feature_manifest=ref(fmp),candidate_row_key_sha256=fm['candidate_row_key_sha256'],scores=ref(sp),output_sha256=sha(sp),shape=list(q.shape),dtype='float32',invalid_quality_exact0=True,rule='Own prescribed3 seeds F64 arithmetic mean then oneF32; never average12',inference_sigmoid='Exact original E GPUF32 sigmoid before CPU NumPyF32',GT_opened=False,labels_read=False,AP_computed=False,threshold_selected=False)
    atomic(rp,receipt);entry=dict(scores=ref(sp),receipt=ref(rp),model_refs=modelrefs,candidate_row_key_sha256=fm['candidate_row_key_sha256'],feature_manifest=ref(fmp),role_protocol=pr,parent_protocol=P['parent_protocol']);ens['groups'][gid]['cases'][case]=entry;nmean+=1;atomic(OUT/'ensemble_index.json',ens);update(completed_ensembles=nmean)
  assert len(ts['runs'])==12 and len(ins['jobs'])==72 and nmean==24 and all(len(g['cases'])==6 for g in ens['groups'].values())
  frozen();ins.update(status='COMPLETE',completed_utc=now());ens.update(status='COMPLETE',completed_utc=now());atomic(OUT/'inference/summary.json',ins);atomic(OUT/'ensemble_index.json',ens)
  completion=dict(status='COMPLETE_ROLE_ACTOR_STOP_AWAIT_INDEPENDENT_REVIEW',created_utc=now(),role='e_visual',protocol=pr,parent_protocol=P['parent_protocol'],inference_clarification=P['inference_clarification'],model_code_freeze=frref,training_summary=ref(OUT/'training/summary.json'),inference_summary=ref(OUT/'inference/summary.json'),ensemble_index=ref(OUT/'ensemble_index.json'),counts=dict(new_trainings=12,actual_GPU_inferences=72,own_triple_means=24,new_AP=0,new_GT=0,new_features=0),actual_GPU_UUID=GPU,CPUenv_threads=8,original_E_math_unchanged=True,all4_groups_all6_preserved=True,independent_acceptance_by_actor=False,STOP=True)
  atomic(OUT/'role_completion.json',completion);update(status='COMPLETE',active=None,group=None,case=None,completed_utc=now(),scientific_writer_STOP=True);print(json.dumps(dict(status='ROLE_ACTOR_STOP',completion=ref(OUT/'role_completion.json'),ensemble_index=ref(OUT/'ensemble_index.json'))),flush=True)
 except BaseException as e:update(status='FAILED',error=repr(e),completion_claimed=False);raise
if __name__=='__main__':main()
