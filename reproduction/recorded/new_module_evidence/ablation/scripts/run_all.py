#!/usr/bin/env python3
from pathlib import Path
import os,sys,json,datetime,subprocess,time,threading,fcntl
from concurrent.futures import ThreadPoolExecutor,as_completed
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ablation')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(p,v):
 q=p.with_name(p.name+f'.tmp.{os.getpid()}');q.write_text(json.dumps(v,indent=2)+'\n');os.replace(q,p)
def run():
 P=json.loads((CODE/'protocol.json').read_text());lock=(OUT/'automation/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not(OUT/'automation/status.json').exists(),'No implicit retry'
 state=dict(status='RUNNING_FROZEN_ABLATION',PID=os.getpid(),started_utc=now(),active={},completed=[]);mutex=threading.Lock();statepath=OUT/'automation/status.json'
 def update():save(statepath,state)
 update()
 def job(case):
  log=OUT/'logs'/f'{case}.log';env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8',NUMEXPR_NUM_THREADS='8',RAXO_PROTOCOL_SHA256=P['latest_protocol']['sha256'])
  argv=[sys.executable,'-B',str(CODE/'scripts/run_dataset.py'),'--case',case]
  with log.open('x')as f:
   proc=subprocess.Popen(argv,stdout=f,stderr=subprocess.STDOUT,env=env)
   with mutex:state['active'][case]=dict(PID=proc.pid,started_utc=now(),log=str(log));update()
   rc=proc.wait()
  with mutex:state['active'].pop(case);state['completed'].append(dict(case=case,exit_code=rc,finished_utc=now(),log=str(log)));update()
  assert rc==0,(case,rc,str(log))
  return case
 try:
  with ThreadPoolExecutor(max_workers=2)as pool:
   fs=[pool.submit(job,c)for c in P['test_cases']]
   for f in as_completed(fs):print('COMPLETE',f.result(),flush=True)
  state.update(status='COMPLETE_24_LOGICAL_ABLATION_CELLS',completed_utc=now());update()
 except BaseException as exc:
  state.update(status='FAILED_REQUIRES_REVIEW',error=repr(exc),finished_utc=now());update();raise
if __name__=='__main__':run()
