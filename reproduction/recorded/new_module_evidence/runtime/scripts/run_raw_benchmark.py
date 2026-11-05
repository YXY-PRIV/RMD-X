#!/usr/bin/env python3
"""Actual raw-image-to-final P5 inference, fixed fresh benchmark workload.

Fresh process for each method/trial, full model weights and all runtime stages;
never estimates complete runtime by adding historical stage measurements.
"""
import os,sys,json,pathlib,time,subprocess,socket,uuid,hashlib,argparse,threading
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from multiprocessing.connection import Listener
import numpy as np
ROOT=pathlib.Path('/mnt/data/yxy/program/RAXO');CODE=ROOT/'experiments/new_module_evidence_20261102/runtime';OUT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime')
PY='/mnt/data/yxy/envs/gdino_cu118_clean/bin/python';AUTH=b'p28-local-runtime-fixed'
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def save(p,x):
 p=pathlib.Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def send(c,x):c.send_bytes(json.dumps(x,separators=(',',':'),allow_nan=False).encode())
def recv(c,proc,timeout=900):
 t=time.monotonic()
 while not c.poll(.2):
  if proc.poll()is not None:raise RuntimeError('Worker exit '+str(proc.returncode))
  if time.monotonic()-t>timeout:raise TimeoutError('Worker timeout preserved trial')
 return json.loads(c.recv_bytes())
def request(c,p,x):send(c,x);return recv(c,p)
def small(e):return dict(original_image_id=e['original_image_id'],file_name=e['file_name'],width=e['width'],height=e['height'],image=dict(path=e['image']['path']))
class Monitor:
 def __init__(self,root):
  self.root=root;self.stop=False;self.samples=[];self.thread=threading.Thread(target=self.run,daemon=True)
  sys.path.insert(0,'/mnt/data/yxy-codex/home/outputs/RAXO/P26_end_revision_20261024/runtime/deps/nvml');import pynvml
  self.nv=pynvml;self.nv.nvmlInit();self.handle=self.nv.nvmlDeviceGetHandleByIndex(0)
 def run(self):
  while not self.stop:
   pids={self.root};changes=True
   while changes:
    changes=False
    for d in pathlib.Path('/proc').iterdir():
     if not d.name.isdigit()or int(d.name)in pids:continue
     try:
      s=(d/'stat').read_text().split(')')[-1].split();parent=int(s[1])
      if parent in pids:pids.add(int(d.name));changes=True
     except(OSError,ValueError,IndexError):pass
   values=[]
   for pid in sorted(pids):
    try:
     kv={x.split(':')[0]:x.split(':')[1].strip()for x in pathlib.Path('/proc',str(pid),'status').read_text().splitlines()if':'in x}
     values.append(dict(PID=pid,RSS_bytes=int(kv['VmRSS'].split()[0])*1024,HWM_bytes=int(kv['VmHWM'].split()[0])*1024))
    except(OSError,KeyError,ValueError):pass
   gpu=[dict(PID=int(p.pid),used_GPU_bytes=int(p.usedGpuMemory))for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)if int(p.pid)in pids]
   self.samples.append(dict(monotonic_ns=time.monotonic_ns(),processes=values,total_RSS_bytes=sum(v['RSS_bytes']for v in values),GPU_processes=gpu,total_processgroup_GPU_bytes=sum(v['used_GPU_bytes']for v in gpu)))
   time.sleep(.05)
 def start(self):self.thread.start()
 def finish(self):self.stop=True;self.thread.join();self.nv.nvmlShutdown();return self.samples

def start(directory,method,track):
 (OUT/'ipc').mkdir(exist_ok=True);sock=OUT/'ipc'/('r_'+uuid.uuid4().hex[:8]+'.sock');assert len(str(sock).encode())<108
 listener=Listener(str(sock),family='AF_UNIX',authkey=AUTH)
 command=[PY,'-B',str(CODE/'src/raw_worker.py'),'--role','main','--socket',str(sock),'--trial',str(directory),'--boundary','C','--method',method]+(['--track']if track else[])
 t=time.perf_counter_ns();proc=subprocess.Popen(command,stdout=(directory/'main_worker.log').open('w'),stderr=subprocess.STDOUT);listener._listener._socket.settimeout(.2)
 while True:
  try:c=listener.accept();break
  except socket.timeout:
   if proc.poll()is not None:raise RuntimeError('Worker startup failed')
   if time.perf_counter_ns()-t>900e9:raise TimeoutError('Startup timeout')
 listener.close();sock.unlink(missing_ok=True);ready=recv(c,proc);startup=time.perf_counter_ns()-t;save(directory/'startup.json',dict(process_to_ready_wall_ns=startup,command=command,ready=ready));return c,proc

def trial(g,proto,compat=False):
 directory=OUT/'raw_trials'/g['id'];assert not directory.exists();directory.mkdir(parents=True);c=None;p=None;monitor=None
 try:
  c,p=start(directory,g['method'],not compat)
  for e in proto['warmup']:request(c,p,dict(op='image',entry=small(e),capture=False))
  request(c,p,dict(op='clear_metadata'));save(directory/'warmup_memory_reset.json',request(c,p,dict(op='reset')))
  monitor=Monitor(p.pid);monitor.start();raw=[];records=[];captures=[]
  cohort_start=time.perf_counter_ns()
  for i,e in enumerate(proto['measured']):
   t=time.perf_counter_ns();z=request(c,p,dict(op='image',entry=small(e),capture=compat));elapsed=time.perf_counter_ns()-t
   raw.append(dict(index=i,image_id=e['original_image_id'],proposals=z['proposal_count'],returned=len(z['records']),request_wall_ns=elapsed,stages_ns=z['stages_ns']));records.extend(z['records'])
   if compat:captures.append(dict(entry=small(e),capture=request(c,p,dict(op='capture'))))
  cohort_wall=time.perf_counter_ns()-cohort_start
  mem=request(c,p,dict(op='memory'));metadata=request(c,p,dict(op='drain_metadata'));samples=monitor.finish();monitor=None
  save(directory/'raw_samples.json',raw);save(directory/'records.json',records);save(directory/'metadata.json',metadata);save(directory/'memory.json',mem);save(directory/'process_memory_samples.json',samples)
  if compat:save(directory/'captures.json',captures)
  imports=request(c,p,dict(op='imports'));save(directory/'imports.json',imports)
  close=request(c,p,dict(op='close'));save(directory/'shutdown.json',close);c.close();c=None;p.wait();assert p.returncode==0
  vals=np.asarray([r['request_wall_ns']/1e6 for r in raw]);summary=dict(group=g,compatibility_only=compat,status='PASS_FRESH_PROCESS_RAW_TO_FINAL_INFERENCE',actual_worker_exit=p.returncode,images=len(raw),proposals=sum(r['proposals']for r in raw),detections=len(records),mean_ms=float(vals.mean()),median_ms=float(np.median(vals)),p90_ms=float(np.quantile(vals,.9)),sample_SD_ms=float(vals.std(ddof=1)),cohort_wall_seconds=cohort_wall/1e9,process_group_peak_RSS_bytes=max(s['total_RSS_bytes']for s in samples),process_group_peak_NVML_GPU_bytes=max(s['total_processgroup_GPU_bytes']for s in samples),GPU=mem,final_output_signature=hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest(),stage_scope=proto['boundary'],new_AP_run=False,historical_accuracy_identity_claimed=False,raw_samples=ref(directory/'raw_samples.json'))
  save(directory/'summary.json',summary);print('DONE_RAW',g['id'],summary['mean_ms'],flush=True)
  return summary
 finally:
  if monitor is not None:monitor.finish()
  if c is not None:
   try:request(c,p,dict(op='close'));c.close();p.wait(timeout=60)
   except Exception:pass

def main():
 a=argparse.ArgumentParser();a.add_argument('--compatibility',action='store_true');a.add_argument('--single',type=int);args=a.parse_args()
 proto=read(CODE/'config/raw_protocol.json');freeze=read(OUT/'automation/raw_code_freeze.json');assert freeze['protocol']==ref(CODE/'config/raw_protocol.json')
 for r in freeze['sources']:assert sha(r['path'])==r['sha256']
 for r in freeze['readonly_upstream_sources']:assert sha(r['path'])==r['sha256']
 b=freeze['actual_raw_input_bindings'];assert sha(b['path'])==b['sha256']
 for r in read(b['path'])['files']:assert pathlib.Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
 for e in proto['measured']+proto['warmup']:assert sha(e['image']['path'])==e['image']['sha256']
 os.sched_setaffinity(0,proto['CPU_affinity'])
 if args.compatibility:
  groups=[dict(id='compatibility_pair',method='pair',trial=-1)]
 else:
  assert read(OUT/'gates/raw_compatibility_review.json')['status']=='PASS'
  groups=proto['groups']
  if args.single is not None:groups=[groups[args.single]]
 for g in groups:trial(g,proto,args.compatibility)
if __name__=='__main__':main()
