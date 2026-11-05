#!/usr/bin/env python3
"""True checkpoint forwards, exact artifact golden, timed boundary is explicit."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',CUBLAS_WORKSPACE_CONFIG=':4096:8')
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='8' if k=='OMP_NUM_THREADS'else'1'
import sys,json,pathlib,time,resource,statistics,argparse,hashlib
import numpy as np
ROOT=pathlib.Path('/mnt/data/yxy/program/RAXO');CODE=ROOT/'experiments/new_module_evidence_20261102/runtime';OUT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime')
sys.path.insert(0,str(CODE/'src'))
from deployment import *
CASES=['DET','CLCXray_test','PIXray','PIDray','DvXray','Hi']
def save(p,j):
 p=pathlib.Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def at(p):return dict(path=str(p),bytes=pathlib.Path(p).stat().st_size,sha256=sha(p))
def memory():
 return dict(peak_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved(),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
def eq(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))
def compare(a,b):
 return dict(shape=list(a.shape),dtype=str(a.dtype),bit_exact=eq(a,b),different_cells=int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))if a.dtype==np.float32 else int(np.count_nonzero(a!=b)),max_absdiff=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)),initial=0.)))
def golden(case,result):
 # No metric/GT annotations opened; exact final endpoint trace and components.
 run=read(pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/pair_conflict_material_projection_20261101/run_index.json'))
 rows=[r for r in run['endpoints'] if r['dataset']==case and r['variant']=='material'and r['route']=='anchor_locked'and r['residual_alpha']==1.]
 assert len(rows)==1,rows
 e=read(rows[0]['export_receipt']['path']);trace=loadz(e['score_trace']['path']);checks=dict(q=compare(result['q'],trace['full_candidate_quality']),q0=compare(result['q0'],trace['strong_anchor_reference_candidate_quality']),selected=compare(result['selected'],trace['strong_anchor_reference_selected_branch']),post=compare(result['post'],trace['post_soft_score']))
 seed_checks={}
 for seed,v in result['details'].items():
  old=loadz(P['cases'][case]['Tavailability_inference'][str(seed)]['components']['path'])
  r=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/material_conditional_ranking_fix1_20261101/inference')/case/'material'/f'seed{seed}/components.npz';cz=loadz(r)
  ez=loadz(pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_e_affine_robustness/inference')/case/'visual_clean'/f'seed{seed}/quality_scores.npz')
  kz=loadz(pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_k_multisource_quality/inference')/case/'geometry'/f'seed{seed}/quality_scores.npz')
  seed_checks[str(seed)]={k:compare(v['T'][k],old[k])for k in ['anchor_logits','correction_logits','full_logits','availability']}
  seed_checks[str(seed)].update(Kfallback=compare(v['Kq'],kz['quality']),E=compare(v['Eq'],ez['quality']),conditional_centered=compare(v['centered'],cz['centered_correction']),conditional_raw=compare(v['rawg'],cz['raw_candidate_correction']))
 all_exact=all(c['bit_exact']for c in checks.values())and all(c['bit_exact']for vv in seed_checks.values()for c in vv.values())
 record=dict(status='PASS_BIT_EXACT_ALL_MODEL_AND_DEPLOYMENT_GOLDEN'if all_exact else'GOLDEN_MISMATCH_REQUIRES_REVIEW',case=case,checks=checks,seed_checks=seed_checks,final_endpoint=rows[0]['export_receipt'],GT_or_AP_opened=False)
 save(OUT/'gates'/f'{case}.json',record);return record

def main():
 a=argparse.ArgumentParser();a.add_argument('--case',choices=CASES);a.add_argument('--gates-only',action='store_true');a.add_argument('--measure-only',action='store_true');args=a.parse_args()
 proto=read(CODE/'config/quality_protocol.json');assert proto['status']=='FROZEN_BEFORE_QUALITY_FORWARD_OR_TIMING'
 freeze=read(OUT/'automation/quality_code_freeze.json');assert freeze['protocol']==at(CODE/'config/quality_protocol.json')
 for r in freeze['sources']:assert sha(r['path'])==r['sha256']
 affinity=proto['CPU_affinity'];assert set(affinity)<=os.sched_getaffinity(0);os.sched_setaffinity(0,affinity)
 torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.set_grad_enabled(False);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
 t=time.perf_counter();model=Deployment();model.sync();load_seconds=time.perf_counter()-t
 save(OUT/'automation'/f'quality_process_{os.getpid()}.json',dict(PID=os.getpid(),argv=sys.argv,protocol=freeze['protocol'],model_load_seconds=load_seconds,model_refs=model.refs,parameters=model.parameter_count,Python=sys.executable,torch=str(torch.__version__),numpy=str(np.__version__),GPU=torch.cuda.get_device_name(),affinity=sorted(os.sched_getaffinity(0)),memory=memory()))
 for case in ([args.case]if args.case else CASES):
  t=time.perf_counter();d=case_data(case);input_load_seconds=time.perf_counter()-t
  nimages=len(np.unique(d['trace']['image_id']));print('CASE',case,len(d['M']),nimages,flush=True)
  if not args.measure_only:
   result=model.run(d,'pair',capture=True);g=golden(case,result);print('GATE',case,g['status'],result['stages_ns']['total']/1e9,flush=True)
   if g['status']!='PASS_BIT_EXACT_ALL_MODEL_AND_DEPLOYMENT_GOLDEN':raise RuntimeError('Golden mismatch; no timing accepted')
  else:
   assert read(OUT/'gates'/f'{case}.json')['status']=='PASS_BIT_EXACT_ALL_MODEL_AND_DEPLOYMENT_GOLDEN'
  if not args.gates_only:
   samples=[]
   for method in proto['warmup_method_order']:model.run(d,method)
   for trial,order in enumerate(proto['trial_method_orders']):
    for method in order:
     torch.cuda.reset_peak_memory_stats();result=model.run(d,method);m=memory()
     item=dict(case=case,trial=trial,method=method,proposals=len(d['M']),retained=len(result['records']),images=nimages,catalogue_classes=d['B'].shape[1],stages_ns=result['stages_ns'],memory=m,prediction_content_sha256=hashlib.sha256(json.dumps(result['records'],sort_keys=True,separators=(',',':')).encode()).hexdigest(),input_load_seconds_excluded=input_load_seconds,model_load_seconds_excluded=load_seconds)
     samples.append(item);save(OUT/'results'/f'quality_{case}_trial{trial}_{method}.json',item);print('MEASURE',case,trial,method,result['stages_ns']['total']/1e9,flush=True)
   summary={}
   for method in ['strong','pair']:
    values=np.asarray([s['stages_ns']['total']/1e9 for s in samples if s['method']==method]);summary[method]=dict(cohort_seconds=values.tolist(),mean_seconds=float(values.mean()),median_seconds=float(np.median(values)),p90_seconds=float(np.quantile(values,.9)),sample_SD_seconds=float(values.std(ddof=1)),mean_ms_per_image=float(values.mean()*1000/nimages),peak_process_RSS_bytes=max(s['memory']['peak_process_RSS_bytes']for s in samples if s['method']==method),peak_cuda_allocated_bytes=max(s['memory']['peak_cuda_allocated_bytes']for s in samples if s['method']==method),peak_cuda_reserved_bytes=max(s['memory']['peak_cuda_reserved_bytes']for s in samples if s['method']==method))
   save(OUT/'results'/f'quality_{case}_summary.json',dict(case=case,boundary=proto['boundary'],summary=summary,trials=3,input_data_loaded_before_warmup=True,proposals=len(d['M']),images=nimages,catalogue_classes=d['B'].shape[1],all_actual_model_forwards=True,full_raw_image_pipeline=False,GT_AP_run=False))
  del d;model.last={};torch.cuda.empty_cache()
 print('DONE',flush=True)
if __name__=='__main__':main()
