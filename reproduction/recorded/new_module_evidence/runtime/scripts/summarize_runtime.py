"""Summarize only complete, golden-verified genuine runtime groups."""
from pathlib import Path
import json,csv,hashlib,datetime
import numpy as np
R=Path('/mnt/data/yxy/program/RAXO');C=R/'experiments/new_module_evidence_20261102/runtime';O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/runtime');CASES=['DET','CLCXray_test','PIXray','PIDray','DvXray','Hi']
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def save(p,j):Path(p).write_text(json.dumps(j,indent=2,allow_nan=False)+'\n')
def stats(v):
 v=np.asarray(v,np.float64);return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),p90=float(np.quantile(v,.9)),sample_SD=float(v.std(ddof=1)),minimum=float(v.min()),maximum=float(v.max()))
P=read(R/'experiments/pair_conflict_material_projection_20261101/config/protocol.json')
run=read('/mnt/data/yxy-codex/home/outputs/RAXO/pair_conflict_material_projection_20261101/run_index.json')
quality=[];quality_checks=[]
for case in CASES:
 s=read(O/'results'/f'quality_{case}_summary.json');assert read(O/'gates'/f'{case}.json')['status']=='PASS_BIT_EXACT_ALL_MODEL_AND_DEPLOYMENT_GOLDEN'
 f=read(P['cases'][case]['C_feature_manifest']['path']);universe=len(f['source_native_records']['V']['params']['imgIds'])
 row=next(r for r in run['endpoints']if r['dataset']==case and r['variant']=='material'and r['route']=='anchor_locked'and r['residual_alpha']==1.)
 e=read(row['export_receipt']['path']);base=read(e['identity']['quality']['strong_anchor_export']['path'])
 for method,predref in [('pair',e['prediction']),('strong',base['prediction'])]:
  expected=hashlib.sha256(json.dumps(read(predref['path']),sort_keys=True,separators=(',',':')).encode()).hexdigest()
  for i in range(3):
   q=read(O/'results'/f'quality_{case}_trial{i}_{method}.json');assert q['prediction_content_sha256']==expected,(case,method,i)
   quality_checks.append(dict(case=case,method=method,trial=i,exact_prediction_content=True,source_prediction=predref))
 quality.append(dict(case=case,proposals=s['proposals'],nonempty_union_images=s['images'],full_COCO_image_universe=universe,catalogue_classes=s['catalogue_classes'],strong_cohort_mean_seconds=s['summary']['strong']['mean_seconds'],pair_cohort_mean_seconds=s['summary']['pair']['mean_seconds'],strong_cohort_SD_seconds=s['summary']['strong']['sample_SD_seconds'],pair_cohort_SD_seconds=s['summary']['pair']['sample_SD_seconds'],strong_amortized_ms_per_nonempty_union_image=s['summary']['strong']['mean_ms_per_image'],pair_amortized_ms_per_nonempty_union_image=s['summary']['pair']['mean_ms_per_image'],delta_amortized_ms_per_nonempty_union_image=s['summary']['pair']['mean_ms_per_image']-s['summary']['strong']['mean_ms_per_image'],effective_pair_parameters=8322,physical_common_resident_parameters=12387))
save(O/'gates/quality_all_trial_prediction_golden.json',dict(status='PASS',checks=quality_checks,all36exact=True,scope='Every timed complete-quality prediction content exactly equals current sealed deployment or strongbaseline;noAPrun'))
with (O/'results/quality_summary.csv').open('w')as f:w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
assert read(O/'gates/raw_compatibility_review.json')['status']=='PASS'
assert read(O/'gates/raw_all_trial_prediction_golden.json')['status']=='PASS'
proto=read(C/'config/raw_protocol.json');groups=[]
for g in proto['groups']:
 d=O/'raw_trials'/g['id'];s=read(d/'summary.json');assert s['status']=='PASS_FRESH_PROCESS_RAW_TO_FINAL_INFERENCE'and s['actual_worker_exit']==0 and not s['compatibility_only'];groups.append(dict(group=g,summary=s,startup=read(d/'startup.json'),raw=read(d/'raw_samples.json')))
raw={}
for method in ['strong','pair']:
 a=[x for x in groups if x['group']['method']==method];v=[r['request_wall_ns']/1e6 for x in a for r in x['raw']]
 raw[method]=dict(request_wall_ms=stats(v),trial_cohort_mean_ms=[x['summary']['mean_ms']for x in a],fresh_process_startup_seconds=[x['startup']['process_to_ready_wall_ns']/1e9 for x in a],maximum_process_group_RSS_bytes=max(x['summary']['process_group_peak_RSS_bytes']for x in a),maximum_concurrent_process_group_NVML_GPU_bytes=max(x['summary']['process_group_peak_NVML_GPU_bytes']for x in a),unique_quality_parameters=7236 if method=='strong'else 8322,actual_raw_requests=len(v),warmup_requests_excluded=12)
 bystage={}
 for stage in sorted({stage for x in a for rr in x['raw']for stage in rr['stages_ns']}):bystage[stage]=stats([rr['stages_ns'].get(stage,0)/1e6 for x in a for rr in x['raw']])
 raw[method]['stage_diagnostic_ms']=bystage
 intervals=[]
 for x in a:
  samples=read(O/'raw_trials'/x['group']['id']/'process_memory_samples.json')
  intervals.extend(np.diff([sample['monotonic_ns']for sample in samples]).astype(np.float64)/1e6)
 raw[method]['actual_memory_sampling_interval_ms']=stats(intervals) if intervals else None
 raw[method]['nominal_memory_monitor_sleep_ms']=50
pair_deltas=[]
for trial in range(3):
 a=next(x for x in groups if x['group']==dict(id=f'trial{trial}_strong',method='strong',trial=trial));b=next(x for x in groups if x['group']==dict(id=f'trial{trial}_pair',method='pair',trial=trial))
 for x,y in zip(a['raw'],b['raw']):assert x['image_id']==y['image_id'];pair_deltas.append((y['request_wall_ns']-x['request_wall_ns'])/1e6)
summary=dict(summary_source=ref(Path(__file__)),status='COMPLETE_ACTUAL_FULL_RAW_AND_FULL_QUALITY_RUNTIME_WITH_GOLDENS',completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),raw_protocol=ref(C/'config/raw_protocol.json'),raw_code_freeze=ref(O/'automation/raw_code_freeze.json'),quality_protocol=ref(C/'config/quality_protocol.json'),quality_code_freeze=ref(O/'automation/quality_code_freeze.json'),quality_scope_clarification=ref(O/'automation/quality_scope_clarification.json'),quality=quality,raw=raw,paired_request_delta_ms=stats(pair_deltas),raw_independent_golden=ref(O/'gates/raw_compatibility_review.json'),raw_all_trial_prediction_golden=ref(O/'gates/raw_all_trial_prediction_golden.json'),quality_all_trial_prediction_golden=ref(O/'gates/quality_all_trial_prediction_golden.json'),measured_groups=dict(quality=36,raw=6,raw_requests=192,raw_compatibility_unmeasured_requests=32),scope_limits=['Rawcosts32fixedCLC12classnewrawdeploymentworkload;notmaincachedAPidentityorall6rawlatency','Cachedall6completequalitycohortbatchedamortizationdenominatornonemptyunionimages;notrequestlatency','Old83timingsnotreusedorsummedtoobtainlatestcompletecost','HostCImemorytrafficcoexisted;CPUcoresandGPU0separate','ProcessgroupRSSandNVMLpeaksareSAMPLEDsteadyinferenceafterwarmup,notguaranteedtransientmaxima;individualtorchCUDAhighwatersexact;modelcoldloadwallseparatewithwarmfilesystempagecache','Cachedbaselinecommonloaded12387physicalweightsdupunusedanchor;rawmethodspecific7236vs8322uniqueweights','Offlineoriginalsupportbankconstructionnotrepeatedormeasured'],new_training=0,new_AP=0)
save(O/'results/runtime_summary.json',summary)  # Final completion is sealed only after report and provenance manifest.
print(json.dumps(dict(status=summary['status'],raw=raw,delta=summary['paired_request_delta_ms']),indent=2))
