"""Fixed-nativeV quality mixing with shared host slots and exact AP reuse."""
from __future__ import annotations
import contextlib,datetime,fcntl,hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_a_quality_mixture')
PSHA='1d905cea6ee8217a28330aeeef83dedced832fbab2058c26d27c4fddfcc18fe9'
BASE=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py')
BSHA='9d818252fcf47334ea2019c4cb0466b6c6244924418d041b0ee9c9644c863fac'
spec=importlib.util.spec_from_file_location('_mixture_readonly_helpers',BASE);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,sha,ref,verify,digest=base.read,base.sha,base.ref,base.verify,base.digest
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fresh(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def atomic(path,x):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);t=path.with_name(path.name+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(path)
@contextlib.contextmanager
def lock(path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a')as f:
  fcntl.flock(f,fcntl.LOCK_EX)
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
def frozen(protocol=CODE/'config/protocol.json'):
 p=Path(protocol).resolve()
 if sha(p)!=PSHA or sha(BASE)!=BSHA:raise ValueError('Frozen mixture protocol or old pure helper changed')
 j=read(p)
 if j['output_root']!=str(OUT)or not j['status'].startswith('FROZEN'):raise ValueError('Wrong new output root or protocol status')
 return j,ref(p)
def evaluation_key(pred,gt,params,sources):return digest(dict(prediction=pred,GT=gt,params=params,sources={k:r['sha256']for k,r in sources.items()}))
def prepare(protocol=CODE/'config/protocol.json'):
 p,pr=frozen(protocol);sp=OUT/'source_index.json'
 with lock(OUT/'locks/prepare.lock'):
  if sp.exists():
   old=read(sp)
   if old['status']!='COMPLETE'or old['protocol']!=pr:raise ValueError('Prepared source identity mismatch')
   return old
  proof=read(verify(p['source_guard_tune_equivalence']))
  if not proof['status'].startswith('PASS'):raise ValueError('Guarded/original tune candidate equality missing')
  cases={}
  for case,c in p['cases'].items():
   fm=read(verify(c['feature_manifest']));status=read(verify(c['status']))
   if status['status']!='COMPLETE'or status['feature_manifest']!=c['feature_manifest']:raise ValueError('Sealed nativeV source not complete')
   src=fm['source_native_record'];outs=fm['outputs'];records=read(verify(c['native_prediction']))
   if outs['native_predictions.json']!=c['native_prediction']or outs['source_trace.npz']!=c['source_trace']or src['GT']!=c['GT']or src['params']!=c['source_params']:raise ValueError('Frozen source native record changed')
   if src['prediction']['sha256']!=c['native_prediction']['sha256']:raise ValueError('Native original prediction bytes differ')
   with np.load(verify(c['source_trace']),allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
   with np.load(verify(src['source_trace']),allow_pickle=False)as z:original={k:z[k].copy()for k in z.files}
   N=len(records);rows=a['global_cache_index']
   for k in ['global_cache_index','image_id','category_id','raw_score','bbox']:
    if not np.array_equal(a[k],original[k]):raise ValueError('Old/new native trace differs '+case+'/'+k)
   if rows.dtype!=np.int64 or len(rows)!=N or len(np.unique(rows))!=N or not np.array_equal(a['source_detection_index'],np.arange(N)):raise ValueError('Native row identity malformed')
   if any(r['image_id']!=int(a['image_id'][i])or r['category_id']!=int(a['category_id'][i])or r['bbox']!=a['bbox'][i].tolist()or float(r['score']).hex()!=float(a['raw_score'][i]).hex()for i,r in enumerate(records)):raise ValueError('Native frozen image/class/bbox/rawscore mismatches')
   components=[];receipts=[]
   for family in p['family_order']:
    d=c['quality_sources'][family];qref=d['scores'];rref=d['receipt'];receipt=read(verify(rref));qp=verify(qref)
    with np.load(qp,allow_pickle=False)as z:q=z['quality'].copy()
    if q.dtype!=np.float32 or q.ndim!=2 or not np.isfinite(q).all()or np.any((q<0)|(q>1)):raise ValueError('Bad source quality')
    if receipt.get('output_sha256',receipt.get('output',{}).get('sha256'))!=qref['sha256']or receipt.get('seeds')!=[17,29,43]:raise ValueError('Source all3 receipt differs')
    if family!='candidate_visual':
     key=receipt.get('global_row_key_sha256',receipt.get('row_key_sha256'))
     if key!=c['source_global_row_key_sha256']or q.shape[1]!=2 or rows.max()>=len(q):raise ValueError('Old global quality key/head mismatch')
     value=q[rows,0]
    else:
     if receipt['candidate_row_key_sha256']!=c['candidate_row_key_sha256']or receipt['variant']!='visual'or q.shape!=(N,5):raise ValueError('Candidate visual row/rank0 mismatch')
     value=q[:,0]
    components.append(value);receipts.append(dict(family=family,scores=qref,receipt=rref))
   values=np.stack(components,axis=1).astype(np.float32);directory=OUT/'aligned'/case;directory.mkdir(parents=True,exist_ok=True);qp=directory/'quality_components.npz'
   with qp.open('xb')as f:np.savez_compressed(f,quality=values,global_cache_index=rows)
   ar=dict(status='COMPLETE_ALIGNED_READONLY_QUALITY_COMPONENTS',case=case,protocol=pr,feature_manifest=c['feature_manifest'],source_native_record=src,source_trace=c['source_trace'],native_prediction=c['native_prediction'],GT=c['GT'],params=c['source_params'],source_global_row_key_sha256=c['source_global_row_key_sha256'],candidate_row_key_sha256=c['candidate_row_key_sha256'],shape=list(values.shape),family_order=p['family_order'],quality_inputs=receipts,output=ref(qp),rawscore_bbox_class_image_order_exact=True,source_GT_annotations_parsed=False,new_GPU_inference=False,new_training=False)
   ap=directory/'alignment.json';fresh(ap,ar);cases[case]=dict(alignment=ref(ap),qualities=ref(qp),rows=N)
  sources=base.pure._native_sources();registry={};skipped=0
  for r in p['reuse']['metric_sources']:
   mp=verify(r);m=read(mp)
   if not m.get('AP_computed_here',m.get('actual_native_evaluation',False))or m.get('AP_reused',False):skipped+=1;continue
   needed=['prediction','original_GT','params','native_log','stats','AP','AP50','AP75']
   if not all(k in m for k in needed)or any(k not in m or m[k]['sha256']!=v['sha256']for k,v in sources.items()):skipped+=1;continue
   key=evaluation_key(m['prediction']['sha256'],m['original_GT']['sha256'],m['params'],sources)
   registry.setdefault(key,r)
  rp=OUT/'prior_actual_registry.json';fresh(rp,dict(status='CANDIDATE_ACTUAL_POINTS_PAYLOAD_VERIFIED_WHEN_MATCHED',protocol=pr,records=registry,registered=len(registry),skipped_nonactual_or_incompatible=skipped,old_metric_refs_verified=True,prediction_GT_and_log_payload_verified_when_exact_key_matches=True,native_sources=sources))
  result=dict(status='COMPLETE',created_utc=now(),protocol=pr,cases=cases,prior_actual_registry=ref(rp),new_AP=0,new_GPU_inference=0,new_training=0)
  fresh(sp,result);return result
def export(dataset,weights,phase,protocol=CODE/'config/protocol.json'):
 p,pr=frozen(protocol);weights=[float(x)for x in weights]
 if dataset not in p['cases']or weights not in p['weights_grid']:raise ValueError('Undeclared dataset/weights')
 idx=read(OUT/'source_index.json')
 if idx['status']!='COMPLETE'or idx['protocol']!=pr:raise ValueError('Aligned source publication incomplete')
 ainfo=read(verify(idx['cases'][dataset]['alignment']));c=p['cases'][dataset];records=read(verify(c['native_prediction']))
 with np.load(verify(c['source_trace']),allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 with np.load(verify(ainfo['output']),allow_pickle=False)as z:components=z['quality'].copy();rows=z['global_cache_index'].copy()
 if not np.array_equal(rows,a['global_cache_index']):raise ValueError('Alignment global rows changed')
 q=np.zeros(len(rows),np.float64)
 for j,w in enumerate(weights):q+=w*components[:,j].astype(np.float64)
 q=q.astype(np.float32);clipped=np.clip(q.astype(np.float64),1e-6,1.);pre=np.power(a['raw_score'],.5)*np.power(clipped,.5);post,rank=base.gaussian(pre,a['bbox'],a['image_id'],a['category_id'],rows)
 identity=dict(dataset=dataset,split='tune'if dataset=='CLC_tune'else'eval',phase=phase,weights=weights,family_order=p['family_order'],protocol=pr,aligned_source=idx['cases'][dataset]['alignment'],original_feature_manifest=c['feature_manifest'],native_prediction=c['native_prediction'],native_source_trace=c['source_trace'],exporter=ref(__file__),readonly_gaussian_and_coco_helpers=ref(BASE),alpha=.5,sigma=1.,category_changes=False)
 key=digest(identity);tag='_'.join(format(w,'.2f').replace('.','p')for w in weights);directory=OUT/'endpoints'/dataset/f'{phase}_w{tag}_{key[:16]}';ep=directory/'export.json'
 with lock(OUT/'locks'/('export_'+key+'.lock')):
  if ep.exists():
   e=read(ep)
   if e['identity_sha256']!=key:raise ValueError('Export resume differs')
   verify(e['prediction']);verify(e['score_trace']);return e
  directory.mkdir(parents=True,exist_ok=True)
  if any(directory.iterdir()):raise FileExistsError('Unsealed new export')
  pp=directory/'predictions.json';fresh(pp,[dict(r,score=float(s))for r,s in zip(records,post)]);tp=directory/'score_trace.npz'
  with tp.open('xb')as f:np.savez_compressed(f,global_cache_index=rows,native_raw_score=a['raw_score'],component_quality=components,weights=np.asarray(weights,np.float64),mixed_quality=q,clipped_quality=clipped,pre_soft_score=pre,post_soft_score=post,soft_selection_rank=rank)
  e=dict(status='EXPORTED_FIXED_NATIVEV_QUALITY_MIXTURE',created_utc=now(),identity=identity,identity_sha256=key,prediction=ref(pp),score_trace=ref(tp),source_GT=c['GT'],source_params=c['source_params'],prediction_count=len(records),active_families=sum(w>0 for w in weights),active_model_count=3*sum(w>0 for w in weights),native_membership_category_bbox_image_order_unchanged=True,GT_used_for_export=False)
  fresh(ep,e);return e
@contextlib.contextmanager
def host_native_slot(directory):
 directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);chosen=None
 try:
  while chosen is None:
   for i in range(4):
    f=(directory/f'slot{i}.lock').open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);chosen=f;break
    except BlockingIOError:f.close()
   if chosen is None:time.sleep(.2)
  yield
 finally:
  if chosen is not None:fcntl.flock(chosen,fcntl.LOCK_UN);chosen.close()
def actual_chain(path,pred,gt,params,sources,seen=None):
 seen=set()if seen is None else seen;path=Path(path).resolve()
 if str(path)in seen:return None
 seen.add(str(path));m=read(path);found=base.pure._completed_actual_source(path,pred,gt,params,sources)
 if found is not None:return found
 if m.get('AP_reused')and m.get('reused_source',{}).get('metrics'):return actual_chain(verify(m['reused_source']['metrics']),pred,gt,params,sources,seen)
 return None
def evaluate(ep,protocol=CODE/'config/protocol.json'):
 p,pr=frozen(protocol);ep=Path(ep).resolve()
 if not ep.is_relative_to(OUT):raise ValueError('Only new A outputs may be evaluated')
 e=read(ep);verify(e['prediction']);verify(e['score_trace']);mp=ep.parent/'metrics.json';sources=base.pure._native_sources()
 if mp.exists():
  m=read(mp)
  if m['export_sha256']!=sha(ep):raise ValueError('Metric resume differs')
  return m
 # Annotation parsing is confined to actual evaluation, never source alignment.
 gtpath=verify(e['source_GT']);gt=read(gtpath);ids=[int(r['id'])for r in gt['annotations']]
 if len(ids)!=len(set(ids))or any(i<=0 for i in ids):raise ValueError('GT annotation IDs must be unique positive')
 params=base.pure._params(gt)
 if params!=e['source_params']:raise ValueError('Source COCO params changed')
 key=evaluation_key(e['prediction']['sha256'],e['source_GT']['sha256'],params,sources);entry=OUT/'evaluation_registry'/(key+'.json')
 with lock(OUT/'locks'/('evaluate_'+key+'.lock')):
  candidates=[]
  if entry.exists():candidates.append(verify(read(entry)['actual_metrics']))
  old=read(OUT/'prior_actual_registry.json')['records'].get(key)
  if old:candidates.append(verify(old))
  reused=None
  for candidate in candidates:
   reused=actual_chain(candidate,e['prediction']['sha256'],e['source_GT']['sha256'],params,sources)
   if reused is not None:break
  if reused:
   original=read(reused['metrics']['path']);m=dict(stats=original['stats'],AP=original['AP'],AP50=original['AP50'],AP75=original['AP75'],native_log=original['native_log'],AP_reused=True,AP_computed_here=False,actual_AP_computed_here=False,actual_native_evaluation=False,original_actual_native_evaluation=True,reused_source=reused)
  else:
   from pycocotools.coco import COCO
   from pycocotools.cocoeval import COCOeval
   log=ep.parent/'native_COCO.log'
   with host_native_slot(p['resource']['shared_native_slot_directory']),log.open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    cg=COCO(str(gtpath))
    if e['prediction_count']:cd=cg.loadRes(e['prediction']['path'])
    else:cd=COCO();cd.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);cd.createIndex()
    ev=COCOeval(cg,cd,'bbox')
    for k in ['imgIds','catIds','maxDets','areaRng','areaRngLbl','useCats']:setattr(ev.params,k,params[k])
    ev.params.iouThrs=np.asarray(params['iouThrs'],np.float64);ev.params.recThrs=np.asarray(params['recThrs'],np.float64);ev.evaluate();ev.accumulate();ev.summarize()
   stats=[float(x)for x in ev.stats];m=dict(stats=stats,AP=stats[0],AP50=stats[1],AP75=stats[2],native_log=ref(log),AP_reused=False,AP_computed_here=True,actual_AP_computed_here=True,actual_native_evaluation=True,reused_source=None)
  m.update(created_utc=now(),export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=e['source_GT'],params=params,**sources,reuse_key=key,GT_access_purpose='CLC tune only weights selection'if e['identity']['dataset']=='CLC_tune'else'Frozen test evaluation only; no per-test selection',native_membership_category_bbox_image_order_unchanged=True)
  fresh(mp,m)
  if not entry.exists():fresh(entry,dict(key=key,actual_metrics=reused['metrics']if reused else ref(mp)))
  return m
