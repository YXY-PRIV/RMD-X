"""Batch B fixed nativeV budget: calibrated cue stress, robust q and diagnostics."""
from pathlib import Path
import contextlib,importlib.util,json,sys
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_b_robust_quality')
PSHA='a27a2cc680f96844388b762745147db12ee97a95bc0332913377f7ee1b6bf651'
HELP=CODE.parent/'batch_a_quality_mixture/src/mixture_endpoint.py'
spec=importlib.util.spec_from_file_location('_B_readonly_A_helpers',HELP);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
base=helpers.base;read,sha,ref,verify,digest=helpers.read,helpers.sha,helpers.ref,helpers.verify,helpers.digest
fresh,lock=helpers.fresh,helpers.lock
def frozen():
 pp=CODE/'config/protocol.json';assert sha(pp)==PSHA;p=read(pp);assert p['status'].startswith('FROZEN');return p,ref(pp)
def load_q(ds,variant,recipe,profile,stage,fm):
 p,pr=frozen();assert variant in p['variants']and recipe in [r['id']for r in p['recipes']]
 if stage=='screen':
  assert ds=='CLC_tune';ip=OUT/'screen/inference_index.json';ix=read(ip);assert ix['status']=='COMPLETE'
  rr=[r for r in ix['records']if(r['recipe_id'],r['variant'],r['profile'],r['seed'])==(recipe,variant,profile,17)];assert len(rr)==1;r=rr[0];rp=verify(r['inference_receipt']);qr=r['scores'];meta=read(rp)
  assert meta['protocol_sha256']==PSHA and meta['case']==ds and meta['profile']==profile and meta['variant']==variant and meta['recipe_id']==recipe and meta['seed']==17
 else:
  assert stage in ['confirmed','final'];sel=read(OUT/'parameters/selected_recipe.json');assert sel['selected_recipe_id']==recipe
  rp=(OUT/'ensembles'/ds/variant/'all3/ensemble.json')if profile=='natural'else(OUT/'confirm/ensembles'/ds/variant/profile/'all3/ensemble.json');meta=read(rp);qr=meta['scores']
  assert meta['protocol_sha256']==PSHA and meta['case']==ds and meta['variant']==variant and meta['profile']==profile and meta['recipe_id']==recipe and meta['seeds']==[17,29,43]and meta['selection']==ref(OUT/'parameters/selected_recipe.json')
 assert meta['feature_manifest']==p['cases'][ds]['feature_manifest']and meta['candidate_row_key_sha256']==fm['candidate_row_key_sha256']and meta['output_sha256']==qr['sha256']
 with np.load(verify(qr),allow_pickle=False)as z:q=z['quality'].copy()
 assert q.dtype==np.float32 and q.shape==(fm['shape'][0],5)and np.isfinite(q).all()and np.all((q>=0)&(q<=1))
 return q,dict(scores=qr,receipt=ref(rp),profile=profile,stage=stage,recipe_id=recipe,variant=variant)
def export(ds,variant,recipe,profile,stage,role,tag):
 p,pr=frozen();assert ds in p['cases']and role in p['final_roles'];assert profile in [r['id']for r in p['stress_profiles']]
 if ds!='CLC_tune':assert profile=='natural'and stage=='final'
 if stage in ['screen','confirmed']:assert ds=='CLC_tune'and role=='score_control'
 _,_,_,fm,fmr,src,outs,a,records,cats,protos,v=base.native_source(ds,p['sources']['candidate_protocol']['path'],p['sources']['candidate_root'])
 assert fmr==p['cases'][ds]['feature_manifest'];q,qr=load_q(ds,variant,recipe,profile,stage,fm)
 beta=p['scoring']['primary_beta']if role=='score_control'else p['scoring']['diagnostic_beta'];which,selection=base.select_candidates(v,q,beta);ii=np.arange(len(which));cat=cats[ii,which];sq=q[ii,which];raw=a['raw_score'].astype(np.float64);clipped=np.clip(sq.astype(np.float64),1e-6,1.)
 if role=='label_only':pre=raw.copy();post=pre.copy();rank=np.full(len(raw),-1,np.int32)
 else:pre=np.power(raw,.5)*np.power(clipped,.5);post,rank=base.gaussian(pre,a['bbox'],a['image_id'],cat,a['global_cache_index'])
 ident=dict(dataset=ds,variant=variant,recipe_id=recipe,profile=profile,stage=stage,role=role,case=tag,beta=beta,protocol=pr,feature_manifest=fmr,quality=qr,source_prediction=outs['native_predictions.json'],source_trace=outs['source_trace.npz'],module=ref(Path(__file__)),readonly_helpers=ref(HELP),readonly_base=ref(helpers.BASE),raw_detector_score_stressed=False)
 key=digest(ident);dr=OUT/'endpoints'/ds/f'{stage}_{recipe}_{variant}_{profile}_{role}_{key[:16]}';ep=dr/'export.json'
 with lock(OUT/'locks'/('export_'+key+'.lock')):
  if ep.exists():
   e=read(ep);assert e['identity_sha256']==key;verify(e['prediction']);verify(e['score_trace']);return e
  dr.mkdir(parents=True,exist_ok=True);assert not any(dr.iterdir());pp=dr/'predictions.json';fresh(pp,[dict(r,category_id=int(c),score=float(s))for r,c,s in zip(records,cat,post)]);tp=dr/'score_trace.npz'
  with tp.open('xb')as f:np.savez_compressed(f,global_cache_index=a['global_cache_index'],native_category_id=a['category_id'],selected_candidate_rank=which,selected_category_id=cat,candidate_selection_score=selection,selected_quality=sq,clipped_quality=clipped,native_raw_score=raw,pre_soft_score=pre,post_soft_score=post,soft_selection_rank=rank)
  e=dict(status='EXPORTED_ROBUST_QUALITY_ENDPOINT',identity=ident,identity_sha256=key,prediction=ref(pp),score_trace=ref(tp),source_GT=src['GT'],source_params=src['params'],source_native_record=src,prediction_count=len(raw),native_class_changes=int(np.sum(cat!=a['category_id'])),native_membership_bbox_image_order_unchanged=True,new_category_UNC_recomputed=False,q_input_stress_only=True)
  fresh(ep,e);return e
def evaluate(ep):
 p,pr=frozen();ep=Path(ep).resolve();assert ep.is_relative_to(OUT);e=read(ep);assert e['identity']['protocol']==pr;verify(e['prediction']);verify(e['score_trace']);mp=ep.parent/'metrics.json';sources=base.pure._native_sources()
 if mp.exists():
  m=read(mp);assert m['export_sha256']==sha(ep);return m
 gtpath=verify(e['source_GT']);gt=read(gtpath);ids=[int(a['id'])for a in gt['annotations']];assert len(set(ids))==len(ids)and min(ids)>0;params=base.pure._params(gt);assert params==e['source_params']
 key=helpers.evaluation_key(e['prediction']['sha256'],e['source_GT']['sha256'],params,sources);entry=OUT/'evaluation_registry'/(key+'.json')
 with lock(OUT/'locks'/('eval_'+key+'.lock')):
  candidates=[]
  if entry.exists():candidates.append(verify(read(entry)['actual_metrics']))
  native=e['source_native_record'].get('metrics_ref')
  if native:candidates.append(verify(native))
  reused=None
  for cp in candidates:
   reused=helpers.actual_chain(cp,e['prediction']['sha256'],e['source_GT']['sha256'],params,sources)
   if reused is not None:break
  if reused:
   old=read(reused['metrics']['path']);m=dict(stats=old['stats'],AP=old['AP'],AP50=old['AP50'],AP75=old['AP75'],native_log=old['native_log'],AP_reused=True,AP_computed_here=False,actual_native_evaluation=False,reused_source=reused)
  else:
   from pycocotools.coco import COCO
   from pycocotools.cocoeval import COCOeval
   log=ep.parent/'native_COCO.log'
   with helpers.host_native_slot(p['resources']['native_lock_directory']),log.open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
    cg=COCO(str(gtpath));cd=cg.loadRes(e['prediction']['path'])if e['prediction_count']else COCO()
    if not e['prediction_count']:cd.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);cd.createIndex()
    ev=COCOeval(cg,cd,'bbox')
    for k in ['imgIds','catIds','maxDets','areaRng','areaRngLbl','useCats']:setattr(ev.params,k,params[k])
    ev.params.iouThrs=np.asarray(params['iouThrs'],np.float64);ev.params.recThrs=np.asarray(params['recThrs'],np.float64);ev.evaluate();ev.accumulate();ev.summarize()
   st=[float(x)for x in ev.stats];m=dict(stats=st,AP=st[0],AP50=st[1],AP75=st[2],native_log=ref(log),AP_reused=False,AP_computed_here=True,actual_native_evaluation=True,reused_source=None)
  m.update(export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=e['source_GT'],params=params,**sources,AR100=m['stats'][8],GT_access_purpose='Calibration-only recipe screening'if e['identity']['dataset']=='CLC_tune'else'Frozen test evaluation only')
  fresh(mp,m)
  if not entry.exists():fresh(entry,dict(key=key,actual_metrics=reused['metrics']if reused else ref(mp)))
  return m
