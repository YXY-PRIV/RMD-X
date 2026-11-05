#!/usr/bin/env python3
"""Frozen saved-correction 2x2 ablation, no fitting or new model inference."""
from pathlib import Path
import os, sys, json, hashlib, datetime, contextlib, fcntl, time
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ablation')
P5CODE=CODE.parents[1]/'pair_conflict_material_projection_20261101'
sys.path.insert(0,str(P5CODE))
from src import pair_endpoint as pe
from src import pair_projection as pp
from src import anchor_reference as ar
SEEDS=[17,29,43]
CELLS=['00_full','10_gate','01_projection_identity_extension','11_both']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def ref(p):return pe.ref(p)
def verify(r):return pe.verify(r)
def save(p,v):
 p=Path(p).resolve(); assert p.is_relative_to(OUT); p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def bitdiff(a,b):assert a.dtype==b.dtype==np.float32;return int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32)))
def frozen():
 P=read(CODE/'protocol.json');pr=ref(CODE/'protocol.json');f=read(OUT/'automation/code_freeze.json')
 assert f['protocol']==pr
 for r in f['sources']+f['readonly_sources']:verify(r)
 assert P['status']=='FROZEN_BEFORE_NEW_ABLATION_EXPORT_OR_AP'
 assert os.environ['RAXO_PROTOCOL_SHA256']==P['latest_protocol']['sha256']
 assert P['cells']==CELLS and P['alpha']==1.0 and P['route']=='anchor_locked'
 return P,pr

def build(case):
 import torch
 torch.set_num_threads(8)
 P,pr=frozen();old,opr,F,Fref,a,C,M,mapping,records=pe.source(case,P['latest_protocol']['path'])
 assert opr==P['latest_protocol'];anchor=ar.rebuild(case,old)
 assert bits(anchor['valid'],M)
 c=old['cases'][case];nf=read(verify(c['auxiliary_feature_manifest']))
 ci=np.load(verify(nf['outputs']['candidate_class_order_index.npy']),allow_pickle=False)
 cats=np.load(verify(nf['outputs']['candidate_category_id.npy']),allow_pickle=False)
 T=np.load(verify(nf['outputs']['supported_reference_T.npy']),allow_pickle=False)
 qv=np.load(verify(nf['outputs']['query_valid.npy']),allow_pickle=False)
 has=np.load(verify(nf['outputs']['actual_supported_class_mask.npy']),allow_pickle=False)
 lookup=pp.source_records(old)
 heads={x:[]for x in CELLS};inputs=[];cellstats={x:[]for x in CELLS};activity=None;seed_corrections={x:[]for x in CELLS}
 for seed in SEEDS:
  rec=lookup[(case,'material',seed)];z=pe.loadnpz(verify(rec['components']));nr=read(verify(rec['receipt']))
  assert nr['checkpoint']==old['native_models']['material'][str(seed)]
  assert bits(z['anchor_logits'],anchor['seed_logits'][seed]) and bits(z['valid'],M)
  assert bits(z['supported_count'],T) and bits(z['candidate_class_order_index'],ci)
  delta=z['centered_correction'];assert delta.dtype==np.float32 and delta.shape==M.shape and np.isfinite(delta).all()
  supported=(z['availability']==1)&M&(T[:,None]>1)
  assert not np.any(delta[~supported])
  projected,A,D,stats=pp.project(delta,M,z['availability'],T,ci,cats,mapping,a['branch_kept'],a['branch_category_id'])
  old_projection=pe.loadnpz(pe.OUT/'projection'/case/'material'/f'seed{seed}'/'components.npz')
  assert bits(projected,old_projection['projected_correction'])and bits(A,old_projection['active'])
  if activity is None:activity=A
  else:assert bits(activity,A)
  corrections={CELLS[0]:delta.copy(),CELLS[1]:np.where(A[:,None],delta,np.float32(0)),CELLS[2]:np.where(A[:,None],projected,delta),CELLS[3]:projected.copy()}
  assert bits(corrections[CELLS[1]][~A],np.zeros_like(delta[~A]))
  assert bits(corrections[CELLS[2]][~A],delta[~A])and bits(corrections[CELLS[2]][A],projected[A])
  assert bits(corrections[CELLS[3]][A],corrections[CELLS[2]][A])
  for cell,dz in corrections.items():
   assert dz.dtype==np.float32 and not np.any(dz[~M])and not np.any(dz[~supported])
   q=torch.sigmoid(torch.from_numpy(z['anchor_logits'])+np.float32(1.)*torch.from_numpy(dz)).numpy().astype(np.float32)
   exact=(dz==0.)|~supported;q[exact]=anchor['seed_quality'][seed][exact]
   pe.check_quality(q,M);heads[cell].append(q.astype(np.float64))
   if cell.startswith('1'):assert bits(q[~A],anchor['seed_quality'][seed][~A])
   cellstats[cell].append(dict(seed=seed,nonzero_residual_tuples=int(np.count_nonzero(dz)),nonzero_residual_proposals=int(np.count_nonzero(np.any(dz!=0,axis=1))),correction_array_sha256=hashlib.sha256(dz.tobytes()).hexdigest(),exact_fallback_candidate_cells=int(np.count_nonzero(exact&M)),valid_correction_pair_sum_max_abs=float(np.abs(dz[A].astype(np.float64).sum(axis=1)).max(initial=0))))
   seed_corrections[cell].append(hashlib.sha256(dz.tobytes()).hexdigest())
  inputs.append(dict(seed=seed,source_components=rec['components'],source_receipt=rec['receipt'],checkpoint=nr['checkpoint'],P5_projection=ref(pe.OUT/'projection'/case/'material'/f'seed{seed}'/'components.npz')))
 A=activity;both=a['branch_kept'].all(1);different=both&(a['branch_category_id'][:,0]!=a['branch_category_id'][:,1]);same=both&~different;single=a['branch_kept'].sum(1)==1
 assert np.all(a['branch_kept'].any(1))
 # Exclusion reason partition has explicit priority and therefore sums to N.
 reasons=dict(same_class_both=int(same.sum()),single_branch=int(single.sum()),different_both_invalid_query_or_Tle1=int(np.sum(different&(T<=1))),different_both_Tgt1_unsupported_pair=int(np.sum(different&(T>1)&~A)),active_supported_conflict=int(A.sum()))
 assert sum(reasons.values())==len(A)
 q0=anchor['mixed'];reference=anchor['reference_selected_branch'];anchor_eff=pe.branch_quality(q0,a['branch_kept'],mapping)
 assert bits(reference,pe.choose(a['branch_kept'],anchor_eff))
 rows=np.flatnonzero(reference>=0);branch=reference[rows].astype(np.int64);slot=mapping[rows,branch];category=a['branch_category_id'][rows,branch]
 assert len(rows)==len(M)
 common=dict(protocol=pr,case=case,alpha=1.,route='anchor_locked',nativeF_lambda_m=.05,nativeF_unc_threshold=.15,
  source_P5_protocol=P['latest_protocol'],source_C_manifest=Fref,source_auxiliary_manifest=c['auxiliary_feature_manifest'],candidate_row_key_sha256=F['candidate_row_key_sha256'],
  formal_anchor_report=anchor['report'],source_components=inputs,mask_partition=reasons,active_mask_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
  original_branch_class_box_rawscore_fixed=True,new_training=0,new_GPU=0,new_model_forward=0,GT_read_for_export=False)
 result=[]
 for cell in CELLS:
  head=np.mean(np.stack(heads[cell]),axis=0).astype(np.float32);q=pe.mix_visual(anchor['base'],head,M);pe.check_quality(q,M)
  if cell.startswith('1'):assert bits(q[~A],q0[~A])
  eff=pe.branch_quality(q,a['branch_kept'],mapping);assert np.array_equal(eff[same,0],eff[same,1])
  sq=q[rows,slot];raw=a['raw_score'][rows].astype(np.float64);clipped=np.clip(sq.astype(np.float64),1e-6,1.);pre=np.power(raw,.5)*np.power(clipped,.5)
  post,rank=pe.gauss.gaussian(pre,a['bbox'][rows],a['image_id'][rows],category,a['global_cache_index'][rows])
  d=OUT/'endpoints'/case/cell;d.mkdir(parents=True,exist_ok=False)
  pred=d/'predictions.json';save(pred,[dict(image_id=int(a['image_id'][i]),category_id=int(cat),bbox=a['bbox'][i].tolist(),score=float(score))for i,cat,score in zip(rows,category,post)])
  predref=ref(pred);gold=None
  if cell in ['00_full','11_both']:
   ir=read(verify(P['reference_indices']['prior_full'if cell=='00_full'else'latest_pair']))
   matches=[r for r in ir['endpoints']if r['dataset']==case and r['variant']=='material'and r['route']=='anchor_locked'and r['residual_alpha']==1.]
   assert len(matches)==1;gr=matches[0];ge=read(verify(gr['export_receipt']))
   assert predref['sha256']==ge['prediction']['sha256']and predref['bytes']==ge['prediction']['bytes'],(case,cell,'Golden prediction mismatch')
   assert ge['source_GT']==records['V']['GT']and ge['source_params']==records['V']['params']
   oldtrace=pe.loadnpz(verify(ge['score_trace']))
   assert bits(q,oldtrace['full_candidate_quality'])and bits(sq,oldtrace['selected_quality'])
   for key,got in [('selected_branch',branch),('selected_category_id',category),('native_raw_score',raw),('pre_soft_score',pre),('post_soft_score',post)]:assert bits(got,oldtrace[key]),(case,cell,key)
   gold=dict(record=gr,prediction=ge['prediction'],same_prediction_bytes=True,full_candidate_q_bitdiff=0)
  tp=d/'score_trace.npz'
  with tp.open('xb')as f:np.savez_compressed(f,global_cache_index=a['global_cache_index'][rows],selected_union_index=rows,selected_branch=branch,selected_candidate_slot=slot,selected_category_id=category,selected_quality=sq,native_raw_score=raw,pre_soft_score=pre,post_soft_score=post,soft_selection_rank=rank,full_candidate_quality=q,strong_anchor_reference_candidate_quality=q0,active_mask=A)
  identity=dict(**common,cell=cell,cell_seed_statistics=cellstats[cell],mixed_quality_sha256=hashlib.sha256(q.tobytes()).hexdigest(),correction_code=ref(__file__))
  e=dict(status='ABLATION_ENDPOINT_EXPORTED',created_utc=now(),identity=identity,identity_sha256=pe.digest(identity),prediction=predref,score_trace=ref(tp),prediction_count=len(rows),source_GT=records['V']['GT'],source_params=records['V']['params'],golden=gold,
   native_metric_candidates=[gold['record']['metrics']]if gold else [],new_class_created=False,UNC_recomputed=False)
  save(d/'export.json',e);result.append(ref(d/'export.json'))
 save(OUT/'results'/f'{case}_exports.json',dict(status='COMPLETE_FOUR_EXPORTS',case=case,protocol=pr,exports=result,mask_partition=reasons))
 return result

@contextlib.contextmanager
def native_slot():
 root=OUT.parent/'resources/native_slots';root.mkdir(parents=True,exist_ok=True);selected=None
 try:
  while selected is None:
   for i in range(4):
    f=(root/f'slot{i}.lock').open('a')
    try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);selected=f;break
    except BlockingIOError:f.close()
   if selected is None:time.sleep(.2)
  mem=dict(x.split(':',1)for x in Path('/proc/meminfo').read_text().splitlines());assert int(mem['MemAvailable'].strip().split()[0])/(2**20)>=64
  yield
 finally:
  if selected:fcntl.flock(selected,fcntl.LOCK_UN);selected.close()

def evaluate(ep):
 P,pr=frozen();ep=verify(ep);e=read(ep);assert e['identity']['protocol']==pr;verify(e['prediction']);verify(e['score_trace'])
 gtpath=verify(e['source_GT']);gt=read(gtpath);ids=[x['id']for x in gt['annotations']];assert len(ids)==len(set(ids))and min(ids)>0
 params=pe.pure._params(gt);assert params==e['source_params'];sources=pe.pure._native_sources()
 reused=None
 for m in e['native_metric_candidates']:
  reused=pe.actual(verify(m),e['prediction']['sha256'],e['source_GT']['sha256'],params,sources)
  if reused is not None:break
 start=time.monotonic()
 if reused:
  old=read(verify(reused['metrics']));stats=old['stats'];log=old['native_log']
 else:
  from pycocotools.coco import COCO
  from pycocotools.cocoeval import COCOeval
  lp=ep.parent/'native_COCO.log'
  with native_slot(),lp.open('x')as f,contextlib.redirect_stdout(f),contextlib.redirect_stderr(f):
   cg=COCO(str(gtpath));cd=cg.loadRes(e['prediction']['path']);ev=COCOeval(cg,cd,'bbox')
   for k in ['imgIds','catIds','maxDets','areaRng','areaRngLbl','useCats']:setattr(ev.params,k,params[k])
   ev.params.iouThrs=np.asarray(params['iouThrs'],np.float64);ev.params.recThrs=np.asarray(params['recThrs'],np.float64);ev.evaluate();ev.accumulate();ev.summarize()
  stats=[float(x)for x in ev.stats];log=ref(lp)
 assert len(stats)==12 and np.isfinite(stats).all()
 m=dict(status='COMPLETE_NATIVE_METRIC',created_utc=now(),protocol=pr,export=ref(ep),identity_sha256=e['identity_sha256'],stats=stats,AP=stats[0],AP50=stats[1],AP75=stats[2],AR100=stats[8],
  prediction=e['prediction'],original_GT=e['source_GT'],params=params,**sources,native_log=log,AP_computed_here=reused is None,AP_reused=reused is not None,reused_source=reused,elapsed_this_metric_s=time.monotonic()-start,
  reference_equality_requires_exact_prediction_bytes_GT_params_native_libraries=True,GT_access_purpose='fixed posthoc ablation evaluation only; no tuning or method selection')
 save(ep.parent/'metrics.json',m)
 return dict(dataset=e['identity']['case'],cell=e['identity']['cell'],AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],stats=stats,AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'],export=ref(ep),metrics=ref(ep.parent/'metrics.json'))

if __name__=='__main__':
 import argparse
 pa=argparse.ArgumentParser();pa.add_argument('--case',required=True);args=pa.parse_args()
 P,pr=frozen();assert args.case in P['test_cases'];exports=build(args.case);rs=[]
 for ep in exports:
  r=evaluate(ep);rs.append(r);print(json.dumps(r),flush=True)
 save(OUT/'results'/f'{args.case}_results.json',dict(status='COMPLETE_FOUR_ABLATION_CELLS',protocol=pr,case=args.case,results=rs))
