"""Immutable quality rescoring of already accepted native prediction lists.

This module never constructs a new box, label, retention decision or proposal.
Native AP is computed only by evaluate_export, or explicitly reused after exact
ordered-prediction/GT/parameter/native-source binding checks.
"""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import numpy as np

sys.dont_write_bytecode = True
CODE = Path(__file__).resolve().parents[1]
OUT = Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_quality_reranking_20261031')
OLD = Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_reliability_pilot_20261031')
PROTOCOL_SHA = '1efa1c18c607496a786607471efbe8784382b1f9efcb7839336eccb1ddca8643'
ADDENDUM_SHA = 'f61f6d1cb76f956a7b66fc0eeed450d58facc8be5d9de611e377dbb3e20a6426'

def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()
def ref(p):
    p=Path(p).resolve()
    return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def verify(r):
    p=Path(r['path'])
    if not p.is_file() or sha(p)!=r['sha256'] or ('bytes'in r and p.stat().st_size!=r['bytes']):raise ValueError(f'Input hash mismatch: {p}')
    return p
def save_new(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def npz_new(p,**arrays):
    with Path(p).open('xb') as f:np.savez_compressed(f,**arrays)
@contextlib.contextmanager
def file_lock(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def frozen(protocol):
    p=Path(protocol).resolve()
    if sha(p)!=PROTOCOL_SHA:raise ValueError('Protocol differs from frozen quality experiment')
    a=CODE/'config/addendum_01_exact_prediction_reuse.json'
    if sha(a)!=ADDENDUM_SHA:raise ValueError('Exact evaluation reuse addendum changed')
    return read(p),ref(p),ref(a)
def checked_root(root):
    root=Path(root).resolve()
    if root!=OUT:raise ValueError('Only the authorized new quality output root is writable')
    root.mkdir(parents=True,exist_ok=True)
    return root

def _metadata_rows(dataset):
    return [json.loads(line) for line in (OLD/'features'/dataset/'rows.jsonl').read_text().splitlines() if line.strip()]

def _source_trace(dataset,split,branch,export,root,rows,decisions):
    e=read(export);verify(e['prediction']);verify(e['state'])
    manifest=read(verify(e['identity']['manifest']))
    mapping={}
    if 'image_id_map' in manifest['assets']:
        mapping={int(k):int(v) for k,v in read(verify(manifest['assets']['image_id_map'])).items()}
    preds=read(e['prediction']['path'])
    with np.load(e['state']['path'],allow_pickle=False) as z:
        idx=z['global_cache_index'];kept=z['after_unc'];selected=idx[kept]
        suffix='visual' if branch=='visual' else 'fused'
        expected_lambda=np.float32(0. if branch=='visual' else .05)
        checks={'fixed_lambda_all_rows_exact':bool(np.all(z['effective_lambda']==expected_lambda)),
                'winner_exact':np.array_equal(z['winner'],decisions[suffix+'_winner'][idx]),
                'category_exact':np.array_equal(z['category'],decisions[suffix+'_category'][idx]),
                'negative_exact':np.array_equal(z['negative'],decisions[suffix+'_negative'][idx]),
                'kept_exact':np.array_equal(kept,decisions[suffix+'_kept'][idx]),
                'one_output_per_retained_row':len(preds)==len(selected),
                'unique_trace_rows':len(np.unique(selected))==len(selected)}
        out_categories=z['category'][kept]
    fields_ok=True;score_ok=True
    for r,global_i,category in zip(preds,selected,out_categories):
        q=rows[int(global_i)]
        fields_ok &= int(r['image_id'])==mapping.get(int(q['image_id']),int(q['image_id'])) and int(r['category_id'])==int(category) and r['bbox']==q['bbox']
        score_ok &= float(r['score']).hex()==float(q['detector_score']).hex()
    checks['all_images_categories_bboxes_exact']=bool(fields_ok)
    checks['all_original_raw_scores_exact']=bool(score_ok)
    if not all(checks.values()):raise ValueError(f'Native source alignment failed {dataset}/{split}/{branch}: {checks}')
    directory=root/'source_traces';directory.mkdir(exist_ok=True)
    path=directory/f'{dataset}_{split}_{branch}.npz'
    arrays={'source_detection_index':np.arange(len(preds),dtype=np.int64),'global_cache_index':selected,
            'original_image_id':np.array([int(rows[int(i)]['image_id']) for i in selected],dtype=np.int64),
            'image_id':np.array([int(p['image_id']) for p in preds],dtype=np.int64),
            'category_id':np.array([int(p['category_id']) for p in preds],dtype=np.int64),
            'bbox':np.array([p['bbox'] for p in preds],dtype=np.float64),
            'raw_score':np.array([p['score'] for p in preds],dtype=np.float64)}
    if path.exists():
        with np.load(path,allow_pickle=False) as old:
            if any(not np.array_equal(old[k],v) for k,v in arrays.items()):raise ValueError('Existing trace differs')
    else:npz_new(path,**arrays)
    return ref(path),checks

def prepare_baselines(protocol,output_root=OUT):
    root=checked_root(output_root);p,pr,ar=frozen(protocol)
    destination=root/'baseline_index.json'
    with file_lock(root/'locks/baseline_index.lock'):
        if destination.exists():
            d=read(destination)
            if d['protocol']['sha256']!=pr['sha256']:raise ValueError('Baseline index protocol mismatch')
            return d
        verify(p['prior_run_index']);prior=read(p['prior_run_index']['path'])
        records=[];align=[]
        sources=[(x['dataset'],'eval',x['method'],x['export_receipt'],x['metrics']) for x in prior['endpoints'] if x['role']=='primary' and x['method'] in ['visual','fixed_additive','post_score']]
        selected=read(OLD/'parameters/selected_parameters.json')
        for method,value,branch in [('learned_gate',1.01,'visual'),('margin_gate',2.,'fixed_additive')]:
            candidate=next(x for x in selected['candidate_results'] if x['method']==method and x['candidate']==value)
            sources.append(('CLC','tune',branch,candidate['export_receipt'],candidate['metrics']))
        if len(sources)!=11:raise ValueError('Expected 9 final context and 2 tune baseline records')
        for ds in ['CLC','DET','Hi']:
            rows=_metadata_rows(ds)
            with np.load(OLD/'features'/ds/'decisions.npz',allow_pickle=False) as z:
                decisions={k:z[k].copy() for k in ['visual_winner','fused_winner','visual_category','fused_category','visual_negative','fused_negative','visual_kept','fused_kept']}
            for dataset,split,branch,ep,mp in [s for s in sources if s[0]==ds]:
                e,m=read(ep),read(mp)
                if not m['actual_native_evaluation'] or m['export_sha256']!=sha(ep):raise ValueError('Baseline has no bound actual native evaluation')
                verify(m['native_log']);verify(m['original_GT']);verify(e['prediction']);verify(e['state'])
                trace=None
                if branch!='post_score':
                    trace,checks=_source_trace(ds,split,branch,ep,root,rows,decisions)
                    align.append({'dataset':ds,'split':split,'branch':branch,'checks':checks,'trace':trace})
                records.append({'dataset':ds,'split':split,'branch':branch,'source_split':e['identity']['split'],
                                'export_receipt':str(ep),'export_ref':ref(ep),'metrics':str(mp),'metrics_ref':ref(mp),
                                'prediction':e['prediction'],'state':e['state'],'GT':m['original_GT'],
                                'params':m['params'],'native_coco_source':m['native_coco_source'],'native_cocoeval_source':m['native_cocoeval_source'],
                                'AP':m['AP'],'AP50':m['AP50'],'AP75':m['AP75'],'count':e['prediction_count'],
                                'source_trace':trace,'purpose':'context_only' if branch=='post_score' else 'fixed_quality_source',
                                'global_row_key_sha256':read(OLD/'features'/ds/'feature_manifest.json')['row_key_sha256']})
            del rows,decisions
        result={'status':'COMPLETE','created_utc':now(),'protocol':pr,'exact_reuse_addendum':ar,'source_run_index':p['prior_run_index'],
                'records':records,'actual_AP_computed_here':False,'AP_recomputed':False,'old_source_modified':False}
        save_new(root/'verification/baseline_alignment.json',{'status':'PASS','records':align,'all_eight_fixed_lists_verified':len(align)==8,'new_AP_runs':0,'old_files_written':False})
        save_new(destination,result)
        return result

def _quality_scores(scores,dataset,variant,seed,expected_protocol):
    path=Path(scores).resolve();receipt_path=path.parent/'inference.json';receipt=read(receipt_path)
    fm=read(OLD/'features'/dataset/'feature_manifest.json')
    if receipt.get('protocol_sha256')!=expected_protocol:raise ValueError('Quality inference protocol mismatch')
    if receipt.get('variant',receipt.get('feature_variant'))!=variant or int(receipt.get('seed',receipt.get('training_seed',-1)))!=seed:raise ValueError('Quality model variant/seed mismatch')
    if receipt.get('global_row_key_sha256',receipt.get('row_key_sha256'))!=fm['row_key_sha256']:raise ValueError('Quality inference proposal order mismatch')
    if receipt.get('output_sha256')!=sha(path) or receipt.get('input_sha256')!=fm['outputs']['score_features.npy']['sha256']:raise ValueError('Quality scores/input feature hash mismatch')
    with np.load(path,allow_pickle=False) as z:
        if 'quality' in z.files:q=z['quality'].copy();key='quality'
        elif 'probabilities' in z.files:q=z['probabilities'].copy();key='probabilities'
        elif {'visual_quality','fixed_fusion_quality'}.issubset(z.files):q=np.stack([z['visual_quality'],z['fixed_fusion_quality']],axis=1);key='visual_quality/fixed_fusion_quality'
        else:raise ValueError('Scores NPZ requires explicit Nx2 quality/probabilities heads')
    if q.dtype!=np.float32 or q.shape!=(fm['shape'][0],2) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):raise ValueError('Invalid sigmoid quality array')
    return q,{'scores':ref(path),'inference_receipt':ref(receipt_path),'quality_key':key,'global_row_key_sha256':fm['row_key_sha256']}

def geometric_scores(raw,quality,alpha):
    raw=np.asarray(raw,dtype=np.float64);q=np.asarray(quality)
    if raw.shape!=q.shape or not np.isfinite(raw).all() or np.any(raw<0) or not np.isfinite(q).all():raise ValueError('Invalid raw-score/quality alignment')
    clipped=np.clip(q.astype(np.float64),1e-6,1.)
    if alpha==0.:new=raw.copy()
    elif alpha==1.:new=clipped.copy()
    else:new=np.power(raw,1.-alpha)*np.power(clipped,alpha)
    if not np.isfinite(new).all():raise ValueError('Nonfinite geometric score')
    return new,clipped

def export_quality(dataset,split,branch,variant,seed,alpha,scores,protocol,output_root=OUT):
    root=checked_root(output_root);p,pr,ar=frozen(protocol)
    if dataset not in ['DET','CLC','Hi'] or split not in ['eval','tune'] or branch not in ['visual','fixed_additive'] or variant not in ['full','visual_only'] or seed not in p['training']['seeds']:raise ValueError('Invalid declared endpoint identity')
    if split=='tune' and (dataset!='CLC' or seed!=17):raise ValueError('Only CLC seed17 tune selection is declared')
    alpha=float(alpha)
    if alpha not in p['scoring']['alpha_candidates']:raise ValueError('Alpha outside frozen grid')
    index=prepare_baselines(protocol,root)
    baseline=next(x for x in index['records'] if (x['dataset'],x['split'],x['branch'])==(dataset,split,branch))
    quality,quality_info=_quality_scores(scores,dataset,variant,seed,pr['sha256'])
    identity={'dataset':dataset,'split':split,'branch':branch,'variant':variant,'seed':seed,'alpha':alpha,
              'protocol':pr,'addendum':ar,'source_baseline_export':baseline['export_ref'],'source_prediction':baseline['prediction'],
              'source_trace':baseline['source_trace'],'quality':quality_info,'exporter':ref(__file__),'scoring_precision':'F64 arithmetic; predicted F32 quality promoted before clip [1e-6,1]'}
    key=digest(identity);directory=root/'endpoints'/f'{dataset}_{split}'/f'{variant}_{branch}_seed{seed}_a{str(alpha).replace(".","p")}_{key[:16]}'
    receipt=directory/'export.json'
    with file_lock(root/'locks'/f'export_{key}.lock'):
        if receipt.exists():
            old=read(receipt)
            if old['identity_sha256']!=key:raise ValueError('Export resume identity mismatch')
            verify(old['prediction']);verify(old['quality_trace']);return old
        if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Unsealed output exists: {directory}')
        directory.mkdir(parents=True,exist_ok=True)
        source=verify(baseline['prediction']);trace_path=verify(baseline['source_trace'])
        with np.load(trace_path,allow_pickle=False) as t:
            global_rows=t['global_cache_index'];raw=t['raw_score'].astype(np.float64);q=quality[global_rows,0 if branch=='visual' else 1]
        new,clipped=geometric_scores(raw,q,alpha)
        prediction=directory/'predictions.json'
        if alpha==0.:shutil.copyfile(source,prediction)
        else:
            records=read(source)
            if len(records)!=len(new):raise ValueError('Source list/trace alignment changed')
            rescored=[dict(record,score=float(value)) for record,value in zip(records,new)]
            if any(any(a[k]!=b[k] for k in ['image_id','category_id','bbox']) for a,b in zip(records,rescored)):raise AssertionError('Fixed list geometry/category changed')
            save_new(prediction,rescored)
        trace=directory/'quality_trace.npz';npz_new(trace,global_cache_index=global_rows,raw_score=raw,quality=q,clipped_quality=clipped,new_score=new)
        result={'status':'QUALITY_SCORE_EXPORT_COMPLETE','created_utc':now(),'identity_sha256':key,'identity':identity,'prediction':ref(prediction),
                'quality_trace':ref(trace),'source_state':baseline['state'],'source_trace':baseline['source_trace'],'source_metrics':baseline['metrics_ref'],
                'source_GT':baseline['GT'],'source_params':baseline['params'],'prediction_count':len(new),'alpha0_exact_original_bytes':alpha==0. and sha(prediction)==baseline['prediction']['sha256'],
                'bbox_category_order_retention_unchanged':True,'original_score_stored_in_trace':True,'states_recomputed':False,'GT_used_for_rescoring':False}
        save_new(receipt,result);return result

def _native_sources():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    return {'native_coco_source':ref(inspect.getfile(COCO)),'native_cocoeval_source':ref(inspect.getfile(COCOeval))}

def _params(gt):
    from pycocotools.cocoeval import COCOeval
    ev=COCOeval(None,None,'bbox');ev.params.imgIds=sorted(int(r['id']) for r in gt['images']);ev.params.catIds=sorted(int(r['id']) for r in gt['categories']);ev.params.maxDets=[1,10,100]
    q=ev.params
    return {'iouType':'bbox','imgIds':q.imgIds,'catIds':q.catIds,'maxDets':q.maxDets,'iouThrs':q.iouThrs.tolist(),'recThrs':q.recThrs.tolist(),'areaRng':q.areaRng,'areaRngLbl':q.areaRngLbl,'useCats':q.useCats}

def _completed_actual_source(path,pred_sha,gt_sha,params,sources):
    path=Path(path);m=read(path)
    actual=m.get('AP_computed_here',m.get('actual_native_evaluation',False))
    if not actual or m.get('AP_reused',False):return None
    if m['prediction']['sha256']!=pred_sha or m['original_GT']['sha256']!=gt_sha or m['params']!=params:return None
    for k in ['native_coco_source','native_cocoeval_source']:
        if m[k]['sha256']!=sources[k]['sha256'] or not verify(m[k]):return None
    verify(m['prediction']);verify(m['original_GT']);verify(m['native_log'])
    return {'metrics':ref(path),'prediction':m['prediction'],'original_GT':m['original_GT'],'native_log':m['native_log'],
            'native_coco_source':m['native_coco_source'],'native_cocoeval_source':m['native_cocoeval_source'],'params':m['params']}

@contextlib.contextmanager
def _native_slot(root):
    handles=[];selected=None
    try:
        while selected is None:
            for i in range(2):
                f=(root/'locks'/f'native_slot{i}.lock').open('a')
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);selected=f;break
                except BlockingIOError:f.close()
            if selected is None:time.sleep(.2)
        yield
    finally:
        if selected is not None:fcntl.flock(selected,fcntl.LOCK_UN);selected.close()

def evaluate_export(export_receipt,output_root=OUT):
    root=checked_root(output_root);path=Path(export_receipt).resolve()
    if not path.is_relative_to(root):raise ValueError('Export outside new quality root')
    e=read(path);verify(e['prediction']);verify(e['quality_trace']);frozen(e['identity']['protocol']['path'])
    metric_path=path.parent/'metrics.json'
    if metric_path.exists():
        m=read(metric_path)
        if m['export_sha256']!=sha(path):raise ValueError('Metric/export resume mismatch')
        verify(e['source_GT'])
        current_sources=_native_sources()
        if any(m[k]['sha256']!=current_sources[k]['sha256'] for k in current_sources):raise ValueError('Native evaluation source changed on resume')
        if m['AP_reused']:
            verify(m['reused_source']['metrics']);verify(m['reused_source']['native_log']);verify(m['reused_source']['prediction']);verify(m['reused_source']['original_GT'])
        else:verify(m['native_log'])
        return m
    gt_ref=e['source_GT'];gt_path=verify(gt_ref);gt=read(gt_path)
    ids=[int(a['id']) for a in gt.get('annotations',[])]
    if len(set(ids))!=len(ids) or any(i<=0 for i in ids):raise ValueError('Accepted source GT must already have positive unique annotation IDs')
    params=_params(gt);sources=_native_sources();pred_sha=e['prediction']['sha256']
    evaluation_key=digest({'ordered_prediction_sha256':pred_sha,'GT_sha256':gt_ref['sha256'],'params':params,
                           'native_coco_sha256':sources['native_coco_source']['sha256'],'native_cocoeval_sha256':sources['native_cocoeval_source']['sha256']})
    registry=root/'evaluation_registry';registry.mkdir(exist_ok=True);entry=registry/(evaluation_key+'.json')
    with file_lock(root/'locks'/('evaluation_'+evaluation_key+'.lock')):
        candidates=[]
        if entry.exists():candidates.append(str(verify(read(entry)['actual_metrics'])))
        candidates.extend(str(verify(x['metrics_ref'])) for x in read(root/'baseline_index.json')['records'])
        reused=None
        for candidate in candidates:
            reused=_completed_actual_source(candidate,pred_sha,gt_ref['sha256'],params,sources)
            if reused is not None:break
        if reused is not None:
            old=read(reused['metrics']['path'])
            m={'created_utc':now(),'export_sha256':sha(path),'identity_sha256':e['identity_sha256'],'prediction':e['prediction'],'original_GT':gt_ref,
               'params':params,'stats':old['stats'],'AP':old['AP'],'AP50':old['AP50'],'AP75':old['AP75'],**sources,
               'native_log':old['native_log'],'AP_reused':True,'AP_computed_here':False,'actual_AP_computed_here':False,'actual_native_evaluation':False,
               'original_actual_native_evaluation':True,'reused_source':reused,'reuse_key':evaluation_key,'reuse_reason':'Exact full ordered prediction SHA, original GT SHA, all cohorts/params and current native COCO sources match'}
        else:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            log=path.parent/'native_COCO.log'
            with _native_slot(root),log.open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                cg=COCO(str(gt_path))
                if e['prediction_count']:cd=cg.loadRes(str(e['prediction']['path']))
                else:cd=COCO();cd.dataset={'images':gt['images'],'categories':gt['categories'],'annotations':[]};cd.createIndex()
                ev=COCOeval(cg,cd,'bbox');ev.params.imgIds=params['imgIds'];ev.params.catIds=params['catIds'];ev.params.maxDets=[1,10,100]
                ev.evaluate();ev.accumulate();ev.summarize()
            stats=[float(x) for x in ev.stats]
            m={'created_utc':now(),'export_sha256':sha(path),'identity_sha256':e['identity_sha256'],'prediction':e['prediction'],'original_GT':gt_ref,
               'params':params,'stats':stats,'AP':stats[0],'AP50':stats[1],'AP75':stats[2],**sources,'native_log':ref(log),
               'AP_reused':False,'AP_computed_here':True,'actual_AP_computed_here':True,'actual_native_evaluation':True,'reused_source':None,'reuse_key':evaluation_key}
        m['GT_access_purpose']='calibration alpha selection' if e['identity']['split']=='tune' else 'evaluation only'
        m['label_box_order_retention_changed']=False
        save_new(metric_path,m)
        if not entry.exists():save_new(entry,{'key':evaluation_key,'actual_metrics':reused['metrics'] if reused else ref(metric_path)})
        return m
