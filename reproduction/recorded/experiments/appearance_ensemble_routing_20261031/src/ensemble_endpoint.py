"""New-root ensemble scoring and strict native-visual-budget routing."""
from __future__ import annotations
import contextlib
import datetime
import fcntl
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_ensemble_routing_20261031')
OLD=Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_quality_reranking_20261031')
OLD_SOURCE=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_quality_reranking_20261031/src/quality_endpoint.py')
PROTOCOL_SHA='712da4597ed12a1245a2208293df3a35881fd34db8179174a8c388f1626e333d'
ADDENDUM_SHA='cb55144c57f665fce8d1c217b459acc250d2b6190e7a4ea712b89b9ab1b5910b'
spec=importlib.util.spec_from_file_location('_readonly_quality_pure_helpers',OLD_SOURCE)
pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
# Only these pure helpers are reused. No call to old export/evaluate/root setup.
sha,read,ref,digest,verify=pure.sha,pure.read,pure.ref,pure.digest,pure.verify
geometric_scores=pure.geometric_scores
native_params,native_sources,completed_actual_source=pure._params,pure._native_sources,pure._completed_actual_source

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def atomic(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def checked_root(root):
    root=Path(root).resolve()
    if root!=OUT:raise ValueError('Only new ensemble root may be written')
    root.mkdir(parents=True,exist_ok=True);return root
def frozen(protocol):
    path=Path(protocol).resolve();a=CODE/'config/addendum_01_visual_budget.json'
    if sha(path)!=PROTOCOL_SHA or sha(a)!=ADDENDUM_SHA:raise ValueError('Frozen ensemble protocol/addendum mismatch')
    return read(path),ref(path),ref(a)

def prepare_sources(protocol,output_root=OUT):
    root=checked_root(output_root);p,pr,ar=frozen(protocol)
    path=root/'source_baseline_index.json'
    with lock(root/'locks/source_baseline_index.lock'):
        if path.exists():
            d=read(path)
            if d['protocol']['sha256']!=pr['sha256'] or d['visual_budget_addendum']['sha256']!=ar['sha256']:raise ValueError('Source index input mismatch')
            return d
        old=read(verify(p['prior_baseline_index']));records=[];checks=[]
        if old['status']!='COMPLETE' or len(old['records'])!=11:raise ValueError('Expected11 complete prior baselines')
        for x in old['records']:
            verify(x['export_ref']);verify(x['metrics_ref']);verify(x['prediction']);verify(x['state'])
            m=read(x['metrics'])
            if not m['actual_native_evaluation'] or m['export_sha256']!=x['export_ref']['sha256']:raise ValueError('Baseline lacks native metric binding')
            r=dict(x,stats=m['stats'])
            if x['branch']!='post_score':
                trace=verify(x['source_trace'])
                with np.load(trace,allow_pickle=False) as t:
                    monotonic=len(t['global_cache_index'])<2 or bool(np.all(np.diff(t['global_cache_index'])>0))
                    count=len(t['global_cache_index'])
                if not monotonic or count!=x['count']:raise ValueError('Native source order/trace count mismatch')
                checks.append({'dataset':x['dataset'],'split':x['split'],'branch':x['branch'],'count':count,'monotonic_global_rows':monotonic})
            records.append(r)
        alignment=read(OLD/'verification/baseline_alignment.json')
        if alignment['status']!='PASS' or not alignment['all_eight_fixed_lists_verified']:raise ValueError('Prior native alignment proof failed')
        result={'status':'IN_PROGRESS_INTERFACE_CONTRACT','created_utc':now(),'protocol':pr,'visual_budget_addendum':ar,
                'source_baseline_index':p['prior_baseline_index'],'records':records,'source_trace_checks':checks,
                'prior_alignment':ref(OLD/'verification/baseline_alignment.json'),'new_AP_runs':0,'old_files_written':False}
        save_new(path,result);return result

def quality_input(scores,dataset,variant,group,protocol_sha):
    path=Path(scores).resolve();rp=path.parent/'inference.json';r=read(rp)
    if r['protocol_sha256']!=protocol_sha or r['variant']!=variant or r['group']!=group or r['output_sha256']!=sha(path):raise ValueError('Ensemble identity/hash mismatch')
    expected=[17,29,43] if group=='all3' else {'pair17_29':[17,29],'pair17_43':[17,43],'pair29_43':[29,43]}[group]
    if r['seeds']!=expected:raise ValueError('Ensemble seed group mismatch')
    with np.load(path,allow_pickle=False) as z:q=z['quality'].copy()
    fm=read(pure.OLD/'features'/dataset/'feature_manifest.json')
    if q.shape!=(fm['shape'][0],2) or q.dtype!=np.float32 or not np.isfinite(q).all() or np.any((q<0)|(q>1)) or r['global_row_key_sha256']!=fm['row_key_sha256']:raise ValueError('Invalid ensemble quality array/row binding')
    return q,{'scores':ref(path),'inference_receipt':ref(rp),'global_row_key_sha256':r['global_row_key_sha256'],'seeds':expected}

def source_record(index,dataset,split,branch):
    return next(x for x in index['records'] if (x['dataset'],x['split'],x['branch'])==(dataset,split,branch))

def source_arrays(record):
    with np.load(verify(record['source_trace']),allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def routing_choice(vrows,frows,quality,tau):
    """Strict V membership; select F only on both-kept rows and strict gain."""
    if not len(frows):return np.zeros(len(vrows),dtype=bool),np.zeros(len(vrows),dtype=bool),np.zeros(len(vrows),dtype=np.int64)
    pos=np.searchsorted(frows,vrows);clipped=np.minimum(pos,len(frows)-1)
    both=(pos<len(frows))&(frows[clipped]==vrows)
    choose=both&((quality[vrows,1]-quality[vrows,0])>tau)
    return choose,both,clipped

def export_endpoint(dataset,split,variant,method,group,alpha,tau,scores,protocol,output_root=OUT):
    root=checked_root(output_root);p,pr,ar=frozen(protocol)
    if dataset not in p['datasets'] or split not in ['tune','eval'] or variant not in p['ensembles']['variants'] or method not in ['visual','fixed_additive','router','matched_priority']:raise ValueError('Undeclared endpoint')
    if group!='all3' and (variant!='full' or group not in p['ensembles']['leave_one_out_groups'] or method not in ['visual','router']):raise ValueError('Undeclared pair diagnostic')
    if split=='tune' and (dataset!='CLC' or group!='all3' or method=='matched_priority'):raise ValueError('Only declared CLC/all3 tune candidates')
    alpha=float(alpha);tau=float(tau)
    if alpha not in p['alpha_candidates']:raise ValueError('Alpha outside frozen grid')
    if method in ['router','matched_priority'] and tau not in p['routing']['candidate_tau']:raise ValueError('Tau outside frozen grid')
    if method=='matched_priority' and (variant!='full' or group!='all3'):raise ValueError('Matched control is full/all3 only')
    index=prepare_sources(protocol,root)
    if index['status']!='COMPLETE' or read(root/'endpoint_ready.json')['status']!='READY':raise RuntimeError('Interface not yet READY')
    q,qr=quality_input(scores,dataset,variant,group,pr['sha256'])
    baseline=source_record(index,dataset,split,'fixed_additive' if method=='fixed_additive' else 'visual')
    if qr['global_row_key_sha256']!=baseline['global_row_key_sha256']:raise ValueError('Ensemble/global native row binding mismatch')
    ident={'dataset':dataset,'split':split,'variant':variant,'method':method,'group':group,'alpha':alpha,
           'tau':tau if method in ['router','matched_priority'] else None,'protocol':pr,'visual_budget_addendum':ar,
           'quality':qr,'source_export':baseline['export_ref'],'source_trace':baseline['source_trace'],
           'new_module':ref(__file__),'readonly_pure_helpers':ref(OLD_SOURCE)}
    key=digest(ident);directory=root/'endpoints'/f'{dataset}_{split}'/f'{variant}_{method}_{group}_a{alpha}_t{ident["tau"]}_{key[:16]}'
    ep=directory/'export.json'
    with lock(root/'locks'/('export_'+key+'.lock')):
        if ep.exists():
            old=read(ep)
            if old['identity_sha256']!=key:raise ValueError('Export resume key mismatch')
            verify(old['prediction']);verify(old['choice_trace']);return old
        if directory.exists() and any(directory.iterdir()):raise FileExistsError(f'Unsealed partial endpoint: {directory}')
        directory.mkdir(parents=True,exist_ok=True)
        a=source_arrays(baseline);rows=a['global_cache_index'];raw=a['raw_score'];source=verify(baseline['prediction'])
        if len(rows) and int(rows.max())>=len(q):raise ValueError('Quality array omits native query rows')
        if method=='fixed_additive':chosen=np.ones(len(rows),dtype=bool);both=None;quality=q[rows,1];categories=a['category_id'].copy();fonly=0
        elif method=='visual':chosen=np.zeros(len(rows),dtype=bool);both=None;quality=q[rows,0];categories=a['category_id'].copy();fonly=0
        else:
            f=source_arrays(source_record(index,dataset,split,'fixed_additive'))
            chosen,both,fpos=routing_choice(rows,f['global_cache_index'],q,tau)
            if method=='matched_priority':chosen[:]=False
            quality=np.where(chosen,q[rows,1],q[rows,0]);categories=a['category_id'].copy()
            if len(f['global_cache_index']):categories=np.where(chosen,f['category_id'][fpos],categories)
            fonly=int(len(f['global_cache_index'])-both.sum())
        new,clipped=geometric_scores(raw,quality,alpha)
        prediction=directory/'predictions.json';records=read(source)
        category_changed=int(np.count_nonzero(categories!=a['category_id']))
        exact_raw=alpha==0. and category_changed==0
        if exact_raw:shutil.copyfile(source,prediction)
        else:
            out=[dict(r,category_id=int(c),score=float(s)) for r,c,s in zip(records,categories,new)]
            if len(out)!=len(records) or any(a0['bbox']!=b0['bbox'] or a0['image_id']!=b0['image_id'] for a0,b0 in zip(records,out)):raise AssertionError('Native budget/image/bbox/order changed')
            save_new(prediction,out)
        trace=directory/'choice_trace.npz'
        with trace.open('xb') as stream:np.savez_compressed(stream,global_cache_index=rows,raw_score=raw,quality=quality,clipped_quality=clipped,new_score=new,chosen_fixed=chosen,category_id=categories,both_kept=np.zeros(len(rows),dtype=bool) if both is None else both)
        result={'status':'ENSEMBLE_ENDPOINT_EXPORTED','created_utc':now(),'identity_sha256':key,'identity':ident,'prediction':ref(prediction),
                'choice_trace':ref(trace),'source_metrics':baseline['metrics_ref'],'source_GT':baseline['GT'],'source_params':baseline['params'],
                'source_prediction':baseline['prediction'],'source_state':baseline['state'],'prediction_count':len(records),
                'membership_reference':'native_fixed_additive' if method=='fixed_additive' else 'native_visual',
                'membership_changed_from_reference':False,'bbox_image_order_original_rawscore_changed':False,
                'category_changed_count':category_changed,'chosen_fixed_count':int(chosen.sum()),
                'both_kept_count':None if both is None else int(both.sum()),'F_only_excluded_count':fonly,
                'subset_native_union':True,'selected_quality_of_nonkept_head_used':False,'states_recomputed':False,
                'category_may_change':method=='router','only_scores_changed':method in ['visual','fixed_additive','matched_priority'],
                'exact_raw_bytes':exact_raw and sha(prediction)==baseline['prediction']['sha256'],'GT_used_for_export':False}
        save_new(ep,result);return result

@contextlib.contextmanager
def native_slot(root):
    selected=None
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
    root=checked_root(output_root);ep=Path(export_receipt).resolve()
    if not ep.is_relative_to(root):raise ValueError('Endpoint outside new root')
    e=read(ep);verify(e['prediction']);verify(e['choice_trace']);frozen(e['identity']['protocol']['path'])
    mp=ep.parent/'metrics.json';sources=native_sources()
    if mp.exists():
        m=read(mp)
        if m['export_sha256']!=sha(ep):raise ValueError('Metric/export binding changed')
        verify(e['source_GT'])
        if any(m[k]['sha256']!=sources[k]['sha256'] for k in sources):raise ValueError('Native evaluator changed')
        verify(m['native_log'])
        if m['AP_reused']:
            for k in ['metrics','prediction','original_GT']:verify(m['reused_source'][k])
        return m
    gt_ref=e['source_GT'];gt_path=verify(gt_ref);gt=read(gt_path)
    ids=[int(a['id']) for a in gt['annotations']]
    if any(i<=0 for i in ids) or len(ids)!=len(set(ids)):raise ValueError('Positive unique annotation IDs required')
    params=native_params(gt);pred_sha=e['prediction']['sha256']
    key=digest({'pred':pred_sha,'GT':gt_ref['sha256'],'params':params,'native_sources':{k:v['sha256'] for k,v in sources.items()}})
    registry=root/'evaluation_registry';registry.mkdir(exist_ok=True);entry=registry/(key+'.json')
    with lock(root/'locks'/('evaluation_'+key+'.lock')):
        candidates=[]
        if entry.exists():candidates.append(str(verify(read(entry)['actual_metrics'])))
        index=read(root/'source_baseline_index.json')
        candidates.extend(str(verify(x['metrics_ref'])) for x in index['records'])
        prior=read(verify(read(e['identity']['protocol']['path'])['prior_run_index']))
        candidates.extend(str(verify({'path':x['metrics'],'sha256':x['metrics_sha256']})) for x in prior['endpoints'])
        # Bounded previous-run registry includes its completed actual tune
        # evaluations, with original metric digests retained by the registry.
        for previous_entry in sorted((OLD/'evaluation_registry').glob('*.json')):
            candidates.append(str(verify(read(previous_entry)['actual_metrics'])))
        source=None
        for path in candidates:
            source=completed_actual_source(path,pred_sha,gt_ref['sha256'],params,sources)
            if source is not None:break
        if source is not None:
            old=read(source['metrics']['path']);m={'stats':old['stats'],'AP':old['AP'],'AP50':old['AP50'],'AP75':old['AP75'],
                'native_log':old['native_log'],'AP_reused':True,'AP_computed_here':False,'actual_AP_computed_here':False,'actual_native_evaluation':False,
                'original_actual_native_evaluation':True,'reused_source':source}
        else:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            log=ep.parent/'native_COCO.log'
            with native_slot(root),log.open('x') as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                cg=COCO(str(gt_path))
                if e['prediction_count']:cd=cg.loadRes(e['prediction']['path'])
                else:cd=COCO();cd.dataset={'images':gt['images'],'categories':gt['categories'],'annotations':[]};cd.createIndex()
                ev=COCOeval(cg,cd,'bbox');ev.params.imgIds=params['imgIds'];ev.params.catIds=params['catIds'];ev.params.maxDets=[1,10,100]
                ev.evaluate();ev.accumulate();ev.summarize()
            stats=[float(x) for x in ev.stats];m={'stats':stats,'AP':stats[0],'AP50':stats[1],'AP75':stats[2],
                'native_log':ref(log),'AP_reused':False,'AP_computed_here':True,'actual_AP_computed_here':True,'actual_native_evaluation':True,'reused_source':None}
        m.update(created_utc=now(),export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=gt_ref,params=params,
                 **sources,reuse_key=key,GT_access_purpose='CLC tune parameter selection' if e['identity']['split']=='tune' else 'evaluation only',
                 routing_membership_reference=e['membership_reference'],routing_category_changed_count=e['category_changed_count'],subset_native_union=True)
        save_new(mp,m)
        if not entry.exists():save_new(entry,{'key':key,'actual_metrics':source['metrics'] if source else ref(mp)})
        return m
