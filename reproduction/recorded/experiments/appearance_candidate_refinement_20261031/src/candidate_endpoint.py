"""Frozen-budget downstream candidate classification and exact native COCO AP.

Every exported row comes from the original visual-retained list. Its class may
change among five GT-free visual candidate classes. Original UNC is a proposal
prefilter, not a certificate for the newly selected category.
"""
from __future__ import annotations
import contextlib,datetime,fcntl,hashlib,importlib.util,json,re,shutil,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/appearance_candidate_refinement_20261031')
PROTOCOL_SHA='b47eedfd0f52c8d33c39b915a0be61029c93f2b18d804dcddb9e41fb1793b534'
CASES=['CLC_tune','DET','CLCXray_test','PIXray','PIDray','DvXray','Hi']
VARIANTS=['full','visual']
BETAS=[0.,.025,.05,.1,.2,.4]
ROLES=['label_only','refine_rescore','score_control']
OLD_HELPERS=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_quality_reranking_20261031/src/quality_endpoint.py')
spec=importlib.util.spec_from_file_location('_candidate_readonly_coco_helpers',OLD_HELPERS)
pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
read,sha,ref,verify,digest=pure.read,pure.sha,pure.ref,pure.verify,pure.digest

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fresh_write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x')as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)
def frozen(protocol,output_root=OUT):
    root=Path(output_root).resolve();pp=Path(protocol).resolve()
    if root!=OUT or PROTOCOL_SHA=='PENDING' or sha(pp)!=PROTOCOL_SHA:raise ValueError('Only frozen candidate protocol/new output root is allowed')
    p=read(pp)
    if not p.get('status','').startswith('FROZEN'):raise ValueError('Protocol not frozen')
    return p,ref(pp),root
def npy(r):
    path=verify(r)
    with path.open('rb')as stream:return np.load(stream,allow_pickle=False)
def native_source(dataset,protocol,output_root=OUT):
    p,pr,root=frozen(protocol,output_root)
    if dataset not in CASES:raise ValueError('Case not declared')
    sp=root/'features'/dataset/'status.json';st=read(sp)
    if st['status']!='COMPLETE':raise RuntimeError('Candidate feature publication is not complete')
    fp=root/'features'/dataset/'feature_manifest.json';fm=read(fp)
    published=st.get('feature_manifest',st.get('features'))
    if published is None or verify(published)!=fp or published['sha256']!=sha(fp):raise ValueError('Published candidate feature manifest mismatch')
    if fm.get('protocol',{}).get('sha256',fm.get('protocol_sha256'))!=PROTOCOL_SHA:raise ValueError('Candidate feature protocol mismatch')
    src=fm['source_native_record'];outs=fm['outputs']
    for name in ['candidate_category_id.npy','candidate_proto_index.npy','candidate_visual_scores.npy','source_trace.npz','native_predictions.json']:verify(outs[name])
    categories=npy(outs['candidate_category_id.npy']);prototypes=npy(outs['candidate_proto_index.npy']);visual=npy(outs['candidate_visual_scores.npy'])
    with np.load(outs['source_trace.npz']['path'],allow_pickle=False)as z:trace={k:z[k].copy()for k in z.files}
    preds=read(outs['native_predictions.json']['path']);N=len(preds)
    if categories.shape!=(N,5)or prototypes.shape!=(N,5)or visual.shape!=(N,5)or categories.dtype!=np.int64 or prototypes.dtype!=np.int64 or visual.dtype!=np.float32:raise ValueError('Malformed candidate shape/dtype')
    if not np.isfinite(visual).all()or np.any(prototypes<0)or any(len(set(r.tolist()))!=5 for r in categories):raise ValueError('Candidate classes must be distinct, positive-prototype and finite')
    rows=trace['global_cache_index'];raw=trace['raw_score'];native_cat=trace['category_id']
    for name in ['global_cache_index','source_detection_index','image_id','original_image_id','category_id']:
        if trace[name].shape!=(N,)or trace[name].dtype!=np.int64:raise ValueError('Native identity shape/dtype: '+name)
    if raw.shape!=(N,)or raw.dtype!=np.float64 or not np.isfinite(raw).all()or np.any(raw<0)or trace['bbox'].shape!=(N,4)or trace['bbox'].dtype!=np.float64 or not np.isfinite(trace['bbox']).all():raise ValueError('Native raw-score/bbox shape, precision or range malformed')
    if not np.array_equal(trace['source_detection_index'],np.arange(N))or len(np.unique(rows))!=N:raise ValueError('Native source row identity malformed')
    if not np.array_equal(categories[:,0],native_cat)or np.any(visual[:,1:]>visual[:,:-1]):raise ValueError('Candidate rank0 must be native visual class and candidates descending V')
    if np.any((visual[:,1:]==visual[:,:-1])&(prototypes[:,1:]<prototypes[:,:-1])):raise ValueError('Visual ties must follow original prototype index')
    if any(r['image_id']!=int(trace['image_id'][i])or r['category_id']!=int(native_cat[i])or r['bbox']!=trace['bbox'][i].tolist()or float(r['score']).hex()!=float(raw[i]).hex()for i,r in enumerate(preds)):raise ValueError('Native predictions/source trace mismatch')
    if 'prediction'in src and src['prediction']['sha256']!=outs['native_predictions.json']['sha256']:raise ValueError('Native prediction bytes changed')
    for k in ['GT']:
        verify(src[k])
    return p,pr,root,fm,ref(fp),src,outs,trace,preds,categories,prototypes,visual
def quality(root,dataset,variant,fm,fmr,N):
    sp=root/'ensembles'/dataset/variant/'status.json';st=read(sp)
    if st['status']!='COMPLETE':raise RuntimeError('Candidate ensemble not complete')
    row=st['ensemble'];qp=verify(row['scores']);rp=verify(row['inference_receipt']);r=read(rp)
    if r['protocol_sha256']!=PROTOCOL_SHA or r['variant']!=variant or r.get('case',r.get('dataset'))!=dataset or r['seeds']!=[17,29,43]or r['candidate_row_key_sha256']!=fm['candidate_row_key_sha256']or r['feature_manifest']['sha256']!=fmr['sha256']or r['output_sha256']!=sha(qp):raise ValueError('Quality ensemble protocol/candidate identity mismatch')
    verify(r['feature_manifest'])
    with np.load(qp,allow_pickle=False)as z:q=z['quality'].copy()
    if q.dtype!=np.float32 or q.shape!=(N,5)or not np.isfinite(q).all()or np.any((q<0)|(q>1)):raise ValueError('Quality must be finite F32[N,5] in[0,1]')
    return q,dict(scores=ref(qp),inference_receipt=ref(rp),candidate_row_key_sha256=r['candidate_row_key_sha256'])
def select_candidates(visual,q,beta):
    # np.argmax chooses the first candidate rank on exact F64 score ties.
    selection_score=visual.astype(np.float64)+float(beta)*q.astype(np.float64)
    chosen=np.argmax(selection_score,axis=1).astype(np.int64)
    if beta==0. and np.any(chosen!=0):raise ValueError('beta0 must choose native visual candidate rank0')
    return chosen,selection_score
def gaussian(scores,boxes,images,categories,rows):
    values=np.array(scores,np.float64,copy=True);boxes=np.asarray(boxes,np.float64);rank=np.full(len(values),-1,np.int32);groups={}
    for j,key in enumerate(zip(images.tolist(),categories.tolist())):groups.setdefault(key,[]).append(j)
    for group in groups.values():
        remaining=np.asarray(group,np.int64);count=0
        while len(remaining):
            ties=remaining[values[remaining]==values[remaining].max()];chosen=int(ties[np.argmin(rows[ties])]);rank[chosen]=count;count+=1;remaining=remaining[remaining!=chosen]
            if not len(remaining):continue
            a=boxes[chosen];b=boxes[remaining];wh=np.maximum(0.,np.minimum(a[:2]+a[2:],b[:,:2]+b[:,2:])-np.maximum(a[:2],b[:,:2]));inter=wh[:,0]*wh[:,1]
            union=max(0.,a[2])*max(0.,a[3])+np.maximum(b[:,2],0.)*np.maximum(b[:,3],0.)-inter
            iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);values[remaining]*=np.exp(-iou*iou)
    return values,rank
def export_endpoint(dataset,variant,beta,role,protocol,output_root=OUT,case='default'):
    if variant not in VARIANTS or role not in ROLES:raise ValueError('Invalid variant/role')
    beta=float(beta)
    if beta not in BETAS or(role=='score_control'and beta!=0.):raise ValueError('beta outside frozen grid or score_control not0')
    p,pr,root,fm,fmr,src,outs,a,records,cats,protos,visual=native_source(dataset,protocol,output_root)
    q,qr=quality(root,dataset,variant,fm,fmr,len(records));which,selection_score=select_candidates(visual,q,beta);ii=np.arange(len(which))
    selected_cat=cats[ii,which];selected_proto=protos[ii,which];selected_q=q[ii,which];raw=a['raw_score'].astype(np.float64)
    if role=='label_only':pre=raw.copy();clipped=np.clip(selected_q.astype(np.float64),1e-6,1.);post=pre.copy();rank=np.full(len(pre),-1,np.int32)
    else:
        clipped=np.clip(selected_q.astype(np.float64),1e-6,1.);pre=np.power(raw,.5)*np.power(clipped,.5)
        post,rank=gaussian(pre,a['bbox'],a['image_id'],selected_cat,a['global_cache_index'])
    identity=dict(dataset=dataset,split='tune'if dataset=='CLC_tune'else'eval',variant=variant,beta=beta,role=role,case=case,protocol=pr,feature_manifest=fmr,source_prediction=outs['native_predictions.json'],native_source_trace=outs['source_trace.npz'],quality=qr,exporter=ref(__file__),readonly_coco_helpers=ref(OLD_HELPERS),candidate_row_key_sha256=fm['candidate_row_key_sha256'],selection_precision='F64(Vcandidate)+beta*F64(q), ties original candidate rank',score_alpha=0. if role=='label_only'else .5,Gaussian_sigma='off'if role=='label_only'else 1.)
    key=digest(identity);tag=re.sub(r'[^A-Za-z0-9_-]','_',str(case))[:40];btag=format(beta,'.12g').replace('.','p')
    directory=root/'endpoints'/dataset/f'{variant}_{role}_b{btag}_{tag}_{key[:16]}';ep=directory/'export.json'
    with lock(root/'locks'/('export_'+key+'.lock')):
        if ep.exists():
            e=read(ep)
            if e['identity_sha256']!=key:raise ValueError('Endpoint resume identity mismatch')
            verify(e['prediction']);verify(e['score_trace']);return e
        if directory.exists()and any(directory.iterdir()):raise FileExistsError('Unsealed endpoint '+str(directory))
        directory.mkdir(parents=True,exist_ok=True);pp=directory/'predictions.json'
        if role=='label_only'and beta==0.:shutil.copyfile(outs['native_predictions.json']['path'],pp)
        else:fresh_write(pp,[dict(r,category_id=int(category),score=float(score))for r,category,score in zip(records,selected_cat,post)])
        tp=directory/'score_trace.npz'
        with tp.open('xb')as stream:np.savez_compressed(stream,global_cache_index=a['global_cache_index'],source_detection_index=a['source_detection_index'],native_category_id=a['category_id'],selected_candidate_rank=which,selected_category_id=selected_cat,selected_proto_index=selected_proto,candidate_selection_score=selection_score,selected_quality=selected_q,clipped_quality=clipped,native_raw_score=raw,pre_soft_score=pre,post_soft_score=post,soft_selection_rank=rank)
        metrics_candidates=[]
        for k in ['metrics_ref','source_metrics','metrics']:
            r=src.get(k)
            if isinstance(r,dict)and'path'in r:metrics_candidates.append(r)
            elif isinstance(r,str)and Path(r).exists():metrics_candidates.append(ref(r))
        result=dict(status='CANDIDATE_REFINEMENT_EXPORTED',created_utc=now(),identity_sha256=key,identity=identity,prediction=ref(pp),score_trace=ref(tp),source_GT=src['GT'],source_params=src['params'],source_native_record=src,native_metric_candidates=metrics_candidates,prediction_count=len(records),native_class_changes=int(np.sum(selected_cat!=a['category_id'])),nativeV_membership_bbox_image_order_unchanged=True,downstream_category_selection=True,UNC_recomputed_for_selected_class=False,original_UNC_use='Original native visual UNC defines proposal budget only; no new-category UNC claim',exact_native_input_bytes=bool(role=='label_only'and beta==0.),GT_used_for_export=False,Gaussian=dict(applied=role!='label_only',sigma=None if role=='label_only'else 1.,threshold_pruning=False,topK=False),quality_scoring_alpha=0. if role=='label_only'else .5)
        fresh_write(ep,result);return result
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
def _actual_chain(path,pred_sha,gt_sha,params,sources,seen=None):
    seen=set()if seen is None else seen;path=Path(path).resolve()
    if str(path)in seen:return None
    seen.add(str(path));m=read(path)
    actual=pure._completed_actual_source(path,pred_sha,gt_sha,params,sources)
    if actual is not None:return actual
    if m.get('AP_reused')and isinstance(m.get('reused_source'),dict)and'metrics'in m['reused_source']:
        r=m['reused_source']['metrics'];return _actual_chain(verify(r),pred_sha,gt_sha,params,sources,seen)
    return None
def evaluate_export(export_receipt,output_root=OUT):
    root=Path(output_root).resolve();ep=Path(export_receipt).resolve()
    if root!=OUT or not ep.is_relative_to(root):raise ValueError('Only new candidate output may be written')
    e=read(ep);frozen(e['identity']['protocol']['path'],root);verify(e['prediction']);verify(e['score_trace']);sources=pure._native_sources();mp=ep.parent/'metrics.json'
    if mp.exists():
        m=read(mp)
        if m['export_sha256']!=sha(ep)or m['identity_sha256']!=e['identity_sha256']:raise ValueError('Metric resume identity mismatch')
        if any(m[k]['sha256']!=sources[k]['sha256']for k in sources):raise ValueError('Native COCO implementation changed')
        verify(m['native_log']);verify(m['original_GT'])
        if m['AP_reused']:
            for k in ['metrics','prediction','original_GT','native_log']:verify(m['reused_source'][k])
        return m
    gtref=e['source_GT'];gtpath=verify(gtref);gt=read(gtpath);ids=[int(x['id'])for x in gt['annotations']]
    if any(x<=0 for x in ids)or len(ids)!=len(set(ids)):raise ValueError('Positive unique annotation IDs required')
    params=pure._params(gt)
    if params!=e['source_params']:raise ValueError('Native COCO parameters differ from exact source cohort')
    key=digest(dict(prediction=e['prediction']['sha256'],GT=gtref['sha256'],params=params,sources={k:v['sha256']for k,v in sources.items()}));registry=root/'evaluation_registry';registry.mkdir(exist_ok=True);entry=registry/(key+'.json')
    with lock(root/'locks'/('evaluation_'+key+'.lock')):
        candidates=[]
        if entry.exists():candidates.append(verify(read(entry)['actual_metrics']))
        candidates.extend(verify(r)for r in e['native_metric_candidates'])
        reused=None
        for old in candidates:
            reused=_actual_chain(old,e['prediction']['sha256'],gtref['sha256'],params,sources)
            if reused is not None:break
        if reused:
            m0=read(reused['metrics']['path']);result=dict(stats=m0['stats'],AP=m0['AP'],AP50=m0['AP50'],AP75=m0['AP75'],native_log=m0['native_log'],AP_reused=True,AP_computed_here=False,actual_AP_computed_here=False,actual_native_evaluation=False,original_actual_native_evaluation=True,reused_source=reused)
        else:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            log=ep.parent/'native_COCO.log'
            with native_slot(root),log.open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                cg=COCO(str(gtpath))
                if e['prediction_count']:cd=cg.loadRes(e['prediction']['path'])
                else:cd=COCO();cd.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);cd.createIndex()
                ev=COCOeval(cg,cd,'bbox')
                for k in ['imgIds','catIds','maxDets','areaRng','areaRngLbl','useCats']:setattr(ev.params,k,params[k])
                ev.params.iouThrs=np.asarray(params['iouThrs'],np.float64);ev.params.recThrs=np.asarray(params['recThrs'],np.float64);ev.evaluate();ev.accumulate();ev.summarize()
            stats=[float(x)for x in ev.stats];result=dict(stats=stats,AP=stats[0],AP50=stats[1],AP75=stats[2],native_log=ref(log),AP_reused=False,AP_computed_here=True,actual_AP_computed_here=True,actual_native_evaluation=True,reused_source=None)
        result.update(created_utc=now(),export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=gtref,params=params,**sources,reuse_key=key,GT_access_purpose='CLC tune-only global beta selection'if e['identity']['dataset']=='CLC_tune'else'evaluation only; no test parameter selection',nativeV_membership_bbox_image_order_unchanged=True,selected_class_changes=e['native_class_changes'])
        fresh_write(mp,result)
        if not entry.exists():fresh_write(entry,dict(key=key,actual_metrics=reused['metrics']if reused else ref(mp)))
        return result
