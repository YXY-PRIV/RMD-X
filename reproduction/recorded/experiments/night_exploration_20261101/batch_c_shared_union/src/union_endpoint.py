"""Native-branch union routing with one exact shared class-candidate quality.

Both available alternatives are published native retained branch predictions.
The router never creates a third class or a new box, and each chosen branch's
original negative/UNC state has already passed. All AP writers use the new root.
"""
from __future__ import annotations
import contextlib,fcntl,importlib.util,json,re,sys,time
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
from .union_data import CODE,OUT,PSHA,frozen,read,sha,ref,verify,digest,save,now,loadnpz
HELPER=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_quality_reranking_20261031/src/quality_endpoint.py')
GAUSSIAN=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py')
GSHA='9d818252fcf47334ea2019c4cb0466b6c6244924418d041b0ee9c9644c863fac'
spec=importlib.util.spec_from_file_location('_union_readonly_coco',HELPER);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
gs=importlib.util.spec_from_file_location('_union_readonly_gaussian',GAUSSIAN);gauss=importlib.util.module_from_spec(gs);gs.loader.exec_module(gauss)
ROUTES=['visual','fixed_additive','router','matched_priority'];VARIANTS=['full','visual','missing']
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def checked(protocol,root=OUT):
    root=Path(root).resolve()
    if root!=OUT or sha(GAUSSIAN)!=GSHA:raise ValueError('New C root/frozen Gaussian source required')
    p,pr,_=frozen(protocol);return p,pr,root
def source(case,protocol,output_root=OUT):
    p,pr,root=checked(protocol,output_root)
    if case not in p['inference_cases']:raise ValueError('Case not declared for endpoints')
    sp=root/'features'/case/'status.json';st=read(sp);fp=root/'features'/case/'feature_manifest.json';F=read(fp)
    if st['status']!='COMPLETE'or st['feature_manifest']!=ref(fp)or F['protocol_sha256']!=PSHA:raise ValueError('Candidate case publication not complete/bound')
    O=F['outputs']
    for name in ['candidate_category_id.npy','candidate_proto_index.npy','candidate_valid.npy','branch_to_candidate.npy','candidate_visual_scores.npy','source_trace.npz']:verify(O[name])
    C=np.load(O['candidate_category_id.npy']['path'],allow_pickle=False);M=np.load(O['candidate_valid.npy']['path'],allow_pickle=False);mapping=np.load(O['branch_to_candidate.npy']['path'],allow_pickle=False);a=loadnpz(O['source_trace.npz']['path']);N=len(a['global_cache_index'])
    if C.shape!=(N,2)or M.shape!=(N,2)or M.dtype!=bool or mapping.shape!=(N,2)or C.dtype!=np.int64 or mapping.dtype!=np.int64 or not np.all(M[:,0]):raise ValueError('Union candidate masks malformed')
    kept=a['branch_kept'];cats=a['branch_category_id']
    if kept.shape!=(N,2)or kept.dtype!=bool or np.any((mapping>=0)!=kept)or np.any(mapping[kept]>1)or np.any(C[~M]!=-1):raise ValueError('Native branch/candidate mapping malformed')
    for branch in [0,1]:
        idx=np.flatnonzero(kept[:,branch]);slot=mapping[idx,branch]
        if not np.all(M[idx,slot])or not np.array_equal(C[idx,slot],cats[idx,branch]):raise ValueError('Native category not exact shared candidate slot')
    same=kept.all(1)&(cats[:,0]==cats[:,1])
    if not np.array_equal(mapping[same,0],mapping[same,1]):raise ValueError('Same-class branches do not share exact q slot')
    records=F['source_native_records']
    if not isinstance(records,dict)or set(records)!=set(['V','F']):raise ValueError('Endpoint requires two native source records')
    if records['V']['GT']!=records['F']['GT']or records['V']['params']!=records['F']['params']:raise ValueError('Mixed native branch GT/cohorts')
    allowed=set(records['V']['params']['catIds'])
    if any(int(c)not in allowed for c in C[M]):raise ValueError('Candidate category outside exact COCO category set')
    for r in records.values():
        for k in ['prediction','source_trace','state','GT','metrics_ref','export_ref']:verify(r[k])
    return p,pr,root,F,ref(fp),a,C,M,mapping,records
def quality(case,variant,root,F,Fref,M):
    st=read(root/'ensembles'/case/variant/'status.json')
    if st['status']!='COMPLETE':raise ValueError('Union quality ensemble incomplete')
    e=st['ensemble'];qp=verify(e['scores']);rp=verify(e['inference_receipt']);r=read(rp)
    if r['protocol_sha256']!=PSHA or(r['case'],r['variant'])!=(case,variant)or r['seeds']!=[17,29,43]or r['feature_manifest']!=Fref or r['candidate_row_key_sha256']!=F['candidate_row_key_sha256']or r['output_sha256']!=sha(qp):raise ValueError('Union quality identity differs')
    q=loadnpz(qp)['quality']
    if q.dtype!=np.float32 or q.shape!=M.shape or np.any(q[~M]!=0)or not np.isfinite(q).all()or np.any((q<0)|(q>1)):raise ValueError('Union quality range/mask invalid')
    return q,dict(scores=ref(qp),inference_receipt=ref(rp),candidate_row_key_sha256=F['candidate_row_key_sha256'])
def branch_quality(q,kept,mapping):
    effective=np.zeros(kept.shape,np.float32)
    for branch in [0,1]:
        idx=np.flatnonzero(kept[:,branch]);effective[idx,branch]=q[idx,mapping[idx,branch]]
    return effective
def choose(kept,effective,route,tau):
    N=len(kept);selected=np.full(N,-1,np.int8);v,f=effective[:,0].astype(np.float64),effective[:,1].astype(np.float64)
    useF=kept[:,1]&((f-v)>float(tau));router=kept[:,0]|useF
    if route=='visual':selected[kept[:,0]]=0
    elif route=='fixed_additive':selected[kept[:,1]]=1
    elif route=='router':selected[kept[:,0]]=0;selected[useF]=1
    elif route=='matched_priority':selected[kept[:,0]]=0;selected[~kept[:,0]&useF]=1
    else:raise ValueError('Unknown route')
    if route in ['router','matched_priority']and not np.array_equal(selected>=0,router):raise ValueError('Matched budget membership differs')
    if tau==1.01 and route in ['router','matched_priority']and(not np.array_equal(selected>=0,kept[:,0])or np.any(selected[selected>=0]!=0)):raise ValueError('Tau1.01 must be exact V route membership')
    return selected,useF,router
def export_endpoint(case,variant,route,tau,protocol,output_root=OUT,phase='final'):
    if variant not in VARIANTS or route not in ROUTES:raise ValueError('Invalid union variant/route')
    p,pr,root,F,Fref,a,candidates,valid,mapping,records=source(case,protocol,output_root);tau=float(tau)
    if tau not in p['routing']['tau_candidates']:raise ValueError('Threshold outside frozen grid')
    if case=='CLC_tune'and route!='router':raise ValueError('Only router tunes global threshold')
    q,qr=quality(case,variant,root,F,Fref,valid);eff=branch_quality(q,a['branch_kept'],mapping);same=a['branch_kept'].all(1)&(a['branch_category_id'][:,0]==a['branch_category_id'][:,1])
    if not np.array_equal(eff[same,0],eff[same,1]):raise ValueError('Same category branch qualities differ')
    selected,useF,routermask=choose(a['branch_kept'],eff,route,tau);rows=np.flatnonzero(selected>=0);branch=selected[rows].astype(np.int64);slot=mapping[rows,branch];category=a['branch_category_id'][rows,branch];sq=q[rows,slot];raw=a['raw_score'][rows].astype(np.float64);clipped=np.clip(sq.astype(np.float64),1e-6,1.);pre=np.power(raw,.5)*np.power(clipped,.5)
    post,rank=gauss.gaussian(pre,a['bbox'][rows],a['image_id'][rows],category,a['global_cache_index'][rows])
    identity=dict(dataset=case,split='tune'if case=='CLC_tune'else'eval',variant=variant,route=route,tau=tau,case=phase,protocol=pr,feature_manifest=Fref,source_prediction_V=records['V']['prediction'],source_prediction_F=records['F']['prediction'],native_source_trace=F['outputs']['source_trace.npz'],quality=qr,exporter=ref(__file__),readonly_gaussian=ref(GAUSSIAN),readonly_coco_helpers=ref(HELPER),candidate_row_key_sha256=F['candidate_row_key_sha256'],alpha=.5,sigma=1.,routing_precision='F32 q promoted toF64, absent branchquality0; strict f-v>tau; equality prefersV')
    key=digest(identity);tag=re.sub(r'[^A-Za-z0-9_-]','_',str(phase))[:24];ttag=format(tau,'.12g').replace('.','p');d=root/'endpoints'/case/f'{variant}_{route}_t{ttag}_{tag}_{key[:16]}';ep=d/'export.json'
    with lock(root/'locks'/('export_'+key+'.lock')):
        if ep.exists():
            e=read(ep)
            if e['identity_sha256']!=key:raise ValueError('Union export resume identity differs')
            verify(e['prediction']);verify(e['score_trace']);return e
        if d.exists()and any(d.iterdir()):raise FileExistsError('Unsealed union endpoint '+str(d))
        d.mkdir(parents=True,exist_ok=True);pred=d/'predictions.json';save(pred,[dict(image_id=int(a['image_id'][i]),category_id=int(c),bbox=a['bbox'][i].tolist(),score=float(s))for i,c,s in zip(rows,category,post)])
        tp=d/'score_trace.npz'
        with tp.open('xb')as stream:np.savez_compressed(stream,global_cache_index=a['global_cache_index'][rows],selected_union_index=rows,selected_branch=branch,selected_candidate_slot=slot,selected_category_id=category,selected_native_proto_index=a['branch_native_proto_index'][rows,branch],selected_quality=sq,clipped_quality=clipped,native_raw_score=raw,pre_soft_score=pre,post_soft_score=post,soft_selection_rank=rank,effective_branch_quality=eff,router_keep_mask=routermask,selected_union_keep_mask=selected>=0)
        counts=dict(union_proposals=len(selected),retained=len(rows),V_selected=int((branch==0).sum()),F_selected=int((branch==1).sum()),F_only_admitted=int((~a['branch_kept'][rows,0]&a['branch_kept'][rows,1]).sum()),overlap_F_selected=int((a['branch_kept'][rows,0]&(branch==1)).sum()),overlap_category_changes=int(((a['branch_category_id'][rows,0]!=category)&a['branch_kept'][rows,0]).sum()))
        e=dict(status='SHARED_UNION_ENDPOINT_EXPORTED',created_utc=now(),identity_sha256=key,identity=identity,prediction=ref(pred),score_trace=ref(tp),prediction_count=len(rows),counts=counts,source_GT=records['V']['GT'],source_params=records['V']['params'],native_metric_candidates=[records['V']['metrics_ref'],records['F']['metrics_ref']],source_native_records=records,at_most_one_output_per_global_proposal=True,selected_class_from_retained_native_branch=True,selected_branch_original_UNC_passed=True,UNC_recomputed=False,new_class_created=False,bbox_rawscore_global_order_preserved=True,sameclass_branch_q_exact=True,matched_priority_membership_equals_router=True if route in ['router','matched_priority']else None,GT_used_for_export=False)
        save(ep,e);return e
@contextlib.contextmanager
def native_slot(p):
    directory=Path(p['resources']['native_lock_directory']);expected=OUT.parent/'resources/native_slots'
    if directory!=expected or p['resources']['native_lock_names']!=[f'slot{i}.lock'for i in range(4)]:raise ValueError('Night global four-slot contract differs')
    directory.mkdir(parents=True,exist_ok=True);selected=None
    try:
        while selected is None:
            for name in p['resources']['native_lock_names']:
                stream=(directory/name).open('a')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);selected=stream;break
                except BlockingIOError:stream.close()
            if selected is None:time.sleep(.2)
        yield
    finally:
        if selected is not None:fcntl.flock(selected,fcntl.LOCK_UN);selected.close()
def actual(path,pred,gt,params,sources,seen=None):
    path=Path(path).resolve();seen=set()if seen is None else seen
    if str(path)in seen:return None
    seen.add(str(path));m=read(path);a=pure._completed_actual_source(path,pred,gt,params,sources)
    if a is not None:return a
    if m.get('AP_reused')and isinstance(m.get('reused_source'),dict)and'metrics'in m['reused_source']:return actual(verify(m['reused_source']['metrics']),pred,gt,params,sources,seen)
    return None
def evaluate_export(export_receipt,output_root=OUT):
    root=Path(output_root).resolve();ep=Path(export_receipt).resolve()
    if root!=OUT or not ep.is_relative_to(root):raise ValueError('Only new C output root may be written')
    e=read(ep);p,pr,_=checked(e['identity']['protocol']['path'],root);verify(e['prediction']);verify(e['score_trace']);sources=pure._native_sources();mp=ep.parent/'metrics.json'
    if mp.exists():
        m=read(mp)
        if m['export_sha256']!=sha(ep)or m['identity_sha256']!=e['identity_sha256']or any(m[k]['sha256']!=sources[k]['sha256']for k in sources):raise ValueError('C metric resume identity/source mismatch')
        verify(m['native_log']);verify(m['original_GT'])
        if m['AP_reused']:
            for k in ['metrics','prediction','original_GT','native_log']:verify(m['reused_source'][k])
        return m
    gtref=e['source_GT'];gtpath=verify(gtref);gt=read(gtpath);ids=[int(a['id'])for a in gt['annotations']]
    if len(set(ids))!=len(ids)or any(x<=0 for x in ids):raise ValueError('Positive unique GT annotation IDs required')
    params=pure._params(gt)
    if params!=e['source_params']:raise ValueError('C endpoint source COCO params/cohort mismatch')
    pred=e['prediction']['sha256'];key=digest(dict(prediction=pred,GT=gtref['sha256'],params=params,sources={k:v['sha256']for k,v in sources.items()}));registry=root/'evaluation_registry';registry.mkdir(exist_ok=True);entry=registry/(key+'.json')
    with lock(root/'locks'/('evaluation_'+key+'.lock')):
        candidates=[]
        if entry.exists():candidates.append(verify(read(entry)['actual_metrics']))
        candidates.extend(verify(r)for r in e['native_metric_candidates']);reused=None
        for path in candidates:
            reused=actual(path,pred,gtref['sha256'],params,sources)
            if reused is not None:break
        if reused:
            old=read(reused['metrics']['path']);m=dict(stats=old['stats'],AP=old['AP'],AP50=old['AP50'],AP75=old['AP75'],AR100=old['stats'][8],native_log=old['native_log'],AP_reused=True,AP_computed_here=False,actual_AP_computed_here=False,actual_native_evaluation=False,reused_source=reused)
        else:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            log=ep.parent/'native_COCO.log'
            with native_slot(p),log.open('x')as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
                cg=COCO(str(gtpath))
                if e['prediction_count']:cd=cg.loadRes(e['prediction']['path'])
                else:cd=COCO();cd.dataset=dict(images=gt['images'],categories=gt['categories'],annotations=[]);cd.createIndex()
                ev=COCOeval(cg,cd,'bbox')
                for k in ['imgIds','catIds','maxDets','areaRng','areaRngLbl','useCats']:setattr(ev.params,k,params[k])
                ev.params.iouThrs=np.asarray(params['iouThrs'],np.float64);ev.params.recThrs=np.asarray(params['recThrs'],np.float64);ev.evaluate();ev.accumulate();ev.summarize()
            stats=[float(x)for x in ev.stats];m=dict(stats=stats,AP=stats[0],AP50=stats[1],AP75=stats[2],AR100=stats[8],native_log=ref(log),AP_reused=False,AP_computed_here=True,actual_AP_computed_here=True,actual_native_evaluation=True,reused_source=None)
        m.update(created_utc=now(),export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=gtref,params=params,**sources,reuse_key=key,GT_access_purpose='CLC tune global threshold selection only'if e['identity']['dataset']=='CLC_tune'else'evaluation only, no per-test parameter selection',native_global_slots=4)
        save(mp,m)
        if not entry.exists():save(entry,dict(key=key,actual_metrics=reused['metrics']if reused else ref(mp)))
        return m
