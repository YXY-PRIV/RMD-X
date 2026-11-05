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
import hashlib,importlib
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_q_supported_rank_seed_stability')
PSHA='0433953caa21af2fff35b8b95e8ca5f3d8db8a95414c5bb7e2cd68867e9065d0'
sys.path.insert(0,str(CODE.parent))
ce=importlib.import_module('batch_c_shared_union.src.union_endpoint')
CPP=CODE.parent/'batch_c_shared_union/config/protocol.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
 p=Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'];return p
def digest(j):return hashlib.sha256(json.dumps(j,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def save(p,j):
 p=Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
def now():return __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
def loadnpz(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy()for k in z.files}
def frozen(protocol):
    pp=Path(protocol).resolve();assert pp==CODE/'config/protocol.json'and sha(pp)==PSHA
    p=read(pp);assert p['status']=='FROZEN_BEFORE_ANY_Q_QUALITY_SMOKE_EXPORT_OR_AP'and p['resources']['CPU_environment_threads']==8 and p['resources']['GPU_runs']==0;return p,ref(pp),None
HELPER=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_quality_reranking_20261031/src/quality_endpoint.py')
GAUSSIAN=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py')
GSHA='9d818252fcf47334ea2019c4cb0466b6c6244924418d041b0ee9c9644c863fac'
spec=importlib.util.spec_from_file_location('_union_readonly_coco',HELPER);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
gs=importlib.util.spec_from_file_location('_union_readonly_gaussian',GAUSSIAN);gauss=importlib.util.module_from_spec(gs);gs.loader.exec_module(gauss)
ROUTES=['router'];VARIANTS=['seed17','seed29','seed43']
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def checked(protocol,root=OUT):
    root=Path(root).resolve()
    if root!=OUT or sha(GAUSSIAN)!=GSHA:raise ValueError('New Q root/frozen Gaussian source required')
    p,pr,_=frozen(protocol);return p,pr,root
def source(case,protocol,output_root=OUT):
    p,pr,root=checked(protocol,output_root)
    assert case in p['inference_cases']
    _,_,_,F,Fref,a,C,M,mapping,records=ce.source(case,CPP,ce.OUT)
    assert p['cases'][case]['feature_manifest']==Fref
    return p,pr,root,F,Fref,a,C,M,mapping,records
def mix_quality(baseq,head,valid):
    for q in [baseq,head]:
        assert q.dtype==np.float32 and q.shape==valid.shape and not np.any(q[~valid])and np.isfinite(q).all()and np.all((q>=0)&(q<=1))
    q=np.zeros(valid.shape,np.float32);b=np.clip(baseq[valid].astype(np.float64),1e-6,1.);h=np.clip(head[valid].astype(np.float64),1e-6,1.)
    q[valid]=(np.power(b,.5)*np.power(h,.5)).astype(np.float32);return q

def quality(case,variant,root,F,Fref,M,mix):
    assert variant in VARIANTS and mix==.5
    p=read(CODE/'config/protocol.json');cfg=p['cases'][case];seed=int(variant[4:]);Nref=cfg['N_feature_manifest'];NF=read(verify(Nref))
    assert NF['protocol_sha256']==p['sources']['N_protocol']['sha256']and NF['source_C_feature_manifest']==Fref and NF['candidate_row_key_sha256']==F['candidate_row_key_sha256']
    assert NF['outputs']['candidate_valid.npy']==F['outputs']['candidate_valid.npy']and NF['outputs']['candidate_category_id.npy']==F['outputs']['candidate_category_id.npy']
    item=cfg['seed_heads'][str(seed)];rp=verify(item['inference_receipt']);r=read(rp);qp=verify(item['scores']);verify(item['checkpoint'])
    assert(r['case'],r['variant'],r['seed'])==(case,'material',seed)and r['protocol_sha256']==p['sources']['N_protocol']['sha256']and r['feature_manifest']==Nref and r['output']==item['scores']and r['checkpoint']==item['checkpoint']and r['candidate_row_key_sha256']==F['candidate_row_key_sha256']and r['output_sha256']==sha(qp)
    assert r['GT_opened']is False and r['labels_read']is False and r['AP_computed']is False
    brp=verify(cfg['base_ensemble']);br=read(brp);bqp=verify(cfg['base_quality']);assert br['protocol_sha256']==p['sources']['E_protocol']['sha256']and(br['case'],br['variant'])==(case,'visual_clean')and br['seeds']==[17,29,43]and br['feature_manifest']==Fref and br['candidate_row_key_sha256']==F['candidate_row_key_sha256']and br['scores']==cfg['base_quality']and br['output_sha256']==sha(bqp)
    head=loadnpz(qp)['quality'];baseq=loadnpz(bqp)['quality'];q=mix_quality(baseq,head,M)
    return q,dict(kind='N_SINGLE_HEAD_SEED_DIAGNOSTIC_WITH_FIXED_E_BASE_ENSEMBLE',head_seed=seed,mix_weight=.5,base_scores=ref(bqp),base_receipt=ref(brp),base_seeds_fixed=[17,29,43],auxiliary_scores=ref(qp),auxiliary_receipt=ref(rp),auxiliary_checkpoint=item['checkpoint'],N_feature_manifest=Nref,source_C_feature_manifest=Fref,mixed_array_sha256=hashlib.sha256(q.tobytes()).hexdigest(),mix_precision='FP64 clip to[1e-6,1], power.5 each, one F32 cast, padding0',active_model_count=4,N_main_ensemble_model_count=6,not_same_budget_ablation=True,legacy_native_pool_unchanged=True)

def branch_quality(q,kept,mapping):
    effective=np.zeros(kept.shape,np.float32)
    for branch in [0,1]:
        idx=np.flatnonzero(kept[:,branch]);effective[idx,branch]=q[idx,mapping[idx,branch]]
    return effective
def choose(kept,effective,route,tau):
    route={'V':'visual','F':'fixed_additive'}.get(route,route)
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
def export_endpoint(case,variant,route,tau,protocol,output_root=OUT,phase='seed_diagnostic',mix=.5):
    if variant not in VARIANTS or route not in ROUTES:raise ValueError('Invalid union variant/route')
    p,pr,root,F,Fref,a,candidates,valid,mapping,records=source(case,protocol,output_root);tau=float(tau)
    if tau!=p['routing']['tau']:raise ValueError('Q fixed tau0')
    assert case in p['test_cases']and phase=='seed_diagnostic'and float(mix)==.5
    q,qr=quality(case,variant,root,F,Fref,valid,float(mix));eff=branch_quality(q,a['branch_kept'],mapping);same=a['branch_kept'].all(1)&(a['branch_category_id'][:,0]==a['branch_category_id'][:,1])
    if not np.array_equal(eff[same,0],eff[same,1]):raise ValueError('Same category branch qualities differ')
    selected,useF,routermask=choose(a['branch_kept'],eff,route,tau);rows=np.flatnonzero(selected>=0);branch=selected[rows].astype(np.int64);slot=mapping[rows,branch];category=a['branch_category_id'][rows,branch];sq=q[rows,slot];raw=a['raw_score'][rows].astype(np.float64);clipped=np.clip(sq.astype(np.float64),1e-6,1.);pre=np.power(raw,.5)*np.power(clipped,.5)
    post,rank=gauss.gaussian(pre,a['bbox'][rows],a['image_id'][rows],category,a['global_cache_index'][rows])
    identity=dict(mix_weight=float(mix),resource_configuration=p['resources'],dataset=case,split='tune'if case=='CLC_tune'else'eval',variant=variant,route=route,tau=tau,case=phase,protocol=pr,feature_manifest=Fref,source_prediction_V=records['V']['prediction'],source_prediction_F=records['F']['prediction'],native_source_trace=F['outputs']['source_trace.npz'],quality=qr,exporter=ref(__file__),readonly_gaussian=ref(GAUSSIAN),readonly_coco_helpers=ref(HELPER),candidate_row_key_sha256=F['candidate_row_key_sha256'],alpha=.5,sigma=1.,routing_precision='F32 q promoted toF64, absent branchquality0; strict f-v>tau; equality prefersV')
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
        e=dict(status='SUPPORTED_RANK_HEAD_SEED_DIAGNOSTIC_EXPORTED',created_utc=now(),identity_sha256=key,identity=identity,prediction=ref(pred),score_trace=ref(tp),prediction_count=len(rows),counts=counts,source_GT=records['V']['GT'],source_params=records['V']['params'],native_metric_candidates=[],source_native_records=records,at_most_one_output_per_global_proposal=True,selected_class_from_retained_native_branch=True,selected_branch_original_UNC_passed=True,UNC_recomputed=False,new_class_created=False,bbox_rawscore_global_order_preserved=True,sameclass_branch_q_exact=True,matched_priority_membership_equals_router=True if route in ['router','matched_priority']else None,GT_used_for_export=False)
        save(ep,e);return e
@contextlib.contextmanager
def native_slot(p):
    directory=Path(p['resources']['native_lock_root']);expected=OUT.parent/'resources/native_slots'
    if directory!=expected or p['resources']['native_global_slots']!=4:raise ValueError('Night global four-slot contract differs')
    directory.mkdir(parents=True,exist_ok=True);selected=None
    try:
        while selected is None:
            for name in [f'slot{i}.lock'for i in range(4)]:
                stream=(directory/name).open('a')
                try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB);selected=stream;break
                except BlockingIOError:stream.close()
            if selected is None:time.sleep(.2)
        yield
    finally:
        if selected is not None:fcntl.flock(selected,fcntl.LOCK_UN);selected.close()
def evaluate_export(export_receipt,output_root=OUT):
    root=Path(output_root).resolve();ep=Path(export_receipt).resolve()
    if root!=OUT or not ep.is_relative_to(root):raise ValueError('Only new Q output root may be written')
    e=read(ep);p,pr,_=checked(e['identity']['protocol']['path'],root);verify(e['prediction']);verify(e['score_trace']);sources=pure._native_sources();mp=ep.parent/'metrics.json'
    if mp.exists():
        m=read(mp);assert m['export_sha256']==sha(ep)and m['identity_sha256']==e['identity_sha256']and m['AP_computed_here']and not m['AP_reused']
        assert all(m[k]['sha256']==sources[k]['sha256']for k in sources);verify(m['native_log']);return m
    gtref=e['source_GT'];gtpath=verify(gtref);gt=read(gtpath);ids=[int(x['id'])for x in gt['annotations']]
    if len(set(ids))!=len(ids)or any(x<=0 for x in ids):raise ValueError('Positive unique GT annotation IDs required')
    params=pure._params(gt);assert params==e['source_params']
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
    stats=[float(x)for x in ev.stats];m=dict(created_utc=now(),stats=stats,AP=stats[0],AP50=stats[1],AP75=stats[2],AR100=stats[8],native_log=ref(log),AP_reused=False,AP_computed_here=True,actual_AP_computed_here=True,actual_native_evaluation=True,reused_source=None,export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=gtref,params=params,**sources,GT_access_purpose='Fixed N-head seed diagnostic evaluation only; no source calibration, seed/method/parameter selection',native_global_slots=4)
    save(mp,m);return m
