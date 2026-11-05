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
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_o_balanced_quality')
PSHA='eb316d97bdcb4035a988d7b5350073049db033a034d7988d92dbe2ab92bb2937'
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
 p=read(pp);assert p['status'].startswith('FROZEN');return p,ref(pp),None
HELPER=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_quality_reranking_20261031/src/quality_endpoint.py')
GAUSSIAN=Path('/mnt/data/yxy/program/RAXO/experiments/appearance_candidate_refinement_20261031/src/candidate_endpoint.py')
GSHA='9d818252fcf47334ea2019c4cb0466b6c6244924418d041b0ee9c9644c863fac'
spec=importlib.util.spec_from_file_location('_union_readonly_coco',HELPER);pure=importlib.util.module_from_spec(spec);spec.loader.exec_module(pure)
gs=importlib.util.spec_from_file_location('_union_readonly_gaussian',GAUSSIAN);gauss=importlib.util.module_from_spec(gs);gs.loader.exec_module(gauss)
ROUTES=['router','matched_priority'];VARIANTS=['K_material','N_material']
@contextlib.contextmanager
def lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a')as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def checked(protocol,root=OUT):
    root=Path(root).resolve()
    if root!=OUT or sha(GAUSSIAN)!=GSHA:raise ValueError('New O root/frozen Gaussian source required')
    p,pr,_=frozen(protocol);return p,pr,root
def source(case,protocol,output_root=OUT):
    p,pr,root=checked(protocol,output_root)
    assert case in p['inference_cases']
    _,_,_,F,Fref,a,C,M,mapping,records=ce.source(case,CPP,ce.OUT)
    assert p['cases'][case]['C_feature_manifest']==Fref
    return p,pr,root,F,Fref,a,C,M,mapping,records
def direct(base,material,valid):
    q=np.zeros(valid.shape,np.float32);b=np.clip(base[valid].astype(np.float64),1e-6,1.);m=np.clip(material[valid].astype(np.float64),1e-6,1.)
    q[valid]=(np.power(b,.5)*np.power(m,.5)).astype(np.float32)
    return q

def residual(base,geometry,material,available,valid):
    q=base.copy();available=available&valid;eps=1e-6
    b,g,m=[np.clip(x[available].astype(np.float64),eps,1.-eps)for x in [base,geometry,material]]
    logit=lambda x:np.log(x)-np.log1p(-x)
    delta=logit(m)-logit(g);value=logit(b)+delta
    s=np.empty(value.shape,np.float64);positive=value>=0;s[positive]=1./(1.+np.exp(-value[positive]));e=np.exp(value[~positive]);s[~positive]=e/(1.+e)
    q[available]=s.astype(np.float32);q[~valid]=0.
    assert np.array_equal(q[~available].view(np.uint32),base[~available].view(np.uint32))
    return q,dict(available_candidates=int(available.sum()),valid_without_material=int((valid&~available).sum()),positive_residuals=int((delta>0).sum()),negative_residuals=int((delta<0).sum()),zero_residuals=int((delta==0).sum()),signed_residual_kept=True,missing_R_F32_exact=True)

def compose(base,geometry,material,available,valid,t):
    D=direct(base,material,valid);R,stats=residual(base,geometry,material,available,valid)
    if t==0.:q=R.copy();assert q.tobytes()==R.tobytes()
    elif t==1.:q=D.copy();assert q.tobytes()==D.tobytes()
    else:
        q=np.zeros(valid.shape,np.float32);d=np.clip(D[valid].astype(np.float64),1e-6,1.);r=np.clip(R[valid].astype(np.float64),1e-6,1.);q[valid]=(np.power(d,float(t))*np.power(r,1.-float(t))).astype(np.float32)
    assert q.dtype==np.float32 and np.isfinite(q).all()and np.all((q>=0)&(q<=1))and not np.any(q[~valid])
    stats.update(t=float(t),t0_exact_R=t==0.,t1_exact_D=t==1.,D_array_sha256=hashlib.sha256(D.tobytes()).hexdigest(),R_array_sha256=hashlib.sha256(R.tobytes()).hexdigest())
    return q,stats

def quality(case,variant,root,F,Fref,M,mix):
    p=read(CODE/'config/protocol.json');assert variant in p['families']and mix in p['mixing']['t_candidates'];i=p['cases'][case]
    brp=verify(i['base_receipt']);br=read(brp);bqp=verify(i['base_scores']);base=loadnpz(bqp)['quality']
    assert br['protocol_sha256']=='e5be43a129dea7e5993f89d4135364e0793a5c4f80a3ff9d3d07cd7c98e48cfb'and(br['case'],br['variant'])==(case,'visual_clean')and br['feature_manifest']==Fref and br['candidate_row_key_sha256']==F['candidate_row_key_sha256']and br['scores']==i['base_scores']and br['seeds']==[17,29,43]and br['output_sha256']==sha(bqp)
    gp=verify(i['geometry_scores']);gr=read(verify(i['geometry_receipt']));geometry=loadnpz(gp)['quality'];assert gr['protocol_sha256']=='86f3a155818607ba8447443ba99cc0f7dfbc09aef1888f496cd2ed32f32a112f'and(gr['case'],gr['variant'])==(case,'geometry')and gr['candidate_row_key_sha256']==F['candidate_row_key_sha256']and gr['seeds']==[17,29,43]and gr['scores']==i['geometry_scores']and gr['output_sha256']==sha(gp)
    family=i['families'][variant];mfref=family['fresh_feature_manifest'];mf=read(verify(mfref));mp=verify(family['material_scores']);mr=read(verify(family['material_receipt']));material=loadnpz(mp)['quality']
    assert mr['protocol_sha256']==family['source_protocol']['sha256']and(mr['case'],mr['variant'])==(case,'material')and mr['feature_manifest']==mfref and mr['candidate_row_key_sha256']==F['candidate_row_key_sha256']and mr['seeds']==[17,29,43]and mr['scores']==family['material_scores']and mr['output_sha256']==sha(mp)
    assert mf['source_C_feature_manifest']==Fref and mf['shape']==[len(M),2,13]
    masks=[]
    for fname in p['families']:
        ff=i['families'][fname];fp=verify(ff['material_feature_array']);feature=np.load(fp,mmap_mode='r',allow_pickle=False);assert feature.shape==M.shape+(13,);masks.append((feature[:,:,12]>0)&M)
    assert np.array_equal(masks[0],masks[1]);available=masks[0]
    for q in [base,geometry,material]:assert q.dtype==np.float32 and q.shape==M.shape and np.isfinite(q).all()and np.all((q>=0)&(q<=1))and not np.any(q[~M])
    q,stats=compose(base,geometry,material,available,M,float(mix))
    qr=dict(kind='GLOBAL_BALANCE_DIRECT_AND_SIGNED_MATERIAL_RESIDUAL',family=variant,balance_t=float(mix),direct_weight=.5,residual_weight=1.,base_scores=i['base_scores'],base_receipt=i['base_receipt'],geometry_scores=i['geometry_scores'],geometry_receipt=i['geometry_receipt'],material_family=family,source_C_feature_manifest=Fref,material_mask_identical_K_N=True,mask_array_sha256=hashlib.sha256(available.tobytes()).hexdigest(),mixed_array_sha256=hashlib.sha256(q.tobytes()).hexdigest(),math_statistics=stats,precision='D/R eachF64->F32; t0exactR/t1exactD; interiorclippedF64powers->F32',active_model_count=6 if mix==1. else 9,shared_material_head_not_double_counted=True,new_training=0,new_GPU=0,new_GT=0,legacy_native_pool_unchanged=True)
    return q,qr
def base_metric_candidates(case,route):
    eroot=OUT.parent/'batch_e_affine_robustness'
    if case=='CLC_tune':return []
    route={'V':'visual','F':'fixed_additive'}.get(route,route)
    if route=='router':return []
    ix=read(eroot/'run_index.json');assert ix['status']=='COMPLETE';rr=[r for r in ix['endpoints']if(r['dataset'],r['variant'],r['route'])==(case,'visual_clean',route)];assert len(rr)==1
    return [ref(rr[0]['metrics'])]
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
def export_endpoint(case,variant,route,tau,protocol,output_root=OUT,phase='final',mix=0.):
    if variant not in VARIANTS or route not in ROUTES:raise ValueError('Invalid union variant/route')
    p,pr,root,F,Fref,a,candidates,valid,mapping,records=source(case,protocol,output_root);tau=float(tau)
    if tau!=p['routing']['tau']:raise ValueError('K uses exactly tau0')
    if case=='CLC_tune'and route!='router':raise ValueError('Only router calibrates the global mixing weight')
    q,qr=quality(case,variant,root,F,Fref,valid,float(mix));eff=branch_quality(q,a['branch_kept'],mapping);same=a['branch_kept'].all(1)&(a['branch_category_id'][:,0]==a['branch_category_id'][:,1])
    if not np.array_equal(eff[same,0],eff[same,1]):raise ValueError('Same category branch qualities differ')
    selected,useF,routermask=choose(a['branch_kept'],eff,route,tau);rows=np.flatnonzero(selected>=0);branch=selected[rows].astype(np.int64);slot=mapping[rows,branch];category=a['branch_category_id'][rows,branch];sq=q[rows,slot];raw=a['raw_score'][rows].astype(np.float64);clipped=np.clip(sq.astype(np.float64),1e-6,1.);pre=np.power(raw,.5)*np.power(clipped,.5)
    post,rank=gauss.gaussian(pre,a['bbox'][rows],a['image_id'][rows],category,a['global_cache_index'][rows])
    identity=dict(mix_weight=float(mix),exploration_timeline=p['exploration_timeline'],dataset=case,split='tune'if case=='CLC_tune'else'eval',variant=variant,route=route,tau=tau,case=phase,protocol=pr,feature_manifest=Fref,source_prediction_V=records['V']['prediction'],source_prediction_F=records['F']['prediction'],native_source_trace=F['outputs']['source_trace.npz'],quality=qr,exporter=ref(__file__),readonly_gaussian=ref(GAUSSIAN),readonly_coco_helpers=ref(HELPER),candidate_row_key_sha256=F['candidate_row_key_sha256'],alpha=.5,sigma=1.,routing_precision='F32 q promoted toF64, absent branchquality0; strict f-v>tau; equality prefersV')
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
        e=dict(status='BALANCED_DIRECT_RESIDUAL_ENDPOINT_EXPORTED',created_utc=now(),identity_sha256=key,identity=identity,prediction=ref(pred),score_trace=ref(tp),prediction_count=len(rows),counts=counts,source_GT=records['V']['GT'],source_params=records['V']['params'],native_metric_candidates=[records['V']['metrics_ref'],records['F']['metrics_ref']]+base_metric_candidates(case,route),source_native_records=records,at_most_one_output_per_global_proposal=True,selected_class_from_retained_native_branch=True,selected_branch_original_UNC_passed=True,UNC_recomputed=False,new_class_created=False,bbox_rawscore_global_order_preserved=True,sameclass_branch_q_exact=True,matched_priority_membership_equals_router=True if route in ['router','matched_priority']else None,GT_used_for_export=False)
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
def actual(path,pred,gt,params,sources,seen=None):
    path=Path(path).resolve();seen=set()if seen is None else seen
    if str(path)in seen:return None
    seen.add(str(path));m=read(path);a=pure._completed_actual_source(path,pred,gt,params,sources)
    if a is not None:return a
    if m.get('AP_reused')and isinstance(m.get('reused_source'),dict)and'metrics'in m['reused_source']:return actual(verify(m['reused_source']['metrics']),pred,gt,params,sources,seen)
    return None
def evaluate_export(export_receipt,output_root=OUT):
    root=Path(output_root).resolve();ep=Path(export_receipt).resolve()
    if root!=OUT or not ep.is_relative_to(root):raise ValueError('Only new O output root may be written')
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
        for previous_registry in p['evaluation']['readonly_external_registry_roots']:
            prior=Path(previous_registry)/(key+'.json')
            if prior.exists():candidates.append(verify(read(prior)['actual_metrics']))
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
        m.update(created_utc=now(),export_sha256=sha(ep),identity_sha256=e['identity_sha256'],prediction=e['prediction'],original_GT=gtref,params=params,**sources,reuse_key=key,GT_access_purpose='CLC tune one global family andbalance t'if e['identity']['dataset']=='CLC_tune'else'evaluation only, no per-test parameter selection',native_global_slots=4)
        save(mp,m)
        if not entry.exists():save(entry,dict(key=key,actual_metrics=reused['metrics']if reused else ref(mp)))
        return m
