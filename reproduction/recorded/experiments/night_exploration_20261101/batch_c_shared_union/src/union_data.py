"""GT-free native V/F union candidates; cal-fit teacher inheritance only.

No native filtering is rerun. Proposal identity is the shared cache's global
row, not an image/bbox approximation. Same-class native branches share a slot.
"""
from __future__ import annotations
import csv,datetime,gc,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
CODE=Path(__file__).resolve().parents[1]
OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_c_shared_union')
PARENT=OUT.parent.parent
PSHA='285b2fd5fcdbc974c762332cf33b3962a6876f75bd8e8a6c0e53e8235d394cd7'
CAND=PARENT/'appearance_candidate_refinement_20261031'
PILOT=PARENT/'appearance_reliability_pilot_20261031'
METRIC=PARENT/'appearance_metric_quality_20261031'
SIX=PARENT/'appearance_six_dataset_transfer_20261031'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def ref(p):
    p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):
    p=Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],str(p);return p
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(d,f,indent=2,allow_nan=False);f.write('\n')
def atomic(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');q.replace(p)
def npy(p,a):
    with Path(p).open('xb')as f:np.save(f,a,allow_pickle=False)
def npz(p,**a):
    with Path(p).open('xb')as f:np.savez_compressed(f,**a)
def loadnpz(p):
    with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def frozen(protocol):
    pp=Path(protocol).resolve()
    if PSHA=='PENDING'or sha(pp)!=PSHA:raise ValueError('Batch C protocol not frozen/mismatch')
    p=read(pp)
    if not p['status'].startswith('FROZEN'):raise ValueError('Batch C not frozen')
    for r in p.get('sources',{}).values():
        if isinstance(r,dict)and{'path','bytes','sha256'}<=set(r):verify(r)
    old=read(CAND/'protocol/protocol.json')
    if sha(CAND/'protocol/protocol.json')!='b47eedfd0f52c8d33c39b915a0be61029c93f2b18d804dcddb9e41fb1793b534':raise ValueError('Sealed original source protocol changed')
    return p,ref(pp),old
def raw_source(case,old):
    ip=verify(old['sources']['metric_source_index']if case in ['DET','Hi','CLC_tune']else old['sources']['six_source_index']);index=read(ip)
    ds='CLC'if case=='CLC_tune'else case;split='tune'if case=='CLC_tune'else'eval';records={}
    for branch,name in [('visual','V'),('fixed_additive','F')]:
        matches=[r for r in index['records']if r['dataset']==ds and r['split']==split and r['branch']==branch and r.get('source','raw')=='raw'and r.get('variant','full')=='full']
        if len(matches)!=1:raise ValueError('Ambiguous native V/F source '+case+'/'+branch)
        r=dict(matches[0]);r['source_index']=ref(ip)
        if case not in ['DET','Hi','CLC_tune']:
            mp=SIX/'endpoints'/case/('raw_'+branch)/'metrics.json';m=read(mp)
            if m['prediction']['sha256']!=r['prediction']['sha256']or m['original_GT']['sha256']!=r['GT']['sha256']or m['params']!=r['params']or not m['AP_computed_here']:raise ValueError('Raw actual metric not exact source')
            r.update(metrics=str(mp),metrics_ref=ref(mp),AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'])
        for key in ['prediction','source_trace','state','GT','export_ref','metrics_ref']:verify(r[key])
        records[name]=r
    if records['V']['GT']!=records['F']['GT']or records['V']['params']!=records['F']['params']:raise ValueError('Native branch GT/cohort/params differ')
    return records
def source_trace(r):
    t=loadnpz(verify(r['source_trace']));pred=read(verify(r['prediction']));n=len(pred);g=t['global_cache_index']
    if g.shape!=(n,)or np.any(np.diff(g)<=0)or not np.array_equal(t['source_detection_index'],np.arange(n)):raise ValueError('Native global rows not uniquely ordered')
    if any(x['image_id']!=int(t['image_id'][i])or x['category_id']!=int(t['category_id'][i])or x['bbox']!=t['bbox'][i].tolist()or float(x['score']).hex()!=float(t['raw_score'][i]).hex()for i,x in enumerate(pred)):raise ValueError('Native prediction/trace mismatch')
    state=loadnpz(verify(r['state']));loc=np.searchsorted(state['global_cache_index'],g)
    if not np.array_equal(state['global_cache_index'][loc],g)or not np.all(state['after_unc'][loc])or not np.array_equal(state['category'][loc],t['category_id']):raise ValueError('Published retained state mismatch')
    t['native_proto_index']=state['winner'][loc].copy()
    return t
def pairs_from_branches(V,F):
    g=np.union1d(V['global_cache_index'],F['global_cache_index']);N=len(g);keep=np.zeros((N,2),bool);category=np.full((N,2),-1,np.int64);indices=np.full((N,2),-1,np.int64)
    trace=dict(global_cache_index=g,source_detection_index=np.arange(N,dtype=np.int64),image_id=np.empty(N,np.int64),original_image_id=np.empty(N,np.int64),bbox=np.empty((N,4),np.float64),raw_score=np.empty(N,np.float64));native_proto=np.full((N,2),-1,np.int64)
    for bi,t in enumerate([V,F]):
        loc=np.searchsorted(g,t['global_cache_index']);already=keep[loc].any(1)
        for key in ['image_id','original_image_id','bbox','raw_score']:
            if np.any(already)and not np.array_equal(trace[key][loc[already]],t[key][already]):raise ValueError('Branches disagree on common proposal '+key)
            trace[key][loc]=t[key]
        keep[loc,bi]=True;category[loc,bi]=t['category_id'];indices[loc,bi]=t['source_detection_index'];native_proto[loc,bi]=t['native_proto_index']
    trace.update(branch_kept=keep,branch_category_id=category,branch_source_detection_index=indices,branch_native_proto_index=native_proto)
    return trace
def calfit_trace(old,manifest):
    cp=verify(old['sources']['calfit_target_csv']);rows=list(csv.DictReader(cp.open()));indices=np.load(verify(manifest['assets']['row_indices']),allow_pickle=False)
    if len(rows)!=7282 or not np.array_equal(indices,[int(r['global_cache_index'])for r in rows]):raise ValueError('Cal-fit CSV role row identities differ')
    categories=np.array([[int(r['visual_category']),int(r['fused_category'])]for r in rows],np.int64);kept=np.array([[r['visual_kept']=='True',r['fused_kept']=='True']for r in rows],bool);union=kept.any(1)
    return indices[union],categories[union],kept[union],dict(csv=ref(cp),all_role_rows=7282,source_mask='Published cal_fit native kept/category fields only; GT CSV fields not used to construct features')
def selected_metadata(path,indices):
    wanted={int(g):j for j,g in enumerate(indices)};result=[None]*len(indices);total=0
    if path.suffix=='.jsonl':
        with path.open()as f:
            for g,line in enumerate(f):
                if g in wanted:result[wanted[g]]=json.loads(line)
                total=g+1
    else:
        allrows=read(path);total=len(allrows);result=[allrows[int(g)]for g in indices]
    if any(x is None for x in result):raise ValueError('Unbound global metadata row')
    return result,total
def array(path):
    import torch
    if path.suffix=='.npy':return np.load(path,allow_pickle=False,mmap_mode='r')
    a=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(a,torch.Tensor):raise TypeError('Expected immutable similarity tensor')
    return a.numpy()
def load_case(case,old):
    mp=verify(old['cases'][case]['asset_manifest']);manifest=read(mp);assets={k:verify(v)for k,v in manifest['assets'].items()};source=None
    if case=='CLC_cal_fit':
        ids,bc,bk,calmeta=calfit_trace(old,manifest);trace=None
    else:
        source=raw_source(case,old);traces={b:source_trace(r)for b,r in source.items()};trace=pairs_from_branches(traces['V'],traces['F']);ids=trace['global_cache_index'];bc,bk=trace['branch_category_id'],trace['branch_kept'];calmeta=None
        for b,r in source.items():
            e=read(r['export_ref']['path']);source_manifest=read(verify(e['identity']['manifest']))
            for k in ['visual_similarity','material_class_similarity','prototype_meta','category_meta','material_bank','proposal_meta']:
                if source_manifest['assets'][k]['sha256']!=manifest['assets'][k]['sha256']:raise ValueError('Mixed V/F/cache family '+case+'/'+b+'/'+k)
    rows,total=selected_metadata(assets['proposal_meta'],ids);mapping={int(k):int(v)for k,v in read(assets['image_id_map']).items()}if'image_id_map'in assets else{}
    actual=dict(global_cache_index=ids,source_detection_index=np.arange(len(ids),dtype=np.int64),image_id=np.array([mapping.get(int(x['image_id']),int(x['image_id']))for x in rows],np.int64),original_image_id=np.array([int(x.get('original_image_id',x['image_id']))for x in rows],np.int64),bbox=np.array([x['bbox']for x in rows],np.float64),raw_score=np.array([x['detector_score']for x in rows],np.float64))
    if trace is not None:
        for key in actual:
            if not np.array_equal(actual[key],trace[key]):raise ValueError('Union source metadata differs '+case+'/'+key)
    else:trace=dict(actual,branch_category_id=bc,branch_kept=bk,branch_source_detection_index=np.full_like(bc,-1),branch_native_proto_index=np.full_like(bc,-1))
    proto=read(assets['prototype_meta']);cats=read(assets['category_meta']);bank=read(assets['material_bank']);C=len(cats);cat_ids=np.array([int(x['category_id'])for x in cats],np.int64);catmap={int(x):i for i,x in enumerate(cat_ids)}
    if len(catmap)!=C or any(int(x)not in catmap for x in bc[bk]):raise ValueError('Native retained category not positive source class')
    if source is not None and set(cat_ids)!=set(source['V']['params']['catIds']):raise ValueError('Cache/GT category vocabulary differs')
    columns=[];canon=[]
    for ci,c in enumerate(cats):
        js=np.array([j for j,p in enumerate(proto)if not p.get('is_negative')and int(p['material_class_index'])==ci],np.int64)
        if len(js)==0:raise ValueError('Positive class without prototype')
        cs={int(proto[int(j)]['canonical_index'])for j in js}
        if len(cs)!=1:raise ValueError('Canonical class reference not fixed')
        canonical=next(iter(cs));canon.append(canonical)
        if proto[canonical].get('is_negative')or int(proto[canonical]['category_id'])!=int(c['category_id']):raise ValueError('Wrong canonical class')
        for j in js:
            p=proto[int(j)]
            if int(p['prototype_index'])!=int(j)or p['category_name']!=c['category_name']or int(p['category_id'])!=int(c['category_id']):raise ValueError('Prototype/category exact mapping differs')
        columns.append(js)
    byname={x['category_name']:x for x in bank['prototypes'].values()};support=np.array([float(byname[x['category_name']]['sample_count'])for x in cats],np.float32)
    if not np.isfinite(support).all()or np.any(support<0):raise ValueError('Invalid actual support count')
    V=array(assets['visual_similarity']);B=array(assets['material_class_similarity'])
    if V.dtype!=np.float32 or B.dtype!=np.float32 or V.shape!=(total,len(proto))or B.shape!=(total,C):raise ValueError('Immutable source matrix shape/dtype mismatch')
    # Read only union rows; original tensor bytes are never changed.
    v=np.asarray(V[ids],np.float32);b=np.asarray(B[ids],np.float32);del V,B;gc.collect()
    return dict(case=case,manifest=ref(mp),assets={k:ref(v)for k,v in assets.items()},V=v,B=b,trace=trace,columns=columns,canon=np.array(canon,np.int64),cats=cats,cat_ids=cat_ids,catmap=catmap,support=support,source=source,calfit_source=calmeta,total_cache_rows=total)
def candidate_mapping(trace):
    kept,bc=trace['branch_kept'],trace['branch_category_id'];N=len(kept);cats=np.full((N,2),-1,np.int64);valid=np.zeros((N,2),bool);mapping=np.full((N,2),-1,np.int64)
    for i in range(N):
        found={}
        for branch in range(2):
            if not kept[i,branch]:continue
            c=int(bc[i,branch])
            if c not in found:
                slot=len(found);found[c]=slot;cats[i,slot]=c;valid[i,slot]=True
            mapping[i,branch]=found[c]
    if not np.all(valid[:,0])or np.any((mapping>=0)!=kept):raise ValueError('Union candidate mask malformed')
    both=kept.all(1);same=both&(bc[:,0]==bc[:,1])
    if not np.all(mapping[same,0]==mapping[same,1]):raise ValueError('Same class failed exact slot sharing')
    return cats,valid,mapping
def features(bundle,chunk=1024):
    import torch
    V,B=bundle['V'],bundle['B'];N,J=V.shape;C=B.shape[1];cats,valid,mapping=candidate_mapping(bundle['trace']);X=np.zeros((N,2,15),np.float32);pp=np.full((N,2),-1,np.int64);counts=np.array([len(j)for j in bundle['columns']],np.float32)
    for start in range(0,N,chunk):
        stop=min(start+chunk,N);v=torch.from_numpy(V[start:stop]);b=B[start:stop];n=stop-start
        maxima=[];winners=[]
        for js in bundle['columns']:
            mx,offset=v[:,js].max(1);maxima.append(mx.numpy());winners.append(js[offset.numpy()])
        cv=np.stack(maxima,1);wp=np.stack(winners,1);order=np.lexsort((wp,-cv),axis=1);vrank=np.empty_like(order);np.put_along_axis(vrank,order,np.broadcast_to(np.arange(C),(n,C)),axis=1)
        br=np.argsort(-b,axis=1,kind='stable');brank=np.argsort(br,axis=1);bt=np.sort(b,axis=1);margin=bt[:,-1]-bt[:,-2]
        top=torch.topk(v,2,dim=1).values.numpy();log=torch.log_softmax(v,dim=1);entropy=(-(log.exp()*log).sum(1)/float(np.log(J))).numpy()
        for slot in range(2):
            mask=valid[start:stop,slot];rr=np.flatnonzero(mask)
            if not len(rr):continue
            ci=np.array([bundle['catmap'][int(x)]for x in cats[start:stop,slot][mask]],np.int64);other=np.stack([bundle['canon'][np.arange(C)!=x]for x in ci]);gap=v[torch.from_numpy(rr),torch.from_numpy(bundle['canon'][ci])]-torch.gather(v[rr],1,torch.from_numpy(other)).mean(1)
            X[start+rr,slot]=np.stack([bundle['trace']['raw_score'][start+rr].astype(np.float32),top[rr,0],top[rr,0]-top[rr,1],entropy[rr],cv[rr,ci],cv[rr,ci]-top[rr,0],np.minimum(vrank[rr,ci],4).astype(np.float32)/np.float32(4),gap.numpy(),np.log1p(counts[ci]),b[rr,ci],brank[rr,ci].astype(np.float32)/np.float32(C-1),margin[rr],np.log1p(bundle['support'][ci]),(bundle['support'][ci]>0).astype(np.float32),(ci==br[rr,0]).astype(np.float32)],axis=1);pp[start+rr,slot]=wp[rr,ci]
    if not np.isfinite(X).all()or np.any(X[~valid]!=0):raise ValueError('Nonfinite features or nonzero padding')
    return X,cats,pp,valid,mapping
def training(bundle,X,cats,valid,key,pr,p,feature_ref):
    old_audit=CAND/'training/calfit_candidate_teacher_audit.npz';A=loadnpz(old_audit);lookup={}
    for g,cc,yy,ii,aa in zip(A['global_cache_index'],A['candidate_category_id'],A['target'],A['max_sameclass_iou'],A['matched_annotation_id']):
        for c,y,i,a in zip(cc,yy,ii,aa):lookup[int(g),int(c)]=(y,i,a)
    old_fm=read(CAND/'features/CLC_cal_fit/feature_manifest.json');old_X=np.load(verify(old_fm['outputs']['candidate_features.npy']),allow_pickle=False);old_C=np.load(verify(old_fm['outputs']['candidate_category_id.npy']),allow_pickle=False);old_T=loadnpz(verify(old_fm['outputs']['source_trace.npz']));feature_lookup={(int(g),int(c)):old_X[i,j]for i,(g,cc)in enumerate(zip(old_T['global_cache_index'],old_C))for j,c in enumerate(cc)}
    N=len(cats);y=np.zeros((N,2),np.float32);iou=np.zeros((N,2),np.float64);ann=np.full((N,2),-1,np.int64)
    for row,slot in np.argwhere(valid):
        k=int(bundle['trace']['global_cache_index'][row]),int(cats[row,slot])
        if k not in lookup:raise ValueError('No sealed same-class teacher key; no label guessing permitted: '+str(k))
        if k not in feature_lookup or not np.array_equal(X[row,slot],feature_lookup[k]):raise ValueError('Cal-fit raw class-conditional feature differs from sealed top5 golden')
        y[row,slot],iou[row,slot],ann[row,slot]=lookup[k]
    if not np.array_equal(y[valid],np.where(iou[valid]>=.5,iou[valid],0.).astype(np.float32)):raise ValueError('Inherited teacher changed')
    splitref=read(CAND/'protocol/protocol.json')['sources']['calfit_split'];split=read(verify(splitref));tr=bundle['trace'];masks={s:np.isin(tr['original_image_id'],split[s+'_original_image_ids'])for s in ['train','val']}
    if np.any(masks['train']&masks['val'])or not np.all(masks['train']|masks['val']):raise ValueError('Original image split not preserved')
    arrays={};stats={}
    for s,n,candidates in [('train',1921,2013),('val',436,452)]:
        m=masks[s]
        if m.sum()!=n or valid[m].sum()!=candidates:raise ValueError('Union calfit population changed')
        arrays.update({f'X_{s}':X[m],f'y_{s}':y[m],f'candidate_valid_{s}':valid[m],f'candidate_category_id_{s}':cats[m],f'image_ids_{s}':tr['original_image_id'][m],f'global_cache_index_{s}':tr['global_cache_index'][m]})
        stats[s]=dict(original_image_scope=len(split[s+'_original_image_ids']),effective_union_images=len(np.unique(tr['original_image_id'][m])),union_proposals=int(m.sum()),unique_valid_candidates=int(valid[m].sum()),positive_candidates=int((y[m][valid[m]]>0).sum()))
    meta=dict(protocol=pr,protocol_sha256=PSHA,candidate_row_key_sha256=key,feature_manifest=feature_ref,source_teacher_audit=ref(old_audit),source_feature_golden=ref(CAND/'features/CLC_cal_fit/feature_manifest.json'),all_valid_calfit_features_exact_sealed_golden=True,teacher='Exact (global_cache_index,category_id) lookup of sealed same-class F64 max-IoU and thresholded F32 utility; no new GT parsed',image_split=splitref,source_calfit=bundle['calfit_source'],statistics=stats,padding_not_training_population=True,shared_sameclass_slots=True,new_GT_annotations_parsed=False,test_labels_used=False)
    arrays['feature_names']=np.asarray(p['feature_schema']['names']);arrays['metadata_json']=np.asarray(json.dumps(meta));path=OUT/'training/union_train_val.npz';path.parent.mkdir(parents=True,exist_ok=True);npz(path,**arrays);npz(OUT/'training/teacher_audit.npz',global_cache_index=tr['global_cache_index'],candidate_category_id=cats,candidate_valid=valid,target=y,max_sameclass_iou=iou,matched_annotation_id=ann,train_mask=masks['train'],val_mask=masks['val'])
    save(OUT/'training/data_receipt.json',dict(status='COMPLETE_INHERITED_UNION_TEACHER_DATA',created_utc=now(),protocol=pr,metadata=meta,training_npz=ref(path),teacher_audit=ref(OUT/'training/teacher_audit.npz'),new_GT_annotations_parsed=False,new_AP_computed=False));atomic(OUT/'training/data_status.json',dict(status='COMPLETE',protocol=pr,data=ref(path),receipt=ref(OUT/'training/data_receipt.json')))
def prepare(case,protocol=CODE/'config/protocol.json',chunk=1024):
    p,pr,old=frozen(protocol)
    if case not in ['CLC_cal_fit','CLC_tune','DET','Hi','CLCXray_test','PIXray','PIDray','DvXray']:raise ValueError('Invalid union case')
    declared=p['sources']['calfit_assets']if case=='CLC_cal_fit'else p['cases'][case]['asset_manifest']
    if declared!=old['cases'][case]['asset_manifest']:raise ValueError('New C protocol/source asset family differs')
    directory=OUT/'features'/case;status=directory/'status.json'
    if status.exists():
        d=read(status)
        if d['status']=='COMPLETE'and d['protocol']==pr:verify(d['feature_manifest']);return d
        raise FileExistsError('Prior incomplete union case; never overwrite')
    directory.mkdir(parents=True,exist_ok=True)
    if any(directory.iterdir()):raise FileExistsError('Unsealed union feature directory')
    bundle=load_case(case,old);X,cats,pp,valid,mapping=features(bundle,chunk)
    for name,a in [('candidate_features.npy',X),('candidate_category_id.npy',cats),('candidate_proto_index.npy',pp),('candidate_valid.npy',valid),('branch_to_candidate.npy',mapping),('candidate_visual_scores.npy',X[:,:,4])]:npy(directory/name,a)
    npz(directory/'source_trace.npz',**bundle['trace']);save(directory/'feature_names.json',p['feature_schema']['names']);outputs={x.name:ref(x)for x in sorted(directory.iterdir())if x.is_file()};binding={k:outputs[k]['sha256']for k in ['source_trace.npz','candidate_category_id.npy','candidate_proto_index.npy','candidate_valid.npy','branch_to_candidate.npy','candidate_visual_scores.npy']};key=digest(binding)
    t=bundle['trace'];both=t['branch_kept'].all(1);different=both&(t['branch_category_id'][:,0]!=t['branch_category_id'][:,1]);counts=dict(union=len(X),native_V=int(t['branch_kept'][:,0].sum()),native_F=int(t['branch_kept'][:,1].sum()),V_only=int((t['branch_kept'][:,0]&~t['branch_kept'][:,1]).sum()),F_only=int((~t['branch_kept'][:,0]&t['branch_kept'][:,1]).sum()),overlap=int(both.sum()),different_class_overlap=int(different.sum()),unique_valid_candidates=int(valid.sum()))
    if case!='CLC_cal_fit'and counts['union']!=p['representation']['source_counts'][case+'_union']:raise ValueError('Frozen union population count differs')
    fm=dict(status='COMPLETE_UNION_CANDIDATE_FEATURES',created_utc=now(),case=case,dataset=case,protocol=pr,protocol_sha256=PSHA,shape=list(X.shape),dtype='float32',outputs=outputs,candidate_row_key_sha256=key,candidate_row_key_binding=binding,row_key_sha256=digest(t['global_cache_index'].tolist()),source_native_records=bundle['source'],source_calfit=bundle['calfit_source'],identity=dict(asset_manifest=bundle['manifest'],assets=bundle['assets'],builder=ref(__file__)),counts=counts,class_metadata=bundle['cats'],canonical_prototype_indices=bundle['canon'].tolist(),class_visual_prototype_count=[len(j)for j in bundle['columns']],material_support_count=bundle['support'].tolist(),rank_definition='Full C positive-class Vmax rank, ties winner prototype index; min(rank,4)/4',candidate_proto_definition='Original V maximizing prototype for this category; native branch winner retained separately in branch_native_proto_index for inference cases',calfit_branch_prototype_note='Accepted calfit CSV records classes/masks only; branch_native_proto_index=-1 is unavailable metadata, not a negative-class flag',sameclass_branch_slot_exact=True,GT_annotations_parsed=False,native_filtering_recomputed=False)
    save(directory/'feature_manifest.json',fm);fmr=ref(directory/'feature_manifest.json');save(directory/'source_index.json',dict(status='COMPLETE',case=case,protocol=pr,feature_manifest=fmr,records=bundle['source'],counts=counts,new_AP=0))
    if case=='CLC_cal_fit':training(bundle,X,cats,valid,key,pr,p,fmr)
    result=dict(status='COMPLETE',case=case,protocol=pr,feature_manifest=fmr,source_index=ref(directory/'source_index.json'),shape=list(X.shape),candidate_row_key_sha256=key,counts=counts,new_AP=0,new_training=0);atomic(status,result);return result
