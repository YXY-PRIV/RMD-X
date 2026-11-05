"""N actor: supported reference set changes only fields9,10,11; no GT/RGB reads."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
CODE=Path(__file__).resolve().parents[1];OUT=Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_n_supported_material_quality');PSHA='94470fb3917d5fc51e98f0a051f2dd3a392cbb8f8c51962befabe05f3c6408dd'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def ref(p):p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(r):p=Path(r['path']);assert p.stat().st_size==r['bytes']and sha(p)==r['sha256'],str(p);return p
def save(p,j):
 p=Path(p);assert p.resolve().is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
def array(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:np.save(f,a,allow_pickle=False)
def savez(p,**a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb')as f:np.savez_compressed(f,**a)
def frozen():
 pp=CODE/'config/protocol.json';assert sha(pp)==PSHA and pp.read_bytes()==(OUT/'protocol/protocol.json').read_bytes();P=read(pp);f=read(OUT/'automation/actor_code_freeze.json');assert f['protocol']==ref(pp)
 for r in f['sources']:verify(r)
 return P
def supported_columns(B,supported):
 B=np.asarray(B,np.float32);supported=np.asarray(supported,bool);assert B.ndim==2 and B.shape==supported.shape and np.isfinite(B).all();n,c=B.shape;out=np.zeros((n,c,3),np.float32);T=supported.sum(1).astype(np.int64)
 # Input columns are already actual source catalogue order; stable sorting never uses CID.
 values=np.where(supported,B.astype(np.float64),-np.inf);order=np.argsort(-values,axis=1,kind='stable');ranks=np.empty((n,c),np.int64);np.put_along_axis(ranks,order,np.broadcast_to(np.arange(c),(n,c)),axis=1);top=values.max(1);second=np.zeros(n,np.float64)
 many=T>=2
 if many.any():second[many]=np.take_along_axis(values,order[:,1:2],axis=1)[many,0]
 for j in range(c):
  mask=supported[:,j];out[mask,j,0]=(ranks[mask,j].astype(np.float64)/np.maximum(T[mask]-1,1)).astype(np.float32);out[mask,j,1]=np.where(T[mask]>=2,(top[mask]-second[mask])/2.,0.).astype(np.float32);out[mask,j,2]=((B[mask,j].astype(np.float64)-top[mask])/2.).astype(np.float32)
 assert np.isfinite(out).all()and np.all(out[~supported]==0.);return out,T
def training():
 P=frozen();source=P['sources']['K_training_npz'];verify(P['sources']['K_training_data_receipt'])
 with np.load(verify(source),allow_pickle=False)as z:d={k:z[k].copy()for k in z.files}
 mappings={};folds=[]
 for fold in ['train','val']:
  X=d['X_'+fold];before=X.copy();keys=np.stack([d[k+'_'+fold]for k in ['source_index','image_index','global_cache_index']],1);groups,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True);assert np.array_equal(counts[inv],d['class_count_'+fold]);sort=np.argsort(inv,kind='stable');cuts=np.r_[0,np.cumsum(counts)];class_indices=np.empty(len(X),np.int64);Ts=np.zeros(len(groups),np.int64);sources=[]
  for sid,name in enumerate(P['source_names']):
   cats=P['source_class_order'][name];assert [r['class_index']for r in cats]==list(range(len(cats)));ids=[int(r['category_id'])for r in cats];sources.append((ids,{cid:i for i,cid in enumerate(ids)},np.asarray(P['source_support_counts'][name])>0))
  for gi,key in enumerate(groups):
   rows=sort[cuts[gi]:cuts[gi+1]];sid=int(key[0]);ids,lookup,has=sources[sid];CID=d['category_id_'+fold][rows];assert len(rows)==len(ids)and len(set(CID.tolist()))==len(ids)and set(CID.tolist())==set(ids);ci=np.asarray([lookup[int(c)]for c in CID],np.int64);class_indices[rows]=ci;qv=d['query_valid_'+fold][rows];assert np.all(qv==qv[0])and np.all(X[rows,:8]==X[rows[0],:8]);actual=np.zeros(len(rows),np.int64);actual[ci]=rows;assert np.array_equal(X[actual,12],(has&qv[0]).astype(np.float32))
   supported=X[actual,12]==1.;B=X[actual,8][None,:];new,T=supported_columns(B,supported[None,:]);X[actual,9:12]=new[0];Ts[gi]=T[0]
  unchanged=np.asarray([0,1,2,3,4,5,6,7,8,12]);assert np.array_equal(X[:,unchanged].view(np.uint32),before[:,unchanged].view(np.uint32))and np.all(X[X[:,12]==0,8:]==0.)
  mappings.update({fold+'_proposal_keys':groups,fold+'_proposal_inverse':inv,fold+'_class_order_index':class_indices,fold+'_supported_reference_T':Ts});changed={str(j):int(np.count_nonzero(X[:,j].view(np.uint32)!=before[:,j].view(np.uint32)))for j in [9,10,11]};folds.append(dict(fold=fold,rows=len(X),proposals=len(groups),supported_T_range=[int(Ts.min()),int(Ts.max())],changed_scalar_counts=changed,unchanged0_through8_and12_exact=True,source_proposal_keys_and_CID_catalogue_proven=True))
 meta=json.loads(str(d['metadata_json'].item()));oldmeta=meta.copy();meta.update(protocol_sha256=PSHA,source_K_protocol_sha256=P['sources']['K_protocol']['sha256'],source_K_training_npz=source,source_K_teacher_metadata=oldmeta,feature_names=P['feature_schema']['names'],N_only_changed_feature_coordinates=[9,10,11]);d['metadata_json']=np.asarray(json.dumps(meta));d['feature_names']=np.asarray(P['feature_schema']['names']);target=OUT/'training/multisource_train_val.npz';savez(target,**d);mapfile=OUT/'training/source_group_mapping.npz';savez(mapfile,**mappings)
 receipt=dict(status='COMPLETE',created_utc=now(),protocol=ref(CODE/'config/protocol.json'),training_npz=ref(target),source_K_training_npz=source,source_K_data_receipt=P['sources']['K_training_data_receipt'],source_mapping=ref(mapfile),source_summaries=read(P['sources']['K_training_data_receipt']['path'])['source_summaries'],folds=folds,source_class_order=P['source_class_order'],only_changed_coordinates=[9,10,11],all_teacher_identity_class_count_queryvalid_unchanged=True,source_image_or_GT_or_RGB_or_encoder_opened=False,feature_names=P['feature_schema']['names']);save(OUT/'training/data_receipt.json',receipt);save(OUT/'training/status.json',dict(status='COMPLETE',protocol=ref(CODE/'config/protocol.json'),training_npz=ref(target),data_receipt=ref(OUT/'training/data_receipt.json')));return receipt
def queries(case):
 P=frozen();c=P['cases'][case];oldfm=read(verify(c['source_K_feature_manifest']));o=oldfm['outputs'];M=np.load(verify(o['candidate_valid.npy']),allow_pickle=False);CID=np.load(verify(o['candidate_category_id.npy']),allow_pickle=False);oldX=np.load(verify(o['candidate_features.npy']),allow_pickle=False);B=np.load(verify(o['fresh_material_B.npy']),allow_pickle=False,mmap_mode='r');QV=np.load(verify(o['query_valid.npy']),allow_pickle=False);n=len(B);classes=c['class_order'];assert [r['class_index']for r in classes]==list(range(len(classes)));lookup={int(r['category_id']):i for i,r in enumerate(classes)};has=np.asarray(c['actual_material_support_counts'])>0;X=oldX.copy();indices=np.full(M.shape,-1,np.int64)
 for slot in [0,1]:
  rr=np.flatnonzero(M[:,slot]);indices[rr,slot]=np.asarray([lookup[int(cid)]for cid in CID[rr,slot]]);assert np.array_equal(oldX[rr,slot,12],(has[indices[rr,slot]]&QV[rr]).astype(np.float32))
 T=np.empty(n,np.int64)
 for lo in range(0,n,4096):
  hi=min(n,lo+4096);support=np.broadcast_to(has,(hi-lo,len(has)))&QV[lo:hi,None];new,t=supported_columns(np.asarray(B[lo:hi]),support);T[lo:hi]=t
  for slot in [0,1]:
   local=np.flatnonzero(M[lo:hi,slot]);rr=local+lo;X[rr,slot,9:12]=new[local,indices[rr,slot]]
 unchanged=np.asarray([0,1,2,3,4,5,6,7,8,12]);assert np.array_equal(X[:,:,unchanged].view(np.uint32),oldX[:,:,unchanged].view(np.uint32))and np.all(X[~M]==0.)and np.all(X[X[:,:,12]==0,8:]==0.)
 folder=OUT/'features'/case;folder.mkdir(parents=True,exist_ok=False);array(folder/'candidate_features.npy',X);array(folder/'candidate_class_order_index.npy',indices);array(folder/'supported_reference_T.npy',T);array(folder/'actual_supported_class_mask.npy',has);outputs=dict(o);outputs.update({p.name:ref(p)for p in folder.iterdir()});fm=dict(oldfm,protocol=ref(CODE/'config/protocol.json'),protocol_sha256=PSHA,feature_names=P['feature_schema']['names'],outputs=outputs,source_K_feature_manifest=c['source_K_feature_manifest'],source_K_candidate_features=o['candidate_features.npy'],supported_rank_reference='Only genuine support classes AND query_valid; stable actual catalogue class_index; F64 math -> F32',N_only_changed_coordinates=[9,10,11]);save(folder/'feature_manifest.json',fm)
 changed={str(j):int(np.count_nonzero(X[:,:,j].view(np.uint32)!=oldX[:,:,j].view(np.uint32)))for j in [9,10,11]};receipt=dict(status='COMPLETE',created_utc=now(),protocol=ref(CODE/'config/protocol.json'),case=case,source_K_feature_manifest=c['source_K_feature_manifest'],feature_manifest=ref(folder/'feature_manifest.json'),source_C_feature_manifest=c['feature_manifest'],candidate_row_key_sha256=c['candidate_row_key_sha256'],rows=n,valid_candidates=int(M.sum()),supported_T_range=[int(T.min()),int(T.max())],changed_scalar_counts=changed,unchanged0_through8_and12_F32_bits_exact=True,source_RGB22_rawB_teacher_and_native_states_untouched=True,GT_or_RGB_or_encoder_opened=False);save(folder/'transformation_receipt.json',receipt);save(folder/'status.json',dict(status='COMPLETE',feature_manifest=ref(folder/'feature_manifest.json'),transformation_receipt=ref(folder/'transformation_receipt.json'),protocol=ref(CODE/'config/protocol.json')));return receipt
