"""CPU-only projection of immutable native learned corrections onto V/F contrast.

The native study protocol and receipts stay unchanged. These are new derived
small arrays under a separate study protocol, never new GPU/model inference.
"""
from pathlib import Path
import datetime as dt
import hashlib
import json
import os
import sys
import numpy as np
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parents[1]
O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/pair_conflict_material_projection_20261101')
ROLES=['material','permuted_material']
SEEDS=[17,29,43]
SOURCE_PROTOCOL_SHA256='3b3c304d90bd71ad6aa55a3702fdf287b79c335450e5efc53082a62d037f6962'


def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def ref(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def verify(r):
 p=Path(r['path']).resolve();assert ref(p)=={k:r[k]for k in ['path','bytes','sha256']},str(p);return p
def loadz(p):
 with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and np.ascontiguousarray(a).tobytes()==np.ascontiguousarray(b).tobytes()
def save(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def atomic(p,v):
 p=Path(p).resolve();assert p.is_relative_to(O);p.parent.mkdir(parents=True,exist_ok=True)
 q=p.with_name(p.name+f'.tmp.{os.getpid()}');save(q,v);os.replace(q,p)


def project(delta,valid,availability,T,ci,category_id,mapping,kept,branch_category):
 """Gather branch V/F, form a F64 half difference, F32 cast, scatter opposites."""
 n=len(valid)
 assert delta.dtype==availability.dtype==np.float32 and delta.shape==availability.shape==valid.shape==(n,2)
 assert valid.dtype==kept.dtype==bool and kept.shape==(n,2)
 assert all(a.dtype==np.int64 and a.shape==(n,2)for a in [ci,category_id,mapping,branch_category])
 assert T.dtype==np.int64 and T.shape==(n,)and np.all(T>=0)
 assert np.isfinite(delta).all()and np.all(np.abs(delta)<=2.)
 assert np.all((availability==0.)|(availability==1.))and not np.any(delta[~valid])and not np.any(availability[~valid])
 assert np.all(category_id[~valid]==-1)and np.all(ci[~valid]==-1)and np.all(category_id[valid]>=0)and np.all(ci[valid]>=0)
 assert np.array_equal(mapping>=0,kept)and np.all(mapping[~kept]==-1)
 bv=np.zeros(n,bool);bf=np.zeros(n,bool);sv=mapping[:,0];sf=mapping[:,1]
 for b in [0,1]:
  rr=np.flatnonzero(kept[:,b]);ss=mapping[rr,b]
  assert np.all((ss>=0)&(ss<2))and np.all(valid[rr,ss])
  assert np.array_equal(category_id[rr,ss],branch_category[rr,b])
  if b==0:bv[rr]=(availability[rr,ss]==1.)
  else:bf[rr]=(availability[rr,ss]==1.)
 same=kept.all(1)&(branch_category[:,0]==branch_category[:,1])
 assert np.array_equal(sv[same],sf[same]),'Identical classes share the same canonical candidate slot'
 active=kept.all(1)&bv&bf&(branch_category[:,0]!=branch_category[:,1])&(sv!=sf)&(T>1)
 rr=np.flatnonzero(active);vs=sv[rr];fs=sf[rr]
 assert np.all(vs!=fs)and np.all(ci[rr,vs]!=ci[rr,fs])and np.all(valid[rr].all(1))
 D=np.zeros(n,np.float32)
 D[rr]=((delta[rr,fs].astype(np.float64)-delta[rr,vs].astype(np.float64))*.5).astype(np.float32)
 out=np.zeros((n,2),np.float32);out[rr,vs]=-D[rr];out[rr,fs]=D[rr]
 assert not np.any(out[~active].view(np.uint32))and not np.any(D[~active].view(np.uint32))and np.all(np.abs(out)<=2.)
 assert np.all(out[rr].astype(np.float64).sum(1)==0.)
 original=delta[rr,fs].astype(np.float64)-delta[rr,vs].astype(np.float64)
 projected=out[rr,fs].astype(np.float64)-out[rr,vs].astype(np.float64)
 error=np.abs(projected-original);assert np.all(error<=2.**-22)
 stats=dict(active_rows=len(rr),inactive_rows=n-len(rr),active_CID0_rows=int(np.sum((branch_category[rr]==0).any(1))),
  active_zero_D_rows=int(np.sum(D[rr]==0.)),maximum_correction_contrast_F64_rounding_error=float(error.max(initial=0.)),
  contrast_rounding_bound=2.**-22,active_pair_sum_exact0=True,inactive_output_positive_zero_bits=True,
  branch_mapping_gather_and_scatter=True,probability_mean_preservation_claimed=False)
 return out,active,D,stats


def frozen(path,expected):
 p=Path(path).resolve();assert p==C/'config/protocol.json';pr=ref(p);assert len(expected)==64 and pr['sha256']==expected
 P=read(p);assert P['status'].startswith('FROZEN')and P['variants']==ROLES and P['seeds']==SEEDS
 assert P['budget']['new_trainings']==P['budget']['new_GPU_inferences']==0
 assert P['sources']['source_study_protocol']['sha256']==SOURCE_PROTOCOL_SHA256
 verify(P['sources']['source_study_protocol'])
 complete=read(verify(P['sources']['source_completion']));assert complete['status'].startswith('COMPLETE')
 freeze=read(O/'automation/projection_code_freeze.json');assert freeze['protocol']==pr
 required={str(C/n)for n in ['src/pair_projection.py','scripts/prepare_projection.py']}
 assert required<={r['path']for r in freeze['sources']}
 for key in ['sources','readonly_sources']:
  for value in freeze[key]:verify(value)
 gate=read(O/'verification/projection_preflight_review.json');assert gate['status']=='PASS'and gate['protocol_sha256']==pr['sha256']
 return P,pr


def source_records(P):
 index=read(verify(P['sources']['source_inference_index']))
 assert index['status']=='COMPLETE_CONDITIONAL_MODELS_AND_FULL_C_INFERENCE'
 assert index['protocol']==P['sources']['source_study_protocol']
 assert len(index['records'])==42 and index['new_trainings']==6 and index['new_GPU_inferences']==42
 lookup={(r['case'],r['variant'],r['seed']):r for r in index['records']}
 expected={(c,v,s)for c in P['inference_cases']for v in ROLES for s in SEEDS}
 assert len(lookup)==42 and set(lookup)==expected
 return lookup


def case_inputs(P,pr,case):
 c=P['cases'][case];cf=read(verify(c['C_feature_manifest']));nf=read(verify(c['auxiliary_feature_manifest']))
 assert cf['candidate_row_key_sha256']==nf['candidate_row_key_sha256']==c['candidate_row_key_sha256']
 assert nf['source_C_feature_manifest']==c['C_feature_manifest']
 assert nf['outputs']['candidate_valid.npy']==cf['outputs']['candidate_valid.npy']
 assert nf['outputs']['candidate_category_id.npy']==cf['outputs']['candidate_category_id.npy']
 trace=loadz(verify(cf['outputs']['source_trace.npz']));mapping=np.load(verify(cf['outputs']['branch_to_candidate.npy']),allow_pickle=False)
 source_outputs=nf['outputs']
 arrays={name:np.load(verify(source_outputs[name+'.npy']),allow_pickle=False)for name in ['candidate_valid','candidate_class_order_index','candidate_category_id','supported_reference_T','query_valid','actual_supported_class_mask']}
 valid,ci,cats,T,qv,has=(arrays[name]for name in ['candidate_valid','candidate_class_order_index','candidate_category_id','supported_reference_T','query_valid','actual_supported_class_mask'])
 assert valid.dtype==qv.dtype==has.dtype==bool and T.dtype==np.int64
 assert valid.shape==ci.shape==cats.shape==mapping.shape==(len(T),2)and qv.shape==T.shape
 assert np.array_equal(T,qv.astype(np.int64)*int(has.sum()))
 catalogue=nf['class_order'];assert [int(c['class_index'])for c in catalogue]==list(range(len(has)))
 ids=np.asarray([int(c['category_id'])for c in catalogue],np.int64)
 assert np.array_equal(ids[ci[valid]],cats[valid])
 available=np.zeros(valid.shape,np.float32);rows,slots=np.nonzero(valid)
 available[rows,slots]=(has[ci[rows,slots]]&qv[rows]).astype(np.float32)
 manifest=dict(status='COMPLETE_PAIR_CONFLICT_SHARED_CASE_MANIFEST',protocol=pr,case=case,
  source_study_protocol=P['sources']['source_study_protocol'],source_C_feature_manifest=c['C_feature_manifest'],source_auxiliary_feature_manifest=c['auxiliary_feature_manifest'],
  source_auxiliary_protocol=nf['protocol'],candidate_row_key_sha256=c['candidate_row_key_sha256'],
  candidate_valid=source_outputs['candidate_valid.npy'],candidate_category_id=source_outputs['candidate_category_id.npy'],
  candidate_class_order_index=source_outputs['candidate_class_order_index.npy'],supported_reference_T=source_outputs['supported_reference_T.npy'],
  query_valid=source_outputs['query_valid.npy'],actual_supported_class_mask=source_outputs['actual_supported_class_mask.npy'],
  branch_to_candidate=cf['outputs']['branch_to_candidate.npy'],native_source_trace=cf['outputs']['source_trace.npz'],
  class_order=catalogue,new_large_array_copies=0,new_GPU_training_or_model_forward=0)
 return cf,nf,trace,mapping,valid,ci,cats,T,available,manifest


def prepare(path,expected):
 P,pr=frozen(path,expected);assert not(O/'projection_index.json').exists(),'No implicit projection retry'
 lookup=source_records(P);records=[];mask_by_case={};case_manifests=[]
 atomic(O/'automation/projection_status.json',dict(status='CPU_PROJECTING',protocol=pr,completed=0,new_trainings=0,new_GPU_inferences=0))
 for case in P['inference_cases']:
  cf,nf,trace,mapping,valid,ci,cats,T,available,manifest=case_inputs(P,pr,case)
  case_folder=O/'projection'/case;case_folder.mkdir(parents=True,exist_ok=False)
  save(case_folder/'case_manifest.json',manifest);case_ref=ref(case_folder/'case_manifest.json');case_manifests.append(case_ref)
  for role in ROLES:
   for seed in SEEDS:
    source=lookup[(case,role,seed)];r=read(verify(source['receipt']));z=loadz(verify(source['components']))
    assert r['status']=='COMPLETE_CONDITIONAL_FULL_C_GPU_INFERENCE'and r['protocol']==P['sources']['source_study_protocol']
    assert(r['case'],r['variant'],r['seed'])==(case,role,seed)and r['output']==source['components']
    assert r['checkpoint']==P['native_models'][role][str(seed)]
    assert r['source_auxiliary_feature_manifest']==P['cases'][case]['auxiliary_feature_manifest']
    assert r['candidate_row_key_sha256']==manifest['candidate_row_key_sha256']
    assert bits(z['valid'],valid)and bits(z['supported_count'],T)and bits(z['candidate_class_order_index'],ci)and bits(z['availability'],available)
    out,active,D,stats=project(z['centered_correction'],valid,available,T,ci,cats,mapping,trace['branch_kept'],trace['branch_category_id'])
    activehash=hashlib.sha256(active.tobytes()).hexdigest()
    if case in mask_by_case:assert mask_by_case[case]==activehash
    else:mask_by_case[case]=activehash
    folder=case_folder/role/f'seed{seed}';folder.mkdir(parents=True,exist_ok=False)
    with(folder/'components.npz').open('xb')as stream:np.savez_compressed(stream,projected_correction=out,active=active,D=D)
    result=dict(status='COMPLETE_PAIR_CONFLICT_MATERIAL_PROJECTION',protocol=pr,source_study_protocol=P['sources']['source_study_protocol'],
     native_model_protocol=r['protocol'],case=case,variant=role,seed=seed,
     source_producer_receipt=source['receipt'],source_components=source['components'],source_checkpoint=r['checkpoint'],
     source_inference_index=P['sources']['source_inference_index'],source_completion=P['sources']['source_completion'],
     source_C_feature_manifest=P['cases'][case]['C_feature_manifest'],source_auxiliary_feature_manifest=P['cases'][case]['auxiliary_feature_manifest'],case_manifest=case_ref,
     output=ref(folder/'components.npz'),branch_to_candidate=manifest['branch_to_candidate'],native_source_trace=manifest['native_source_trace'],
     candidate_row_key_sha256=manifest['candidate_row_key_sha256'],active_mask_SHA256=activehash,statistics=stats,created_utc=now(),
     actual_new_CPU_projection=True,new_training=0,new_GPU_inference=0,new_model_forward=0,new_query_encoder=0,
     GT_opened=False,labels_read=False,AP_computed=False,producer=ref(__file__))
    save(folder/'receipt.json',result)
    records.append(dict(case=case,variant=role,seed=seed,receipt=ref(folder/'receipt.json'),components=result['output'],projection=result['output'],source_components=source['components']))
    atomic(O/'automation/projection_status.json',dict(status='CPU_PROJECTING',protocol=pr,completed=len(records),new_trainings=0,new_GPU_inferences=0))
 assert len(records)==42
 result=dict(status='COMPLETE_42_CPU_PAIR_CONFLICT_PROJECTIONS',protocol=pr,created_utc=now(),records=records,case_manifests=case_manifests,
  shared_active_mask_SHA256=mask_by_case,new_trainings=0,new_GPU_inferences=0,new_model_forwards=0,new_query_encoders=0,
  actual_new_CPU_projections=42,old_fitted_models_referenced=6,accepted_native_component_references=42,
  source_study_protocol=P['sources']['source_study_protocol'],source_inference_index=P['sources']['source_inference_index'],source_completion=P['sources']['source_completion'])
 save(O/'projection_index.json',result)
 save(O/'projection_completion.json',dict(status='COMPLETE_42_CPU_PAIR_CONFLICT_PROJECTIONS',protocol=pr,index=ref(O/'projection_index.json'),new_trainings=0,new_GPU_inferences=0))
 atomic(O/'automation/projection_status.json',dict(status='COMPLETE',protocol=pr,completed=42,new_trainings=0,new_GPU_inferences=0))
 return result
