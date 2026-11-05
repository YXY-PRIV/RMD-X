"""Portable exact-function adapters for the recorded RMD-X score contracts.

Only functions extracted verbatim from sealed sources are included. Importing
this module reads no protocol, checkpoint or external data and performs no I/O.
This is the pair/routing/suppression contract, not a stand-alone trained detector.
"""
import numpy as np

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

def check_quality(q, M):
    assert q.dtype == np.float32 and q.shape == M.shape
    assert np.isfinite(q).all() and np.all((q >= 0) & (q <= 1)) and not np.any(q[~M])

def mix_visual(base, head, valid):
    q = np.zeros(valid.shape, np.float32)
    b = np.clip(base[valid].astype(np.float64), 1e-6, 1.)
    h = np.clip(head[valid].astype(np.float64), 1e-6, 1.)
    q[valid] = (np.power(b, .5)*np.power(h, .5)).astype(np.float32)
    return q

def branch_quality(q, kept, mapping):
    effective = np.zeros(kept.shape, np.float32)
    for branch in [0, 1]:
        rows = np.flatnonzero(kept[:, branch])
        effective[rows, branch] = q[rows, mapping[rows, branch]]
    return effective

def choose(kept, effective):
    selected = np.full(len(kept), -1, np.int8)
    selected[kept[:, 0]] = 0
    use_f = kept[:, 1] & ((effective[:, 1].astype(np.float64)-effective[:, 0].astype(np.float64)) > 0.)
    selected[use_f] = 1
    return selected

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
