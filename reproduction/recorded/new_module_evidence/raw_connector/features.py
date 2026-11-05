"""GT-free dynamic native-union feature construction for runtime deployment.

This constructs current feature definitions on fresh detector outputs. It does
not recreate unauthenticated historical encoder/cache runs or claim their AP.
Original scientific feature function bodies are loaded read-only via AST.
"""
from pathlib import Path
import ast, hashlib, json, sys
import numpy as np
sys.dont_write_bytecode = True
ROOT = Path('/mnt/data/yxy/program/RAXO')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
NIGHT = ROOT/'experiments/night_exploration_20261101'
SOURCES = {
 'C': NIGHT/'batch_c_shared_union/src/union_data.py',
 'K': NIGHT/'batch_k_multisource_quality/src/data_pipeline.py',
 'N': NIGHT/'batch_n_supported_material_quality/src/supported_data.py',
 'G': ROOT/'experiments/material_conditional_ranking_fix1_20261101/src/group_data.py',
 'native': ROOT/'paper_revision_V2_20261102/publication_reproduction/native_stage/source_evidence/runtime/scripts/runtime_common.py',
}

def _load(path, names, scope):
    nodes = [x for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in names]
    assert {x.name for x in nodes} == set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return {k:scope[k] for k in names}

_scope = dict(np=np)
_C = _load(SOURCES['C'], ['candidate_mapping','features'], _scope)
_K = _load(SOURCES['K'], ['geometry','append_material'], dict(np=np))
_N = _load(SOURCES['N'], ['supported_columns'], dict(np=np))
_G = _load(SOURCES['G'], ['gate_context'], dict(np=np))
_D = _load(SOURCES['native'], ['native_modules','Decision'], dict(np=np,sys=sys,PROJECT=ROOT))


def source_refs():
    def digest(p):
        h=hashlib.sha256()
        with p.open('rb')as f:
            for b in iter(lambda:f.read(8<<20),b''):h.update(b)
        return h.hexdigest()
    return {k:dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for k,p in SOURCES.items()}


def native_states(V, B, rows, proto, cats):
    """Exact original full-J addition and strict canonical-gap UNC helper."""
    import torch
    from rmdx import cache_fusion, preargmax_fusion
    d = _D['Decision'](proto,cats)
    v = torch.from_numpy(np.asarray(V,np.float32))
    b = torch.from_numpy(np.asarray(B,np.float32))
    expanded = cache_fusion.material_class_to_prototype(b,proto)
    fused = preargmax_fusion.fuse_similarity_material(v,expanded,0.05)
    # States include negative decisions before category/UNC acceptance.
    return d.minimal(rows,v,capture=True)[2],d.minimal(rows,fused,capture=True)[2]


def _gather(full, ci, valid):
    out = np.zeros((*valid.shape,full.shape[-1]),np.float32)
    for slot in (0,1):
        r = np.flatnonzero(valid[:,slot]);out[r,slot] = full[r,ci[r,slot]]
    return out


def build_features(V, Bnative, Bfresh, rows, image_sizes, proto, category_meta,
                   native_support, aux_support=None, query_valid=None,
                   Baux=None, branch_state=None, case='CLCXray_test',
                   original_support=None):
    """Return runtime's C15/T13/aux13 schema, including exact native routing.

    All supplied arrays/rows describe ALL detector proposals for each image,
    before native filtering. Geometry percentile and max-IoU use that universe.
    image_sizes is (W,H) for a single image or {image_id:(W,H)}. Bfresh uses the
    original quality bank; Baux is independently current-aligned if different.
    Native branch filtering ALWAYS uses Bnative with lambda=.05, UNC>.15.
    No images or GT are opened here; no unavailable descriptor is synthesized.
    """
    V=np.asarray(V,np.float32);bn=np.asarray(Bnative,np.float32)
    bf=np.asarray(Bfresh,np.float32);ba=bf if Baux is None else np.asarray(Baux,np.float32)
    rows=[dict(r) for r in rows];n=len(rows);C=len(category_meta)
    assert V.shape==(n,len(proto)) and bn.shape==bf.shape==ba.shape==(n,C)
    assert all(np.isfinite(x).all() for x in (V,bn,bf,ba))
    support=np.asarray(native_support,np.float32);original=np.asarray(native_support if original_support is None else original_support,np.int64)
    actual=np.asarray(original if aux_support is None else aux_support,np.int64)
    assert support.shape==original.shape==actual.shape==(C,)
    assert np.all(support>=0) and np.all(original>=0) and np.all(actual>=0)
    qv=np.ones(n,bool) if query_valid is None else np.asarray(query_valid,bool)
    assert qv.shape==(n,)
    for i,r in enumerate(rows):
        r.setdefault('global_cache_index',r.get('cache_index',i))
        r['cache_index']=int(r['global_cache_index'])
        r.setdefault('original_image_id',int(r['image_id']))
    gidx=np.asarray([r['cache_index'] for r in rows],np.int64)
    assert len(np.unique(gidx))==n
    if n:
        if branch_state is None:branch_state=native_states(V,bn,rows,proto,category_meta)
        vs,fs=branch_state
        kept=np.stack([np.asarray(s['after_unc'],bool)for s in (vs,fs)],1)
        bc=np.stack([np.asarray(s['category'],np.int64)for s in (vs,fs)],1)
        wp=np.stack([np.asarray(s['winner'],np.int64)for s in (vs,fs)],1)
        assert kept.shape==bc.shape==wp.shape==(n,2)
    else:
        kept=np.zeros((0,2),bool);bc=np.empty((0,2),np.int64);wp=bc.copy()
    take=np.flatnonzero(kept.any(1));u=len(take)
    G=np.zeros((n,8),np.float32)
    ids=np.asarray([r['image_id']for r in rows],np.int64)
    for iid in np.unique(ids):
        ii=np.flatnonzero(ids==iid)
        wh=image_sizes if isinstance(image_sizes,(tuple,list)) else image_sizes[int(iid)]
        assert len(wh)==2 and wh[0]>0 and wh[1]>0
        G[ii]=_K['geometry']([rows[j]for j in ii],*wh)
    cols=[];canon=[];catids=np.asarray([int(c['category_id'])for c in category_meta],np.int64)
    for ci,c in enumerate(category_meta):
        js=np.asarray([j for j,p in enumerate(proto)if not p['is_negative'] and int(p['material_class_index'])==ci],np.int64)
        assert len(js)>0
        cc={int(proto[j]['canonical_index'])for j in js};assert len(cc)==1
        cols.append(js);canon.append(next(iter(cc)))
    trace=dict(global_cache_index=gidx[take],source_detection_index=np.arange(u,dtype=np.int64),
        image_id=ids[take],original_image_id=np.asarray([r['original_image_id']for r in rows],np.int64)[take],
        bbox=np.asarray([r['bbox']for r in rows],np.float64).reshape(n,4)[take],
        raw_score=np.asarray([r['detector_score']for r in rows],np.float64)[take],
        branch_kept=kept[take],branch_category_id=bc[take],branch_native_proto_index=wp[take],
        branch_source_detection_index=np.full((u,2),-1,np.int64))
    if u:
        bundle=dict(V=V[take],B=bn[take],trace=trace,columns=cols,canon=np.asarray(canon,np.int64),
            catmap={int(cid):i for i,cid in enumerate(catids)},support=support)
        CX,CID,protoindex,M,mapping=_C['features'](bundle)
    else:
        CX=np.zeros((0,2,15),np.float32);CID=np.full((0,2),-1,np.int64);protoindex=CID.copy();M=np.zeros((0,2),bool);mapping=CID.copy()
    ci=np.full((u,2),-1,np.int64)
    lookup={int(c):i for i,c in enumerate(catids)}
    for i,j in np.argwhere(M):ci[i,j]=lookup[int(CID[i,j])]
    def full(B, counts, selected):
        z=_K['append_material'](G[selected],B[selected],dict(actual_material_support_counts=counts),qv[selected])
        s=np.broadcast_to(np.asarray(counts)>0,(len(selected),C)) & qv[selected,None]
        normalized,T=_N['supported_columns'](B[selected],s)
        z[:,:,9:12]=normalized
        assert np.all(z[:,:,12]==s.astype(np.float32))
        return z,s,T
    TX,ts,_=full(bf,original,take);AX,ass,T=full(ba,actual,take)
    context,ctxT=_G['gate_context'](AX,ass,qv[take]);assert np.array_equal(T,ctxT)
    out=dict(case=case,C_X=CX,T_X=_gather(TX,ci,M),aux_X=_gather(AX,ci,M),
        M=M,ci=ci,cats=CID,mapping=mapping,B=ba[take],has=actual>0,qv=qv[take],T=T,
        geometry=G[take],trace=trace,full_material=AX[:,:,8:],gate_context=context,
        union_indices=take,all_proposals=n,native_lambda=0.05,native_UNC=0.15,
        native_prototype_index=protoindex,native_branch_states=(branch_state if n else (None,None)))
    for k in ['C_X','T_X','aux_X','B','geometry','full_material','gate_context']:
        assert np.isfinite(out[k]).all(),k
    assert not np.any(out['C_X'][~M]) and not np.any(out['T_X'][~M]) and not np.any(out['aux_X'][~M])
    return out
