"""Byte-identical O pure composition functions; no old writer imported."""
import numpy as np
import hashlib

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
