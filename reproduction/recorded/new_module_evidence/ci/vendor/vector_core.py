from native_core import *
from fast_accumulate import fast_accumulate
def replay_fast(cache,draw):
 n=len(draw);go=np.r_[0,np.cumsum(cache['gcount'][draw])[:-1]];do=np.r_[0,np.cumsum(cache['dcount'][draw])[:-1]]
 records=[None]*(cache['K']*cache['A']*n)
 for j,source in enumerate(draw):
  for block,old in cache['records_by_image'][int(source)]:
   r=dict(old);r['image_id']=j+1
   r['gtIds']=(cache['gpos'][np.asarray(old['gtIds'],dtype=np.int64)]+go[j]+1).tolist()
   r['dtIds']=(cache['dpos'][np.asarray(old['dtIds'],dtype=np.int64)]+do[j]+1).tolist()
   r['dtMatches']=map_matches(old['dtMatches'],cache['gpos'],go[j])
   r['gtMatches']=map_matches(old['gtMatches'],cache['dpos'],do[j])
   records[block*n+j]=r
 e=COCOeval(iouType='bbox');p=copy.deepcopy(cache['params']);p.imgIds=list(range(1,n+1))
 e.params=copy.deepcopy(p);e._paramsEval=copy.deepcopy(p);e.evalImgs=records
 fast_accumulate(e);e.summarize();return e
