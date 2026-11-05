import copy,collections,hashlib,json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ATOL=1e-12
def canonical(records):
 h=hashlib.sha256()
 for r in records:
  t=[int(r['image_id']),int(r['category_id']),*(float(v).hex() for v in r['bbox']),float(r['score']).hex()]
  h.update((json.dumps(t,separators=(',',':'))+'\n').encode())
 return {'count':len(records),'sha256':h.hexdigest()}

def params(p):
 return {'iouType':p.iouType,'imgIds':list(map(int,p.imgIds)),'catIds':list(map(int,p.catIds)),
 'iouThrs':p.iouThrs.tolist(),'recThrs':p.recThrs.tolist(),'maxDets':list(map(int,p.maxDets)),
 'areaRng':p.areaRng,'areaRngLbl':p.areaRngLbl,'useCats':int(p.useCats)}

def native(gt,pred,stream=None):
 g=COCO();g.dataset=copy.deepcopy(gt);g.createIndex()
 if pred:d=g.loadRes(copy.deepcopy(pred))
 else:d=COCO();d.dataset={'images':copy.deepcopy(gt['images']),'categories':copy.deepcopy(gt['categories']),'annotations':[]};d.createIndex()
 e=COCOeval(g,d,'bbox');e.evaluate();e.accumulate();e.summarize();return e

def clone(gt,pred,draw,image_ids):
 images={int(x['id']):x for x in gt['images']};gg=collections.defaultdict(list);dd=collections.defaultdict(list)
 for a in gt['annotations']:gg[int(a['image_id'])].append(a)
 for a in pred:dd[int(a['image_id'])].append(a)
 newgt={k:copy.deepcopy(v) for k,v in gt.items() if k not in ('images','annotations')}
 newgt['images']=[];newgt['annotations']=[];newdt=[]
 for j,pos in enumerate(draw):
  old_id=image_ids[int(pos)];im=copy.deepcopy(images[old_id]);im['id']=j+1;newgt['images'].append(im)
  for a in gg[old_id]:
   a=copy.deepcopy(a);a['id']=len(newgt['annotations'])+1;a['image_id']=j+1;newgt['annotations'].append(a)
  for d in dd[old_id]:
   d=copy.deepcopy(d);d['image_id']=j+1;newdt.append(d)
 assert all(int(a['id'])>0 for a in newgt['annotations'])
 return newgt,newdt

def make_cache(e,gt,pred,image_ids):
 I=len(image_ids);A=len(e.params.areaRng);K=len(e.params.catIds);index={v:i for i,v in enumerate(image_ids)}
 gcount=np.zeros(I,np.int64);dcount=np.zeros(I,np.int64)
 gpos=np.zeros(max(int(a['id']) for a in gt['annotations'])+1,np.int64);dpos=np.zeros(len(pred)+1,np.int64)
 for a in gt['annotations']:
  i=index[int(a['image_id'])];gpos[int(a['id'])]=gcount[i];gcount[i]+=1
 for did,d in enumerate(pred,1):
  i=index[int(d['image_id'])];dpos[did]=dcount[i];dcount[i]+=1
 records=[[] for _ in image_ids];incidence=np.zeros((K,I),bool);gtcounts=np.zeros((K,I),np.int32)
 for offset,r in enumerate(e.evalImgs):
  if r is None:continue
  block,i=divmod(offset,I);records[i].append((block,r))
  if block%A==0:
   k=block//A;gtcounts[k,i]=int(np.count_nonzero(np.asarray(r['gtIgnore'])==0));incidence[k,i]=gtcounts[k,i]>0
 return {'params':copy.deepcopy(e._paramsEval),'records_by_image':records,'I':I,'A':A,'K':K,
 'gcount':gcount,'dcount':dcount,'gpos':gpos,'dpos':dpos,'incidence':incidence,'gtcounts':gtcounts,
 'image_ids':image_ids,'stats':e.stats.copy()}

def map_matches(arr,pos,offset):
 a=np.asarray(arr,dtype=np.int64);return np.where(a>0,pos[a]+offset+1,0).astype(np.float64)

def replay(cache,draw):
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
 e.accumulate();e.summarize();return e

def valid_mask(e):return np.any(e.eval['precision'][:,:,:,0,2]>=0,axis=(0,1))

def compare(a,b):
 x=np.asarray(a);y=np.asarray(b)
 return {'passed':x.shape==y.shape and np.array_equal(np.isnan(x),np.isnan(y)) and np.array_equal(x==-1,y==-1) and bool(np.allclose(x,y,atol=ATOL,rtol=0,equal_nan=True)),
 'bitwise_equal':bool(np.array_equal(x,y,equal_nan=True)) if x.shape==y.shape else False,
 'max_abs':float(np.nanmax(np.abs(x-y))) if x.shape==y.shape and x.size else None,'shape':list(x.shape)}

def compare_records(native_e,fast_e):
 errors=[];count=0
 for i,(a,b) in enumerate(zip(native_e.evalImgs,fast_e.evalImgs)):
  if (a is None)!=(b is None):errors.append({'index':i,'field':'None'});continue
  if a is None:continue
  count+=1
  for key in ('image_id','category_id','aRng','maxDet','dtIds','gtIds','dtMatches','gtMatches','dtScores','gtIgnore','dtIgnore'):
   if not np.array_equal(np.asarray(a[key]),np.asarray(b[key]),equal_nan=True):errors.append({'index':i,'field':key})
 return {'passed':not errors and len(native_e.evalImgs)==len(fast_e.evalImgs),'nonNone_records':count,'mismatches':errors[:20]}
