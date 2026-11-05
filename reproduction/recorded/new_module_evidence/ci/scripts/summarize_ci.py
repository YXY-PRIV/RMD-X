#!/usr/bin/env python3
"""Pair fixed endpoint vectors, retaining all metric signs and masks."""
import argparse,csv,datetime,hashlib,json,os,pathlib,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
CODE=pathlib.Path(__file__).resolve().parents[1];OUT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ci')
sys.path.insert(0,str(CODE/'scripts'));from run_ci import frozen,read,ref,check,save,npz,now,NAMES

def measure(x,mask):
 x=np.asarray(x,np.float64);mask=np.asarray(mask,bool);assert x.shape==mask.shape and np.isfinite(x).all()
 v=x[mask];head=x[:1000][mask[:1000]];q=np.quantile(v,[.025,.975],method='linear')if len(v)else None;h=np.quantile(head,[.025,.975],method='linear')if len(head)else None
 return dict(valid_replicates=int(mask.sum()),missing_GT_sentinel_replicates=int((~mask).sum()),full_B2000_percentile=q,first1000_percentile=h,endpoint_shift_full_minus_first=q-h if q is not None and h is not None else None,interval_sign='strictly_positive'if q is not None and q[0]>0 else 'strictly_negative'if q is not None and q[1]<0 else 'includes_zero'if q is not None else 'undefined')
def load(method,P,pr,fr):
 cp=OUT/'results'/method/'completion.json'
 if not cp.exists():return None
 r=read(cp);assert r['B']==2000 and r['all_unique_ids_0_1999']and r['binding']['protocol_sha256']==pr['sha256']and r['binding']['code_freeze_sha256']==fr['sha256'];p=read(check(r['point']));g=read(check(r['gate']));check(g['cache']);check(p['arrays']);check(r['resource_plan']);check(r['summary'])
 assert g['status']=='PASS'and g['actual_fresh_native_matching_calls']==4 and len(g['reports'])==3 and all(z['passed']for z in g['reports'])
 c=P['cases'][method];assert p['GT']==c['GT']and p['params']==c['params']and p['prediction']==c['prediction']and p['source_stats_comparison']['passed']
 with np.load(check(r['arrays']),allow_pickle=False)as z:a={k:z[k].copy()for k in z.files}
 assert a['stats'].shape==(2000,12)and np.isfinite(a['stats']).all()and np.array_equal(a['bootstrap_id'],np.arange(2000))and list(a['stat_names'])==NAMES
 return a,p,r,g,ref(cp)
def main(partial):
 P,pr,fr=frozen();loaded={n:load(n,P,pr,fr)for n in P['cases']};pairs={};rows=[];arrays={};macros={}
 for comparison,base in [('material_minus_Tavail','Tavail'),('material_minus_permuted','permuted')]:
  result={};deltas=[];pts=[];fixeddeltas=[];fixedpts=[]
  for cohort in P['cohorts']:
   name=cohort+'__material';other=cohort+'__'+base;ca=loaded[name];ba=loaded[other]
   if ca is None or ba is None:continue
   c,cp,cr,cg,cref=ca;b,bp,br,bg,bref=ba
   for k in ['GT','params','draws','gate_draws','images','categories']:assert P['cases'][name][k]==P['cases'][other][k],(cohort,k)
   for k in ['valid_category_mask','category_image_multiplicity','unique_original_images']:assert np.array_equal(c[k],b[k]),(cohort,k)
   valid=(c['stats']>=0)&(b['stats']>=0);assert np.array_equal(c['stats']>=0,b['stats']>=0)
   d=c['stats']-b['stats'];pt=np.asarray(cp['stats'])-bp['stats'];deltas.append(d);pts.append(pt);stats={nm:dict(point_delta=float(pt[i]),**measure(d[:,i],valid[:,i]))for i,nm in enumerate(NAMES)}
   record=dict(candidate_point_stats=cp['stats'],comparator_point_stats=bp['stats'],paired_stats=stats,point_valid_GT_categories=int(cg['reports'][0]['valid_GT_categories']),valid_GT_category_count_range=[int(c['valid_category_mask'].sum(1).min()),int(c['valid_category_mask'].sum(1).max())],unique_sampled_images_range=[int(c['unique_original_images'].min()),int(c['unique_original_images'].max())],sampled_occurrences=P['cases'][name]['images'],draws=P['cases'][name]['draws'],GT=P['cases'][name]['GT'],params=P['cases'][name]['params'],same_category_masks_and_multiplicity=True,candidate_completion=cref,comparator_completion=bref)
   fixed=d[:,:3].copy();fixedpt=pt[:3].copy()
   if cohort=='DET':
    weights=c['valid_category_mask'].sum(1)/307.;fixed=d[:,:3]*weights[:,None];fixedpt=pt[:3]*(cg['reports'][0]['valid_GT_categories']/307.)
    assert np.array_equal(c['fixed307_zero_allarea_AP_AP50_AP75'],c['stats'][:,:3]*weights[:,None])and np.array_equal(b['fixed307_zero_allarea_AP_AP50_AP75'],b['stats'][:,:3]*weights[:,None])
    record['fixed307_missing_GT_zero_allarea']={nm:dict(point_delta=float(fixedpt[i]),**measure(fixed[:,i],valid[:,i]))for i,nm in enumerate(NAMES[:3])}
   fixeddeltas.append(fixed);fixedpts.append(fixedpt);result[cohort]=record
   arrays[comparison+'__'+cohort+'__delta']=d;arrays[comparison+'__'+cohort+'__valid']=valid
   for i,nm in enumerate(NAMES[:3]):
    z=stats[nm];rows.append(dict(comparison=comparison,dataset=cohort,metric=nm,point_candidate_AP_percent=float(cp['stats'][i])*100,point_comparator_AP_percent=float(bp['stats'][i])*100,point_delta_pp=float(pt[i])*100,CI95_lower_pp=float(z['full_B2000_percentile'][0])*100,CI95_upper_pp=float(z['full_B2000_percentile'][1])*100,first1000_lower_pp=float(z['first1000_percentile'][0])*100,first1000_upper_pp=float(z['first1000_percentile'][1])*100,interval_sign=z['interval_sign'],valid_replicates=z['valid_replicates']))
  pairs[comparison]=result
  if len(result)==6:
   d=np.asarray(deltas)[:,:,:3].mean(axis=0);p=np.asarray(pts)[:,:3].mean(axis=0);fd=np.asarray(fixeddeltas).mean(axis=0);fp=np.asarray(fixedpts).mean(axis=0)
   macros[comparison]=dict(fixed_six_macro={nm:dict(point_delta=float(p[i]),**measure(d[:,i],np.ones(2000,bool)))for i,nm in enumerate(NAMES[:3])},fixed_six_macro_DET307_sensitivity={nm:dict(point_delta=float(fp[i]),**measure(fd[:,i],np.ones(2000,bool)))for i,nm in enumerate(NAMES[:3])})
   arrays[comparison+'__macro_delta_AP_AP50_AP75']=d;arrays[comparison+'__macro_DET307_delta_AP_AP50_AP75']=fd
 state=dict(status='PARTIAL_PAIRED_FIXED_ENDPOINT_CI'if partial else'COMPLETE_PAIRED_SIX_TWO_COMPARISONS_B2000',updated_utc=now(),protocol=pr,code_freeze=fr,completed_endpoints=[k for k,v in loaded.items()if v is not None],planned_endpoints=18,B=2000,first_B=1000,primary_metric='AP',secondary_metrics=['AP50','AP75'],pairs=pairs,macros=macros,estimand=P['estimand'],macro_estimand=P['macro'],all_signs_retained=True,pointwise_unadjusted=True,no_new_training_or_test_selection=True,publication_edits=False)
 if partial:save(OUT/'paired_partial.json',state);print(json.dumps(dict(status=state['status'],completed_endpoints=len(state['completed_endpoints']),completed_primary_pairs=len(pairs['material_minus_Tavail']))));return
 assert len(state['completed_endpoints'])==18 and all(len(v)==6 for v in pairs.values())
 ap=OUT/'paired_replicates.npz';npz(ap,**arrays,stat_names=np.asarray(NAMES),bootstrap_id=np.arange(2000,dtype=np.int64));state['paired_arrays']=ref(ap);state['actual_new_native_matching_calls']=72;state['gate_replays']=54;state['resource_probe_replays']=18;state['formal_vector_replays']=36000
 save(OUT/'paired_summary.json',state)
 csvp=OUT/'paired_results.csv'
 with csvp.open('x',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 completion=dict(status=state['status'],completed_utc=now(),protocol=pr,code_freeze=fr,summary=ref(OUT/'paired_summary.json'),arrays=ref(ap),csv=ref(csvp),run_index=ref(OUT/'run_index.json'))
 save(OUT/'completion.json',completion);print(json.dumps(dict(status=state['status'],summary=completion['summary'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');main(p.parse_args().partial)
