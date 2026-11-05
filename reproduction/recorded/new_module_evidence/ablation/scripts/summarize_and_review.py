#!/usr/bin/env python3
"""Read-only independent actual-array and sealed metric review, then descriptive contrasts."""
from pathlib import Path
import os,sys,json,hashlib,csv,datetime
import numpy as np
sys.dont_write_bytecode=True
C=Path(__file__).resolve().parents[1];O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ablation');R=C.parents[1]
sys.path.insert(0,str(R/'pair_conflict_material_projection_20261101'))
from src import pair_endpoint as pe
CELLS=['00_full','10_gate','01_projection_identity_extension','11_both']
def read(p):return json.loads(Path(p).read_text())
def ref(p):return pe.ref(p)
def verify(r):return pe.verify(r)
def save(p,v):
 p=Path(p)
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def bits(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def stats_ref(r):return read(verify(r))
def independent_review(case,rs,oldP):
 # No pair_projection helper in this reviewer. Re-derive admissibility and all
 # four gathered residuals from sealed original components and branch mappings.
 import torch
 torch.set_num_threads(8)
 c=oldP['cases'][case];cf=read(verify(c['C_feature_manifest']));af=read(verify(c['auxiliary_feature_manifest']))
 a=pe.loadnpz(verify(cf['outputs']['source_trace.npz']));mapping=np.load(verify(cf['outputs']['branch_to_candidate.npy']),allow_pickle=False)
 valid=np.load(verify(af['outputs']['candidate_valid.npy']),allow_pickle=False);T=np.load(verify(af['outputs']['supported_reference_T.npy']),allow_pickle=False)
 ci=np.load(verify(af['outputs']['candidate_class_order_index.npy']),allow_pickle=False);has=np.load(verify(af['outputs']['actual_supported_class_mask.npy']),allow_pickle=False);qv=np.load(verify(af['outputs']['query_valid.npy']),allow_pickle=False)
 av=np.zeros(valid.shape,bool);row,sl=np.nonzero(valid);av[row,sl]=qv[row]&has[ci[row,sl]]
 rr=np.flatnonzero(a['branch_kept'].all(1));A=np.zeros(len(valid),bool)
 vs=mapping[rr,0];fs=mapping[rr,1]
 A[rr]=(a['branch_category_id'][rr,0]!=a['branch_category_id'][rr,1])&(vs!=fs)&av[rr,vs]&av[rr,fs]&(T[rr]>1)
 trace={r['cell']:pe.loadnpz(verify(read(verify(r['export']))['score_trace']))for r in rs}
 for cell in CELLS:assert bits(trace[cell]['active_mask'],A)
 reference=trace[CELLS[0]];rows=reference['selected_union_index'];branch=reference['selected_branch'];slot=reference['selected_candidate_slot']
 for cell in CELLS:
  z=trace[cell]
  for key in ['global_cache_index','selected_union_index','selected_branch','selected_candidate_slot','selected_category_id','native_raw_score','strong_anchor_reference_candidate_quality']:assert bits(z[key],reference[key]),(case,cell,key)
  assert bits(z['selected_category_id'],a['branch_category_id'][rows,branch])
  assert bits(z['native_raw_score'],a['raw_score'][rows].astype(np.float64))
  pred=read(verify(read(verify(next(r['export']for r in rs if r['cell']==cell)))['prediction']))
  assert len(pred)==len(rows)
  assert all(int(p['image_id'])==int(a['image_id'][i])and int(p['category_id'])==int(a['branch_category_id'][i,b])and p['bbox']==a['bbox'][i].tolist()and float(p['score'])==float(s)for p,i,b,s in zip(pred,rows,branch,z['post_soft_score']))
 # Source references are recorded independently for all seeds; exact non-active
 # fallbacks and cross-cell q equalities directly test the factor decomposition.
 for cell in CELLS:
  z=trace[cell]
  if cell.startswith('1'):assert bits(z['full_candidate_quality'][~A],z['strong_anchor_reference_candidate_quality'][~A])
 assert bits(trace[CELLS[2]]['full_candidate_quality'][~A],trace[CELLS[0]]['full_candidate_quality'][~A])
 assert bits(trace[CELLS[2]]['full_candidate_quality'][A],trace[CELLS[3]]['full_candidate_quality'][A])
 assert bits(trace[CELLS[1]]['full_candidate_quality'][A],trace[CELLS[0]]['full_candidate_quality'][A])
 # These identities precede Gaussian; final off-mask rows can be suppressed by
 # modified same-class conflict rows and need not keep final scores.
 e0=read(verify(next(r['export']for r in rs if r['cell']==CELLS[0])))
 source_components=e0['identity']['source_components'];checks=[]
 for x in source_components:
  z=pe.loadnpz(verify(x['source_components']));d=z['centered_correction'];assert bits(z['valid'],valid)
  assert bits(z['availability'],av.astype(np.float32));assert np.isfinite(d).all()and not np.any(d[~(av&(T[:,None]>1))])
  ri=np.flatnonzero(A);vi=mapping[ri,0];fi=mapping[ri,1]
  D=((d[ri,fi].astype(np.float64)-d[ri,vi].astype(np.float64))/2).astype(np.float32)
  p=np.zeros(d.shape,np.float32);p[ri,vi]=-D;p[ri,fi]=D
  assert np.all(p[ri].astype(np.float64).sum(1)==0)
  arr={CELLS[0]:d.copy(),CELLS[1]:np.zeros_like(d),CELLS[2]:d.copy(),CELLS[3]:p}
  arr[CELLS[1]][A]=d[A];arr[CELLS[2]][A]=p[A]
  for cell in CELLS:
   expect_hash=hashlib.sha256(arr[cell].tobytes()).hexdigest();ex=read(verify(next(r['export']for r in rs if r['cell']==cell)))
   rec=next(s for s in ex['identity']['cell_seed_statistics']if s['seed']==x['seed']);assert rec['correction_array_sha256']==expect_hash
  checks.append(dict(seed=x['seed'],original_components=x['source_components'],independently_reconstructed_four_correction_hashes_match=True,projected_pair_sum_exact0=True))
 for r in rs:
  e=read(verify(r['export']));m=read(verify(r['metrics']));assert m['prediction']==e['prediction']and m['original_GT']==e['source_GT']and m['params']==e['source_params']
  assert r['stats']==m['stats']and r['AP']==m['stats'][0]and r['AP50']==m['stats'][1]and r['AP75']==m['stats'][2]
  for k,source in pe.pure._native_sources().items():assert m[k]==source
  if r['cell']in ['00_full','11_both']:
   assert e['golden']['same_prediction_bytes']and e['golden']['full_candidate_q_bitdiff']==0 and m['AP_reused']
  else:assert m['AP_computed_here']and not m['AP_reused']
 return dict(case=case,status='PASS',active_proposals=int(A.sum()),total_proposals=len(A),independent_mask_and_projection_checks=checks,
  fixed_branch_category_box_rawscore_order=True,full_six_field_prediction_trace_consistency=True,cross_cell_factor_identities_bitexact_preGaussian=True,
  same_param_native_COCO_source_bindings=True,golden00_and11_prediction_bytes_and_all_q_exact=True)

def main():
 P=read(C/'protocol.json');pr=ref(C/'protocol.json');oldP=read(verify(P['latest_protocol']));state=read(O/'automation/status.json');assert state['status']=='COMPLETE_24_LOGICAL_ABLATION_CELLS'
 freeze=read(O/'automation/code_freeze.json');assert freeze['protocol']==pr
 for r in freeze['sources']+freeze['readonly_sources']:verify(r)
 allrows=[];comparisons=[];reviews=[];case_partitions=[]
 effect_defs=dict(gate_without_projection=['10_gate','00_full'],gate_with_projection=['11_both','01_projection_identity_extension'],projection_without_gate_identity_extended=['01_projection_identity_extension','00_full'],projection_with_gate=['11_both','10_gate'])
 for case in P['test_cases']:
  d=read(O/'results'/f'{case}_results.json')
  assert d['protocol']==pr and d['status']=='COMPLETE_FOUR_ABLATION_CELLS';rs=d['results'];assert [r['cell']for r in rs]==CELLS
  allrows.extend(rs);reviews.append(independent_review(case,rs,oldP));by={r['cell']:r for r in rs}
  e=read(verify(rs[0]['export']));case_partitions.append(dict(dataset=case,**e['identity']['mask_partition']))
  baseline_m=read(verify(e['identity']['formal_anchor_report']['archived_metrics']));bc={k:baseline_m[k]for k in ['AP','AP50','AP75']}
  effects={name:{k:100*(by[left][k]-by[right][k])for k in ['AP','AP50','AP75']}for name,(left,right)in effect_defs.items()}
  interaction={k:100*(by['11_both'][k]-by['10_gate'][k]-by['01_projection_identity_extension'][k]+by['00_full'][k])for k in ['AP','AP50','AP75']}
  comp=dict(dataset=case,AP_fraction={cell:by[cell]['AP']for cell in CELLS},formal_anchor_AP_fraction=bc['AP'],effects_pp=effects,interaction_pp=interaction,
    each_cell_minus_formal_anchor_pp={cell:100*(by[cell]['AP']-bc['AP'])for cell in CELLS},posthoc_no_selection=True)
  comparisons.append(comp)
 assert len(allrows)==24 and sum(r['AP_computed_here']for r in allrows)==12 and sum(r['AP_reused']for r in allrows)==12
 summary=dict(status='COMPLETE_24_LOGICAL_12_NEW_NATIVE_12_STRICT_REUSE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol=pr,code_freeze=ref(O/'automation/code_freeze.json'),
  cases=comparisons,effect_definitions=effect_defs,interaction_definition='AP11-AP10-AP01+AP00 (pp), descriptive, not a significance test or causal estimate',
  mean_simple_effect_pp={name:float(np.mean([c['effects_pp'][name]['AP']for c in comparisons]))for name in effect_defs},
  mean_interaction_pp=float(np.mean([c['interaction_pp']['AP']for c in comparisons])),
  positive_negative_tie_counts={name:dict(positive=sum(c['effects_pp'][name]['AP']>0 for c in comparisons),negative=sum(c['effects_pp'][name]['AP']<0 for c in comparisons),tie=sum(c['effects_pp'][name]['AP']==0 for c in comparisons))for name in effect_defs},
  logical_native_AP=24,actual_new_native_AP=12,strict_exact_reuse_AP=12,new_training=0,new_GPU_inference=0,new_model_forward=0,
  selected_method_changed=False,all_signs_retained=True,scope=P['scope'])
 save(O/'results/summary.json',summary);save(O/'verification/independent_array_metric_review.json',dict(status='PASS',protocol=pr,cases=reviews,summary=ref(O/'results/summary.json')))
 fields=['dataset','cell','AP','AP50','AP75','AP_computed_here','AP_reused']
 with(O/'results/all_results.csv').open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=fields+['export_path','metrics_path']);w.writeheader()
  for r in allrows:w.writerow({**{k:r[k]for k in fields},'export_path':r['export']['path'],'metrics_path':r['metrics']['path']})
 with(O/'results/factorial_effects.csv').open('x',newline='')as f:
  fields=['dataset']+CELLS+list(effect_defs)+['interaction','formal_anchor','both_minus_formal_anchor'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for r in comparisons:w.writerow(dict(dataset=r['dataset'],**{c:100*r['AP_fraction'][c]for c in CELLS},**{name:r['effects_pp'][name]['AP']for name in effect_defs},interaction=r['interaction_pp']['AP'],formal_anchor=100*r['formal_anchor_AP_fraction'],both_minus_formal_anchor=r['each_cell_minus_formal_anchor_pp']['11_both']))
 with(O/'results/activity_counts.csv').open('x',newline='')as f:
  w=csv.DictWriter(f,fieldnames=list(case_partitions[0]));w.writeheader();w.writerows(case_partitions)
 lines=['# 新模块冲突筛选与成对投影分项消融','', '状态：完成。24个逻辑设置=12次新原生COCO评测+12项全输入严格复用；不重训、不新推理、不改当前主方法选择。','',
 '固定原材料模型、alpha=1与anchor_locked，不按任何测试结果重新选参。δ表示既有全支持类残差，A表示原P5可用冲突对掩码。00=δ；10=Aδ；01=A·pair(δ)+(1−A)δ；11=A·pair(δ)。01仅在可定义成对投影的A上去共同分量，A外使用恒等延拓；没有给单分支、同类或缺支持候选创造一对。', '',
 '|数据集|00 full AP%|10 gate AP%|01 projection+identity AP%|11 both AP%|gate effect no projection pp|gate effect with projection pp|projection effect no gate pp|projection effect with gate pp|interaction pp|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
 for r in comparisons:
  vals=[r['dataset']]+[f"{100*r['AP_fraction'][c]:.6f}"for c in CELLS]+[f"{r['effects_pp'][name]['AP']:+.6f}"for name in effect_defs]+[f"{r['interaction_pp']['AP']:+.6f}"]
  lines.append('|'+ '|'.join(vals)+'|')
 lines+=['','所有差异均为固定预测的描述性点AP差，不属于未看过测试集验证、训练方差或独立因果证据。交互项只表示AP非线性下的算术差，不能称为统计显著的协同作用。', '', '独立复核已逐行重建原支持掩码、分支映射、四组每seed残差hash及成对和0；核对四组评分交叉恒等式、所有输出类别/原框/rawscore顺序、实际预测score和score_trace，以及全部原生COCO源库/参数。00与前一轮固定alpha1/anchor_locked全类头、11与当前P5预测JSON逐byte且完整quality精确一致。', '', '可用于正文解释：筛选和投影的效果依赖上下文，必须报告四个简单效应与所有正负结果；不能只用完整方法相对全类头的一个差把两种机制的作用合并归因。']
 (C/'执行报告.md').write_text('\n'.join(lines)+'\n')
 save(O/'completion.json',dict(status=summary['status'],summary=ref(O/'results/summary.json'),review=ref(O/'verification/independent_array_metric_review.json'),report=ref(C/'执行报告.md'),protocol=pr,new_training=0,new_GPU=0,new_model_forward=0))
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
