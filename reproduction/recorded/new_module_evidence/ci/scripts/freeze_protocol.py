#!/usr/bin/env python3
"""Freeze source-selected P5 endpoints and inherited image draws before new matching."""
import datetime,hashlib,json,pathlib,sys
CODE=pathlib.Path(__file__).resolve().parents[1]
OUT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ci')
OLD=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/pair_conflict_material_projection_20261101')
RAXO=CODE.parents[2]
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):
 p=pathlib.Path(p).resolve()
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 h=h.hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def check(r):assert ref(r['path'])==r
p5p=RAXO/'experiments/pair_conflict_material_projection_20261101/config/protocol.json';indexp=OLD/'run_index.json';idx=read(indexp);assert idx['protocol']==ref(p5p)
oldp=RAXO/'experiments/night_exploration_20261101/batch_p_supported_rank_bootstrap/config/protocol.json';drawproto=read(oldp)
cohorts=['DET','CLCXray_test','PIXray','PIDray','DvXray','Hi'];aliases={'CLCXray_test':'CLCXray'};cases={};source_refs=[ref(p5p),ref(indexp),ref(OLD/'completion.json'),ref(OLD/'parameters/selected_parameters.json'),ref(oldp)]
for cohort in cohorts:
 main=next(e for e in idx['endpoints']if e['dataset']==cohort and e['variant']=='material'and e['route']=='anchor_locked'and e['residual_alpha']==1.)
 perm=next(e for e in idx['endpoints']if e['dataset']==cohort and e['variant']=='permuted_material'and e['route']=='anchor_locked'and e['residual_alpha']==1.)
 me=read(main['export_receipt']['path']);ar=me['identity']['quality']['strong_anchor_export'];check(ar)
 endpoints={'material':(main['export_receipt'],main['metrics']),'Tavail':(ar,ref(pathlib.Path(ar['path']).parent/'metrics.json')),'permuted':(perm['export_receipt'],perm['metrics'])}
 dc=drawproto['cases'][aliases.get(cohort,cohort)]
 for role,(er,mr)in endpoints.items():
  check(er);check(mr);e=read(er['path']);m=read(mr['path']);gt=m['original_GT'];pred=e['prediction'];check(gt);check(pred)
  assert m['prediction']['sha256']==pred['sha256']and m['params']==dc['params']and gt==dc['GT'],(cohort,role)
  cases[cohort+'__'+role]=dict(cohort=cohort,role=role,method='P5_'+role,source_metric=mr,source_export=er,prediction=pred,GT=gt,params=m['params'],expected_point_stats=m['stats'],expected_point_AP=m['AP'],prediction_count=e['prediction_count'],images=len(m['params']['imgIds']),categories=len(m['params']['catIds']),draws=dc['draws'],gate_draws=dc['gate_draws'])
# Exact input comparison against every completed prior bootstrap endpoint; never reuse by AP closeness.
prior=[]
for cp in (OLD.parent/'night_exploration_20261101').glob('**/results/*/completion.json'):
 j=read(cp)
 if 'gate'not in j or 'point'not in j:continue
 pt=read(j['point']['path'])
 if 'prediction'not in pt or 'GT'not in pt or 'params'not in pt:continue
 dr=read(j['gate']['path'])['input_binding']['draws']
 prior.append(dict(completion=ref(cp),prediction=pt['prediction'],GT=pt['GT'],params=pt['params'],draws=dr))
reuse={}
for name,c in cases.items():
 matches=[p for p in prior if all(p[k]==c[k]for k in ['prediction','GT','params','draws'])]
 assert not matches,'If exact reuse is possible integrate and freeze explicitly before running: '+name
 reuse[name]=dict(prior_completed_methods_considered=len(prior),exact_matches=0,new_matching_and_replay_required=True)
lib=pathlib.Path('/mnt/data/yxy/envs/gdino_cu118_clean/lib/python3.10/site-packages/pycocotools')
P=dict(study_id='new_module_evidence_20261102_ci',status='FROZEN_BEFORE_NEW_MATCHING_OR_BOOTSTRAP',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),hypotheses=['Source-selected latest pair material minus fixed formal Tavail union','Same latest material minus trained permuted-material pair control'],source_selection='P5 alpha=1 anchor_locked selected on CLC_tune before this CI; endpoints fixed, no refit/reselection',cohorts=cohorts,roles=['material','Tavail','permuted'],cases=cases,method_order=[c+'__'+r for c in cohorts for r in ['material','Tavail']]+[c+'__permuted'for c in cohorts],B=2000,first_B_sensitivity=1000,quantiles=[.025,.975],quantile_method='linear',source_refs=source_refs,reuse_audit=reuse,native_libraries={f:ref(lib/f)for f in ['coco.py','cocoeval.py','mask.py','_mask.cpython-310-x86_64-linux-gnu.so']},matching='Fresh installed unmodified native COCO bbox matching for each fixed endpoint; identity plus bootstrap0/1 cloned native gates at1e-12 for precision/recall/scores/all12stats/params/matchrecords. Gate image occurrences receive positive newGT/DTIDs.',formal_engine='Gated vector replay of fresh native matching cache, B2000 fixed inherited image draws; never2000 fresh matching calls.',estimand='Conditional empirical image bootstrap of fixed, historically test-exposed predictions. Does not include training, model selection or physical-specimen uncertainty. Pointwise intervals unadjusted for multiplicity, not universal superiority/confirmatory evidence.',macro='Equal weight of the fixed six cohorts and same replicate indices across independent cohort draws. Not a population/domain bootstrap. DET fixed307 allarea sensitivity retained.',resources=dict(max_workers=24,threads_per_worker=1,reserved_RAM_GiB=64,guard_factor=1.25,native_workers=2,CPU_affinity=list(range(96,120)),shared_native_slot_directory='/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/resources/native_slots',native_slot_names=['slot0.lock','slot1.lock','slot2.lock','slot3.lock'],shared_formal_lock='/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/resources/ci_formal.lock',runtime_pause_request='/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/resources/runtime_quiet_request.json',worker_rule='After identity/draw0/draw1 PASS, isolated draw0 resource probe only; workers=min24,floor((MemAvailableGiB-64)/(1.25*peakRSSGiB)), independent of AP sign.'),code_root=str(CODE),output_root=str(OUT),publication_edits=False,training_or_model_selection=False,all_old_roots_readonly=True)
pp=CODE/'config/protocol.json'
with pp.open('x')as f:json.dump(P,f,indent=2,allow_nan=False);f.write('\n')
fr=dict(status='FROZEN_BEFORE_ANY_NEW_MATCHING',protocol=ref(pp),sources=[ref(p)for p in sorted(CODE.rglob('*.py'))])
with(OUT/'automation/code_freeze.json').open('x')as f:json.dump(fr,f,indent=2);f.write('\n')
print(json.dumps(dict(protocol=ref(pp),freeze=ref(OUT/'automation/code_freeze.json'),methods=len(cases),prior_completed_methods_considered=len(prior),exact_reuse=0)))
