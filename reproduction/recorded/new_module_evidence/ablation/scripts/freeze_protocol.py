#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,datetime,os
C=Path(__file__).resolve().parents[1];O=Path('/mnt/data/yxy-codex/home/outputs/RAXO/new_module_evidence_20261102/ablation');R=C.parents[1];L=R/'pair_conflict_material_projection_20261101';Q=Path('/mnt/data/yxy-codex/home/outputs/RAXO')
def ref(p):
 p=Path(p).resolve()
 hobj=hashlib.sha256()
 with p.open('rb')as f:
  for block in iter(lambda:f.read(8<<20),b''):hobj.update(block)
 h=hobj.hexdigest()
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h)
def save(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2);f.write('\n')
P=dict(status='FROZEN_BEFORE_NEW_ABLATION_EXPORT_OR_AP',study_id='new_module_evidence_20261102_factorial_ablation',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 latest_protocol=ref(L/'config/protocol.json'),test_cases=['DET','CLCXray_test','PIXray','PIDray','DvXray','Hi'],seeds=[17,29,43],variant='material',alpha=1.,route='anchor_locked',
 cells=['00_full','10_gate','01_projection_identity_extension','11_both'],
 definitions=dict(delta='Immutable source full-supported-class centered F32 correction gathered into unique original native candidate slots; never retrained.',A='Existing P5 mask: both original branches retained, unequal classes/canonical slots, valid query T>1, actual support for both classes.',pair='Gather explicit branch V/F corrections; D=F32((F64(delta_F)-F64(delta_V))/2), then scatter [-D,+D].',cell00='delta',cell10='A*delta',cell01='A*pair(delta)+(1-A)*delta; identity extension outside the admissible pair domain, not an all-proposal projection',cell11='A*pair(delta)',
  no_defined_pair='Same-class, single-branch, invalid-query and unsupported-pair cases retain delta in cell01 and zero it in gated cells. No absent candidate is created.',
  operator_identity='The gate and identity-extended projection commute. Outside A cell01==cell00 and cell11==cell10; on A cell01==cell11, cell10==cell00, at correction level.',
  statistics='All 4 simple effects and descriptive AP interaction11-10-01+00, every sign retained. No parameter selection on these outcomes.'),
 pipeline=dict(sigmoid='Dense CPU torch F32 sigmoid of saved formal T per-seed logits plus correction; correction0 copies original formal T per-seed quality bitwise.',seed_mean='F64 mean3 thenF32',mix='Independent E3 visual-clean three-seed mean and corrected head geometric mix .5/.5 with clip1e-6..1, F64 thenF32',router='Original formal Tavail selected branch fixed for every cell',score='raw^.5 * quality^.5; original same-class Gaussian sigma1',nativeF_lambda_m=.05,nativeF_UNC=.15,GT='Original positive unique annotation IDs, same full params/images/categories; no new matching in exporters.'),
 reference_indices=dict(prior_full=ref(Q/'material_conditional_ranking_fix1_20261101/run_index.json'),latest_pair=ref(Q/'pair_conflict_material_projection_20261101/run_index.json')),
 reuse='00 and11 must exactly reproduce same-alpha/same-route prior full and published pair prediction bytes, entire q and key score traces; then only exact prediction/GT/params/native library native metrics can be reused.',
 budget=dict(logical_AP=24,expected_new_native_AP=12,expected_strict_reuses=12,new_training=0,new_GPU=0,new_model_forward=0),
 resources=dict(max_native_workers=2,global_native_slots=4,shared_slot_directory=str(O.parent/'resources/native_slots'),torch_CPU_threads=8,reserve_RAM_GiB=64,no_foreign_process_termination=True),
 scope='Fixed-weight posthoc diagnostic after test exposure, not confirmation or an untouched-test evaluation. Zero-sum logit correction is not probability-mean invariance; Gaussian can indirectly change inactive final scores.')
save(C/'protocol.json',P)
local=[ref(p)for p in sorted((C/'scripts').glob('*.py'))]
readonly=[L/'src/pair_endpoint.py',L/'src/pair_projection.py',L/'src/anchor_reference.py',R/'appearance_quality_reranking_20261031/src/quality_endpoint.py',R/'appearance_candidate_refinement_20261031/src/candidate_endpoint.py',R/'night_exploration_20261101/batch_c_shared_union/src/union_endpoint.py',R/'night_exploration_20261101/batch_c_shared_union/src/union_data.py',R/'material_reliability_20261101/src/reliability_endpoint.py']
freeze=dict(status='FROZEN_BEFORE_NEW_ABLATION_EXPORT_OR_AP',protocol=ref(C/'protocol.json'),sources=local,readonly_sources=[ref(p)for p in readonly],created_utc=P['created_utc'])
save(O/'automation/code_freeze.json',freeze);print(json.dumps(dict(protocol=freeze['protocol'],code_freeze=ref(O/'automation/code_freeze.json'))))
