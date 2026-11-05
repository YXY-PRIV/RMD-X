"""Freeze H0 metadata, deterministic old-query rows, image hashes and sources."""
from __future__ import annotations
import csv,datetime,hashlib,heapq,json,pathlib
ROOT=pathlib.Path('/mnt/data/yxy/program/RAXO');CODE=pathlib.Path(__file__).resolve().parents[1];OUT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/night_exploration_20261101/batch_h_multisource_material');AUDIT=pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/batch01_20261023/data_audit')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):
 p=pathlib.Path(p).resolve();h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def fresh(p,j):
 with pathlib.Path(p).open('x')as f:json.dump(j,f,indent=2,allow_nan=False);f.write('\n')
def select_rows(p):
 def gen():
  with p.open()as f:
   for i,line in enumerate(f):
    r=json.loads(line);assert r['cache_index']==i
    key=dict(global_cache_index=i,image_id=int(r['image_id']),annotation_id=r.get('annotation_id'),file_name=r['file_name'],bbox=[float(x).hex()for x in r['bbox']],detector_score=float(r['detector_score']).hex())
    yield digest(key),i,r,key
 return heapq.nsmallest(32,gen(),key=lambda x:(x[0],x[1]))
def main():
 target=CODE/'config/gate_protocol.json';assert not target.exists();source_names=['rmdx/material.py','rmdx/preargmax_fusion.py','rmdx/cache_builder.py'];sources={n:ref(ROOT/n)for n in source_names};driver={n:ref(CODE/n)for n in ['scripts/prepare_gate.py','scripts/run_gate.py']}
 datasets={}
 for ds in ['pidray','HiXray']:
  cache=ROOT/'outputs_rmdx/cache_multidataset'/f'{ds}_groundingDINO_100_0_full';rp=cache/'proposal_meta.jsonl';meta=read(cache/'cache_summary.json');catp=cache/'category_meta.json';cats=read(catp);assert [x['class_index']for x in cats]==list(range(len(cats)))
  selected=[];images={};image_root=pathlib.Path('/mnt/data/raxo/datasets/xray-datasets')/ds/'test'
  for h,i,r,key in select_rows(rp):
   path=image_root/r['file_name'];ir=ref(path);images[str(path)]=ir;selected.append(dict(row_key_sha256=h,cache_index=i,key=key,proposal=r,image=ir))
  datasets[ds]=dict(cache_summary=ref(cache/'cache_summary.json'),old_B=ref(cache/'material_class_similarity.pt'),proposal_meta=ref(rp),category_meta=ref(catp),actual_bank=ref(ROOT/'outputs_rmdx/material_multidataset/banks'/f'{ds}_material_bank_transfer_from_det_compass_support.json'),category_order=[x['category_name']for x in cats],category_ids=[x['category_id']for x in cats],old_B_expected_shape=[meta['proposal_count'],len(cats)],old_B_expected_dtype='float32',image_root=str(image_root),selected_rows=selected,unique_image_refs=list(images.values()))
 P=dict(study_id='H0_frozen_material_query_producer_gate_20261101',status='FROZEN_BEFORE_ANY_H0_MATERIAL_QUERY_EXTRACTION',created_utc=now(),scope='CPU numeric compatibility gate only; no officialtrain feature extraction, GDINO, SAM, DINO, training, GPU, AP or test GT annotations.',selection='For each PID/Hi full ordered proposal_meta stream, compute SHA256 of canonical sorted JSON key(global_cache_index,image_id,annotation_id,file_name,bbox float.hex,detector_score float.hex); take exactly32 smallest(hash,index), no source error/value-based selection. Compare all target classes for each selected row.',selection_per_case=32,production_sources=sources,gate_driver_sources=driver,actual_producer='Unmodified rmdx.preargmax_fusion.build_category_material_matrix(names,names,bank), then compute_material_similarity_for_proposals with actual RGB bbox_only acquisition and scalar material F64 zscore/cosine, final TorchF32. Only trace-wrap original _proposal_features to save returned actual22 values; no arithmetic/inputs stubbed.',normalization='Actual lookup reconstructs class-equal F64 mean/popstd from unchanged bank mean_feature vectors in frozen cache category order; do not use stored bank_feature_stats, deduplicate equal means, refit or patch B.',acceptance='Strict uint32 bitwise zero differences in every32*C compared value. Any mismatch blocks further H science; preserve old protocol/code/attempt, report ULPs and sources before any repair.',datasets=datasets,prior_DET_evidence=ref(pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/P26_end_revision_20261024/runtime/verification/A2_GD_complete_source_hex_evidence.json')),source_catalogs={n:ref(AUDIT/n)for n in ['extended_image_fingerprints.csv','image_fingerprints.csv','rebuilt_bank_image_fingerprints.csv']},CLC_role_exclusion=ref(pathlib.Path('/mnt/data/yxy-codex/home/outputs/RAXO/P16_P25_revision_20261024/validation/data/roles.csv')),future_train_exclusion='All six eval image names/fileSHA/RGBSHA + ALL known legacy/rebuilt support images + explicit CLC cal_fit191/tune191 names/fileSHA/RGBSHA. Physical specimen/capture-session identity remains unverified. Candidate inventory work is separate from actual extraction.',new_science_outside_H0_authorized=False,publication_edits=False)
 fresh(target,P);fresh(OUT/'protocol/gate_protocol.json',P);fresh(OUT/'protocol/gate_code_freeze.json',dict(status='FROZEN_BEFORE_MATERIAL_EXTRACTION',created_utc=now(),protocol=ref(target),production_sources=sources,driver_sources=driver));print(json.dumps(dict(status='FROZEN',protocol=ref(target),cases={k:[r['cache_index']for r in v['selected_rows']]for k,v in datasets.items()})),flush=True)
if __name__=='__main__':main()
