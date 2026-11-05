#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.supported_data import training,queries,frozen,OUT,save,ref,now
P=frozen();r=training();print(json.dumps(dict(stage='TRAINING_TRANSFORM_COMPLETE',receipt=ref(OUT/'training/data_receipt.json'))),flush=True);records=[]
for case in P['inference_cases']:
 r=queries(case);records.append(r);print(json.dumps(dict(stage='QUERY_TRANSFORM_COMPLETE',case=case,changed=r['changed_scalar_counts'])),flush=True)
save(OUT/'data_completion.json',dict(status='COMPLETE',completed_utc=now(),protocol=ref(Path(__file__).resolve().parents[1]/'config/protocol.json'),training=ref(OUT/'training/data_receipt.json'),query_transformations=records,new_GT_RGB_encoder_AP_runs=0));print('N_DATA_ALL_COMPLETE',flush=True)
