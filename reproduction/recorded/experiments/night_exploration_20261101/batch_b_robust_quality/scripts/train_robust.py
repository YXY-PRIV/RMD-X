#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from robust_model import train,write
def main():
 p=argparse.ArgumentParser()
 for n in ['input','protocol','recipe-id','variant','output-dir']:p.add_argument('--'+n,required=True)
 p.add_argument('--seed',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--resume');a=p.parse_args()
 try:r=train(a)
 except BaseException as e:
  if Path(a.output_dir).exists():write(Path(a.output_dir)/'run_status.json',dict(status='FAILED',error=repr(e),completion_claimed=False))
  raise
 print(json.dumps({k:r[k]for k in ['status','recipe_id','variant','seed','best_epoch','epochs_completed','optimizer_steps_completed']}),flush=True)
if __name__=='__main__':main()
