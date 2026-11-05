#!/usr/bin/env python3
"""One new material training or explicitly bridged label-free CUDA inference."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model import SEEDS,train,infer

def main():
 p=argparse.ArgumentParser();commands=p.add_subparsers(dest='mode',required=True)
 for name in ['train','infer']:
  s=commands.add_parser(name)
  for key in ['protocol','protocol-sha256','output-dir']:s.add_argument('--'+key,required=True)
  s.add_argument('--device',default='cuda:0')
  if name=='train':s.add_argument('--variant',required=True,choices=['material']);s.add_argument('--seed',required=True,type=int,choices=SEEDS)
  else:s.add_argument('--role',required=True,choices=['material','legacy_material']);s.add_argument('--case',required=True);s.add_argument('--checkpoint',required=True)
 a=p.parse_args();print(json.dumps(train(a)if a.mode=='train'else infer(a)),flush=True)

if __name__=='__main__':main()
