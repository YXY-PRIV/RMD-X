#!/usr/bin/env python3
"""One source-only conditional training or label-free full-C CUDA inference."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model import VARIANTS,SEEDS,train,infer

def main():
 p=argparse.ArgumentParser();commands=p.add_subparsers(dest='mode',required=True)
 for mode in ['train','infer']:
  s=commands.add_parser(mode)
  for key in ['protocol','protocol-sha256','output-dir','variant']:s.add_argument('--'+key,required=True,**({'choices':VARIANTS}if key=='variant'else{}))
  s.add_argument('--device',default='cuda:0')
  if mode=='train':s.add_argument('--seed',required=True,type=int,choices=SEEDS)
  else:s.add_argument('--case',required=True);s.add_argument('--checkpoint',required=True)
 a=p.parse_args();print(json.dumps(train(a)if a.mode=='train'else infer(a)),flush=True)

if __name__=='__main__':main()
