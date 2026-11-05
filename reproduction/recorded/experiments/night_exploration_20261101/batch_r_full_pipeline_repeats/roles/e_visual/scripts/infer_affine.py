#!/usr/bin/env python3
import argparse,sys,json
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.affine_model import infer
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--feature-manifest',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--device',default='cuda:0');print(json.dumps(infer(p.parse_args())),flush=True)
