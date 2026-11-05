#!/usr/bin/env python3
"""One CPU-only preparation after frozen study/source and independent gates."""
import os,sys,json,argparse
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.pair_projection import prepare

def main():
 p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--protocol-sha256',required=True);a=p.parse_args()
 print(json.dumps(prepare(a.protocol,a.protocol_sha256)),flush=True)

if __name__=='__main__':main()
