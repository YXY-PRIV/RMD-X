#!/usr/bin/env python3
"""Only a previously incomplete label-free inference; never a training CLI."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.quality_model_continued import VARIANTS,infer

def main():
 p=argparse.ArgumentParser()
 for key in ['protocol','protocol-sha256','output-dir','case','checkpoint']:p.add_argument('--'+key,required=True)
 p.add_argument('--variant',required=True,choices=VARIANTS);p.add_argument('--device',default='cuda:0')
 a=p.parse_args();print(json.dumps(infer(a)),flush=True)

if __name__=='__main__':main()
