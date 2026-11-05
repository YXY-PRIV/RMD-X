#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.union_model import train
def main():
    p=argparse.ArgumentParser()
    for name in ['input','protocol','output-dir']:p.add_argument('--'+name,required=True)
    p.add_argument('--variant',required=True,choices=['full','visual','missing']);p.add_argument('--seed',required=True,type=int);p.add_argument('--device',default='cuda:0');a=p.parse_args();print(json.dumps(train(a)),flush=True)
if __name__=='__main__':main()
