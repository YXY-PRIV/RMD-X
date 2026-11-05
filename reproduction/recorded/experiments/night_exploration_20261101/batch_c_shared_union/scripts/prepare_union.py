#!/usr/bin/env python3
"""Prepare one frozen native-union pair feature case; never run AP or training."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.union_data import prepare,CODE
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True,choices=['CLC_cal_fit','CLC_tune','DET','Hi','CLCXray_test','PIXray','PIDray','DvXray']);p.add_argument('--protocol',default=str(CODE/'config/protocol.json'));p.add_argument('--chunk-size',type=int,default=1024);a=p.parse_args();print(json.dumps(prepare(a.case,a.protocol,a.chunk_size)),flush=True)
if __name__=='__main__':main()
