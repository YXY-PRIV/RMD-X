#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.mixture_endpoint import CODE,prepare,export,evaluate
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--weights');p.add_argument('--phase',default='default');p.add_argument('--protocol',default=str(CODE/'config/protocol.json'));p.add_argument('--prepare',action='store_true');a=p.parse_args()
 if a.prepare:
  x=prepare(a.protocol);print(json.dumps(dict(status=x['status'],cases=len(x['cases']))));return
 e=export(a.dataset,[float(x)for x in a.weights.split(',')],a.phase,a.protocol);ep=Path(e['prediction']['path']).parent/'export.json';m=evaluate(ep,a.protocol)
 print(json.dumps(dict(export_receipt=str(ep),metrics=str(ep.parent/'metrics.json'),AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'])),flush=True)
if __name__=='__main__':main()
