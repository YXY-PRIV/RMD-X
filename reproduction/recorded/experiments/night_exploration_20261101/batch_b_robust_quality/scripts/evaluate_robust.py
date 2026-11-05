#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.robust_endpoint import export,evaluate,OUT
def main():
 p=argparse.ArgumentParser()
 for k in ['dataset','variant','recipe','profile','stage','role']:p.add_argument('--'+k,required=True)
 p.add_argument('--case',default='default');a=p.parse_args();e=export(a.dataset,a.variant,a.recipe,a.profile,a.stage,a.role,a.case);ep=Path(e['prediction']['path']).parent/'export.json';m=evaluate(ep)
 print(json.dumps(dict(export_receipt=str(ep),metrics=str(ep.parent/'metrics.json'),AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'])),flush=True)
if __name__=='__main__':main()
