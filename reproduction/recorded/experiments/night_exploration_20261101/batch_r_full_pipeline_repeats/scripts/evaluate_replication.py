#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.replication_endpoint import export_endpoint,evaluate_export,read,ref
p=argparse.ArgumentParser()
for k in ['dataset','variant','route','case','protocol','output-root']:p.add_argument('--'+k,required=True)
p.add_argument('--tau',type=float,required=True)
p.add_argument('--mix',type=float,required=True)
a=p.parse_args();e=export_endpoint(a.dataset,a.variant,a.route,a.tau,a.protocol,a.output_root,a.case,a.mix)
ep=Path(e['prediction']['path']).parent/'export.json';m=evaluate_export(ep,a.output_root)
print(json.dumps(dict(export_receipt=str(ep),metrics=str(ep.parent/'metrics.json'),AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'])),flush=True)
