#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True;sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.fallback_endpoint import export_endpoint,evaluate_export
p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--route',required=True);a=p.parse_args();C=Path(__file__).resolve().parents[1];e=export_endpoint(a.dataset,'missing_fallback',a.route,0.,C/'config/protocol.json',phase='final');ep=Path(e['prediction']['path']).parent/'export.json';m=evaluate_export(ep);print(json.dumps(dict(export_receipt=str(ep),metrics=str(ep.parent/'metrics.json'),AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'])),flush=True)
