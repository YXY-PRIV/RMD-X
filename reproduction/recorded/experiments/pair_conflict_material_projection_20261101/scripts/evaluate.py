#!/usr/bin/env python3
"""One fixed endpoint and actual native COCO evaluation or exact reuse."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pair_endpoint import OUT, export_endpoint, evaluate_export

parser = argparse.ArgumentParser()
for name in ['dataset', 'variant', 'route', 'protocol']:
    parser.add_argument('--'+name, required=True)
parser.add_argument('--alpha', required=True, type=float)
parser.add_argument('--phase', required=True, choices=['calibration', 'final'])
args = parser.parse_args()
e = export_endpoint(args.dataset, args.variant, args.route, args.alpha, args.protocol, OUT, args.phase)
ep = Path(e['prediction']['path']).parent/'export.json'
m = evaluate_export(ep)
print(json.dumps(dict(export_receipt=str(ep), metrics=str(ep.parent/'metrics.json'), AP=m['AP'],
    AP50=m['AP50'], AP75=m['AP75'], AR100=m['stats'][8],
    AP_computed_here=m['AP_computed_here'], AP_reused=m['AP_reused'])), flush=True)
