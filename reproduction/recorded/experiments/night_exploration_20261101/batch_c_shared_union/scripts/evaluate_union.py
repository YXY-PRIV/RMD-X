#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True;os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1');os.environ.setdefault('CUDA_VISIBLE_DEVICES','');sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.union_endpoint import CODE,OUT,VARIANTS,ROUTES,export_endpoint,evaluate_export
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['CLC_tune','DET','CLCXray_test','PIXray','PIDray','DvXray','Hi']);p.add_argument('--variant',required=True,choices=VARIANTS);p.add_argument('--route',required=True,choices=ROUTES);p.add_argument('--tau',type=float,required=True);p.add_argument('--case',default='final');p.add_argument('--protocol',default=str(CODE/'config/protocol.json'));p.add_argument('--output-root',default=str(OUT));p.add_argument('--stage',choices=['export','all'],default='all');a=p.parse_args();e=export_endpoint(a.dataset,a.variant,a.route,a.tau,a.protocol,a.output_root,a.case);ep=Path(e['prediction']['path']).parent/'export.json';out=dict(export_receipt=str(ep),metrics=str(ep.parent/'metrics.json')if a.stage=='all'else None,prediction_count=e['prediction_count'],counts=e['counts'])
    if a.stage=='all':
        m=evaluate_export(ep,a.output_root);out.update(AP=m['AP'],AP50=m['AP50'],AP75=m['AP75'],AR100=m['AR100'],AP_computed_here=m['AP_computed_here'],AP_reused=m['AP_reused'])
    print(json.dumps(out),flush=True)
if __name__=='__main__':main()
