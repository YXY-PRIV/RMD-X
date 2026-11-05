import os,sys,pathlib,argparse
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'src'))
import data_pipeline as d
a=argparse.ArgumentParser();a.add_argument('stage',choices=['freeze','eligibility','training','inference']);a.add_argument('--case');args=a.parse_args()
if args.stage=='freeze':d.freeze()
elif args.stage=='eligibility':d.eligibility()
elif args.stage=='training':d.training()
else:d.inference_case(args.case)
