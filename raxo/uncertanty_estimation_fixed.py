from tqdm.auto import tqdm
import argparse
import json
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


parser = argparse.ArgumentParser(description='Analyze pseudo-labels')
parser.add_argument('--dets', required=True, help='path to the detection results in coco res format')
parser.add_argument('--ths', default=0.15, help='ths')

####################################################################################
####################################################################################

def main(args):
    coco_dt = json.load(open(args.dets))
    
    final_dets = [det for det in coco_dt if det['dist_cosine_inter']>float(args.ths)]
    # Save file
    save_name = args.dets.replace(".json", "_uncertanty.json")
    with open(save_name, 'w') as f:
        json.dump(final_dets, f)
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)