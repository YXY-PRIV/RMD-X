"""
Final AP method experimentation protocol. Unify database/web branchs. Split the categories of the dataset in base
and novel. The base AP is the database branch, the novel AP de web branch. Repeat x times to better results and report mean
Repeat it for different percentajes: 100/0; 80/20; 50/50; 20/80; 0/100
"""

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
from metrics import compute_recall
from tidecv import TIDE, datasets
import random
import numpy as np

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Evaluate COCO detections.")
parser.add_argument('--cocoGt', type=str, required=True, help="Path to the ground truth COCO annotations (coco_annotations.json).")
parser.add_argument('--cocoDt_database', type=str, required=True, help="Path to the detection result file of database-branch.")
parser.add_argument('--cocoDt_web', type=str, required=True, help="Path to the detection result file of web branch.")
parser.add_argument('--n', type=int, default=3, help="Number of repetitions.")


def format_results(data):
    for sublist in data:
        print(" ".join(map(str, sublist)))
    
def random_split_categories(categories, percentages):
    """
    Splits the categories randomly into base and novel sets based on the provided percentages.

    Args:
        categories (list): List of category dictionaries from COCO annotations.
        percentages (tuple): A tuple of two values (base_percentage, novel_percentage).

    Returns:
        tuple: (base_categories, novel_categories)
    """
    random.shuffle(categories)
    total_cats = len(categories)
    base_count = int(total_cats * percentages[0])

    base_cats = categories[:base_count]
    novel_cats = categories[base_count:]
    all_cats_ids = base_cats + novel_cats
    
    return base_cats, novel_cats, all_cats_ids

def compute_AP(cocoGt, cocoDt, cat_ids, eval_type="bbox"):
    """
    Compute the Average Precision (AP) for the given category IDs.

    Args:
        cocoGt (COCO): COCO ground truth object.
        cocoDt (COCO): COCO detections object.
        cat_ids (list): List of category IDs to evaluate.
        eval_type (str): Type of evaluation (e.g., "bbox").

    Returns:
        list: AP values for each category.
    """
    ap = {}
    ap50 = {}
    ap75 = {}
    for cat_id in cat_ids:
        cocoEval = COCOeval(cocoGt, cocoDt, eval_type)
        cocoEval.params.catIds = [cat_id]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        ap[cat_id] = float(cocoEval.stats[0]) * 100  # AP@[IoU=0.50:0.95]
        ap50[cat_id] = float(cocoEval.stats[1]) * 100
        ap75[cat_id] = float(cocoEval.stats[2]) * 100
    
    return ap, ap50, ap75


def main(args):

    # Load the ground truth and detection results
    cocoGt = COCO(args.cocoGt)
    cocoDt_database = cocoGt.loadRes(args.cocoDt_database)
    cocoDt_web = cocoGt.loadRes(args.cocoDt_web)
    # Define the percentage splits
    splits = [(1.0, 0.0), (0.8, 0.2), (0.5, 0.5), (0.2, 0.8), (0.0, 1.0)]
    
    # Obtain categories, and categorie to id:
    cats = cocoGt.dataset['categories']
    mapper = {x['id']:x['name'] for x in cats}
    #print(mapper) # debug!!
    cat_ids = list(mapper.keys())

    # Compute the AP per category for each branch:
    ap_database, ap50_database, ap75_database = compute_AP(cocoGt, cocoDt_database, cat_ids, eval_type="bbox")
    #print(ap_database.values()) # debug
    ap_web, ap50_web, ap75_web = compute_AP(cocoGt, cocoDt_web, cat_ids, eval_type="bbox")
    #print(ap_web.values()) # debug

    
    # All metrics computed just combine them properly.
    results_per_split = []
    for split in splits:
        print(f"Evaluating split: Base={split[0] * 100}%, Novel={split[1] * 100}%")
        
        repetition_results = {}
        for i in range(int(args.n)):
            #print(f"Repetition {i + 1}/{args.n}")
            base_cats_ids, novel_cats_ids, all_cats_ids = random_split_categories(cat_ids, split)
            #print(f"Base cats: {base_cats_ids}") # debug
            #print(f"Novel cats: {novel_cats_ids}") # debug
            #print(f"All cats: {all_cats_ids}") # debug

            # Base APs:
            AP_b = [ap_database[x] for x in base_cats_ids] 
            AP50_b = [ap50_database[x] for x in base_cats_ids] 
            AP75_b = [ap75_database[x] for x in base_cats_ids] 
            # Novel APs:
            AP_n = [ap_web[x] for x in novel_cats_ids] 
            AP50_n = [ap50_web[x] for x in novel_cats_ids] 
            AP75_n = [ap75_web[x] for x in novel_cats_ids] 
            # Overall AP:
            AP = AP_b + AP_n
            AP50 = AP50_b + AP50_n
            AP75 = AP75_b + AP75_n
            
            
            # here computation
            AP_b = sum(AP_b) / len(AP_b) if len(AP_b)>0 else 0
            AP50_b = sum(AP50_b) / len(AP50_b) if len(AP50_b)>0 else 0
            AP75_b = sum(AP75_b) / len(AP75_b) if len(AP75_b)>0 else 0
            
            AP_n = sum(AP_n) / len(AP_n) if len(AP_n)>0 else 0
            AP50_n = sum(AP50_n) / len(AP50_n) if len(AP50_n)>0 else 0
            AP75_n = sum(AP75_n) / len(AP75_n) if len(AP75_n)>0 else 0
            
            AP = sum(AP) / len(AP) if len(AP)>0 else 0
            AP50 = sum(AP50) / len(AP50) if len(AP50)>0 else 0
            AP75 = sum(AP75) / len(AP75) if len(AP75)>0 else 0
            
            aux = [AP, AP50, AP75, AP_b, AP50_b, AP75_b, AP_n, AP50_n, AP75_n]
            #print(aux) # debug
            repetition_results[i] = aux
            
        # Aggegrate repetition results
        data = np.array(list(repetition_results.values()))
        # Compute the mean for each column
        column_means = np.mean(data, axis=0)
        #print(f"Final results : {column_means}")
        results_per_split.append(column_means.tolist())   
        
    print("Final results all splits:")
    format_results(results_per_split)
   


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

