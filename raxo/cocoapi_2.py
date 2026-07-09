from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
from metrics import compute_recall
# from tidecv import TIDE, datasets

# Setup command line argument parsing
parser = argparse.ArgumentParser(description="Evaluate COCO detections.")
parser.add_argument('--cocoGt', type=str, required=True, help="Path to the ground truth COCO annotations (coco_annotations.json).")
parser.add_argument('--cocoDt', type=str, required=True, help="Path to the detection result file (e.g., detections.json).")
args = parser.parse_args()

# Load the ground truth and detection results
cocoGt = COCO(args.cocoGt)
cocoDt = cocoGt.loadRes(args.cocoDt)

# Perform COCO evaluation
cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()



# Compute recall
compute_recall(cocoDt, cocoGt)


# # tide
# gt = datasets.COCO(args.cocoGt)
# results=datasets.COCOResult(args.cocoDt)
# tide = TIDE()
# tide.evaluate_range(gt, results, mode=TIDE.BOX) # Use TIDE.MASK for masks
# tide.summarize()  # Summarize the results as tables in the console