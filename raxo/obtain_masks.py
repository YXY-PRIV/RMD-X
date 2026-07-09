"""
This file extract the mask for the prototype-images. This is:
1) The annotated support set
2) The web-retrieved images
"""

import os
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import nms
import argparse
import copy
import json

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import cv2
import supervision as sv
from supervision.draw.color import ColorPalette
import pycocotools.mask as mask_util


CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

DINO_DIMENSION=768
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "This script requires a GPU to run."


parser = argparse.ArgumentParser(description='Obtain masks with SAM2')
parser.add_argument('--gt', required=True, help='Path to the ground truth file (in COCO format for database_branch)')
parser.add_argument('--image_path', required=True, help='Path to the directory containing the images referenced in gt')
parser.add_argument('--branch', required=False, choices=['known', 'unknown'], default="known", help='Branch of the method')


def save_coco(coco_dt, filtradas_anns, save_name):
    save_dict = {}
    for k in coco_dt.dataset.keys():
        if k != 'annotations':
            save_dict[k] = coco_dt.dataset[k]
    
    # Add anotations
    save_dict['annotations'] = filtradas_anns
    
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(save_dict, indent=4, sort_keys=True)
        fp.write(s)
        
        
def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def plot(img_path, boxes_xyxy, masks, imag_id_name):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3d = cv2.merge([img_gray, img_gray, img_gray])
    detections = sv.Detections(
        xyxy=boxes_xyxy,  # (n, 4)
        mask=masks.astype(bool),  # (n, h, w)
    )
    
    box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = box_annotator.annotate(scene=gray_3d.copy(), detections=detections)

    mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP), color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    cv2.imwrite(os.path.join("/models/borrar_sam/", f"sam2_{imag_id_name}.jpg"), annotated_frame)


def build_sam_model():
    sam2_checkpoint = "/models/sam/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    return sam2_predictor




def main_known(args, plot_result=False):
    # 1) Create SAM 
    sam2_predictor = build_sam_model()
    
    # 2) OPEN DATASET 
    dataset = COCO(args.gt)
    imgs = dataset.loadImgs(dataset.getImgIds())
    
    # Take into accoutn each image can have more than 1 annotation:
    annotations_with_masks = []
    imag_id_name = 0
    for img in tqdm(imgs):
        # Open image
        img_path = os.path.join(args.image_path, img['file_name'])
        image = Image.open(img_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))
        
        # Load annotations of this image:
        annotations = dataset.loadAnns(dataset.getAnnIds(imgIds=img['id']))
        
        # For each annotation:
        for anot in annotations:
            # box (np.ndarray or None): A length 4 array given a box prompt to the model, in XYXY format.
            boxes_xywh = np.array([anot['bbox']]) 
            boxes_xyxy = np.column_stack([boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 0] + boxes_xywh[:, 2], boxes_xywh[:, 1] + boxes_xywh[:, 3]])
                    
            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,
                multimask_output=False,
            )
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            if plot_result:
                plot(img_path, boxes_xyxy, masks, imag_id_name)
                imag_id_name += 1
                
            # Now save results in COCO format. We need to keep the same anot_id than the box to be
            # able to load the mask in the next step.
            new_anot = copy.deepcopy(anot)
            new_anot['mask'] = single_mask_to_rle(masks[0])
            annotations_with_masks.append(new_anot)
            
    # Save json file
    save_coco(dataset, annotations_with_masks, args.gt.replace(".json", "_with_masks.json"))
        
        
def main_unknown(args, plot_result=True):
    # 1) Create SAM 
    sam2_predictor = build_sam_model()
    
    # 2) OPEN DATASET 
    dataset = json.load(open(args.gt))
    images_anots = dataset['images']
    
    # 3) Obtain masks of each image. 
    # Take into accoutn each image IS AN OBJECT CROP
    imag_id_name = 0
    images_anots_with_masks = []
    for image_anot in tqdm(images_anots):
        img_path = os.path.join(args.image_path, image_anot['image_name'])
        image = Image.open(img_path)
        sam2_predictor.set_image(np.array(image.convert("RGB")))

        boxes_xyxy = np.array([image_anot["bbox_xyxy"]])

        masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes_xyxy,   # this is the full image
                multimask_output=False,
            )
        
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        
        if plot_result:
            plot(img_path, boxes_xyxy, masks, imag_id_name)
            imag_id_name += 1

        # Here we have to save the result
        new_image_anot = copy.deepcopy(image_anot)
        new_image_anot['mask'] = single_mask_to_rle(masks[0])
        images_anots_with_masks.append(new_image_anot)
        

    # Save new annotations with masks
    save_name = args.gt.replace(".json", "_with_masks.json")
    print(save_name)
    with open(save_name, 'w') as fp:
        s = json.dumps(images_anots_with_masks, indent=4, sort_keys=True)
        fp.write(s)

    
    
if __name__ == "__main__":
    args = parser.parse_args()
    if args.branch == "known":
        main_known(args, False)
    elif args.branch == "unknown":
       main_unknown(args, False)
    else:
        raise Exception("Method not implemented")