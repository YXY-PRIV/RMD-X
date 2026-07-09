import argparse
import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import torch
import torchvision.transforms as T
from torchvision.ops import nms
from dataset import CocoPrototypes_with_masks
from torch.utils.data import DataLoader
DINO_PATH_SIZE=14
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "This script requires a GPU to run."


parser = argparse.ArgumentParser(description='Proces the output of a OVOD detector to improve results')
parser.add_argument('--json_res', required=True, help='Path to the detections of OVOD detector in COCOC_RES format')
parser.add_argument('--image_path', required=True, help='Path to the directory containing the images referenced in gt')
parser.add_argument('--prototypes', required=True, help='Path to the prototypes .pt file')
parser.add_argument('--nms', required=True, help='NMS threshold')
parser.add_argument('--name', required=False, default="", help='Name to append to the final name')
parser.add_argument('--branch', required=True, choices=['known', 'unknown'], default="known", help='Branch of the method')


def custom_collate_fn(batch):
    # Separate images and dictionaries
    images = [item[0] for item in batch]
    dicts = [item[1] for item in batch]
    masks = [item[2] for item in batch]


    # Stack images into a single tensor along a new batch dimension
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    # Return images tensor and list of dictionaries
    return images, dicts, masks


def apply_nms(coco, iou_threshold=0.5):
    
    new_annotations_nms = []
    image_ids = coco.getImgIds()
    for image_id in tqdm(image_ids):
        annotation_ids = coco.getAnnIds(imgIds=image_id)
        annotations = coco.loadAnns(annotation_ids)

        boxes = []
        scores = []
        for ann in annotations:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            scores.append(ann["score"])
        
        if len(boxes)>0:    
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores)
            keep_indices = nms(boxes, scores, iou_threshold)
            filtered_annotations = [annotations[i] for i in keep_indices]
            new_annotations_nms += filtered_annotations
        
    return new_annotations_nms
    

def build_model():
    transform = T.Compose([
        T.Resize((224,224) ,interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    dinov2_vitb14_reg = dinov2_vitb14_reg.cuda()
    
    return transform, dinov2_vitb14_reg


def assemble_prototypes(prototypes_path, vocab):
    
    if not os.path.exists(prototypes_path):
        raise Exception(f"{prototypes_path} not exists")
    
    # Load the support set
    support_set = torch.load(prototypes_path)
    prototypes = []
    mapper = {}
    cat_in_each_index = []
    cat_in_each_index_original = []
    
    for cat, values in support_set.items():
        if values['supercategory'] in vocab or cat=="negative":
            if "neg_" in cat or cat=="negative":
                prototypes.append(values['prototype'])
                cat_in_each_index.append('negative')
                cat_in_each_index_original.append(cat)
            else:
                prototypes.append(values['prototype'])
                mapper[cat] = values['supercategory']
                cat_in_each_index.append(values['supercategory'])
                cat_in_each_index_original.append(cat)
    
    
    support_set_tensor = torch.stack(prototypes)
    # Sanity check
    # print(support_set_tensor.shape[0])
    # expected_number = (len(vocab) * 31) + len(vocab)*30 + 1
    # print(expected_number)
    # assert support_set_tensor.shape[0] == expected_number
    #import pdb; pdb.set_trace()
    
    return support_set_tensor, cat_in_each_index, cat_in_each_index_original
 


def main_known(args):
    
    coco = COCO(args.json_res)
    # Obtain vocab from categories
    mapper_name_to_real = {c['name']:c['id'] for c in coco.dataset['categories']}
    vocab = list(mapper_name_to_real.keys())
    
    # 1) Load prototypes:
    print("Assambling prototypes...")
    prototypes, cat_name_in_each_index, cat_in_each_index_original = assemble_prototypes(args.prototypes, vocab)
    cat_in_each_index_original_mapper = {x: i for i, x in enumerate(cat_in_each_index_original)}
    prototypes = prototypes.cuda()
    
    # 2) Apply nms
    print("Applying NMS...")
    annotations_nms = apply_nms(coco, float(args.nms))
    coco.dataset['annotations'] = annotations_nms
    
    # 3) Build dino model
    transforms, model = build_model()
    
    # 3) Build dataloader
    data = CocoPrototypes_with_masks(coco, args.image_path, transforms)
    dataloader = DataLoader(data, batch_size=256, shuffle=False, num_workers=8, collate_fn=custom_collate_fn)
    
    # 4) Main re-clasification loop
    print("Reclasification...")
    new_anots = []
    for batch in tqdm(dataloader):
        crops, annotations, masks = batch
        crops = crops.cuda()
        masks = masks.cuda()
        
        with torch.no_grad():
            outputs = model.get_intermediate_layers(crops)
            outputs = outputs[0]
        
        # Expand the mask to match the dimensions of T
        masks_expanded = masks.unsqueeze(-1)  # Shape: [n_batch, 256, 1]
        # Apply the mask
        masked_T = outputs * masks_expanded  # Zero out unmasked values
        # Compute the mean along the selected channels
        sum_masked = masked_T.sum(dim=1, keepdim=True)  # Sum over channels, shape [n_batch, 1, 768]
        count_masked = masks_expanded.sum(dim=1, keepdim=True)  # Count of selected elements per batch, shape [n_batch, 1, 1]
        # Avoid division by zero
        count_masked = count_masked.clamp(min=1) # Should not happen never, as mask always exist
        # Compute the mean
        outs = sum_masked / count_masked  # Shape: [n_batch, 1, 768]
        
        
        similarity = torch.nn.functional.cosine_similarity(outs, prototypes.unsqueeze(0), dim=-1)
        similarity = similarity.cpu()
        
        # Save the new results  
        for s, anot in zip(similarity, annotations):
            
            _, max_indice = s.max(dim=0)
            if cat_name_in_each_index[max_indice] == "negative":
                continue
            predicted_class = mapper_name_to_real[cat_name_in_each_index[max_indice]]
            
            # Compute here the 2 uncertanti metrics:
            # 1) Distance from the predicted class mean - average other class mean
            s_predicted = s[cat_in_each_index_original_mapper[cat_name_in_each_index[max_indice]]]
            indices = [cat_in_each_index_original_mapper[c_name] for c_name in vocab if c_name != cat_name_in_each_index[max_indice]]
            s_mean = np.mean([s[i] for i in indices])
            dist_cosine_inter = s_predicted - s_mean
            # 2) Distance from the predicted class - average the predicted class
            dist_cosine_intra = s[max_indice] - s_predicted
            
            anot_new = {
                "category_id": predicted_class,
                "image_id": anot['image_id'],
                "bbox": anot['bbox'],
                "score": anot['score'],
                #"dist_cosine_1_rest": (s[max_indice] - s[torch.arange(len(s)) != max_indice][:-1].mean()).item()
                "dist_cosine_inter": dist_cosine_inter.item(),
                "dist_cosine_intra": dist_cosine_intra.item()
            }
            new_anots.append(anot_new)
    
    # Save final results
    with open(args.json_res.replace(".json", f"_nms_{args.nms}_our_method_{args.name}.json"), 'w') as f:
        json.dump(new_anots, f)
       


 
if __name__ == "__main__":
    args = parser.parse_args()
    if args.branch == "known":
        main_known(args)
    elif args.branch == "unknown":
        pass
    else:
        raise Exception("Method not implemented")

   
        
