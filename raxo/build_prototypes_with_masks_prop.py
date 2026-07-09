"""
This version v1: 
Passing crop image and crop mask. Patchifing the mask (0.3/0.7), and keeping 2 negative prototypes: 
pixeles outside the mask, and full image.
"""

from pycocotools.coco import COCO
from tqdm import tqdm
import torch
import torchvision.transforms as T
import argparse

from dataset import CocoPrototypes_with_masks, CocoPrototypesFullImage
from torch.utils.data import DataLoader
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F

DINO_DIMENSION=768
DINO_PATH_SIZE=14
device = "cuda" if torch.cuda.is_available() else "cpu"
assert device == "cuda", "This script requires a GPU to run."


parser = argparse.ArgumentParser(description='Build prototypes from GT in coco format')
parser.add_argument('--gt', required=True, help='Path to the ground truth file in COCO format')
parser.add_argument('--image_path', required=True, help='Path to the directory containing the images referenced in gt')
parser.add_argument('--out', required=True, help='Path to the where positive and negative prototypes will be saved')


def build_model():
    transform = T.Compose([
        T.Resize((224,224) ,interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    dinov2_vitb14_reg = dinov2_vitb14_reg.cuda()
    
    return transform, dinov2_vitb14_reg


def compute_full_image_negative_prototype(model, dataloader):
    sum_tensor = torch.zeros(DINO_DIMENSION)
    n_elements = 0
    for batch in tqdm(dataloader):
        images = batch
        images = images.cuda()
        with torch.no_grad():
            outputs = model(images).cpu()
            sum_tensor += outputs.sum(dim=0)
            n_elements += outputs.shape[0]
    
    negative_prot_full_image = sum_tensor / n_elements
    return negative_prot_full_image


def compute_prototypes(model, dataloader, mapper_real_to_name, vocab):
    """
    This function computes the positive prototype per class and the full negative one
    """
    # Keep track dictionary per class
    pos_prototypes = {class_name: [] for class_name in vocab}
    neg_prototypes = {class_name: [] for class_name in vocab}
    
    for batch in tqdm(dataloader):
        images, annotations, masks = batch
        images = images.cuda()

        with torch.no_grad():
            outputs = model.get_intermediate_layers(images)
            outputs = outputs[0].cpu()
            """
            64: This is the batch size. 
            256: This is the number of tokens per image. 
            In a Vision Transformer (ViT), each image is divided into patches, and each patch is treated as a token. 
            For example: If the input image is of size 224x224 (as typically used in ViTs), And the patch 
            size is 14x14 (a common configuration for DINOv2-base), The number of patches (tokens) 
            per image = ( 224 / 14 ) x ( 224 / 14 ) = 16 x 16 = 256 
            768: This is the hidden dimension of the ViT. Each token is represented as a 768-dimensional embedding vector
            in the DINOv2-base model.
            """

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
        outs = outs.squeeze()
        
        # Append element to pos_prototypes
        for out, anot in zip(outs, annotations):            
            pos_prototypes[mapper_real_to_name[anot['category_id']]].append(out)
            
        
        # TO SAME BUT WITH INVERSE MASK
        negative_masks = 1 - masks  # Where the original mask is 1, it becomes 0, and where it is 0, it becomes 1
        masks_expanded = negative_masks.unsqueeze(-1)  # Shape: [n_batch, 256, 1]
        masked_T = outputs * masks_expanded  # Zero out unmasked values based on the inverse of the mask
        sum_masked = masked_T.sum(dim=1, keepdim=True)  # Sum over channels, shape [n_batch, 1, 768]
        count_masked = masks_expanded.sum(dim=1, keepdim=True)  # Count of selected elements per batch, shape [n_batch, 1, 1]
        count_masked = count_masked.clamp(min=1)  # Ensure no zero counts
        outs = sum_masked / count_masked  # Shape: [n_batch, 1, 768]
        outs = outs.squeeze()
        for out, anot in zip(outs, annotations):
            neg_prototypes[mapper_real_to_name[anot['category_id']]].append(out)
            
    
    final_prototypes = {}
    for class_name, prototypes in pos_prototypes.items():
        # Add individual prototypes with suffix
        for i, prototype in enumerate(prototypes, start=1):
            new_key = f"{class_name}_{i}"
            final_prototypes[new_key] = {
                "prototype": prototype,
                "supercategory": class_name
            }

        # Compute and append the mean prototype
        if prototypes:  # Ensure there are elements to compute the mean
            mean_prototype = torch.stack(prototypes).mean(dim=0)
            final_prototypes[class_name] = {
                "prototype": mean_prototype,
                "supercategory": class_name
            }
            
    for class_name, prototypes in neg_prototypes.items():
        # Add individual negative prototypes with suffix
        for i, prototype in enumerate(prototypes, start=1):
            new_key = f"{class_name}_neg_{i}"  # Use 'neg' as a suffix to distinguish from positive prototypes
            final_prototypes[new_key] = {
                "prototype": prototype,
                "supercategory": class_name
            }
    
    return final_prototypes
    
    




###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# MAINS!!!

def main(args):
    """
    Computes the positive C prototypes (c=card(classes)); and the background full image one
    """
    
    def custom_collate_fn_CocoPrototypes(batch):
        # Separate images and dictionaries
        images = [item[0] for item in batch]
        dicts = [item[1] for item in batch]
        masks = [item[2] for item in batch]


        # Stack images into a single tensor along a new batch dimension
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)

        # Return images tensor and list of dictionaries
        return images, dicts, masks


    gt = COCO(args.gt)
    
    # Obtain vocab from categories
    cats = gt.loadCats(gt.getCatIds())
    mapper_real_to_name = {x['id']:x['name'] for x in cats}
    vocab = [x['name'] for x in cats] 
    
    # 1) Build dino model
    transforms, model = build_model()
    
    # 2) Class-specific positive proposals and hard-negative prototype
    data = CocoPrototypes_with_masks(gt, args.image_path, transforms)
    dataloader = DataLoader(data, batch_size=64, shuffle=False, num_workers=8, collate_fn=custom_collate_fn_CocoPrototypes)
    prototypes = compute_prototypes(model, dataloader, mapper_real_to_name, vocab)
    
    ## 3) Full image negative prototype
    data_full_image = CocoPrototypesFullImage(gt, args.image_path, transforms)
    dataloader_full_image = DataLoader(data_full_image, batch_size=64, shuffle=False, num_workers=8)
    negative_prot_full_image = compute_full_image_negative_prototype(model, dataloader_full_image)
    prototypes['negative'] = {"prototype": negative_prot_full_image, "supercategory":"negative"}

    
    # Save prototypes in file
    torch.save(prototypes, args.out)
    
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
