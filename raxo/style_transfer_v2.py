"""
In this style transfer we do:
1) Take the mask of the detection
2) Retrieve the material-colour
3) Paint the object with the colour and save the result
"""

import argparse
import pycocotools.mask as mask_util
import json
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import random


parser = argparse.ArgumentParser(description='Style-transfer custom method')
parser.add_argument('--gt', required=True, help='Path to the ground truth file (in custom format)')
parser.add_argument('--image_path', required=True, help='Path to the directory containing the images referenced in gt')
parser.add_argument('--out', required=True, help='path to the folder where style-transfer images will be saved')
parser.add_argument('--colours', required=True, help='Path to the colour knowledge')



def main(args):
    
    os.makedirs(args.out, exist_ok=True)
    
    # 1) Load data with masks
    crops = json.load(open(args.gt))
    
    # 2) Load material-information.
    category_avg_colors = json.load(open(args.colours))
    
    # 3) Transform each of the images, and save the result
    for crop in tqdm(crops):
        img_path = os.path.join(args.image_path, crop['image_name'])
        image = Image.open(img_path).convert("RGB")  # Ensure it's RGB
        image_array = np.array(image)
        mask = mask_util.decode(crop['mask'])  # Decodifica la máscara a un np.array
        
        # Paint the part of the mask with blue color and the rest of the image with black
        output = np.zeros_like(image_array)
        if "super_category" in crop.keys():
            output[mask > 0] = category_avg_colors[crop['super_category']]
        else:
            output[mask > 0] = category_avg_colors[crop['category']]
        output_image = Image.fromarray(output)
        try:
            output_image.save(os.path.join(args.out, f"{crop['image_name']}"))
        except:
            print(crop['image_name'])
            continue
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


