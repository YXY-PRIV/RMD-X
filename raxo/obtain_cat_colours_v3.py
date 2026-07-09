"""
Final version of the method to build the colour databse: 
1) given all the categories names, extract the materials (super-categories)
2) take the colour of each supercategory removing 
3) save the colour per_cat
"""

import json
import openai
from tqdm import tqdm
import argparse
import re  # For extracting the RGB values
import os
from sklearn.cluster import KMeans
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import pycocotools.mask as mask_util


openai.api_key = ""
MODEL = "gpt-4o"
PROMPT="""
You are a computer expert specializing in material classification. Your task is to analyze a given list of objects, determine their primary material composition, and group them accordingly.

Instructions:
Identify the main materials present among the objects (e.g., metal, organic, inorganic, plastic, ceramic, etc.).
Assign each object to the most appropriate material category. Each object should belong to only one category based on its primary composition.
Return the results in JSON format, where the keys are material categories, and the values are lists of objects belonging to those categories.
Example:
Input:
Objects: gun, bat, pressure_vessel, beer_glass, fur_coat, lemon

Expected Output (JSON):

  "metal": ['gun', 'bat'],
  "inorganic": ['pressure_vessel', 'beer_glass'],
  "organic": ['fur_coat', 'lemon']

Now, classify the following list of objects: {}. Return only the json format
"""
PATH = "/datasets/xray-datasets/"
IMAGES_PATH = "/datasets/xray-datasets/support_full/images"
dataset_names = ("PIXray", "pidray", "CLCXray", "HiXray", "DvXray", "xray_fsod", "eds", "COMPASS-XP")



parser = argparse.ArgumentParser(description='Build prototypes from GT in coco format')
parser.add_argument('--cats_gt', required=True, help='')
parser.add_argument('--out', required=True, help='')


def prepare_chatgpt_message(main_prompt):
    messages = [{"role": "user", "content": main_prompt}]
    return messages


def call_chatgpt_for_rgb(cats):
    """Call the OpenAI API and extract the RGB values"""
    custom_prompt = PROMPT.format(cats)
    total_prompt = prepare_chatgpt_message(custom_prompt)

    # Sending the request to OpenAI's GPT model
    response = openai.chat.completions.create(
        model=MODEL,
        messages=total_prompt,
        n=1,
        stop=None,
        temperature=0.3
    )
    
    # Parse the response text to extract RGB values
    generated_text = response.choices[0].message.content.strip()
    cleaned_string = generated_text.replace("```json", "").replace("```","").replace("\n", "").replace(" ", "").strip()    
    parsed_json = json.loads(cleaned_string)
    print(parsed_json)
    print(parsed_json.keys())
            
    return parsed_json


def clustering(d_name, support, material):
    # Here we need to iterate over the category names in material. For each cat
    # Build a mapper category_id -> category_name

    # Loop through each category in the COCO annotations
    mean_colors = []
    for category in support.loadCats(support.getCatIds()):
        if category['name'] not in material[1]:
            continue
        category_id = category['id']
        category_name = category['name']
        
        img_ids = support.getImgIds(catIds=[category_id])
        img_info_list = support.loadImgs(img_ids)
        # Here we have to remove imgs which have d_name in the name:
        imgs_filtered = [img for img in img_info_list if d_name not in img['file_name']]


        # Process each selected image
        for img_info in imgs_filtered:
            #img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(IMAGES_PATH, img_info['file_name'])

            # Load the image using PIL
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)

            # Get the annotations (masks) for the selected image
            ann_ids = support.getAnnIds(imgIds=[img_info['id']], catIds=[category_id])
            anns = support.loadAnns(ann_ids)

            for ann in anns:
                # Get the segmentation mask of the object
                mask = mask_util.decode(ann['mask'])

                # Skip if no valid mask (empty object)
                if np.sum(mask) == 0:
                    continue

                # Crop the object from the image using the mask
                cropped_object = image_np * np.expand_dims(mask, axis=-1)

                # Calculate the mean color for the cropped object
                mean_color = np.mean(cropped_object[mask > 0], axis=0)
                mean_colors.append(mean_color)
        
    # If we collected any colors, calculate the average mean color ALL categories -> material
    if mean_colors:
        avg_mean_color = np.mean(mean_colors, axis=0)
    
    return avg_mean_color.astype(int)
                
                
                

def main(args):
    with open(args.cats_gt, 'r') as f:
        file = json.load(f)
    
    if not os.path.exists("/models/ICCV25_experimentation/colour_knowledge_material_clustering.json"):
        # 1) Obtain the material-based supercategories
        cats = [x['name'] for x in file['categories']]
        materials = call_chatgpt_for_rgb(cats)
        
        # Save the results to the output file
        with open("/models/ICCV25_experimentation/colour_knowledge_material_clustering.json", 'w') as out_file:
            json.dump(materials, out_file, indent=4)
        print(f"Results saved to /models/ICCV25_experimentation/colour_knowledge_material_clustering.json")
    else:
        materials = json.load(open("/models/ICCV25_experimentation/colour_knowledge_material_clustering.json"))
    
    
    # 2) Obtain the colours from the material excluding the prohibited dataset. 1 each time
    for d_name in tqdm(dataset_names):
        # Open file:
        support = COCO("/datasets/xray-datasets/support_full/full_30_support_set_with_masks.json")
        
        # Colour of each material:
        mat_colour = {}
        color_dict_per_cat = {}
        for mat in materials.items():
            # Get colours 
            color_mat = clustering(d_name, support, mat)
            color_mat = color_mat.tolist()
            mat_colour[mat[0]] = color_mat

            # Finallly save the colour per cat
            for cat in mat[1]:
                rgb_values = color_mat
                color_dict_per_cat[cat] = rgb_values
                
        # Save results. Should be call n times, where n is the number of datasets
        save_name = os.path.join(args.out, f"{d_name}_colour.pt")
        with open(save_name, 'w') as out_file:
            json.dump(mat_colour, out_file, indent=4)
        print(f"Results saved to {save_name}")
        
        save_name = os.path.join(args.out, f"{d_name}_colour_per_cat.pt")
        with open(save_name, 'w') as out_file:
            json.dump(color_dict_per_cat, out_file, indent=4)
        print(f"Results saved to {save_name}")
        
        
       

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
