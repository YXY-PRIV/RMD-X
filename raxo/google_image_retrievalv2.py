"""
Thanks to: https://github.com/cvlab-columbia/DoubleRight/blob/master/google_image_search_url.py and mmdetection inference script
"""

##############################################################
import argparse
import os
import json
from google_images_search import GoogleImagesSearch
##############################################################
from PIL import Image

from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
import json
import os

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
##############################################################


API_KEY = ''
CX = ''
INIT_IMAGES_DOWNLOAD_DIR = 'imgs/'
INIT_IMAGES_JSON = 'annotations.json'
FINAL_IMAGES_ = 'imgs_final/'
promt = "A photo of a "
##############################################################


##############################################################
"""
{'inputs': '/datasets/xray-datasets/pidray/full_test/xray_easy00026.png', 'out_dir': './borrar/', 'texts': 'Baton .', 'pred_score_thr': 0.3, 'batch_size': 1, 'show': False, 'no_save_vis': False, 'no_save_pred': False, 'print_result': False, 'custom_entities': False, 'tokens_positive': None}

{'model': 'projects/GroundingDINO_modify/configs/grounding_dino_swin-b_finetune_16xb2_1x_pidray.py', 
'weights': '/models/groundingDINO/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth', 'device': 'cuda:0', 'palette': 'none'}
"""

DEVICE='cuda:0'
SCORE_THS = 0.5
BATCH_SIZE = 1
CHUNKED_SIZE = -1
LIMIT = 60  # Limit of tries per categories. Number of images that will be searched from internet

call_args = {
    'inputs': None, 
    'out_dir': None,
    'texts': None,
    'pred_score_thr': SCORE_THS,
    'batch_size': BATCH_SIZE,
    'show': False,
    'no_save_vis': False,
    'no_save_pred': True,
    'print_result': False, 
    'custom_entities': False,
    'tokens_positive': None
}




parser = argparse.ArgumentParser(description='Web image search of given categories')
parser.add_argument('--cats', required=True, help='Path the .json file where categories are stores')
parser.add_argument('--n', required=True, help='Number of examples images per class')
parser.add_argument('--model_config', required=True, help='Path to the .py with the configuration info')
parser.add_argument('--model_weights', required=True, help='Checkpoint file')
parser.add_argument('--out', required=True, help='Path to the directory where results will be saved')
parser.add_argument('--cats_from_gt', required=True, help='Path to the directory where GT test info is located. COCO format')


def build_inferencer(args):
    init_args = {'model':args.model_config, 'weights':args.model_weights, 'device':DEVICE, 'palette':'none'}
    inferencer = DetInferencer(**init_args)
    inferencer.model.test_cfg.chunked_size = CHUNKED_SIZE

    "NOTE: call arguments must be writted"
    return inferencer


# This main keep the top1 detection for retrieved image, if its confidence is bigger than the threshold
def main2(args):
    # 1) Build model inferencer
    inferencer = build_inferencer(args)
    os.makedirs(args.out, exist_ok=True)
    
    # 2) Obtain categories for wich images from internet will be retrieved
    # Support the LLM augmentation, and the classic behaviour
    if os.path.exists(args.cats):
        data_cats = json.load(open(args.cats))
        if 'category_list_real' in data_cats.keys():
            real_cats = data_cats['category_list_real']
            find_cats = data_cats['category_list_find']
            mapper = {find:real for find,real in zip(find_cats, real_cats)}
            print(mapper)
        else:
            real_cats = data_cats['categories']
            find_cats = data_cats['categories']
            mapper = {find:real for find,real in zip(find_cats, real_cats)}
            print(mapper)
    else:
        from pycocotools.coco import COCO
        coco = COCO(args.cats_from_gt)
        # Get category IDs and names
        categories = coco.loadCats(coco.getCatIds())
        real_cats = [cat["name"] for cat in categories]
        find_cats = [cat["name"] for cat in categories]
        mapper = {find:real for find,real in zip(find_cats, real_cats)}
        print(mapper)


    
    final_json = {"images":[]}
    try:
        img_id = 0
        for category in find_cats:
            if category!="open-end wrench":
                continue
            # 3) For each category obtain the web images
            query = promt + f'{category}'
            print('qi', query, API_KEY)
            _search_params = {
                    'q': query,
                    'num': LIMIT,
                    'searchType': 'image',
                    'dateRestrict': 'y7',  # Results from the last 6 years
                    'safe': '',
                    'fileType': 'jpg|png',
                    'imgType': 'photo',
                    'imgSize': '',
                    'imgDominantColor': '',
                    'rights': '',
                    'imgColorType': "",
                    'lr': 'lang_en',
                }

            # Init Google Image Search every iteration to avoid repetition error # trying
            gis = GoogleImagesSearch(API_KEY, CX)
            try :
                gis.search(search_params=_search_params)
            except Exception as Error:
                print ('error during query:', Error)
                print('Query error with :', query)

            # 4) While the number of good images in the category is not args.n (or still images)
            n_images_cat_i = 0
            for image in gis.results():
                try:
                    if n_images_cat_i >= int(args.n):
                        break

                    path = os.path.join(args.out, INIT_IMAGES_DOWNLOAD_DIR)
                    image.download(path)
                    
                    original_name = os.path.basename(image.path)
                    _, file_extension = os.path.splitext(original_name)

                    new_img_name = f"cat_{mapper[category]}_{img_id}{file_extension}"
                    img_id += 1
                    new_image_path = image.path.replace(original_name, new_img_name)
                    os.rename(image.path, new_image_path)


                    # Load image and run OVOD
                    call_args['inputs'] = new_image_path
                    call_args['texts'] = f"{category} ."
                    call_args['out_dir'] = os.path.join(args.out, FINAL_IMAGES_)
                    #print(call_args)  # debug

                    try:
                        res = inferencer(**call_args)
                    except Exception:
                        continue
                    # Get predictions with score greater than defined threhold
                    bbox_final = res['predictions'][0]["bboxes"][0]
                    score = res['predictions'][0]["scores"][0]
                    if score > SCORE_THS:
                        # Keep it
                        try:
                            img_aux = Image.open(new_image_path)
                        except Exception:
                            continue
                        
                        # Modify this to save bbox instead of crop image!!!!
                        # Append annotations
                        final_json['images'].append({"image_name": new_img_name, "url":image.url, "category": category, "super_category": mapper[category], "bbox_xyxy":bbox_final})
                        n_images_cat_i += 1
                except:
                    continue


            print(f"Category {category} of super_category {mapper[category]} ends with {n_images_cat_i} objects !!")   

        # Save final json
        with open(os.path.join(args.out, INIT_IMAGES_JSON), 'w') as f:
            json.dump(final_json, f)
    
    except:
        with open(os.path.join(args.out, INIT_IMAGES_JSON), 'w') as f:
            json.dump(final_json, f)



# def main(args):
#     """
#     TO keep all the detections with confidence bigger than 0.5
#     """
#     # 1) Build model inferencer
#     inferencer = build_inferencer(args)
#     os.makedirs(args.out, exist_ok=True)
    
#     # 2) Obtain categories for wich images from internet will be retrieved
#     data_cats = json.load(open(args.cats))
#     real_cats = data_cats['category_list_real']
#     find_cats = data_cats['category_list_find']
#     mapper = {find:real for find,real in zip(find_cats, real_cats)}
    
    
#     final_json = {"images":[]}
#     for category in find_cats:
#         # 3) For each category obtain the web images
#         query = promt + f'{category}'
#         print('qi', query, API_KEY)
#         _search_params = {
#                 'q': query,
#                 'num': LIMIT,
#                 'searchType': 'image',
#                 'dateRestrict': 'y6',  # Results from the last 6 years
#                 'safe': '',
#                 'fileType': 'jpg|png',
#                 'imgType': 'photo',
#                 'imgSize': '',
#                 'imgDominantColor': '',
#                 'rights': '',
#                 'imgColorType': "",
#                 'lr': 'lang_en',
#             }

#         # Init Google Image Search every iteration to avoid repetition error # trying
#         gis = GoogleImagesSearch(API_KEY, CX)
#         try :
#             gis.search(search_params=_search_params)
#         except Exception as Error:
#             print ('error during query:', Error)
#             print('Query error with :', query)

#         # 4) While the number of good images in the category is not args.n (or still images)
#         n_images_cat_i = 0
#         for image in gis.results():
#             path = os.path.join(args.out, INIT_IMAGES_DOWNLOAD_DIR)
#             image.download(path)

#             # Load image and run OVOD
#             call_args['inputs'] = image.path
#             call_args['texts'] = f"{category} ."
#             call_args['out_dir'] = os.path.join(args.out, FINAL_IMAGES_)
#             #print(call_args)  # debug

#             try:
#                 res = inferencer(**call_args)
#             except Exception:
#                 continue
#             # Get predictions with score greater than defined threhold
#             bboxes = [bbox for bbox, score in zip(res['predictions'][0]["bboxes"], res['predictions'][0]["scores"]) if score > SCORE_THS]

#             # Keep the bboxes 
#             for idx, bbox_final in enumerate(bboxes):
#                 # If we already have N objets for the class -> break
#                 if n_images_cat_i >= int(args.n):
#                     break 
#                 try:
#                     img_aux = Image.open(image.path)
#                 except Exception:
#                     continue
#                 img_aux = img_aux.crop(bbox_final)
#                 name = os.path.join(os.path.join(args.out, FINAL_IMAGES_),os.path.basename(image.path))
#                 _, extension = os.path.splitext(name)
#                 name = name.replace(extension, f"_{idx}.png")
#                 img_aux.save(name)
#                 # Append annotations
#                 final_json['images'].append({"image_name": os.path.basename(name), "url":image.url, "category": mapper[category]})
#                 n_images_cat_i += 1
            
#             # If we already get the n images break
#             if n_images_cat_i >= int(args.n):
#                 break

#         print(f"Category {mapper[category]} ends with {n_images_cat_i} objects !!")   

#     # Save final json
#     with open(os.path.join(args.out, INIT_IMAGES_JSON), 'w') as f:
#         json.dump(final_json, f)



if __name__ == "__main__":
    args = parser.parse_args()
    #main(args)
    main2(args)
