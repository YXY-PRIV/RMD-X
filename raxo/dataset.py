from torch.utils.data import Dataset
import random
import os
from PIL import Image
import numpy as np
from tabulate import tabulate

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from metrics import compute_recall
from utils2 import *
import pycocotools.mask as mask_util
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F


def compute_dataset_statics(coco, gt):
    """
    Compute number of images, number of boxes per image, mean box size, recall, class-agnostic AP.
    """
    n_images = len(coco.dataset['images'])
    n_bboxes = len(coco.dataset['annotations'])
    density = n_bboxes/n_images
    area_mean = np.mean([x['bbox'] for x in coco.dataset['annotations']])
    
    print(f"Number of images: {n_images}")
    print(f"Number of bboxes: {n_bboxes}")
    print(f"n_boxes/n_images: {density}")
    print(f"Mean area: {area_mean}")
    
    # Recall
    compute_recall(coco, gt)
    
    # AP
    # TO DO    
    
def plot_mask(crop, mask_cropped_resized):
    # Convertir el recorte y la máscara a formatos compatibles
    crop_array = np.array(crop)  # Convertir el crop de PIL a un array numpy
    crop_array_masked = crop_array * np.stack([mask_cropped_resized] * 3, axis=-1)

    # Mostrar la imagen recortada y la máscara superpuesta
    plt.figure(figsize=(8, 8))
    plt.imshow(crop_array_masked)
    plt.axis('off')
    plt.savefig("./debug_mask.png")

class CocoCrops(Dataset):
    """
    This dataset class, takes the annotation result .json file in coco format, and 
    crop each of the objects. The GT is used to get the images path, because the .json 
    result file only includes the image id.


    Parameters
    ----------
    json_res : cocoapi
        Coco format .json results
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, json_res, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dets = json_res
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        
        #compute_dataset_statics(self.dets, gt)

    def __len__(self):
        # len() is the number of objects
        return len(self.dets.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.dets.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        bbox = anot['bbox']
        crop = image_original.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))            
        crop = self.transforms(crop)
        return (crop, anot)


class CocoPrototypes(Dataset):
    """
    This dataset class, takes the GT coco format of a support set and crops the objects.



    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        

    def __len__(self):
        # len() is the number of objects
        return len(self.gt.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.gt.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        bbox = anot['bbox']
        crop = image_original.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))            
        crop = self.transforms(crop)
        return (crop, anot)


class CocoPrototypesFullImage(Dataset):
    """
    This dataset class, takes the GT coco format of a support set return the full image.



    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        self.images_ids = self.gt.getImgIds()
        

    def __len__(self):
        # len() is the number of objects
        return len(self.images_ids)


    def __getitem__(self, idx):
        img_id = self.images_ids[idx]
        image_name = self.gt.loadImgs(self.gt.getImgIds(img_id))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        image_original = self.transforms(image_original)
        return image_original
    


class CocoPrototypesWithNegative(Dataset):
    """
    This dataset class, takes the GT coco format of a support set and crops the objects
    to get the positive prototypes. It also generates negative random boxes as "hard negative" 


    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, iou_min=0.1, iou_max=0.25, n_negative_boxes=10, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        self.iou_max = iou_max
        self.iou_min = iou_min
        self.n_negative_boxes = n_negative_boxes
        

    def __len__(self):
        # len() is the number of objects
        return len(self.gt.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.gt.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        image = image_original.copy()
        
        bbox = anot['bbox']  # This is xywh
        crop = image.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))            
        crop = self.transforms(crop)
        
        # Obtain the hard negatives:
        negative_boxes = generate_negative_boxes(bbox, self.n_negative_boxes, iou_min=self.iou_min, iou_max=self.iou_max) # Returned boxes in xyxy format
        negative_crops = []
        for box_negative in negative_boxes:
            image = image_original.copy()
            negative_crop =  image.crop(box_negative)
            negative_crop_tensor = self.transforms(negative_crop)
            negative_crops.append(negative_crop_tensor)

        
        return (crop, negative_crops, anot)
    
    
    
class CocoPrototypesUnknown(Dataset):
    """
    This dataset class, takes the GT unknown images where the full image represents the object.
    The custom format of the .json is:
    {
        "images": [
            {"image_name":--, "url":-----, "category"},
        ]
    }


    Parameters
    ----------
    gt : str
        Unknown images in custom format
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt['images']

        # To maintain interoperability with previous versions where there was no supercategory.
        if not "super_category" in self.gt[0]:
            self.cat_names = list(set([x['category'] for x in self.gt]))
        else:
            # Take into account sub and supercategories
            cat_names = []
            supercategory_names = []
            mapper = {}

            for x in self.gt:
                cat_names.append(x['category'])
                supercategory_names.append(x['super_category'])
                mapper[x['category']] = x['super_category']

            self.cat_names = list(set(cat_names))
            self.supercategory_names = list(set(supercategory_names))
            self.mapper = mapper
        
    def getCatNames(self):
        return self.cat_names
    

    def getMapper(self):
        return self.mapper
      

    def __len__(self):
        # len() is the number of objects
        return len(self.gt)


    def __getitem__(self, idx):
        image = self.gt[idx]

        image_name = image['image_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path).convert('RGB')
        image_original = self.transforms(image_original)
        cat = image['category']
        
        if not "super_category" in image.keys():  # Old version interoperability
            supercat = cat
        else:
            supercat = self.mapper[cat]

        return image_original, cat, supercat
    
    
class CocoPrototypesUnknown_negative(Dataset):
    """
    This dataset class, takes the GT unknown images where the full image represents the object.
    The custom format of the .json is:
    {
        "images": [
            {"image_name":--, "url":-----, "category"},
        ]
    }

    It generates a random crop for the image

    Parameters
    ----------
    gt : str
        Unknown images in custom format
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt['images']
        

    def __len__(self):
        # len() is the number of objects
        return len(self.gt)


    def __getitem__(self, idx):
        image = self.gt[idx]

        image_name = image['image_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path).convert('RGB')
        
        # Here perform the random crop
        width, height = image_original.size
        crop_width = max(1, width // 4)  # Asegura al menos 1 píxel de ancho
        crop_height = max(1, height // 4)  # Asegura al menos 1 píxel de alto
        
        x = random.randint(0, max(0, width - crop_width))
        y = random.randint(0, max(0, height - crop_height))
        crop = image_original.crop((x, y, x + crop_width, y + crop_height))

        crop = self.transforms(crop)
        
        return crop
    
    
    

class CocoPrototypes_with_masks_v1(Dataset):
    """
    This dataset class, takes the GT (with masks) coco format of a support set and return (per each annotation) 
    1 full image, and the mask. Does not perform crop, but mask is transformed according new image dimensions.



    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        
        # Extract the size from T.Resize
        for t in transforms.transforms:
            if isinstance(t, T.Resize):
                self.target_size = t.size  # This should be (224, 224) in default setting
                break
                

    def __len__(self):
        # len() is the number of objects
        return len(self.gt.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.gt.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        
        # Need to apply transforms to bbox and mask
        mask = mask_util.decode(anot['mask'])
        resized_mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)  # Shape: (224, 224)
        mask_tensor = torch.from_numpy(resized_mask) 

        image = self.transforms(image_original)
        return (image, anot, mask_tensor)
    
    
    
class CocoPrototypes_with_masks_v2(Dataset):
    """
    This dataset class, takes the GT (with masks) coco format of a support set and return (per each annotation):
    1) the crop image
    2) the crop mask
    
    Note: mask is transformed according new image dimensions.



    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        
        # Extract the size from T.Resize
        for t in transforms.transforms:
            if isinstance(t, T.Resize):
                self.target_size = t.size  # This should be (224, 224) in default setting
                break
                

    def __len__(self):
        # len() is the number of objects
        return len(self.gt.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.gt.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        
        # Neeed to crop image
        bbox = list(map(int, anot['bbox']))
        crop = image_original.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))    
        # crop.save("./debug.png")
        crop_tensor = self.transforms(crop)
        
        # Need to crop mask and apply transforms to mask
        mask = mask_util.decode(anot['mask'])  # Decodifica la máscara a un np.array
        x_min, y_min, width, height = bbox 
        mask_cropped = mask[y_min:y_min + height, x_min:x_min + width]
        mask_cropped_resized = cv2.resize(mask_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)  # Shape: (224, 224)
        mask_cropped_resized_tensor = torch.from_numpy(mask_cropped_resized) 

        #plot_mask(crop, mask_cropped) #Plot to debug
        return (crop_tensor, anot, mask_cropped_resized_tensor)
    
    

class CocoPrototypes_with_masks(Dataset):
    """
    This dataset class takes the GT (with masks) in COCO format from a support set and returns, for each annotation:
        1) The cropped image.
        2) The cropped mask, patchified to match the final DinoV2 dimensions.
    



    Parameters
    ----------
    gt : cocoapi
        Coco format GT results.
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88, DINO_PATH_SIZE=14):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        self.DINO_PATH_SIZE=DINO_PATH_SIZE

        # Extract the size from T.Resize
        for t in transforms.transforms:
            if isinstance(t, T.Resize):
                self.target_size = t.size  # This should be (224, 224) in default setting
                break
            
        print(f"Applying sanity check! Before # annotations = {len(self.gt.dataset['annotations'])}")
        # Perform a sanity check of the masks to remove the ones which are empty. 
        self.gt.dataset["annotations"] = [
            ann for ann in self.gt.dataset["annotations"]
            if all(coord >= 2 for coord in ann["bbox"])  # Ensure bbox values are non-negative
        ]
        print(f"After # annotations = {len(self.gt.dataset['annotations'])}")
 

    def __len__(self):
        # len() is the number of objects
        return len(self.gt.dataset['annotations'])


    def __getitem__(self, idx):
        anot = self.gt.dataset['annotations'][idx]
        # Load image and crop object
        image_name = self.gt.loadImgs(self.gt.getImgIds(anot['image_id']))[0]['file_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path)
        
        # Neeed to crop image
        bbox = list(map(int, anot['bbox']))
        crop = image_original.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))    
        # crop.save("./debug.png")
        crop_tensor = self.transforms(crop)
        
        # Need to crop mask and apply transforms to mask
        try:
            mask = mask_util.decode(anot['mask'])  # Decodifica la máscara a un np.array
            x_min, y_min, width, height = bbox 
            mask_cropped = mask[y_min:y_min + height, x_min:x_min + width]
            mask_cropped_resized = cv2.resize(mask_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)  # Shape: (224, 224)
            mask_cropped_resized_tensor = torch.from_numpy(mask_cropped_resized) 
            # Patchify mask:
            final_mask = mask_cropped_resized_tensor.to(dtype=torch.float32)
            final_mask = F.avg_pool2d(final_mask.unsqueeze(0), kernel_size=self.DINO_PATH_SIZE, stride=self.DINO_PATH_SIZE) 
            final_mask = final_mask.view(-1)
            final_mask = (final_mask > 0).float()
        except:
            print(anot)
            print(mask.shape)

        #plot_mask(crop, mask_cropped) #Plot to debug
        return (crop_tensor, anot, final_mask)
    
    
    
class CocoPrototypesUnknown_with_masks(Dataset):
    """
    This dataset class, takes the GT unknown images where the full image represents the object.
    The custom format of the .json is:
    {
        "images": [
            {"image_name":--, "url":-----, "category", "bbox_xyxy":--, "mask":---},
        ]
    }


    Parameters
    ----------
    gt : str
        Unknown images in custom format
    dtpath: str
        Path to the images
    transforms: torchvision.transforms
    seed: int

    Returns
    -------

    """
    def __init__(self, gt, dtpath, transforms, seed=88, DINO_PATH_SIZE=14):
        random.seed(seed)
                
        self.dtpath = dtpath
        self.transforms = transforms
        self.gt = gt
        self.DINO_PATH_SIZE=DINO_PATH_SIZE

        # Extract the size from T.Resize
        for t in transforms.transforms:
            if isinstance(t, T.Resize):
                self.target_size = t.size  # This should be (224, 224) in default setting
                break
        
        # Initial total count
        total_initial_count = len(self.gt)
        
        # Remove annotations with images that do not exist
        initial_count = len(self.gt)
        self.gt = [anot for anot in self.gt if os.path.exists(os.path.join(self.dtpath, anot['image_name']))]
        removed_nonexistent = initial_count - len(self.gt)
        
        # Remove all annotations that have duplicate image names
        image_counts = {}
        for anot in self.gt:
            image_counts[anot['image_name']] = image_counts.get(anot['image_name'], 0) + 1
        
        initial_count = len(self.gt)
        self.gt = [anot for anot in self.gt if image_counts[anot['image_name']] == 1]
        removed_duplicates = initial_count - len(self.gt)
        
        # Final total count
        total_final_count = len(self.gt)
        
        print(f"Total annotations before filtering: {total_initial_count}")
        print(f"Removed {removed_nonexistent} annotations with nonexistent images.")
        print(f"Removed {removed_duplicates} duplicate annotations.")
        print(f"Total annotations after filtering: {total_final_count}")

        
        

        # To maintain interoperability with previous versions where there was no supercategory. 
        # Add a supercategory to all detections, with same value as category
        if not "super_category" in self.gt[0]:
            self.cat_names = list(set([x['category'] for x in self.gt]))
            self.supercategory_names = list(set([x['category'] for x in self.gt]))
            new_self_gt = [
                {**item, "super_category": item["category"]} for item in self.gt
            ]
            self.gt = new_self_gt
        else:
            # Take into account sub and supercategories
            cat_names = []
            supercategory_names = []
            mapper = {}

            for x in self.gt:
                cat_names.append(x['category'])
                supercategory_names.append(x['super_category'])
                mapper[x['category']] = x['super_category']

            self.cat_names = list(set(cat_names))
            self.supercategory_names = list(set(supercategory_names))
            self.mapper = mapper
        
    def getCatNames(self):
        return self.cat_names
    
    def getSuperCatNames(self):
        return self.supercategory_names

    def getMapper(self):
        return self.mapper

    def __len__(self):
        # len() is the number of objects
        return len(self.gt)

    def __getitem__(self, idx):
        anot = self.gt[idx]

        image_name = anot['image_name']
        image_path = os.path.join(self.dtpath, image_name)
        image_original = Image.open(image_path).convert('RGB')
        
        # Neeed to crop image
        bbox = anot["bbox_xyxy"]
        bbox = list(map(int, bbox))
        crop = image_original.crop((bbox[0], bbox[1], bbox[2], bbox[3]))    
        # crop.save("./debug.png")
        crop_tensor = self.transforms(crop)
        
        # Need to crop mask and apply transforms to mask
        mask = mask_util.decode(anot['mask'])  # Decodifica la máscara a un np.array
        x_min, y_min, x_max, y_max = bbox 
        mask_cropped = mask[y_min:y_max, x_min:x_max]
        mask_cropped_resized = cv2.resize(mask_cropped, self.target_size, interpolation=cv2.INTER_NEAREST)  # Shape: (224, 224)
        mask_cropped_resized_tensor = torch.from_numpy(mask_cropped_resized) 
        # Patchify mask:
        final_mask = mask_cropped_resized_tensor.to(dtype=torch.float32)
        final_mask = F.avg_pool2d(final_mask.unsqueeze(0), kernel_size=self.DINO_PATH_SIZE, stride=self.DINO_PATH_SIZE) 
        final_mask = final_mask.view(-1)
        final_mask = (final_mask > 0).float()

        #plot_mask(crop, mask_cropped) #Plot to debug
        return (crop_tensor, anot, final_mask)