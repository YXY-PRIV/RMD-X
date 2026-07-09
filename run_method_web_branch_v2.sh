
# UNKNOWN BRANCH with masks

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH=/models/ICCV25_experimentation/
DATASET_PATH="/datasets/xray-datasets/"
datasets=("PIXray" "pidray" "CLCXray" "DET-COMPASS" "HiXray" "DvXray")
detectors=("groundingDINO" "detic" "CoDet" "VLDet")
########################################################




#############################################################
#1) BUILD PROTOTYPES FOR EACH DATASET
# Note that images are retrieved from the internet using the Google API. 
# You must configure it or use our precomputed visual descriptors.
#############################################################

# python raxo/obtain_cat_colours_v3.py # add arguments

# for dataset in "${datasets[@]}"
# do
#     echo "Building web-powered visual descriptors in dataset: $dataset"

    # # 1) GENERATE HYPONYMYS   (not in the ICCV version)
    # python3 raxo/build_hrchy.py \
    #     --categories  ${DATASET_PATH}${dataset}/annotations/full_test.json \
    #     --method llm \
    #     --out ${RESULTS_PATH}prototypes_web_2/${detector}/${dataset}/ \
    #     --n_hyponyms 3

    # 2) WEB RETRIEVAL
    # CUDA_VISIBLE_DEVICES=0 python raxo/google_image_retrievalv2.py \
    #     --cats ${RESULTS_PATH}prototypes_web_2/${dataset}/categories.json \
    #     --n 30 \
    #     --model_config projects/GroundingDINO_modify/configs/grounding_dino_swin-b_finetune_16xb2_1x_pidray.py \
    #     --model_weights /models/groundingDINO/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth \
    #     --out ${RESULTS_PATH}prototypes_web_2/${dataset}/images/ \
    #     --cats_from_gt ${DATASET_PATH}${dataset}/annotations/full_test.json
    
    # # 3) OBTAIN MASKS
    # CUDA_VISIBLE_DEVICES=0 python raxo/obtain_masks.py \
    #     --gt /models/ICCV25_experimentation/prototypes_web_2/${dataset}/images/annotations.json \
    #     --image_path /models/ICCV25_experimentation/prototypes_web_2/${dataset}/images/imgs \
    #     --branch unknown
    
    # 4) STYLE TRANSFER v2
    # python raxo/style_transfer_v2.py \
    #     --gt /models/ICCV25_experimentation/prototypes_web_2/${dataset}/images/annotations_with_masks.json \
    #     --image_path /models/ICCV25_experimentation/prototypes_web_2/${dataset}/images/imgs \
    #     --out /models/ICCV25_experimentation/prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
    #     --colours /models/ICCV25_experimentation/colour_knowledge_material_clustering.json"

    # # 5) BUILD the prototypes
    # CUDA_VISIBLE_DEVICES=0 python raxo/build_prototypes_unknown_concepts_with_masks_prop.py \
    #     --gt /models/ICCV25_experimentation/prototypes_web_2/${dataset}/images/annotations_with_masks.json \
    #     --image_path /models/ICCV25_experimentation/prototypes_web_2/${dataset}/imgs_style_transfer_v2_v2 \
    #     --out ${RESULTS_PATH}prototypes_web_2/${dataset}_prot_sam2_style2_v2.pt
# done


#############################################################
# 2) CLASSIFICATION
#############################################################
for detector in "${detectors[@]}"
do
    # Loop through each dataset
    for dataset in "${datasets[@]}"
    do
        echo "Obtaining results with web-branch in dataset: $dataset"

        # Nota para pixray: models/ICCV25_experimentation/prototypes_web/PIXray_prot_sam2_style2_v2.pt
        
        CUDA_VISIBLE_DEVICES=0 python raxo/main_with_masks_prop.py \
            --json_res ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks.json \
            --image_path ${DATASET_PATH}${dataset}/test \
            --prototypes ${RESULTS_PATH}prototypes_web_2/${dataset}_prot_sam2_style2_v2.pt \
            --nms 0.8 \
            --name web_branch_v2 \
            --branch known

        python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json


        python raxo/uncertanty_estimation_fixed.py \
            --dets /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json

        python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json

        
        # This is only when the in-house results are already computed
        python raxo/final_cocoapi.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt_database /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_known_final_uncertanty.json \
            --cocoDt_web /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_web_branch_v2.json
        
    done
done


