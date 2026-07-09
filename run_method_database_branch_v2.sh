# KNOWN BRANCH with masks

########################################################
#### CHANGE THIS VARIABLES
########################################################
RESULTS_PATH=/models/ICCV25_experimentation/
DATASET_PATH="/datasets/xray-datasets/"
datasets=("PIXray" "pidray" "CLCXray" "DET-COMPASS" "HiXray" "DvXray")
detectors=("groundingDINO" "detic" "CoDet" "VLDet")
########################################################




#############################################################
# 1) BASELINE RESULTS
#############################################################

echo "----OBTAINING BASELINE RESULTS----"

for detector in "${detectors[@]}"
do
    for dataset in "${datasets[@]}"
    do
    echo "Baseline results in dataset $dataset" with detector $detector
    python3 raxo/cocoapi_2.py \
        --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
        --cocoDt ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}.json
    done
done
echo
echo


#############################################################
# 2) TO BUILD VISUAL DESCRIPTORS (only needed if you do not use the precomputed ones)
#############################################################

# echo "----BUILDING IN-HOUSE VISUAL DESCRIPTORS----"
# CUDA_VISIBLE_DEVICES=3 python raxo/obtain_masks.py \
#     --gt /datasets/xray-datasets/support_full/full_30_support_set.json \
#     --image_path /datasets/xray-datasets/support_full/images \

# CUDA_VISIBLE_DEVICES=3 python raxo/build_prototypes_with_masks_prop.py \
#     --gt /datasets/xray-datasets/support_full/full_30_support_set_with_masks.json \
#     --image_path /datasets/xray-datasets/support_full/images \
#     --out /models/ICCV25_experimentation/known_prototypes_sam2.pt



#############################################################
# 3) CLASSIFICATION
#############################################################


for detector in "${detectors[@]}"
do
    for dataset in "${datasets[@]}"
    do

        echo "Obtaining results with database-branch in dataset: $dataset"

        # 1) # Extract masks. See extract_masks_inference.sh and extract_masks_inference_run.sh
        CUDA_VISIBLE_DEVICES=0 python raxo/obtain_masks_res_inference.py \
            --gt /datasets/xray-datasets/${dataset}/annotations/full_test.json \
            --res /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}.json \
            --image_path /datasets/xray-datasets/${dataset}/test/


        # 2) clasification
        CUDA_VISIBLE_DEVICES=0 python raxo/main_with_masks_prop.py \
            --json_res ${RESULTS_PATH}initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks.json \
            --image_path ${DATASET_PATH}${dataset}/test \
            --prototypes /models/ICCV25_experimentation/known_prototypes_sam2.pt \
            --nms 0.8 \
            --name known_final \
            --branch known

        python raxo/uncertanty_estimation_fixed.py \
            --dets /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_known_final.json

        # Theese are the 100/0 results (using only the in-house samples)
        python raxo/cocoapi_2.py \
            --cocoGt ${DATASET_PATH}${dataset}/annotations/full_test.json \
            --cocoDt /models/ICCV25_experimentation/initial_detections/${detector}/coco_results.bbox_${dataset}_with_masks_nms_0.8_our_method_known_final_uncertanty.json

    done
done
echo
echo