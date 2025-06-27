#!/bin/bash

BASE_DIR=/path/to/Causal-LLaVA

COUNF_DIR=$BASE_DIR/confounders

WEIGHT_PATH=/path/to/weight

DATASET_PATH=/path/to/datasets

# Run the script to extract projector confounders (projector_confounders.bin)
python $COUNF_DIR/get_projector_confounders.py \
    --projector_weight_path $WEIGHT_PATH/causal-llava-v1.5-7b-pretrain/mm_projector.bin \
    --vision_tower $BASE_DIR/openai/clip-vit-large-patch14-336 \
    --output_dir $COUNF_DIR/output \
    --image_patch_json_path $COUNF_DIR/annotations/detail_5k_bbox_patch_numbers.json \
    --category_mapping_path $COUNF_DIR/annotations/COCO-id-name-mapping.json \
    --base_image_dir $DATASET_PATH/coco/train2017 \
    --mm_hidden_size 1024 \
    --hidden_size 4096 \
    --mm_projector_type mlp2x_gelu \
    --num_attention_heads 32
    # These arguments are specific for llava-v1.5