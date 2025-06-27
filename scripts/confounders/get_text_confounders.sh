#!/bin/bash

BASE_DIR=/path/to/Causal-LLaVA

COUNF_DIR=$BASE_DIR/confounders

WEIGHT_PATH=/path/to/weight

DATASET_PATH=/path/to/datasets

# Run the script to extract ALL visual confounders
python $COUNF_DIR/get_text_confounders.py \
    --model_path $WEIGHT_PATH/causal-llava-v1.5-7b-fineunte \
    --model_name llava-v1.5-7b \
    --output_dir $COUNF_DIR/output \
    --qa_pairs_path $COUNF_DIR/annotations/detail_5k.json \
    --category_mapping_path $COUNF_DIR/annotations/COCO-id-name-mapping.json \
    --conv_mode llava_v1    # conv_mode llava_v1 is also suitable for llava-v1.5