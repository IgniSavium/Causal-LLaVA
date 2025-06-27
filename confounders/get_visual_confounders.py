# Import core libraries
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm  # Progress bar
import re
from PIL import Image
import requests
from io import BytesIO

# Import LLaVA-specific modules
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
import warnings
import os

# Constants for LLaVA-v1.5 architecture
IMG_OFFSET = 34  # Offset in prompt to locate image token start (model-specific)
NUM_LAYER = 33   # Total transformer layers in the model
MAX_SAMPLES_PER_CATEGORY = 20000  # Max samples to aggregate per object category

# Global dictionaries to accumulate hidden states per category
category_hidden_states_sum = {}
category_counts = {}

def parse_arguments():
    """
    Parse command-line arguments for model paths, data directories, and processing parameters.
    """
    parser = argparse.ArgumentParser(description="Process visual confounders from images.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model weights.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pretrained model architecture.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--image_patch_json_path", type=str, required=True, help="Path to JSON with image patch annotations.")
    parser.add_argument("--category_mapping_path", type=str, required=True, help="Path to JSON mapping category IDs to names.")
    parser.add_argument("--base_image_dir", type=str, required=True, help="Root directory containing image files.")
    parser.add_argument("--query", type=str, default="Describe this image in detail.", help="Prompt for model inference.")
    return parser.parse_args()

def load_image(image_file):
    """
    Load an image from a local path or URL.
    Converts to RGB format for consistent input processing.
    """
    if image_file.startswith(("http", "https")):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def extract_category_hidden_state(hidden_state, patch_numbers, category_id):
    """
    Extract and average hidden states for specified image patches.
    
    Args:
        hidden_state: Model outputs (tensor of shape [layers, batch, tokens, features])
        patch_numbers: Indices of patches corresponding to the category
        category_id: Object category ID (e.g., 0 for 'car', 1 for 'person')
    
    Returns:
        Averaged hidden state vector for the category.
    """
    if not patch_numbers:
        print(f"Skipping category {category_id} due to empty patch_numbers.")
        return torch.zeros_like(hidden_state[:, 0, :])  # Zero fallback
    
    # Adjust patch indices by IMG_OFFSET to align with model's token position
    patch_indices = [num + IMG_OFFSET for num in patch_numbers]
    
    # Validate indices to prevent out-of-bounds errors
    for idx in patch_indices:
        if idx < 0 or idx >= hidden_state.size(1):
            raise ValueError(f"Patch index {idx} invalid for hidden_state size {hidden_state.size()}")
    
    # Extract and average patch-specific hidden states
    patch_hidden_states = hidden_state[:, patch_indices, :]  # Shape: [layers, patches, features]
    category_hidden_state = patch_hidden_states.mean(dim=1)  # Average across patches
    return category_hidden_state

def process_category_hidden_states(image_id, hidden_state, bbox_info):
    """
    Process hidden states for all categories present in an image.
    Accumulates results in global dictionaries for later averaging.
    """
    for category_info in bbox_info[image_id]["bbox_patch_info"]:
        category_id = category_info["category_id"]
        
        # Skip if category has reached sample limit
        if category_id in category_counts and category_counts[category_id] >= MAX_SAMPLES_PER_CATEGORY:
            continue
            
        patch_numbers = category_info["patch_numbers"]
        try:
            # Get category-specific hidden state
            category_hidden_state = extract_category_hidden_state(hidden_state, patch_numbers, category_id)
        except ValueError as e:
            print(f"Error processing image {image_id}, category {category_id}: {e}")
            continue
        
        # Initialize storage on first encounter
        if category_id not in category_hidden_states_sum:
            category_hidden_states_sum[category_id] = torch.zeros_like(category_hidden_state, dtype=torch.float64)
            category_counts[category_id] = 0
        
        # Accumulate results
        category_hidden_states_sum[category_id] += category_hidden_state
        category_counts[category_id] += 1

def calculate_average_category_hidden_state(category_mapping):
    """
    Average accumulated hidden states per category.
    Falls back to zero vectors for categories with no valid samples.
    """
    average_category_hidden_states = {}
    example_hidden_state = next(iter(category_hidden_states_sum.values()), None)
    hidden_size = example_hidden_state.size(-1) if example_hidden_state is not None else 0
    
    for category_id in category_mapping.keys():
        if category_id in category_counts and category_counts[category_id] > 0:
            # Compute average for valid categories
            total_sum = category_hidden_states_sum[category_id]
            avg_hidden_state = total_sum / category_counts[category_id]
            average_category_hidden_states[category_id] = avg_hidden_state
        else:
            # Zero fallback for unused categories
            print(f"Warning: Category ID {category_id} missing samples. Using zero vector.")
            average_category_hidden_states[category_id] = torch.zeros((NUM_LAYER, hidden_size), dtype=torch.float64)
    
    return average_category_hidden_states

def save_category_hidden_states_as_bin(average_category_hidden_states, output_dir):
    """
    Save category-specific hidden states in HuggingFace-compatible format.
    Each layer's states are saved under a specific key structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    state_dict = {}
    
    for layer in range(NUM_LAYER):
        layer_hidden_states = []
        sorted_ids = sorted(average_category_hidden_states.keys())
        
        for category_id in sorted_ids:
            # Extract layer-specific hidden state and move to CPU
            layer_hidden_states.append(average_category_hidden_states[category_id][layer].cpu())
        
        # Stack into matrix [num_categories, hidden_size]
        layer_matrix = torch.stack(layer_hidden_states, dim=0)
        
        # Define key based on LLaVA layer convention
        if layer == 0:
            key = "model.mm_projector.visual_confounders"
        else:
            key = f"model.layers.{layer-1}.causal_intervention.visual_confounders"
        
        state_dict[key] = layer_matrix
    
    # Save to binary file
    bin_file = Path(output_dir) / "visual_confounders.bin"
    torch.save(state_dict, bin_file)
    print(f"Saved confounder matrices to {bin_file}")

def run_inference_and_process(args):
    """
    Main pipeline: 
    1. Load model
    2. Process images
    3. Extract and aggregate hidden states
    4. Save results
    """
    # Initialize model (disable init for faster loading)
    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, None, args.model_name)
    
    # Prepare prompt with image token
    qs = args.query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    # Set conversation template based on model version
    if 'llama-2' in args.model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in args.model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in args.model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Load annotation data
    with open(args.image_patch_json_path, 'r') as f:
        bbox_patch_info = json.load(f)
    
    with open(args.category_mapping_path, 'r') as f:
        category_mapping = {item["ID"]: item["OBJECT (PAPER)"] for item in json.load(f)}
    
    # Process each image
    for image_id, image_info in tqdm(bbox_patch_info.items()):
        image_file = image_info["file_name"]
        image_path = Path(args.base_image_dir) / image_file
        image = load_image(str(image_path))
        
        # Preprocess image
        images_tensor = process_images([image], image_processor, model.config).to(model.device, dtype=torch.float16)
        
        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        
        # Forward pass
        with torch.inference_mode():
            outputs = model(input_ids, images=images_tensor, output_hidden_states=True, return_dict=True)
        
        # Process outputs
        hidden_states = outputs.hidden_states
        hidden_states_tensor = torch.stack(hidden_states, dim=0).squeeze(1)
        
        if torch.isnan(hidden_states_tensor).any():
            print(f"NaN detected in hidden_states for image {image_id}")
            continue
        
        process_category_hidden_states(image_id, hidden_states_tensor, bbox_patch_info)
    
    # Finalize and save results
    average_category_hidden_states = calculate_average_category_hidden_state(category_mapping)
    save_category_hidden_states_as_bin(average_category_hidden_states, args.output_dir)

def main():
    args = parse_arguments()
    run_inference_and_process(args)

if __name__ == "__main__":
    main()