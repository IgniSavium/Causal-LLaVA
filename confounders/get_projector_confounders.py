import argparse
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re
from PIL import Image
import requests
from io import BytesIO
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
import warnings
import os

# Constants
offset = -1  # Offset for the image token start (used in llava-v1.5)
NUM_LAYER = 33  # Number of transformer layers in the model
MAX_SAMPLES_PER_CATEGORY = 10000  # Maximum number of samples to accumulate per category before stopping

# Global dictionary to store category-specific cumulative hidden states and counts
category_hidden_states_sum = {}
category_counts = {}

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process visual confounders from images.")
    parser.add_argument("--projector_weight_path", type=str, required=True, 
                      help="Path to the projector weights .bin file")
    parser.add_argument("--vision_tower", type=str, required=True, help="Path to the frozen vision tower.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store results.")
    parser.add_argument("--image_patch_json_path", type=str, required=True, help="Path to the JSON file containing image patch info.")
    parser.add_argument("--category_mapping_path", type=str, required=True, help="Path to the JSON file containing category mapping.")
    parser.add_argument("--base_image_dir", type=str, required=True, help="Base directory containing the images.")
    parser.add_argument("--mm_hidden_size", type=int, default=1024, help="Hidden size for mm_projector.")
    parser.add_argument("--hidden_size", type=int, default=4096, help="Hidden size for language model.")
    parser.add_argument("--mm_projector_type", type=str, default="mlp2x_gelu", help="Type of mm_projector (e.g., mlp2x_gelu).")
    parser.add_argument("--num_attention_heads", type=int, default=32, help="Number of attention heads.")
    return parser.parse_args()

def load_image(image_file):
    """
    Load an image either from a local path or URL.
    """
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def extract_category_hidden_state(hidden_state, patch_numbers, category_id):
    """
    Extract and average hidden states for specific patches corresponding to a given category.
    """
    if len(patch_numbers) == 0:
        print(f"Skipping category {category_id} due to empty patch_numbers.")
        return torch.zeros_like(hidden_state[0, :])  # Return zero tensor if no patches are available

    patch_indices = [num + offset for num in patch_numbers]  # Adjust indices by offset

    # Ensure all indices are within bounds
    for idx in patch_indices:
        if idx < 0 or idx >= hidden_state.size(0):
            raise ValueError(f"Patch index {idx} is out of bounds for hidden_state with size {hidden_state.size()}")

    patch_hidden_states = hidden_state[patch_indices, :]  # Extract relevant patches
    category_hidden_state = patch_hidden_states.mean(dim=0)  # Average across patches
    return category_hidden_state

def process_category_hidden_states(image_id, hidden_state, bbox_info):
    """
    Process and compute hidden states for each category based on patch information from bounding boxes.
    Stop accumulating when the sample count exceeds MAX_SAMPLES_PER_CATEGORY.
    """
    for category_info in bbox_info[image_id]["bbox_patch_info"]:
        category_id = category_info["category_id"]
        if category_id in category_counts and category_counts[category_id] >= MAX_SAMPLES_PER_CATEGORY:
            continue  # Skip categories that have reached the sample limit
        patch_numbers = category_info["patch_numbers"]
        try:
            category_hidden_state = extract_category_hidden_state(hidden_state, patch_numbers, category_id)
        except ValueError as e:
            print(f"Error processing image {image_id}, category {category_id}: {e}")
            continue
        # Initialize sums and counts for this category if not already done
        if category_id not in category_hidden_states_sum:
            category_hidden_states_sum[category_id] = torch.zeros_like(category_hidden_state, dtype=torch.float64)
            category_counts[category_id] = 0
        # Accumulate hidden state and increment count
        category_hidden_states_sum[category_id] += category_hidden_state
        category_counts[category_id] += 1

def calculate_average_category_hidden_state(category_mapping):
    """
    Calculate the average hidden state for each category using accumulated sums and counts.
    Warn about categories that were never involved in processing.
    """
    average_category_hidden_states = {}
    uninvolved_categories = []
    example_hidden_state = next(iter(category_hidden_states_sum.values()), None)
    hidden_size = example_hidden_state.size(-1) if example_hidden_state is not None else 0
    for category_id in category_mapping.keys():
        if category_id in category_counts and category_counts[category_id] > 0:
            total_sum = category_hidden_states_sum[category_id]
            count = category_counts[category_id]
            avg_hidden_state = total_sum / count  # Compute average
            average_category_hidden_states[category_id] = avg_hidden_state
        else:
            uninvolved_categories.append(category_id)
            print(f"Warning: Category ID {category_id} was never involved in processing. Setting its hidden states to all zeros.")
            zero_hidden_state = torch.zeros((NUM_LAYER, hidden_size), dtype=torch.float64)
            average_category_hidden_states[category_id] = zero_hidden_state
    return average_category_hidden_states

def save_category_hidden_states_as_bin(average_category_hidden_states, output_dir):
    """
    Save the computed average hidden states as a single .bin file compatible with Hugging Face's `from_pretrained()`.
    The key naming follows the format "model.mm_projector.visual_confounders".
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    
    state_dict = {}  # Dictionary to hold the state dict for saving
    layer_hidden_states = []  # List to collect hidden states for all categories
    
    sorted_category_ids = sorted(average_category_hidden_states.keys())  # Sort categories for consistent ordering
    
    for category_id in sorted_category_ids:
        hidden_state = average_category_hidden_states[category_id].cpu()
        layer_hidden_states.append(hidden_state)
    
    layer_hidden_states_matrix = torch.stack(layer_hidden_states, dim=0)  # Stack into matrix [num_categories, hidden_size]
    
    key = "model.mm_projector.visual_confounders"  # Key for saving in state_dict
    state_dict[key] = layer_hidden_states_matrix
    
    bin_file = Path(output_dir) / "projector_confounders.bin"  # Define output file path
    torch.save(state_dict, bin_file)  # Save state_dict to .bin file
    
    print(f"Saved projector_confounders to {bin_file}")  # Print confirmation message

def load_mm_projector_weights(projector_weight_path):
    """
    Load mm_projector weights from the specified path.
    Args:
        projector_weight_path (str): Path to the projector weights file.
    Returns:
        dict: Loaded mm_projector weights.
    """
    # Check if the file exists at the given path
    if not os.path.exists(projector_weight_path):
        raise FileNotFoundError(f"mm_projector weights not found at {projector_weight_path}")
    
    # Load the weights using PyTorch
    mm_projector_weights = torch.load(projector_weight_path)
    return mm_projector_weights

from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_vision_projector
from transformers import CLIPImageProcessor

def load_vision_components(vision_tower_cfg, projector_config, projector_weight_path):
    """
    Load vision tower, mm_projector, and image processor components.
    Additionally, load mm_projector weights from the specified path.
    Args:
        vision_tower_cfg: Configuration object for the vision tower.
        projector_config: Configuration object containing settings for the mm_projector.
        projector_weight_path (str): Path to the mm_projector weights file.
    Returns:
        tuple: (vision_tower, mm_projector, image_processor)
    """
    # Step 1: Load the vision tower model
    vision_tower = build_vision_tower(vision_tower_cfg, delay_load=False)
    vision_tower.load_model()

    # Step 2: Build the mm_projector using the provided configuration
    mm_projector = build_vision_projector(projector_config)

    # Step 3: Load the image processor for preprocessing images
    if not os.path.exists(vision_tower_cfg.mm_vision_tower):
        raise FileNotFoundError(f"Vision tower path does not exist: {vision_tower_cfg.mm_vision_tower}")
    image_processor = CLIPImageProcessor.from_pretrained(vision_tower_cfg.mm_vision_tower)

    # Step 4: Load mm_projector weights and modify state_dict keys
    mm_projector_weights = load_mm_projector_weights(projector_weight_path)
    new_state_dict = {k.replace("model.mm_projector.", ""): v for k, v in mm_projector_weights.items()}

    # Verify required keys are present in the modified state_dict
    required_keys = [
        "visual_confounders",
        "mlp.0.weight",
        "mlp.0.bias",
        "mlp.2.weight",
        "mlp.2.bias",
        "visual_cross_attn.kv_proj.weight",
        "visual_cross_attn.kv_proj.bias"
    ]
    missing_keys = [key for key in required_keys if key not in new_state_dict]
    if missing_keys:
        raise KeyError(f"Missing required keys in mm_projector weights: {missing_keys}")

    # Load the modified state_dict into the mm_projector
    mm_projector.load_state_dict(new_state_dict)

    # Convert mm_projector to half precision for efficiency
    mm_projector.half()
    
    return vision_tower, mm_projector, image_processor

# Define the VisionTowerConfig class
class VisionTowerConfig:
    def __init__(
        self,
        mm_vision_select_feature="patch",
        mm_vision_select_layer=-2,
        mm_vision_tower=None,
        unfreeze_mm_vision_tower=False
    ):
        """
        Initialize the VisionTowerConfig with individual parameters.
        Args:
            mm_vision_select_feature (str): The feature selection method (default: "patch").
            mm_vision_select_layer (int): The layer to select features from (default: -2).
            mm_vision_tower (str): Path to the vision tower model.
            unfreeze_mm_vision_tower (bool): Whether to unfreeze the vision tower (default: False).
        """
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_tower = mm_vision_tower
        self.unfreeze_mm_vision_tower = unfreeze_mm_vision_tower

# Define the ProjectorConfig class
class ProjectorConfig:
    def __init__(self, mm_hidden_size, hidden_size, mm_projector_type, num_attention_heads):
        """
        Initialize the ProjectorConfig with individual parameters.
        Args:
            mm_hidden_size (int): Hidden size for multi-modal projection.
            hidden_size (int): Hidden size for the model.
            mm_projector_type (str): Type of the multi-modal projector.
            num_attention_heads (int): Number of attention heads.
        """
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size
        self.mm_projector_type = mm_projector_type
        self.num_attention_heads = num_attention_heads
        
def run_inference_and_process(vision_tower, mm_projector, image_processor, bbox_patch_info, category_mapping, output_dir, base_image_dir):
    """
    Run inference and process category-specific hidden states for all images.
    """
    # Move models to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_tower.to(device)
    mm_projector.to(device)
    
    # Process each image in the dataset
    for image_id, image_info in tqdm(bbox_patch_info.items()):
        image_file = image_info["file_name"]
        image_path = Path(base_image_dir) / image_file
        image = load_image(str(image_path))
        images_tensor = process_images([image], image_processor, None).to(vision_tower.device, dtype=torch.float16)

        # Encode images to get features from vision tower
        with torch.inference_mode():
            vision_features = vision_tower(images_tensor)
            mm_projector_hidden_states = mm_projector.forward_mlp_only(vision_features)

        # Remove the batch dimension if present
        if mm_projector_hidden_states.dim() == 3 and mm_projector_hidden_states.size(0) == 1:
            mm_projector_hidden_states = mm_projector_hidden_states.squeeze(0)
            
        # Check for NaN values in hidden states
        if torch.isnan(mm_projector_hidden_states).any():
            print(f"NaN detected in mm_projector_hidden_states for image {image_id}")
            continue

        # Process category-specific hidden states
        process_category_hidden_states(image_id, mm_projector_hidden_states, bbox_patch_info)

    # Calculate average hidden state for each category
    average_category_hidden_states = calculate_average_category_hidden_state(category_mapping)

    # Save results as .bin files
    save_category_hidden_states_as_bin(average_category_hidden_states, output_dir)

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Build vision tower configuration
    vision_tower_cfg = VisionTowerConfig(
        mm_vision_select_feature="patch",
        mm_vision_select_layer=-2,
        mm_vision_tower=args.vision_tower,
        unfreeze_mm_vision_tower=False
    )
    
    # Build projector configuration
    projector_config = ProjectorConfig(
        mm_hidden_size=args.mm_hidden_size,
        hidden_size=args.hidden_size,
        mm_projector_type=args.mm_projector_type,
        num_attention_heads=args.num_attention_heads
    )

    # Load vision components and mm_projector weights
    vision_tower, mm_projector, image_processor = load_vision_components(
        vision_tower_cfg, projector_config, args.projector_weight_path
    )

    # Load bounding box patch info from JSON
    with open(args.image_patch_json_path, 'r') as f:
        bbox_patch_info = json.load(f)

    # Load category mapping from JSON
    with open(args.category_mapping_path, 'r') as f:
        category_mapping = {item["ID"]: item["OBJECT (PAPER)"] for item in json.load(f)}

    # Run inference and process hidden states
    run_inference_and_process(
        vision_tower=vision_tower,
        mm_projector=mm_projector,
        image_processor=image_processor,
        bbox_patch_info=bbox_patch_info,
        category_mapping=category_mapping,
        output_dir=args.output_dir,
        base_image_dir=args.base_image_dir
    )

if __name__ == "__main__":
    main()