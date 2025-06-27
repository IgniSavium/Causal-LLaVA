import argparse
import json
import torch
from pathlib import Path
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from llava.conversation import conv_templates
from llava.utils import disable_torch_init
import os

# Constants - take LLaVA-v1.5 as an example
NUM_LAYER = 33  # Number of transformer layers
MAX_SAMPLES_PER_CATEGORY = 20000  # Stop accumulating when the number of samples in a category reaches this value

# Global variables to store category-specific hidden states and counts
category_hidden_states_sum = {}
category_counts = {}

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process textual confounders from model outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pretrained model.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--qa_pairs_path", type=str, required=True, help="Path to the JSON file containing question-answer pairs.")
    parser.add_argument("--category_mapping_path", type=str, required=True, help="Path to the JSON file containing category mapping.")
    parser.add_argument("--conv_mode", type=str, default="llava_v1.5", help="Conversation mode for the model.")
    return parser.parse_args()

def extract_category_word_hidden_state(hidden_state, input_ids, category_word, tokenizer):
    """
    Extract the hidden states for the given category word and return their average.
    If the category word is not found, return None.
    """
    # Encode the category word into tokens
    category_tokens = tokenizer.encode(category_word, add_special_tokens=False)
    token_indices = []
    # Find all occurrences of the category word in the input_ids
    for i in range(len(input_ids) - len(category_tokens) + 1):
        if input_ids[i:i+len(category_tokens)].tolist() == category_tokens:
            token_indices.extend(range(i, i + len(category_tokens)))
    if not token_indices:
        return None
    # Compute the average hidden state for the category word
    category_hidden_states = hidden_state[:, token_indices, :]
    category_hidden_state = category_hidden_states.mean(dim=1)  # Average across tokens
    return category_hidden_state

def process_category_hidden_states(image_id, hidden_state, input_ids, tokenizer, category_mapping):
    """
    Process and compute category-specific hidden states for a given image's answer.
    Stop accumulating when the sample count for a category exceeds the threshold.
    """
    global category_hidden_states_sum, category_counts
    for category_id, category_word in category_mapping.items():
        # Skip if the category has reached the maximum sample limit
        if category_id in category_counts and category_counts[category_id] >= MAX_SAMPLES_PER_CATEGORY:
            continue
        try:
            # Extract hidden states for the category word
            category_hidden_state = extract_category_word_hidden_state(hidden_state, input_ids, category_word, tokenizer)
        except Exception as e:
            print(f"Error processing image {image_id}, category {category_id}: {e}")
            continue
        if category_hidden_state is None:
            continue
        # Initialize the sum and count for the category if not already done
        if category_id not in category_hidden_states_sum:
            category_hidden_states_sum[category_id] = torch.zeros_like(category_hidden_state, dtype=torch.float64)
            category_counts[category_id] = 0
        category_hidden_states_sum[category_id] += category_hidden_state
        category_counts[category_id] += 1

def calculate_average_category_hidden_state(category_mapping):
    """
    Calculate the average hidden state for each category using the accumulated sum and count.
    If a category has not been involved, set its average hidden state to all zeros.
    """
    average_category_hidden_states = {}
    for category_id in category_mapping.keys():
        if category_id in category_counts and category_counts[category_id] > 0:
            total_sum = category_hidden_states_sum[category_id]
            count = category_counts[category_id]
            avg_hidden_state = total_sum / count
            average_category_hidden_states[category_id] = avg_hidden_state
        else:
            # Set to zero tensor if no samples were processed for the category
            hidden_size = next(iter(category_hidden_states_sum.values())).size(-1)
            zero_hidden_state = torch.zeros((NUM_LAYER, hidden_size), dtype=torch.float64)
            average_category_hidden_states[category_id] = zero_hidden_state
    return average_category_hidden_states

def save_category_hidden_states_as_bin(average_category_hidden_states, output_dir):
    """
    Save the average hidden states in the format compatible with Hugging Face's from_pretrained().
    Each layer's category hidden states are stacked into a matrix of shape [num_categories, hidden_size].
    These matrices are saved into a single .bin file for all layers.
    
    Modified to rename keys as:
    - layer_1 -> "model.layers.0.causal_intervention.text_confounders"
    - layer_2 -> "model.layers.1.causal_intervention.text_confounders"
    ...
    - layer_32 -> "model.layers.31.causal_intervention.text_confounders"
    - layer_0 is discarded.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    num_layers = NUM_LAYER
    state_dict = {}
    
    # Start from layer_1 (skip layer_0)
    for layer in range(1, num_layers):  # Start from 1 instead of 0
        layer_hidden_states = []
        for category_id in sorted(average_category_hidden_states.keys()):
            hidden_state = average_category_hidden_states[category_id][layer].cpu()
            layer_hidden_states.append(hidden_state)
        layer_hidden_states_matrix = torch.stack(layer_hidden_states, dim=0)
        
        # New key format: "model.layers.{layer-1}.causal_intervention.text_confounders"
        key = f"model.layers.{layer-1}.causal_intervention.text_confounders"
        state_dict[key] = layer_hidden_states_matrix
    
    # Save the state_dict to a .bin file
    bin_file = Path(output_dir) / "text_confounders.bin"
    torch.save(state_dict, bin_file)
    print(f"Saved all layer matrices to {bin_file}")

def run_inference_and_process(args):
    """
    Run model inference and process category-specific hidden states for all images.
    """
    disable_torch_init()
    tokenizer, model, _, _ = load_pretrained_model(args.model_path, None, args.model_name)
    
    # Load QA pairs and category mapping
    with open(args.qa_pairs_path, 'r') as f:
        qa_pairs = {item["id"]: item for item in json.load(f)}
    with open(args.category_mapping_path, 'r') as f:
        category_mapping = {item["ID"]: item["OBJECT (PAPER)"] for item in json.load(f)}
        
    for image_id, qa_info in tqdm(qa_pairs.items()):
        question = qa_info["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        answer = qa_info["conversations"][1]["value"]
        
        # Prepare conversation prompt
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        prompt = conv.get_prompt()
        
        # Tokenize input
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    
        with torch.inference_mode():
            outputs = model(input_ids, output_hidden_states=True, return_dict=True)
            
        # Extract fully_connected_hidden_states instead of hidden_states
        hidden_states = outputs.hidden_states
        hidden_states_tensor = torch.stack(hidden_states, dim=0).squeeze(1)
        
        # Process category-specific hidden states
        process_category_hidden_states(image_id, hidden_states_tensor, input_ids[0], tokenizer, category_mapping)
    
    # Calculate average hidden states for all categories
    average_category_hidden_states = calculate_average_category_hidden_state(category_mapping)
    
    # Save results to a .bin file
    save_category_hidden_states_as_bin(average_category_hidden_states, args.output_dir)

def main():
    """
    Main function to parse arguments and run inference.
    """
    args = parse_arguments()
    run_inference_and_process(args)

if __name__ == "__main__":
    main()