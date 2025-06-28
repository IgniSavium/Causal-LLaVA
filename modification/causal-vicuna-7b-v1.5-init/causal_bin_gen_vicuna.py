import torch

initialization_dict = {
    "model.pre_causal_intervention_layernorm.weight": torch.ones([4096]),
    "model.causal_intervention.visual_confounders": torch.zeros([80, 4096]),
    "model.causal_intervention.text_confounders": torch.zeros([80, 4096]),
    "model.causal_intervention.visual_cross_attn.kv_proj.weight": torch.zeros([4096, 4096]),
    "model.causal_intervention.visual_cross_attn.kv_proj.bias": torch.zeros([4096]),
    "model.causal_intervention.text_cross_attn.kv_proj.weight": torch.zeros([4096, 4096]),
    "model.causal_intervention.text_cross_attn.kv_proj.bias": torch.zeros([4096])
}

# Save the dictionary as a binary file
torch.save(initialization_dict, "/path/to/vicuna-7b-v1.5/causal-initialization.bin")

print("The weight initialization file 'causal-initialization.bin' was created successfully.")