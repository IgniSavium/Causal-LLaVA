from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import torch
from argparse import Namespace

model_path = "/path/to/weight/causal-llava-v1.5-7b"

prompt = "Describe this image in detail?"

# image_file = "https://...png"
image_file = "/path/to/Causal-LLaVA/images/yellow_car.jpg"

# Using argparse.Namespace to create an argument object
args = Namespace(
    model_path=model_path,
    model_base=None,
    query=prompt,
    image_file=image_file,
    conv_mode=None,  # Automatic inference
    do_sample=True,
    temperature=0.2,
    max_new_tokens=1024,
)

eval_model(args)
