

# %%
from transformers import AutoConfig, MllamaForCausalLM
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaForCausalLM, BitsAndBytesConfig
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16,  # or torch.float16 if needed
    bnb_8bit_use_double_quant=True,
)

# Step 1: Load the full pretrained model with both vision and language components
full_model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    # device_map="auto"
).to(device)

language_model_checkpoint = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_noquant"

# config = AutoConfig.from_pretrained(language_model_checkpoint)
# print(config)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel

from safetensors.torch import load_file
from transformers import AutoConfig

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaForCausalLM, BitsAndBytesConfig
from datetime import date
from peft import LoraConfig, get_peft_model


# Quantization configuration for both models
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16,  # or torch.float16 if needed
    bnb_8bit_use_double_quant=True,
)

lora_config = LoraConfig(
   r=16,  # Rank of the decomposition, typically a small number (e.g., 8, 16)
   lora_alpha=32,  # Scaling factor for LoRA parameters
   target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model layers (e.g., Q, V projections in attention layers)
   lora_dropout=0.1,  # Dropout for LoRA layers
   bias="none"  # LoRA doesn’t update biases by default; change to "all" if needed
)


# Define model paths
model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"

lora_config = LoraConfig(
   r=16,  # Rank of the decomposition, typically a small number (e.g., 8, 16)
   lora_alpha=32,  # Scaling factor for LoRA parameters
   target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model layers (e.g., Q, V projections in attention layers)
   lora_dropout=0.1,  # Dropout for LoRA layers
   bias="none"  # LoRA doesn’t update biases by default; change to "all" if needed
)
language_model_checkpoint = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_captions_noquant"

from safetensors.torch import load_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load the state dictionary from the safetensors checkpoint
# # checkpoint_path = "/home/oe2015/final_model_checkpoint_512_quant/model-00001-of-00003.safetensors"
# # state_dict = load_file(checkpoint_path)

# # checkpoint_path = "/home/oe2015/final_model_checkpoint_512_quant/model-00002-of-00003.safetensors"
# # state_dict2 = load_file(checkpoint_path)

# # checkpoint_path = "/home/oe2015/final_model_checkpoint_512_quant/model-00003-of-00003.safetensors"
# # state_dict3 = load_file(checkpoint_path)

# # # Fix the state_dict to ensure compatibility
# # for key, tensor in state_dict.items():
# #     if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
# #         state_dict[key] = tensor.requires_grad_(False)

# # for key, tensor in state_dict2.items():
# #     if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
# #         state_dict[key] = tensor.requires_grad_(False)

# # for key, tensor in state_dict3.items():
# #     if tensor.dtype.is_floating_point or tensor.dtype.is_complex:
# #         state_dict[key] = tensor.requires_grad_(False)
# # for key, tensor in state_dict.items():
# #     print(f"{key}: dtype={tensor.dtype}, requires_grad={tensor.requires_grad}, shape={tensor.shape}")

model = MllamaForCausalLM.from_pretrained(
    language_model_checkpoint,
    # config=config,
    torch_dtype=torch.bfloat16,
    # device_map="auto",
    use_safetensors=True,  # Explicitly use safetensors
    local_files_only=True
).to(device)

from transformers import AutoConfig, MllamaForCausalLM
from safetensors.torch import load_file

# # Load LoRA layers dynamically if needed
model = get_peft_model(model, lora_config)
full_model = get_peft_model(full_model, lora_config)

# # Verify model layers
# # for name, param in model.named_parameters():
# #     print(name, param.requires_grad)


# from safetensors.torch import load_file

# safetensors_shard = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_noquant/model-00001-of-00009.safetensors"
# state_dict = load_file(safetensors_shard)

# # Step 5: Assign weights to the model's LoRA layers
# model_dict = dict(model.named_parameters())
# for key, value in state_dict.items():
#     if "lora_A" in key or "lora_B" in key:
#         model_key = f"base_model.model.{key}"  # Adjust based on your model's structure
#         if model_key in model_dict:
#             model_dict[model_key].data.copy_(value)


# # Path to the specific safetensors shard
# shard_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_noquant/model-00001-of-00009.safetensors"

# # Load the state dictionary from the shard
# state_dict = load_file(shard_path)


# # Compare the LoRA-specific weights
# # Compare the LoRA-specific weights
# for key, value in state_dict.items():
#     if "lora_A" in key or "lora_B" in key:
#         # Match the keys to the model's parameter names
#         model_key = f"base_model.model.{key}"  # Adjust this if necessary to match the model's naming
#         if model_key in dict(model.named_parameters()):
#             model_tensor = dict(model.named_parameters())[model_key]
            
#             # Move value to the same device as model_tensor
#             value = value.to(model_tensor.device)
            
#             # Compare tensors
#             are_equal = torch.allclose(model_tensor.data, value, atol=1e-6)
#             print(f"{key} matches: {are_equal}")
#             if not are_equal:
#                 print(f"Difference for {key}: {torch.abs(model_tensor.data - value).max().item()}")


# for name, param in model.named_parameters():
#     print(name, param.requires_grad)


# for name, param in full_model.named_parameters():
#     print(name, param.requires_grad)

# # %%
# from safetensors.torch import load_file


# # Define the path to the shards and the range of indices
# base_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_noquant"
# shard_prefix = "model-"
# shard_suffix = ".safetensors"
# start_index = 1  # Start index for shard files
# end_index = 9    # End index for shard files

# # Get the model's parameter dictionary
# model_dict = dict(model.named_parameters())

# # Loop through each shard file and load the weights
# for i in range(start_index, end_index + 1):
#     shard_path = f"{base_path}/{shard_prefix}{i:05d}-of-{end_index:05d}{shard_suffix}"
#     print(f"Loading shard: {shard_path}")
    
#     # Load the state dictionary from the shard
#     state_dict = load_file(shard_path)
    
#     # Assign weights to the model's LoRA layers
#     for key, value in state_dict.items():
#         if "lora_A" in key or "lora_B" in key:
#             # Adjust the key based on the model's structure
#             model_key = f"base_model.model.{key}"
#             if model_key in model_dict:
#                 print(f"Loading weights for: {model_key}")
#                 model_dict[model_key].data.copy_(value)
#             else:
#                 print(f"Key not found in model: {model_key}")

# # %%
# shard_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_noquant/model-00001-of-00009.safetensors"

# # Load the state dictionary from the shard
# state_dict = load_file(shard_path)

# # Compare the LoRA-specific weights
# # Compare the LoRA-specific weights
# for key, value in state_dict.items():
#     if "lora_A" in key or "lora_B" in key:
#         # Match the keys to the model's parameter names
#         model_key = f"base_model.model.{key}"  # Adjust this if necessary to match the model's naming
#         if model_key in dict(model.named_parameters()):
#             model_tensor = dict(model.named_parameters())[model_key]
            
#             # Move value to the same device as model_tensor
#             value = value.to(model_tensor.device)
            
#             # Compare tensors
#             are_equal = torch.allclose(model_tensor.data, value, atol=1e-6)
#             print(f"{key} matches: {are_equal}")
#             if not are_equal:
#                 print(f"Difference for {key}: {torch.abs(model_tensor.data - value).max().item()}")

import torch

import torch

# def compare_and_replace_lora_layers(full_model, checkpoint_model):
#     # Move models to the same device if necessary
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # full_model.to(device)
#     # checkpoint_model.to(device)

#     # Extract the LoRA parameters from the model checkpoint
#     checkpoint_state_dict = {
#         name.replace("base_model.model.", "base_model.model.language_model."): param
#         for name, param in checkpoint_model.named_parameters()
#         if "lora" in name
#     }
    
#     # Extract LoRA layers from the full_model
#     full_model_state_dict = {
#         name: param
#         for name, param in full_model.named_parameters()
#         if "lora" in name
#     }
    
#     all_matched = True

#     for name, full_model_param in full_model_state_dict.items():
#         if name in checkpoint_state_dict:
#             checkpoint_param = checkpoint_state_dict[name]
            
#             # Ensure both tensors are on the same device
#             # if full_model_param.device != device:
#             #     full_model_param.data = full_model_param.data.to(device)
#             # if checkpoint_param.device != device:
#             #     checkpoint_param.data = checkpoint_param.data.to(device)
            
#             # Check if the tensors match
#             if torch.allclose(full_model_param.data, checkpoint_param.data, atol=1e-6):
#                 print(f"Matched: {name}")
#             else:
#                 all_matched = False
#                 print(f"MISMATCH: {name}")
#                 print(f"Difference: {torch.abs(full_model_param.data - checkpoint_param.data).max().item()}")
                
#                 # Replace the parameter in the full_model with the one from the checkpoint
#                 full_model_param.data = checkpoint_param.data.clone()
#         else:
#             all_matched = False
#             print(f"Missing in checkpoint: {name}")

#     # Check for extra layers in the checkpoint
#     for name in checkpoint_state_dict.keys():
#         if name not in full_model_state_dict:
#             all_matched = False
#             print(f"Extra in checkpoint: {name}")

#     if all_matched:
#         print("All LoRA layers matched and replaced.")
#     else:
#         print("Some LoRA layers did not match.")

# def compare_and_replace_lora_layers(full_model, checkpoint_model):
#     # Define the target device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Extract the LoRA parameters from the model checkpoint
#     checkpoint_state_dict = {
#         name.replace("base_model.model.", "base_model.model.language_model."): param
#         for name, param in checkpoint_model.named_parameters()
#         if "lora" in name
#     }
    
#     # Extract LoRA layers from the full_model
#     full_model_state_dict = {
#         name: param
#         for name, param in full_model.named_parameters()
#         if "lora" in name
#     }
    
#     all_matched = True

#     for name, full_model_param in full_model_state_dict.items():
#         if name in checkpoint_state_dict:
#             checkpoint_param = checkpoint_state_dict[name]
            
#             # Ensure both tensors are moved to the same device for comparison
#             full_model_tensor = full_model_param.data.to(device)
#             checkpoint_tensor = checkpoint_param.data.to(device)
            
#             # Check if the tensors match
#             if torch.allclose(full_model_tensor, checkpoint_tensor, atol=1e-6):
#                 print(f"Matched: {name}")
#             else:
#                 all_matched = False
#                 print(f"MISMATCH: {name}")
#                 print(f"Difference: {torch.abs(full_model_tensor - checkpoint_tensor).max().item()}")
                
#                 # Replace the parameter in the full_model with the one from the checkpoint
#                 full_model_param.data = checkpoint_tensor.clone()
#         else:
#             all_matched = False
#             print(f"Missing in checkpoint: {name}")

#     # Check for extra layers in the checkpoint
#     for name in checkpoint_state_dict.keys():
#         if name not in full_model_state_dict:
#             all_matched = False
#             print(f"Extra in checkpoint: {name}")

#     if all_matched:
#         print("All LoRA layers matched and replaced.")
#     else:
#         print("Some LoRA layers did not match.")

# Apply the comparison and replacement
# compare_and_replace_lora_layers(full_model.base_model.model.language_model, model.base_model.model)

from safetensors.torch import load_file
import torch

def normalize_layer_name(name):
    """
    Normalize layer names to account for structural differences between model and checkpoint.
    """
    return name.replace("language_model.model.", "model.").replace("base_model.model.", "")

# Define the path to the shards and the range of indices
base_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_2048_captions_noquant"
shard_prefix = "model-"
shard_suffix = ".safetensors"
start_index = 1  # Start index for shard files
end_index = 9    # End index for shard files

# Get the model's parameter dictionary
model_dict = dict(model.named_parameters())

# Function to load all layers from shard files into the model
def load_shards_into_model(model, base_path, shard_prefix, shard_suffix, start_index, end_index):
    for i in range(start_index, end_index + 1):
        shard_path = f"{base_path}/{shard_prefix}{i:05d}-of-{end_index:05d}{shard_suffix}"
        print(f"Loading shard: {shard_path}")
        
        # Load the state dictionary from the shard
        state_dict = load_file(shard_path)
        
        # Assign weights to the model's parameters
        for key, value in state_dict.items():
            model_key = f"base_model.model.{key}"  # Adjust based on your model's structure
            if model_key in model_dict:
                print(f"Loading weights for: {model_key}")
                model_dict[model_key].data.copy_(value)
            else:
                print(f"Key not found in model: {model_key}")

# Function to compare and replace all layers from the checkpoint model to the full model
def compare_and_replace_all_layers(full_model, checkpoint_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # checkpoint_state_dict = {
    #     name.replace("base_model.model.", "base_model.model.language_model."): param
    #     for name, param in checkpoint_model.named_parameters()
    # }

    # Extract all parameters from the checkpoint model
    # checkpoint_state_dict = {
    #     name: param
    #     for name, param in checkpoint_model.named_parameters()
    # }
    
    # Extract all parameters from the full model
    # full_model_state_dict = {
    #     name: param
    #     for name, param in full_model.named_parameters()
    # }

    checkpoint_state_dict = {
        normalize_layer_name(name): param
        for name, param in checkpoint_model.named_parameters()
    }
    
    # Extract all parameters from the full model
    full_model_state_dict = {
        normalize_layer_name(name): param
        for name, param in full_model.named_parameters()
    }
    
    all_matched = True

    for name, full_model_param in full_model_state_dict.items():
        if name in checkpoint_state_dict:
            checkpoint_param = checkpoint_state_dict[name]
            
            # Ensure tensors are on the same device for comparison
            full_model_tensor = full_model_param.data.to(device)
            checkpoint_tensor = checkpoint_param.data.to(device)
            
            # Check if tensors match
            if torch.allclose(full_model_tensor, checkpoint_tensor, atol=1e-6):
                print(f"Matched: {name}")
            else:
                all_matched = False
                print(f"MISMATCH: {name}")
                print(f"Difference: {torch.abs(full_model_tensor - checkpoint_tensor).max().item()}")
                
                # Replace the parameter in the full_model with the one from the checkpoint
                full_model_param.data = checkpoint_tensor.clone()
        else:
            all_matched = False
            print(f"Missing in checkpoint: {name}")

    # Check for extra layers in the checkpoint
    for name in checkpoint_state_dict.keys():
        if name not in full_model_state_dict:
            all_matched = False
            print(f"Extra in checkpoint: {name}")

    if all_matched:
        print("All layers matched and replaced.")
    else:
        print("Some layers did not match.")

# Load all shards into the model

for name, param in model.named_parameters():
    print(name, param.requires_grad)

for name, param in full_model.named_parameters():
    print(name, param.requires_grad)

load_shards_into_model(model, base_path, shard_prefix, shard_suffix, start_index, end_index)

# Compare and replace layers between the full model and the checkpointed model
compare_and_replace_all_layers(full_model.base_model.model, model.base_model.model)


import torch.nn as nn
import torch

# Device setup
device = torch.device("cuda")  # Or "cuda" or "cpu" based on your environment

# # Define dimensions
# hidden_size_q_proj = 4096  # Dimensions for q_proj
# hidden_size_v_proj_in = 4096  # Input dimensions for v_proj
# hidden_size_v_proj_out = 1024  # Output dimensions for v_proj

# # Loop through the specified layers and redefine the q_proj and v_proj layers
# for layer_idx in [20, 21, 22, 24, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 39]:
#     # Set q_proj
#     model.base_model.model.model.layers[layer_idx].self_attn.q_proj = nn.Linear(
#         in_features=hidden_size_q_proj,
#         out_features=hidden_size_q_proj,
#         bias=False
#     ).to(device)

#     # Set v_proj
#     model.base_model.model.model.layers[layer_idx].self_attn.v_proj = nn.Linear(
#         in_features=hidden_size_v_proj_in,
#         out_features=hidden_size_v_proj_out,
#         bias=False
#     ).to(device)

#     # Reinitialize weights for q_proj and v_proj
#     nn.init.xavier_uniform_(model.base_model.model.model.layers[layer_idx].self_attn.q_proj.weight)
#     nn.init.xavier_uniform_(model.base_model.model.model.layers[layer_idx].self_attn.v_proj.weight)

# # Print confirmation
# print("Layers 20, 21, and 22 have been updated with new q_proj and v_proj configurations.")


# %%

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig
# import torch

# # Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Path to the LoRA adapter files
# adapter_json_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision/adapter_config.json"
# adapter_model_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision/adapter_model.safetensors"

# from peft import PeftModel, PeftConfig
# from transformers import AutoModelForCausalLM

# config = PeftConfig.from_pretrained("/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision")
# full_model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     # quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     # device_map="auto"
# ).to(device)

# model = PeftModel.from_pretrained(full_model, 
# "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision")
# # check if it's working
# model.print_trainable_parameters()

# from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel, PeftConfig
# import torch

# # Set the device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Path to the LoRA adapter files
# adapter_json_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision"
# adapter_model_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_1024_captions_noquant_vision/adapter_model.safetensors"

# # Load the adapter configuration
# peft_config = PeftConfig.from_pretrained(adapter_json_path)

# # Load the base model
# # base_model_name = peft_config.base_model_name_or_path
# full_model = MllamaForConditionalGeneration.from_pretrained(
#     model_id,
#     # quantization_config=bnb_config,
#     torch_dtype=torch.bfloat16,
#     # device_map="auto"
# ).to(device)

# # Load the LoRA adapter into the base model
# model = PeftModel.from_pretrained(
#     full_model,
#     adapter_model_path,
#     adapter_name="default",  # Adjust if you have multiple adapters
# )

# model.print_trainable_parameters()


# # Move the model to the desired device
# model = model.to(device)

# # Optional: Load the tokenizer
# print("LoRA model loaded successfully!")


# Move the model to the desired device
model = model.to(device)


# Optional: Load the tokenizer
processor = AutoProcessor.from_pretrained(model_id)

print("LoRA model loaded successfully!")


# Load tokenizer from the trained language model checkpoint
# tokenizer = AutoTokenizer.from_pretrained(language_model_checkpoint)

# Update the processor to use the new tokenizer
# processor.tokenizer = tokenizer

print("bbbbb")


import pandas as pd
from PIL import Image
import io

output_path = "extracted_tikz_codes_version_test_4.jsonl"

# # Import Metrics
from torchmetrics.image.kid import KernelInceptionDistance as KID
from torchmetrics.text import ExtendedEditDistance
from eed import TexEditDistance
# from dreamsim1 import DreamSim  # Ensure dreamsim is installed and imported
from crystalbleu1 import CrystalBLEU  # Ensure crystalbleu is installed and imported

# Define Metric Classes
# class TexEditDistance(ExtendedEditDistance):
#     def compute(self, preds, target):
#         return super().compute(preds, target)

class ImageSim:
    def compute(self, img1, img2):
        return torch.cosine_similarity(img1, img2).item()

class DreamSim:
    def __init__(self, model_name="ensemble"):
        self.model, self.processor = model, processor

    def compute(self, img1, img2):
        img1, img2 = self.processor(img1), self.processor(img2)
        with torch.no_grad():
            return 1 - self.model(img1, img2).item()  # Higher is better

# class CrystalBLEU:
#     def __init__(self, corpus, k=500, n=4):
#         self.corpus = corpus
#         self.k = k
#         self.n = n

#     def compute(self, references, hypotheses):
#         return CrystalBLEU(list_of_references=references, hypotheses=hypotheses)

# Initialize Metrics
kid_metric = KID()
tex_edit_distance = TexEditDistance()
# image_sim_metric = ImageSim()
# dreamsim_metric = DreamSim()
# corpus_bleu_metric = CrystalBLEU(corpus=0)  # Corpus from the test dataset

# corpus_bleu_metric = CrystalBLEU()  # Corpus from the test dataset

def generate_tikz_code(image, caption):


    # image_bytes = image['bytes']
    # image = Image.open(io.BytesIO(image_bytes))

    # text_query = (
    #     "This is a picture of a scientific figure <|image|>. "
    #     "Generate LaTeX code that draws this scientific figure using TikZ. "
    #     "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
    #     "Make sure the TikZ code accurately reflects the image details as much as possible. "
    #     "Make sure the Tikz code compiles with no errors and is syntactically correct"
    #     "Make sure the output is not more than 500 words"
    #     "Don't forget to include \\usepackage{tikz}! Return your result in a ```latex code block."
    # )
    
    text_query = (
        "This is a picture of a scientific figure <|image|> as well as its caption. "
        f"The caption reads: '{caption}'."
        "Generate LaTeX code that draws this scientific figure using TikZ. "
        "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
        "Don't forget to include \\usepackage{tikz}! I understand that this is a challenging task, so do your best. Return your result in a ```latex code block."
    )

    # text_query = (
    #     "This is a caption of a scientific figure"
    #     f"The caption reads: '{caption}'."
    #     "Generate LaTeX code that draws this scientific figure using TikZ. "
    #     "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
    #     "Don't forget to include \\usepackage{tikz}! I understand that this is a challenging task, so do your best. Return your result in a ```latex code block."
    # )

    # text_query = (
    #     "This is a picture of a scientific figure <|image|>"
    #     "Generate LaTeX code that draws this scientific figure using TikZ. "
    #     "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
    #     "Don't forget to include \\usepackage{tikz}! I understand that this is a challenging task, so do your best. Return your result in a ```latex code block."
    # )

    # text_query = "describe this image <|image|>"
    chat_template = [
        {"role": "system", "content": "You are a helpful AI that can generate tikz code from images."},
        {"role": "user", "content": text_query}
    ]

    # Apply the chat template
    input_text = processor.apply_chat_template(chat_template, add_generation_prompt=True)

    # Since the processor expects a list of images or a single image, wrap it in a list if necessary
    images = [image]  # Or just use 'image' if the processor can handle single image objects directly

    # Check if the number of image tokens in the text matches the number of provided images
    # This is important for models expecting specific tokens to align with image inputs
    total_image_tokens = input_text.count(processor.image_token)
    if total_image_tokens != len(images):
        print(f"Mismatch between image tokens ({total_image_tokens}) and images provided ({len(images)}).")

    # Prepare inputs for the model
    inputs = processor(images=images, text=input_text, return_tensors="pt").to(device)
    # inputs = processor(text=input_text, return_tensors="pt").to(device)

    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the assistant's response
    output = model.generate(**inputs, max_new_tokens=1000)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    return generated_text

# def generate_tikz_code(image, caption):
#     # Text query for TikZ code generation
#     text_query = (
#         "This is a picture of a scientific figure <|image|>. "
#         "Generate LaTeX code that draws this scientific figure using TikZ. "
#         "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
#         "Don't forget to include \\usepackage{tikz}! Return your result in a ```latex code block."
#     )

#     chat_template = [
#         {"role": "system", "content": "You are a helpful AI that can generate TikZ code from images."},
#         {"role": "user", "content": text_query}
#     ]

#     # Apply the chat template
#     input_text = processor.apply_chat_template(chat_template, add_generation_prompt=True)

#     # Wrap the image in a list if necessary
#     images = [image]

#     # Count the number of image tokens in the text
#     total_image_tokens = input_text.count(processor.image_token)
#     if total_image_tokens != len(images):
#         print(f"Mismatch between image tokens ({total_image_tokens}) and images provided ({len(images)}).")

#     # Prepare inputs for the model
#     inputs = processor(images=images, text=input_text, return_tensors="pt").to(device)

#     # Filter inputs for the `generate` method
#     filtered_inputs = {k: v for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

#     # Generate the assistant's response
#     output = model.generate(**filtered_inputs, max_new_tokens=1000)
#     generated_text = processor.decode(output[0], skip_special_tokens=True)

#     return generated_text


def extract_latex_code(full_output):
    start_marker = "```latex"
    end_marker = "```"
    alt_start_marker = r"\begin{tikzpicture}"
    alt_end_marker = r"\end{tikzpicture}"
    
    try:
        # Primary method: Look for "```latex" and "```" markers
        first_occurrence = full_output.index(start_marker)
        start = full_output.index(start_marker, first_occurrence + len(start_marker)) + len(start_marker)
        end = full_output.index(end_marker, start)
        latex_code = full_output[start:end].strip()
        
        # Return if successful extraction
        if latex_code:
            return latex_code
        else:
            raise ValueError("Empty LaTeX code extracted")
    
    except ValueError:
        # Alternative method: Look for "\begin{tikzpicture}" and "\end{tikzpicture}" markers
        try:
            start = full_output.index(alt_start_marker)
            end = full_output.index(alt_end_marker, start) + len(alt_end_marker)
            latex_code = full_output[start:end].strip()
            return latex_code if latex_code else "Error: Extracted LaTeX code is empty."
        
        except ValueError:
            # If neither method succeeds, return an error message
            return "Error: Could not locate LaTeX code markers."

import json
from datasets import load_dataset
from tqdm import tqdm

model = full_model

model = model.to(device)
# model = model.language_model

print(model.device)

# Load your merged captions from the JSON file
with open('final_merged_captions.json', 'r') as file:
    merged_captions = json.load(file)

# Load the original dataset
ds_train = load_dataset("nllg/datikz", split="train")

# Step 1: Filter the dataset to keep only entries with matching URIs
filtered_dataset = ds_train.filter(lambda x: x['uri'] in merged_captions)

# Step 2: Update captions in the filtered dataset
def update_caption(example):
    uri = example['uri']
    example['caption'] = merged_captions[uri][0]  # Update with the caption
    return example

# Apply the update function with a progress bar
updated_dataset = filtered_dataset.map(
    update_caption,
    num_proc=4  # Adjust based on CPU cores for parallel processing
)

# Print the length of the updated dataset
print(f"Length of the updated dataset: {len(updated_dataset)}")

df = updated_dataset
corpus = updated_dataset["code"]  # Extract the "code" column as a list

corpus_bleu_metric = CrystalBLEU(corpus=corpus)  # Corpus from the test dataset

image = df[0]['image']
code = df[0]['code']
caption = df[0]['caption']
generated_code = generate_tikz_code(image, caption)
print(generated_code)

print("########################################################")
# Extract the LaTeX code block
extracted_code = extract_latex_code(generated_code)
print(extracted_code)

import sys
import time

# Evaluation Loop without repetition
for i, example in enumerate(df, start=1):
    reference_code = example['code']
    image = example['image']
    caption = example['caption']
    print(f"Processing sample {i}/{len(df)} - Caption: {caption}")
    sys.stdout.flush()

    # Start processing the sample
    start_time = time.time()

    # Generate TikZ code and extract LaTeX code in one pass
    generated_code = generate_tikz_code(image, caption)
    extracted_code = extract_latex_code(generated_code)

    # Update metrics
    tex_edit_distance.update([extracted_code], [[reference_code]])
    print(f"TEX Edit Distance for sample {i}: {tex_edit_distance.sentence_eed[-1]}")
    sys.stdout.flush()

    corpus_bleu_metric.update(
        list_of_references=[[reference_code]],
        hypotheses=[extracted_code]
    )
    cbleu_score = corpus_bleu_metric.compute()
    print(f"CrystalBLEU Score for sample {i}: {cbleu_score}")
    sys.stdout.flush()

    # Save extracted TikZ code and related info to a JSONL file
    with open(output_path, "a") as file:
        json.dump({
            "sample_id": i,
            "caption": caption,
            "reference_code": reference_code,
            "extracted_code": extracted_code,
            "tex_edit_distance": tex_edit_distance.sentence_eed[-1].item(),
            "cbleu_score": cbleu_score
        }, file)
        file.write("\n")

    elapsed_time = time.time() - start_time
    print(f"Sample {i} processing time: {elapsed_time:.2f} seconds\n")
    sys.stdout.flush()

# Compute and print final scores
print("Final TEX Edit Distance:", tex_edit_distance.compute())
print("Final CrystalBLEU Score:", corpus_bleu_metric.compute())
sys.stdout.flush()