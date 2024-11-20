# import torch
# from datetime import date
# from PIL import Image, ImageTk
# from transformers import MllamaForConditionalGeneration, AutoProcessor
# import tkinter as tk
# from tkinter import filedialog, ttk, messagebox
# import logging
# import json
# import os
# from peft import PeftModel
# from transformers import MllamaForCausalLM


# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
# # # Get today's date
# date_string: str = date.today().strftime("%d %b %Y")

# final_model_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_512_quant"

# # Load the base model and apply LoRA weights
# model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"
# # model = MllamaForConditionalGeneration.from_pretrained(
# #     model_id,
# #     torch_dtype=torch.bfloat16,
# #     device_map="auto",
# # ).to(device)

# # Define the path to your model checkpoint
# final_model_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_512_quant"

# # Load the model directly from the saved checkpoint
# model = MllamaForCausalLM.from_pretrained(
#     final_model_path,
#     torch_dtype=torch.bfloat16,  # Set dtype if required by your model
#     device_map="auto"            # Automatically allocate model layers to available devices
# ).to(device)

# print(f"Model loaded from {final_model_path} without LoRA weights.")

##############################################################################

import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor, MllamaForCausalLM, BitsAndBytesConfig
from datetime import date

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Quantization configuration for both models
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16,  # or torch.float16 if needed
    bnb_8bit_use_double_quant=True,
)

# Define model paths
model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"
language_model_checkpoint = "/scratch/oe2015/ai_project/DeTikZify/detikzify/evaluate/final_model_checkpoint_512_quant"

# Step 1: Load the full pretrained model with both vision and language components
full_model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Step 2: Load only the language model part from your trained checkpoint
trained_language_model = MllamaForCausalLM.from_pretrained(
    language_model_checkpoint,
    # quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    # device_map={"": device}  # Explicitly set the device for the language model
)

for name, param in trained_language_model.named_parameters():
    print(name, param.requires_grad)

print("########################################################################")

for name, param in full_model.named_parameters():
    print(name, param.requires_grad)

# Step 3: Replace the language model component in the full model with the trained language model
full_model.language_model.load_state_dict(trained_language_model.state_dict())

print("Loaded pretrained vision model and replaced language model with trained weights.")

model = full_model
# Load LoRA configuration
# model = PeftModel.from_pretrained(model, final_model_path)
# print(f"Model loaded from {final_model_path}")

print("bbbbb")

processor = AutoProcessor.from_pretrained(model_id)

from transformers import AutoTokenizer

# Load tokenizer from the trained language model checkpoint
# tokenizer = AutoTokenizer.from_pretrained(language_model_checkpoint)

# Update the processor to use the new tokenizer
# processor.tokenizer = tokenizer

print("bbbbb")

from PIL import Image

from datasets import load_dataset
# # full dataset
ds_train = load_dataset("nllg/datikz-v2", split="train")
# print(ds_test[0]['code'])
print(ds_train)

import pandas as pd
from PIL import Image
import io


# Path to your Parquet file
file_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/test-00000-of-00001-e9214e6870e54ffa.parquet"
output_path = "extracted_tikz_codes_version_test.jsonl"

# Load the Parquet file into a DataFrame
df = pd.read_parquet(file_path)

# Display the first few rows to check the data
# image = df['image'].iloc[0]
image_bytes = df['image'].iloc[0]['bytes']
image = Image.open(io.BytesIO(image_bytes))
code = df['code'].iloc[0]
caption = df['caption'].iloc[0]
# print(code)
# print(caption)

# If the image data is in bytes (common in DataFrames storing raw image data)
# image = ds_test[0]['image']
# Assuming 'processor' and 'model' are already initialized and configured

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
corpus_bleu_metric = CrystalBLEU(corpus=[ex.code for ex in df.itertuples()])  # Corpus from the test dataset
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
    
    # text_query = (
    #     "This is a picture of a scientific figure <|image|> as well as its caption. "
    #     f"The caption reads: '{caption}'."
    #     "Generate LaTeX code that draws this scientific figure using TikZ. "
    #     "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
    #     "Don't forget to include \\usepackage{tikz}! I understand that this is a challenging task, so do your best. Return your result in a ```latex code block."
    # )

    text_query = (
        "This is a picture of a scientific figure <|image|>"
        "Generate LaTeX code that draws this scientific figure using TikZ. "
        "Ensure that the LaTeX code is self-contained and does not require any packages except TikZ-related imports. "
        "Don't forget to include \\usepackage{tikz}! I understand that this is a challenging task, so do your best. Return your result in a ```latex code block."
    )

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

    # Generate the assistant's response
    output = model.generate(**inputs, max_new_tokens=1000)
    generated_text = processor.decode(output[0], skip_special_tokens=True)

    return generated_text

# def extract_latex_code(full_output):
#     start_marker = "```latex"
#     end_marker = "```"
    
#     try:
#         # Find the first occurrence of the start marker
#         first_occurrence = full_output.index(start_marker)
        
#         # Find the second occurrence of the start marker after the first
#         start = full_output.index(start_marker, first_occurrence + len(start_marker)) + len(start_marker)
        
#         # Find the end marker after the second start marker
#         end = full_output.index(end_marker, start)
        
#         # Extract and clean up the LaTeX code
#         latex_code = full_output[start:end].strip()
        
#         # Validate extracted code
#         if latex_code:
#             return latex_code
#         else:
#             return "Error: Extracted LaTeX code is empty. Retry generating the code."
    
#     except ValueError:
#         # If markers are missing, return an error message
#         return "Error: Could not locate LaTeX code markers. Retry generating the code."

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

# Load your merged captions from the JSON file
with open('final_merged_captions_test.json', 'r') as file:
    merged_captions = json.load(file)

# Load the original dataset
ds_train = load_dataset("nllg/datikz", split="test")

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