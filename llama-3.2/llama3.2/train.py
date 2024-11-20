import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MllamaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments, DataCollatorForSeq2Seq, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
# from PIL import Image
# import pandas as pd
# import io
from bitsandbytes.optim import Adam8bit
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from accelerate import Accelerator
# from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

from huggingface_hub import login

bnb_config = BitsAndBytesConfig(
   load_in_8bit=True,
   bnb_8bit_quant_type="nf4",
   bnb_8bit_compute_dtype=torch.bfloat16,  # or torch.float16 if needed
   bnb_8bit_use_double_quant=True,
)

# login(token = "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")
# Define LoRA configuration
lora_config = LoraConfig(
   r=16,  # Rank of the decomposition, typically a small number (e.g., 8, 16)
   lora_alpha=32,  # Scaling factor for LoRA parameters
   target_modules=["q_proj", "v_proj"],  # Apply LoRA to specific model layers (e.g., Q, V projections in attention layers)
   lora_dropout=0.1,  # Dropout for LoRA layers
   bias="none"  # LoRA doesn’t update biases by default; change to "all" if needed
)

# Initialize distributed training
def init_distributed():
   # Set up environment variables for multi-node multi-GPU training
   if not dist.is_initialized():
       dist.init_process_group(backend="nccl")  # "nccl" is the preferred backend for GPUs
       torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))  # Set the GPU for each process
       print(f"Running on GPU {int(os.environ['LOCAL_RANK'])} with process group initialized.")

# Run this at the start of your script
# if torch.cuda.device_count() > 1:
#     print("omars cuda")
#     init_distributed()

# Load model and processor
model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"

device = torch.device("cuda")

model_kwargs = {
   "text_config": {
       "_attn_implementation_autoset": False,
       "_name_or_path": "",
       "add_cross_attention": False,
       "architectures": None,
       "bad_words_ids": None,
       "begin_suppress_tokens": None,
       "bos_token_id": 128000,
       "chunk_size_feed_forward": 0,
       "cross_attention_hidden_size": None,
       "cross_attention_layers": [
       3,
       8,
       13,
       18,
       23,
       28,
       33,
       38
       ],
       "decoder_start_token_id": None,
       "diversity_penalty": 0.0,
       "do_sample": False,
       "dropout": 0,
       "early_stopping": False,
       "encoder_no_repeat_ngram_size": 0,
       "eos_token_id": [
       128001,
       128008,
       128009
       ],
       "exponential_decay_length_penalty": None,
       "finetuning_task": None,
       "forced_bos_token_id": None,
       "forced_eos_token_id": None,
       "hidden_act": "silu",
       "hidden_size": 4096,
       "id2label": {
       "0": "LABEL_0",
       "1": "LABEL_1"
       },
       "initializer_range": 0.02,
       "intermediate_size": 14336,
       "is_decoder": False,
       "is_encoder_decoder": False,
       "label2id": {
       "LABEL_0": 0,
       "LABEL_1": 1
       },
       "length_penalty": 1.0,
       "max_length": 20,
       "max_position_embeddings": 131072,
       "min_length": 0,
       "model_type": "mllama_text_model",
       "no_repeat_ngram_size": 0,
       "num_attention_heads": 32,
       "num_beam_groups": 1,
       "num_beams": 1,
       "num_hidden_layers": 40,
       "num_key_value_heads": 8,
       "num_return_sequences": 1,
       "output_attentions": False,
       "output_hidden_states": False,
       "output_scores": False,
       "pad_token_id": 128004,
       "prefix": None,
       "problem_type": None,
       "pruned_heads": {},
       "remove_invalid_values": False,
       "repetition_penalty": 1.0,
       "return_dict": True,
       "return_dict_in_generate": False,
       "rms_norm_eps": 1e-05,
       "rope_scaling": {
       "factor": 8.0,
       "high_freq_factor": 4.0,
       "low_freq_factor": 1.0,
       "original_max_position_embeddings": 8192,
       "rope_type": "llama3"
       },
       "rope_theta": 500000.0,
       "sep_token_id": None,
       "suppress_tokens": None,
       "task_specific_params": None,
       "temperature": 1.0,
       "tf_legacy_loss": False,
       "tie_encoder_decoder": False,
       "tie_word_embeddings": False,
       "tokenizer_class": None,
       "top_k": 50,
       "top_p": 1.0,
       "torch_dtype": "bfloat16",
       "torchscript": False,
       "typical_p": 1.0,
       "use_bfloat16": False,
       "use_cache": False,
       "vocab_size": 128256
   },
}

model = MllamaForConditionalGeneration.from_pretrained(
   model_id,
#    quantization_config=bnb_config,
   **model_kwargs
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")

model.config.gradient_checkpointing = True

# for name, param in model.named_parameters():
#     if any(layer in name for layer in lora_config.target_modules):
#         param.requires_grad = False
#     else:
#         param.requires_grad = False


# Flag to identify LoRA parameters in the language model only
# for name, param in model.named_parameters():
#    # Check if parameter is in the language model and part of LoRA-targeted modules, and is a floating-point tensor
#    if "language_model" in name and param.dtype.is_floating_point:
#        param.requires_grad = True  # Enable gradient for LoRA parameters in the language model
#    else:
#        param.requires_grad = False  # Disable gradient for all other parameters

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# optimizer = Adam8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

import torch.optim as optim

# Initialize the optimizer with Adam and filter the parameters that require gradients
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

# model.config.text_config["use_cache"] = False
print("#############################")
# print(model.config.text_config["use_cache"])
# model.config.attention_type = "memory_efficient"  # If supported

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# model = accelerator.prepare(model)

# model.parallelize()
# model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

processor = AutoProcessor.from_pretrained(model_id)

# Define prompt and message structure
system_message = "You are a helpful assistant that generates TikZ code for scientific figures using the figures and their captions"
# prompt =  ("This is a picture of a scientific figure <|image|> as well as its caption. "f"The caption reads: '{caption}'." )


# Dataset formatter
class TikZDataset(Dataset):
   def __init__(self, dataset):
       self.dataset = dataset
       self.processor = processor
       self.max_tiles = 8  # Max number of tiles per image

   def __len__(self):
       return len(self.dataset)

   def __getitem__(self, idx):
       row = self.dataset[idx]
       image = row['image']
       caption = row['caption']
       code = row['code']

       prompt = (
           "This is a picture of a scientific figure <|image|> as well as its caption. "
           f"The caption reads: '{caption}'."
       )

       messages = [
           {"role": "system", "content": [{"type": "text", "text": system_message}]},
           {"role": "user", "content": [{"type": "text", "text": prompt}]},
           {"role": "assistant", "content": [{"type": "text", "text": code}]}
       ]

       return {"messages": messages, "image": image}

# Load dataset and apply formatting

ds_train = load_dataset("nllg/datikz", split="train")

import json
from datasets import load_dataset
from tqdm import tqdm

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

ds_train = updated_dataset
formatted_dataset = TikZDataset(ds_train)


# Collate function
def collate_fn(examples):
   # Format messages without the image token for token counting
   texts_with_code = [
       processor.apply_chat_template(
           [
               {"role": "system", "content": [{"type": "text", "text": system_message}]},
               {"role": "assistant", "content": [{"type": "text", "text": example["messages"][-1]["content"][0]["text"]}]}
           ],
           tokenize=False
       )
       for example in examples
   ]

   images = [example["image"] for example in examples]  # Extract images separately

   # Process text and image separately without max_length to get original token counts
   text_batch_no_trunc = processor(
       text=texts_with_code,
       return_tensors="pt",
       padding=True, 
       truncation=False  # No truncation to count original text tokens
   )

#    image_batch_no_trunc = processor(
#        text=prompt,
#        images=images,
#        return_tensors="pt",
#        padding=True, 
#        truncation=False  # No truncation to count original text tokens
#    )

   # Format messages with the image token included in the prompt for actual processing
   texts_with_image_token = [
       processor.apply_chat_template(example["messages"], tokenize=False) for example in examples
   ]

   combined_batch_no_trunc = processor(
       text=texts_with_image_token,
       images=images,
       return_tensors="pt",
       padding=True,
       truncation=False  # No truncation to count original combined tokens
   )

   text_token_counts = [len(input_ids) for input_ids in text_batch_no_trunc["input_ids"]]

#    image_token_counts = [len(input_ids) for input_ids in image_batch_no_trunc["input_ids"]]


   print("Number of tokens in text before truncation:", text_token_counts)
   combined_token_counts = [len(input_ids) for input_ids in combined_batch_no_trunc["input_ids"]]
   print("Number of tokens in combined text and images before truncation:", combined_token_counts)

   # Calculate image token counts by subtracting text tokens from combined tokens
#    image_token_counts = [combined - text for combined, text in zip(combined_token_counts, text_token_counts)]
#    print("Number of tokens in image before truncation:", image_token_counts)

   # Process batch with max_length=512 to apply truncation
   batch = processor(
       text=texts_with_image_token,
       images=images,
       return_tensors="pt",
       padding=True,
       max_length=1024,  # Apply max length
       truncation=True  # Enable truncation to ensure no inputs exceed 512 tokens
   )

   labels = batch["input_ids"].clone()
   labels[labels == processor.tokenizer.pad_token_id] = -100

   # Mask out image tokens
   image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
   for image_token_id in image_tokens:
       labels[labels == image_token_id] = -100
   batch["labels"] = labels

   return batch


# Collate function
# def collate_fn(examples):
#     # Format messages
#     # print("Examples received in collate_fn:", examples)
#     texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
#     images = [example["image"] for example in examples]  # Extract images separately

#     # Process batch with max_length=512 and padding enabled
#     batch = processor(
#         text=texts, 
#         images=images, 
#         return_tensors="pt", 
#         padding=True, 
#         max_length=512, 
#         truncation=True  # Enable truncation to ensure no inputs exceed 512 tokens
#     )

#     labels = batch["input_ids"].clone()
#     labels[labels == processor.tokenizer.pad_token_id] = -100

#     # Mask out image tokens
#     image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
#     for image_token_id in image_tokens:
#         labels[labels == image_token_id] = -100
#     batch["labels"] = labels

#     return batch


# Define a custom trainer to override the data loader
class CustomTrainer(Trainer):
   def get_train_dataloader(self):
       return DataLoader(
           self.train_dataset,
           batch_size=1,
           collate_fn=collate_fn,
           shuffle=True
       )

from tqdm import tqdm
import torch

# Define training loop
# Training loop with logging
def train_model(model, train_dataloader, optimizer, epochs=3, log_interval=100, log_file="training_log_1024_captions_noquant_vision.txt"):
   model.train()
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)

   with open(log_file, "w") as log:
       for epoch in range(epochs):
           log.write(f"Epoch {epoch + 1}/{epochs}\n")
           log.flush()
           print(f"Epoch {epoch + 1}/{epochs}")
           epoch_loss = 0.0

           avg_epoch_loss = epoch_loss / len(train_dataloader)
           epoch_msg = f"Average epoch loss: {avg_epoch_loss:.4f}\n"
           print(epoch_msg)
           log.write(epoch_msg + "\n")
           log.flush()  # Immediately write to disk

            # Save final model checkpoint
           final_model_path = "final_model_checkpoint_1024_captions_noquant_vision"
           model.save_pretrained(final_model_path)
           log.write(f"Model saved to {final_model_path}\n")
           log.flush()  # Ensure final message is written
           print(f"Model saved to {final_model_path}")

           for i, batch in enumerate(tqdm(train_dataloader, desc="Training")):
               batch = {k: v.to(device) for k, v in batch.items()}

               # Forward pass
               outputs = model(**batch)
               loss = outputs.loss

               # Backward pass and optimization
               optimizer.zero_grad()
               loss.backward()
               optimizer.step()

               epoch_loss += loss.item()

               # Log loss every 100 iterations
               if (i + 1) % log_interval == 0:
                   avg_loss = epoch_loss / (i + 1)
                   log_msg = f"Iteration {i + 1}, Loss: {avg_loss:.4f}"
                   print(log_msg)
                   log.write(log_msg + "\n")
                   log.flush()  # Immediately write to disk
                   
                   final_model_path = "final_model_checkpoint_1024_captions_noquant_vision"
                   model.save_pretrained(final_model_path)
                   log.write(f"Model saved to {final_model_path}\n")
                   log.flush()  # Ensure final message is written
                   print(f"Model saved to {final_model_path}")

               # Clean up to free memory
               del loss, outputs

    #    avg_epoch_loss = epoch_loss / len(train_dataloader)
    #    epoch_msg = f"Average epoch loss: {avg_epoch_loss:.4f}\n"
    #    print(epoch_msg)
    #    log.write(epoch_msg + "\n")
    #    log.flush()  # Immediately write to disk

    #     # Save final model checkpoint
    #    final_model_path = "final_model_checkpoint_2048"
    #    model.save_pretrained(final_model_path)
    #    log.write(f"Model saved to {final_model_path}\n")
    #    log.flush()  # Ensure final message is written
    #    print(f"Model saved to {final_model_path}")


# Load your dataset
train_dataloader = DataLoader(formatted_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

# Start training with the custom loop
train_model(model, train_dataloader, optimizer, epochs=3)

avg_epoch_loss = epoch_loss / len(train_dataloader)
epoch_msg = f"Average epoch loss: {avg_epoch_loss:.4f}\n"
print(epoch_msg)
log.write(epoch_msg + "\n")
log.flush()  # Immediately write to disk

# Save final model checkpoint
final_model_path = "final_model_checkpoint_1024_captions_noquant_vision"
model.save_pretrained(final_model_path)
log.write(f"Model saved to {final_model_path}\n")
log.flush()  # Ensure final message is written
print(f"Model saved to {final_model_path}")



