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
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # or torch.float16 if needed
    bnb_4bit_use_double_quant=True,
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
    quantization_config=bnb_config,
    **model_kwargs
).to("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
accelerator = Accelerator(mixed_precision="bf16" if torch.cuda.is_bf16_supported() else "fp16")

model = get_peft_model(model, lora_config)

model.config.gradient_checkpointing = True

# for name, param in model.named_parameters():
#     if any(layer in name for layer in lora_config.target_modules) and param.dtype.is_floating_point:
#         param.requires_grad = True
#     else:
#         param.requires_grad = False

# Flag to identify LoRA parameters in the language model only
model.vision_model.eval()
for name, param in model.named_parameters():
    # Check if parameter is in the language model and part of LoRA-targeted modules, and is a floating-point tensor
    if "base_model.model.language_model.model.layers.2.self_attn.q_proj.lora_A.default.weight" in name and any(layer in name for layer in lora_config.target_modules) and param.dtype.is_floating_point:
        param.requires_grad = True  # Enable gradient for LoRA parameters in the language model
    else:
        param.requires_grad = False  # Disable gradient for all other parameters

model.print_trainable_parameters()

optimizer = Adam8bit(filter(lambda p: p.requires_grad, model.language_model.parameters()), lr=1e-5)

# model.config.text_config["use_cache"] = False
print("#############################")
# print(model.config.text_config["use_cache"])
# model.config.attention_type = "memory_efficient"  # If supported

for name, param in model.named_parameters():
    print(name, param.requires_grad)

# model = accelerator.prepare(model)

# model.parallelize()
# model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

processor = AutoProcessor.from_pretrained(model_id)

# Define prompt and message structure
system_message = "You are a helpful assistant that generates TikZ code for scientific figures."
prompt = """This is a picture of a scientific figure <|image|>."""

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

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_message}]},
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": code}]}
        ]

        return {"messages": messages, "image": image}

# Load dataset and apply formatting
ds_train = load_dataset("nllg/datikz-v2", split="train")
formatted_dataset = TikZDataset(ds_train)

# Collate function
def collate_fn(examples):
    # Format messages
    # print("Examples received in collate_fn:", examples)
    texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
    images = [example["image"] for example in examples]  # Extract images separately

    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask out image tokens
    image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
    batch["labels"] = labels

    return batch

# Define a custom trainer to override the data loader
class CustomTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=True
        )

train_dataloader = DataLoader(formatted_dataset, batch_size=16, collate_fn=collate_fn)

from tqdm import tqdm
import torch

# Define training loop
def train_model(model, train_dataloader, optimizer, epochs=3):
    model.train()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, desc="Training"):
            # Move inputs to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            # epoch_loss += loss.item()

            # Delete loss and outputs to free memory
            del loss, outputs
            torch.cuda.empty_cache()  # Free up GPU memory

        # avg_epoch_loss = epoch_loss / len(train_dataloader)
        # print(f"Average epoch loss: {avg_epoch_loss}")

# Load your dataset
train_dataloader = DataLoader(formatted_dataset, batch_size=1, collate_fn=collate_fn, shuffle=True)

# Start training with the custom loop
train_model(model.language_model, train_dataloader, optimizer, epochs=3)

args = SFTConfig(
    output_dir="fine-tuned-visionllama-unsloth",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    # optim="adamw_torch_4bit",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    report_to="tensorboard",
    dataset_kwargs={"skip_prepare_dataset": True},
    gradient_checkpointing_kwargs={'use_reentrant':False},
    torch_empty_cache_steps=1  # Clear GPU cache every 10 steps
)

args.remove_unused_columns = False
args.dataset_kwargs = {"skip_prepare_dataset": True}

# trainer = SFTTrainer(
#     model=model.language_model.to("cuda"),
#     args=args,
#     train_dataset=formatted_dataset,
#     data_collator=collate_fn,
#     optimizers=(optimizer, None),          
#     processing_class=processor.tokenizer,
#     peft_config=lora_config
# )

# Initialize CustomTrainer with the custom data loader
# trainer = CustomTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=formatted_dataset,
#     data_collator=collate_fn,  # Use the custom collate function here
# )

# trainer = FastLanguageModel.get_peft_model(
#     trainer,
#     r=8,
#     target_modules=["q_proj", "v_proj"],
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     use_gradient_checkpointing=True,
#     random_state=3407,
#     use_rslora=False,
#     loftq_config=None,
# )

# Start training
# trainer.train()
