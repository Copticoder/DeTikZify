import torch
from torch.utils.data import Dataset, DataLoader
from transformers import MllamaForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from PIL import Image
import pandas as pd
import io

from trl import SFTConfig

args = SFTConfig(
    output_dir="fine-tuned-visionllama-unsloth",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    logging_steps=5,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    push_to_hub=True,
    report_to="tensorboard",
    dataset_kwargs={"skip_prepare_dataset": True},
)

# Load your model and processor
model_id = "mylesgoose/Llama-3.2-11B-Vision-Instruct"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

from PIL import Image

from datasets import load_dataset
# # full dataset
ds_train = load_dataset("nllg/datikz-v2", split="train")
# print(ds_test[0]['code'])
print(ds_train)

# Load and preprocess your training dataset
class TikZDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.processor = processor
        self.max_tiles = 8  # Define max number of tiles per image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        image = row['image']
        # image = Image.open(io.BytesIO(image_bytes))
        caption = row['caption']
        code = row['code']

        caption_with_image_token = f"This is a picture of a scientific figure <|image|>. {caption}"
        # Preprocess image and text
        # Compute aspect_ratio_id based on image dimensions
        aspect_ratio = image.size[0] / image.size[1]  # width / height
   
        # Map aspect ratios to IDs: 1 = square, 2 = portrait, 3 = landscape
        if abs(aspect_ratio - 1) < 0.1:
            aspect_ratio_id = 1  # Square
        elif aspect_ratio < 1:
            aspect_ratio_id = 2  # Portrait
        else:
            aspect_ratio_id = 3  # Landscape

        # Determine the number of active tiles based on the aspect ratio
        num_tiles = min(int(self.max_tiles * aspect_ratio), self.max_tiles)
        
        # Generate aspect ratio mask: 1 for active tiles (image content), 0 for padding tiles
        aspect_ratio_mask = torch.cat([
            torch.ones(num_tiles),  # Active tiles
            torch.zeros(self.max_tiles - num_tiles)  # Padding
        ]).unsqueeze(0)  # Shape: (1, max_tiles)

        # Preprocess inputs and labels with padding and truncation
        inputs = self.processor(
            images=image,
            text=caption_with_image_token,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512  # Adjust max length as needed
        )

        labels = self.processor(
            text=code,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512  # Adjust max length as needed
        )["input_ids"]

        labels[labels == processor.tokenizer.pad_token_id] = -100  # Ignore pad tokens in loss

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "aspect_ratio_ids": torch.tensor([aspect_ratio_id], dtype=torch.long),
            "aspect_ratio_mask": aspect_ratio_mask,
            "labels": labels.squeeze(0)
        }

# Load data from your Parquet file
# file_path = "/scratch/oe2015/ai_project/DeTikZify/detikzify/train.parquet"
# df = pd.read_parquet(file_path)
from transformers import DataCollatorWithPadding

dataset = TikZDataset(ds_train)
# Use DataCollatorForSeq2Seq to handle padding
data_collator = DataCollatorForSeq2Seq(
    tokenizer=processor.tokenizer,
    model=model,
    padding="longest"
)
# Create DataLoader
# train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./fine_tuned_llama",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    fp16=True,
    logging_dir='./logs',
    logging_steps=500,
    save_steps=1000,
    save_total_limit=2,
    report_to="none"
)


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator  # Use the data collator with padding
)

# Start training
trainer.train()
