# Multimodal TikZ Code Generation from Captions and Images

Authors: Omar El Herraoui, Kareem Eleky, Ahmad Attia  
Institution: Mohamed Bin Zayed University of Artificial Intelligence, Abu Dhabi, UAE  
Emails: {omar.el-herraoui, kareem.elzeky, ahmad.attia}@mbzuai.ac.ae  
Group ID: G-06  

## Abstract

In this study, we address the challenge of multimodal TikZ code generation, aiming to enhance scientific visualization by integrating image and caption inputs. Using the DaTikZ dataset, we improved textual alignment by replacing the original captions with more descriptive versions generated via GPT-4o. Our approach combines LLaMA 3.2 with its native image encoder, DeepSeek 1.3B with SigLIP for detailed visual embedding extraction, and applying DNG-MCTS which is a variant of Monte Carlo tree search (MCTS) for inference refinement. Experiments reveal that incorporating captions significantly improves TikZ code generation quality.

## Contributions

- **Benchmarking and Fine-Tuning the Multimodal LLaMA 3.2 Model:** The newly released multimodal LLaMA 3.2 11B model was benchmarked and fine-tuned on the DeTikZify dataset.
- **Dataset Augmentation with Newly Generated Captions:** To overcome the limitations of legacy captions, we generated 50,000 descriptive captions using GPT-4o, tailored to support TikZ code generation.

## Repository Contents

- `train.py`: Script for training the model.
- `eval_normal.py`: Script for evaluating the model performance.
- `generate_captions.py`: Script for generating captions from images.

## Hardware Requirements

- **GPU:** NVIDIA A100 (or equivalent) with at least 40 GB of VRAM recommended for training and inference tasks.
- **Memory:** At least 80 GB of RAM recommended for handling large datasets and model parameters.

## Installation

Clone this repository and install the required dependencies:

```bash
conda env create -f environment.yml
conda activate detikzify-env
```

## Usage

## Training the Model

To train the model, you can use the corresponding bash script for the task (generate captions, training, evaluation):

For training

```bash
sbatch ./run_train.sh
```

For Evaluation

```bash
sbatch ./run_eval.sh
```

For Generating Captions

```bash
sbatch ./run_generate.sh
```

## Citation

If you use our code or approach in your research, please cite our paper.