---
title: Phi-2 GRPO Fine-tuned Chat
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.0.0
app_file: app.py
pinned: false
license: mit
---

# Phi-2 GRPO Fine-tuned Chat

Chat interface for a fine-tuned Phi-2 model using GRPO (Group Relative Policy Optimization) training on OpenAssistant dataset.

## ⚠️ Hardware Requirements

This Space requires a **GPU with at least 14GB VRAM** to run properly. 

- **Recommended**: T4, A10G, or better
- **Free tier**: May experience memory issues or slow performance
- **Upgrade to GPU**: Go to Space settings → Hardware → Select GPU

If you see memory errors, the Space needs upgraded hardware.

## Model

This Space uses a fine-tuned version of [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2) that has been trained using:
- **GRPO**: Group Relative Policy Optimization
- **QLoRA**: 4-bit quantization with LoRA adapters
- **Dataset**: OpenAssistant/oasst1

## Usage

Simply type your message in the chat interface and the model will respond. You can adjust generation parameters:
- **Max Tokens**: Maximum length of the response
- **Temperature**: Controls randomness (lower = more deterministic)
- **Top-p**: Nucleus sampling parameter

## Model Details

- **Base Model**: microsoft/phi-2 (2.7B parameters)
- **Fine-tuning Method**: GRPO with QLoRA
- **Training Data**: OpenAssistant/oasst1 (English conversations)
- **Quantization**: 4-bit NF4
- **Memory Usage**: ~5-6GB GPU VRAM (with 4-bit quantization)

## Limitations

- The model may generate incorrect or biased information
- Responses are limited by the training data and model size
- May not handle all types of queries effectively
- Requires GPU for reasonable inference speed

## Citation

If you use this model, please cite:
- Phi-2: Microsoft
- OpenAssistant Dataset: OpenAssistant team
- GRPO: TRL library from Hugging Face
