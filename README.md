# Phi-2 GRPO Fine-tuning

Fine-tune Microsoft Phi-2 model using GRPO (Group Relative Policy Optimization) on the OpenAssistant/oasst1 dataset with QLoRA 4-bit quantization.

## Overview

This project provides a complete pipeline for fine-tuning the Phi-2 model using:
- **GRPO Trainer**: Group Relative Policy Optimization from TRL
- **QLoRA**: Efficient fine-tuning with 4-bit quantization
- **OpenAssistant Dataset**: High-quality conversational data
- **Gradio Interface**: Easy-to-use chat interface

## Project Structure

```
phi2-grpo-finetune/
├── config/
│   └── training_config.yaml      # Training hyperparameters
├── scripts/
│   ├── train.py                   # Main training script
│   ├── prepare_dataset.py         # Dataset preprocessing
│   └── inference.py               # Inference script
├── notebooks/
│   └── train_colab.ipynb          # Google Colab training notebook
├── app.py                         # Gradio chat interface
├── requirements.txt               # Python dependencies
└── space_config/                  # Hugging Face Space config
    ├── README.md
    └── app.py
```

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd phi2-grpo-finetune
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Authenticate with Hugging Face (optional, for pushing models):
```bash
huggingface-cli login
```

## Quick Start

### 1. Prepare Dataset

First, prepare the OpenAssistant dataset:

```bash
python scripts/prepare_dataset.py \
    --dataset_name "OpenAssistant/oasst1" \
    --language "en" \
    --model_name "microsoft/phi-2" \
    --max_length 512 \
    --output_dir "./data" \
    --val_size 0.1
```

### 2. Train the Model

Train using the configuration file:

```bash
python scripts/train.py --config config/training_config.yaml
```

Or with custom parameters:

```bash
python scripts/train.py \
    --config config/training_config.yaml \
    --resume_from_checkpoint ./outputs/checkpoint-500
```

### 3. Test the Model

Run inference to test your fine-tuned model:

```bash
python scripts/inference.py \
    --model_path ./outputs \
    --interactive
```

Or test with a single prompt:

```bash
python scripts/inference.py \
    --model_path ./outputs \
    --prompt "What is machine learning?"
```

### 4. Launch Gradio Interface

Start the chat interface:

```bash
python app.py
```

Then open your browser to `http://localhost:7860`

## Google Colab Training

For training on Google Colab with a T4 GPU, use the provided notebook:

1. Open `notebooks/train_colab.ipynb` in Google Colab
2. Make sure you're using a GPU runtime (Runtime → Change runtime type → GPU → T4)
3. Run all cells in order
4. Optionally mount Google Drive to save checkpoints

The notebook is optimized for T4 GPU with:
- Batch size: 2
- Gradient accumulation: 16 steps
- Max sequence length: 512 tokens
- Mixed precision training (fp16)

## Configuration

Edit `config/training_config.yaml` to customize training parameters:

### Key Parameters

- **Model**: `microsoft/phi-2`
- **Dataset**: `OpenAssistant/oasst1`
- **Learning Rate**: 1e-4
- **Batch Size**: 4-8 (adjust based on GPU memory)
- **LoRA r**: 16
- **LoRA alpha**: 32
- **Max Sequence Length**: 512-1024

### T4 GPU Recommendations

For T4 GPU (16GB VRAM):
- Batch size: 2-4
- Gradient accumulation: 8-16 steps
- Max sequence length: 512-768 tokens
- Enable gradient checkpointing

## Dataset Format

The dataset preparation script extracts conversation pairs from oasst1's tree structure and formats them as:

```
Human: {user_prompt}

Assistant: {assistant_response}
```

## Training Process

1. **Dataset Loading**: Loads and preprocesses OpenAssistant/oasst1
2. **Model Setup**: Loads Phi-2 with 4-bit quantization
3. **QLoRA**: Applies LoRA adapters for efficient fine-tuning
4. **GRPO Training**: Trains using Group Relative Policy Optimization
5. **Checkpointing**: Saves checkpoints during training
6. **Evaluation**: Evaluates on validation set

## Model Output

After training, the model is saved to `./outputs/` with:
- LoRA adapter weights
- Tokenizer files
- Training metrics
- Configuration files

## Hugging Face Space Deployment

To deploy your model as a Hugging Face Space:

1. Push your model to Hugging Face Hub:
```bash
# In your training script or notebook
trainer.push_to_hub(hub_model_id="your-username/phi2-grpo-finetuned")
```

2. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Create new Space
   - Select "Gradio" SDK

3. Copy files from `space_config/` to your Space repository:
   - `app.py`
   - `README.md`
   - `requirements.txt`

4. Update `app.py` to point to your model:
```python
MODEL_PATH = "your-username/phi2-grpo-finetuned"
```

5. Your Space will automatically build and deploy!

## Troubleshooting

### Out of Memory Errors

- Reduce batch size
- Increase gradient accumulation steps
- Reduce max sequence length
- Enable gradient checkpointing

### Slow Training

- Use mixed precision (fp16/bf16)
- Enable gradient checkpointing
- Use paged optimizer (paged_adamw_8bit)
- Reduce max sequence length

### Dataset Issues

- Ensure you've run `prepare_dataset.py` first
- Check that the dataset path is correct
- Verify language filter is set correctly

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ GPU memory for training
- 8GB+ RAM

## Dependencies

See `requirements.txt` for full list. Key packages:
- `torch>=2.0.0`
- `transformers>=4.35.0`
- `trl>=0.7.0` (for GRPO)
- `peft>=0.6.0` (for LoRA)
- `bitsandbytes>=0.41.0` (for quantization)
- `gradio>=4.0.0` (for UI)

## References

- [Phi-2 Model](https://huggingface.co/microsoft/phi-2)
- [OpenAssistant Dataset](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [GRPO Trainer Documentation](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

## License

This project follows the licenses of the underlying models and datasets:
- Phi-2: [MIT License](https://huggingface.co/microsoft/phi-2)
- OpenAssistant: [Apache 2.0](https://huggingface.co/datasets/OpenAssistant/oasst1)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Microsoft for the Phi-2 model
- OpenAssistant for the dataset
- Hugging Face for TRL and PEFT libraries
