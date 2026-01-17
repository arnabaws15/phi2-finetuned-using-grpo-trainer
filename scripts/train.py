"""
Training script for Phi-2 fine-tuning using GRPO (Group Relative Policy Optimization).
Uses QLoRA with 4-bit quantization for efficient training.
"""

import os
import yaml
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
import bitsandbytes as bnb


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: dict):
    """Setup model with 4-bit quantization and QLoRA adapters."""
    model_name = config["model"]["name"]
    trust_remote_code = config["model"].get("trust_remote_code", True)
    
    # Load tokenizer
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Setup 4-bit quantization
    quant_config = config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_config["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, quant_config["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type=quant_config["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_config["bnb_4bit_use_double_quant"],
    )
    
    # Load model with quantization
    # Use float16 (T4 GPU doesn't support bfloat16)
    print(f"Loading model: {model_name} with 4-bit quantization (fp16)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.float16,  # T4 GPU compatibility
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Setup LoRA
    lora_config = config["lora"]
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        lora_dropout=lora_config["lora_dropout"],
        bias=lora_config["bias"],
        task_type=lora_config["task_type"],
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


def load_dataset(config: dict):
    """Load and prepare dataset."""
    dataset_config = config["dataset"]
    
    # Check if preprocessed dataset exists
    data_dir = Path("./data")
    if data_dir.exists():
        print(f"Loading preprocessed dataset from {data_dir}")
        dataset = load_from_disk(str(data_dir))
    else:
        print("Preprocessed dataset not found. Please run prepare_dataset.py first.")
        raise FileNotFoundError(f"Dataset not found at {data_dir}")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train Phi-2 with GRPO")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/training_config.yaml",
        help="Path to training configuration file",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load dataset
    dataset = load_dataset(config)
    
    # Setup training arguments
    training_config = config["training"]
    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        lr_scheduler_type=training_config["lr_scheduler_type"],
        warmup_steps=training_config["warmup_steps"],
        max_steps=training_config["max_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        save_total_limit=training_config["save_total_limit"],
        eval_strategy=training_config.get("eval_strategy") or training_config.get("evaluation_strategy", "steps"),
        save_strategy=training_config["save_strategy"],
        load_best_model_at_end=training_config["load_best_model_at_end"],
        metric_for_best_model=training_config["metric_for_best_model"],
        greater_is_better=training_config["greater_is_better"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        optim=training_config["optim"],
        report_to=training_config["report_to"],
        remove_unused_columns=False,
    )
    
    # Setup GRPO config
    grpo_config = config.get("grpo", {})
    grpo_trainer_config = GRPOConfig(
        output_dir=training_config["output_dir"],
        learning_rate=training_config["learning_rate"],
        num_train_epochs=training_config["num_train_epochs"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        per_device_eval_batch_size=training_config["per_device_eval_batch_size"],  # Must be divisible by 8 for GRPO
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        max_steps=training_config["max_steps"],
        logging_steps=training_config["logging_steps"],
        save_steps=training_config["save_steps"],
        eval_steps=training_config["eval_steps"],
        save_total_limit=training_config["save_total_limit"],
        eval_strategy=training_config.get("eval_strategy") or training_config.get("evaluation_strategy", "steps"),
        save_strategy=training_config["save_strategy"],
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        gradient_checkpointing=training_config["gradient_checkpointing"],
        optim=training_config["optim"],
        report_to=training_config["report_to"],
        # Note: group_size is not a parameter of GRPOConfig
        # GRPO uses default group size internally
    )
    
    # Define reward function for GRPO
    # This function evaluates the quality of generated completions
    def reward_function(prompts, completions, **kwargs):
        """
        Simple reward function for GRPO training.
        Returns rewards for each completion.
        You can replace this with a more sophisticated reward model.
        """
        rewards = []
        for completion in completions:
            # Simple reward: encourage reasonable length (not too short, not too long)
            # You can customize this based on your needs
            length = len(completion.split())
            if length < 5:
                reward = -1.0  # Penalize very short responses
            elif length > 500:
                reward = -0.5  # Slightly penalize very long responses
            else:
                reward = 1.0  # Reward reasonable length responses
            
            rewards.append(reward)
        
        return rewards
    
    # Initialize GRPO trainer
    print("Initializing GRPO trainer...")
    # Note: GRPOTrainer requires reward_funcs parameter
    # The tokenizer is typically accessed from the model or passed via processing_class
    trainer = GRPOTrainer(
        model=model,
        args=grpo_trainer_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        reward_funcs=reward_function,  # Required: reward function(s) for GRPO
    )
    
    # Start training
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # Save final model
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(training_config["output_dir"])
    
    # Save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Push to hub if configured
    hub_config = config.get("hub", {})
    if hub_config.get("push_to_hub", False):
        hub_model_id = hub_config.get("hub_model_id")
        if hub_model_id:
            print(f"Pushing model to hub: {hub_model_id}")
            trainer.push_to_hub(hub_model_id=hub_model_id)
        else:
            print("Warning: push_to_hub is True but hub_model_id is not set")
    
    print("Training complete!")


if __name__ == "__main__":
    main()
