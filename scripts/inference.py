"""
Inference script for testing the fine-tuned Phi-2 model.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import BitsAndBytesConfig


def load_model_and_tokenizer(model_path: str, base_model_name: str = "microsoft/phi-2"):
    """Load fine-tuned model with QLoRA adapters."""
    print(f"Loading base model: {base_model_name}")
    
    # Setup quantization (same as training)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(base_model, model_path)
    model = model.merge_and_unload()  # Merge adapters for faster inference
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
):
    """Generate response from model."""
    # Format prompt
    formatted_prompt = f"Human: {prompt}\n\nAssistant: "
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant response
    if "Assistant:" in full_text:
        response = full_text.split("Assistant:")[-1].strip()
    else:
        response = full_text[len(formatted_prompt):].strip()
    
    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/phi-2",
        help="Base model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to test",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    model.eval()
    
    print("Model loaded successfully!")
    print("=" * 50)
    
    if args.interactive:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.")
        print("=" * 50)
        
        while True:
            prompt = input("\nYou: ")
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            
            if not prompt.strip():
                continue
            
            print("Assistant: ", end="", flush=True)
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)
    
    elif args.prompt:
        # Single prompt mode
        print(f"Prompt: {args.prompt}")
        print("=" * 50)
        response = generate_response(
            model,
            tokenizer,
            args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"Response: {response}")
    
    else:
        # Default test prompts
        test_prompts = [
            "What is machine learning?",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate factorial.",
        ]
        
        for prompt in test_prompts:
            print(f"\nPrompt: {prompt}")
            print("=" * 50)
            response = generate_response(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
