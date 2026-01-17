"""
Dataset preparation script for OpenAssistant/oasst1 dataset.
Extracts conversation pairs and formats them for GRPO training.
"""

import json
from typing import Dict, List, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
import argparse


def extract_conversation_threads(dataset, language: str = "en") -> List[Dict]:
    """
    Extract conversation threads from oasst1 dataset.
    oasst1 has a tree structure with parent_id relationships.
    """
    conversations = []
    
    # Filter for specified language and non-deleted messages
    filtered_data = [
        msg for msg in dataset
        if msg.get("lang") == language 
        and not msg.get("deleted", False)
        and msg.get("review_result", False)  # Only approved messages
    ]
    
    # Build message tree
    message_dict = {msg["message_id"]: msg for msg in filtered_data}
    
    # Find root messages (prompts without parents)
    root_messages = [msg for msg in filtered_data if msg.get("parent_id") is None]
    
    def get_conversation_thread(message_id: str, thread: List[Dict] = None) -> List[Dict]:
        """Recursively extract conversation thread."""
        if thread is None:
            thread = []
        
        if message_id not in message_dict:
            return thread
        
        msg = message_dict[message_id]
        thread.append(msg)
        
        # Find children (replies)
        children = [
            m for m in filtered_data 
            if m.get("parent_id") == message_id
        ]
        
        # Sort by rank if available, otherwise by created_date
        # Handle None values in rank - use 0 if rank is None or missing
        children.sort(key=lambda x: (x.get("rank") if x.get("rank") is not None else 0, x.get("created_date", "")))
        
        # Follow the best-ranked child (rank 0) or first child
        if children:
            best_child = children[0]
            return get_conversation_thread(best_child["message_id"], thread)
        
        return thread
    
    # Extract conversations from root messages
    for root in root_messages:
        thread = get_conversation_thread(root["message_id"])
        
        # Extract prompt-completion pairs
        for i in range(len(thread) - 1):
            if thread[i]["role"] == "prompter" and thread[i + 1]["role"] == "assistant":
                conversations.append({
                    "prompt": thread[i]["text"],
                    "completion": thread[i + 1]["text"],
                    "message_tree_id": root.get("message_tree_id"),
                })
    
    return conversations


def format_for_grpo(conversations: List[Dict], tokenizer, max_length: int = 512) -> Dataset:
    """
    Format conversations for GRPO training.
    GRPO expects prompt-completion pairs with proper formatting.
    """
    formatted_data = []
    
    for conv in conversations:
        prompt = conv["prompt"]
        completion = conv["completion"]
        
        # Format using phi-2 chat template if available
        # Phi-2 uses a simple format: "Human: {prompt}\n\nAssistant: {completion}"
        formatted_text = f"Human: {prompt}\n\nAssistant: {completion}"
        
        # Tokenize
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )
        
        # For GRPO, we need to identify prompt and completion tokens
        prompt_tokenized = tokenizer(
            f"Human: {prompt}\n\nAssistant: ",
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        
        prompt_length = len(prompt_tokenized["input_ids"])
        completion_length = len(tokenized["input_ids"]) - prompt_length
        
        # Format prompt text (what GRPOTrainer expects)
        prompt_text = f"Human: {prompt}\n\nAssistant: "
        
        formatted_data.append({
            "prompt": prompt_text,  # Required by GRPOTrainer
            "text": formatted_text,  # Full text for reference
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "prompt_length": prompt_length,
            "completion_length": completion_length,
        })
    
    return Dataset.from_list(formatted_data)


def main():
    parser = argparse.ArgumentParser(description="Prepare oasst1 dataset for GRPO training")
    parser.add_argument("--dataset_name", type=str, default="OpenAssistant/oasst1",
                        help="Hugging Face dataset name")
    parser.add_argument("--language", type=str, default="en",
                       help="Language to filter (default: en)")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-2",
                        help="Model name for tokenizer")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for processed dataset")
    parser.add_argument("--val_size", type=float, default=0.1,
                       help="Validation split size")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        print(f"Limited to {len(dataset)} samples")
    
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Extracting conversation threads...")
    conversations = extract_conversation_threads(dataset, language=args.language)
    print(f"Extracted {len(conversations)} conversation pairs")
    
    print("Formatting for GRPO...")
    formatted_dataset = format_for_grpo(conversations, tokenizer, max_length=args.max_length)
    
    # Split into train/validation
    if args.val_size > 0:
        split_dataset = formatted_dataset.train_test_split(test_size=args.val_size, seed=42)
        dataset_dict = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
        })
    else:
        dataset_dict = DatasetDict({"train": formatted_dataset})
    
    print(f"Saving dataset to {args.output_dir}")
    dataset_dict.save_to_disk(args.output_dir)
    
    print(f"Train samples: {len(dataset_dict['train'])}")
    if "validation" in dataset_dict:
        print(f"Validation samples: {len(dataset_dict['validation'])}")
    
    print("Dataset preparation complete!")


if __name__ == "__main__":
    main()
