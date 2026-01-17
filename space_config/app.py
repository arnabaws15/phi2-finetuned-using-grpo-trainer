"""
Gradio chat interface for the fine-tuned Phi-2 model.
Hugging Face Space version - expects model to be loaded from Hub.
"""

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from transformers import BitsAndBytesConfig
from threading import Thread
import os


# Configuration - Update MODEL_PATH to your Hugging Face model ID
MODEL_PATH = os.getenv("MODEL_PATH", "arisin/phi2-grpo-finetuned")  # Update this!
BASE_MODEL_NAME = os.getenv("BASE_MODEL", "microsoft/phi-2")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))


# Global variables for model and tokenizer
model = None
tokenizer = None


def load_model():
    """Load the fine-tuned model from Hugging Face Hub."""
    global model, tokenizer
    
    if model is not None:
        return "Model already loaded!"
    
    try:
        print(f"Loading model: {MODEL_PATH}")
        
        # Check if GPU is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if device == "cuda":
            # GPU available - use 4-bit quantization
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for memory
            )
            
            # Try to load fine-tuned model directly
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "14GB", "cpu": "30GB"},  # Adjust for Spaces GPU
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                )
                print("Loaded fine-tuned model from Hub")
            except:
                # Fallback: load base model and try to load adapters
                print(f"Loading base model: {BASE_MODEL_NAME}")
                base_model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={0: "14GB", "cpu": "30GB"},
                )
                
                try:
                    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
                    model = model.merge_and_unload()
                    print("Loaded model with LoRA adapters")
                except:
                    model = base_model
                    print("Using base model (adapters not found)")
                
                tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL_NAME,
                    trust_remote_code=True,
                )
        else:
            # CPU fallback - no quantization
            print("⚠️ No GPU available, loading on CPU (will be slow)")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_PATH,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_PATH,
                    trust_remote_code=True,
                )
            except:
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL_NAME,
                    device_map="cpu",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    BASE_MODEL_NAME,
                    trust_remote_code=True,
                )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model.eval()
        return f"Model loaded successfully on {device}!"
    
    except Exception as e:
        return f"Error loading model: {str(e)}"


def generate_response(
    message: str,
    history: list,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate response from the model with streaming."""
    global model, tokenizer
    
    # Lazy load model on first use
    if model is None or tokenizer is None:
        status = load_model()
        if "Error" in status:
            yield status
            return
        if model is None or tokenizer is None:
            yield "Error: Model not loaded. Please reload the model."
            return
    
    # Format prompt
    formatted_prompt = f"Human: {message}\n\nAssistant: "
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)
    
    # Define stop tokens/strings
    stop_strings = ["Alien", "\nHuman:", "\n\nHuman:", "###"]
    
    # Setup streaming
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    generation_kwargs = dict(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    
    # Generate in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream the response with stop token detection
    partial_response = ""
    try:
        for new_text in streamer:
            partial_response += new_text
            
            # Check if any stop string is in the response
            should_stop = False
            for stop_str in stop_strings:
                if stop_str in partial_response:
                    # Truncate at the stop string
                    partial_response = partial_response.split(stop_str)[0].strip()
                    should_stop = True
                    break
            
            yield partial_response
            
            if should_stop:
                break
                
    except Exception as e:
        yield f"Error generating response: {str(e)}"


def clear_chat():
    """Clear the chat history."""
    return [], ""  # Return empty list for Gradio 6.0 chatbot format


# Don't load model on startup - load lazily on first use to avoid async issues
print("Ready to load model on first inference...")


# Create Gradio interface
with gr.Blocks(title="Phi-2 Fine-tuned Chat") as demo:
    gr.Markdown(
        """
        # Phi-2 Fine-tuned Chat Interface
        
        Chat with the fine-tuned Phi-2 model using GRPO training on OpenAssistant dataset.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Model Information")
            model_status = gr.Textbox(
                label="Model Status",
                value="Ready to load model on first inference",
                interactive=False,
            )
            load_btn = gr.Button("Reload Model", variant="secondary")
            load_btn.click(load_model, outputs=model_status)
            
            gr.Markdown("### Generation Parameters")
            max_tokens = gr.Slider(
                minimum=50,
                maximum=512,
                value=MAX_NEW_TOKENS,
                step=10,
                label="Max New Tokens",
            )
            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=TEMPERATURE,
                step=0.1,
                label="Temperature",
            )
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=TOP_P,
                step=0.05,
                label="Top-p",
            )
        
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Chat",
                height=500,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    scale=4,
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Chat", variant="secondary")
    
    # Event handlers
    def update_chatbot(message, history, max_tokens, temperature, top_p):
        history = history or []
        if message.strip():
            # Add user message in Gradio 6.0 format
            history.append({"role": "user", "content": message})
            
            # Add placeholder for assistant response
            history.append({"role": "assistant", "content": ""})
            
            for response in generate_response(message, history, max_tokens, temperature, top_p):
                # Update the last assistant message with the response
                history[-1]["content"] = response
                yield history
    
    submit_btn.click(
        update_chatbot,
        inputs=[msg, chatbot, max_tokens, temperature, top_p],
        outputs=[chatbot],
    ).then(
        lambda: "",
        outputs=[msg],
    )
    
    msg.submit(
        update_chatbot,
        inputs=[msg, chatbot, max_tokens, temperature, top_p],
        outputs=[chatbot],
    ).then(
        lambda: "",
        outputs=[msg],
    )
    
    clear_btn.click(clear_chat, outputs=[chatbot, msg])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())  # Moved theme to launch() for Gradio 6.0
