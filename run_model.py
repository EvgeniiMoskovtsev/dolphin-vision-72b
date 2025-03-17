import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import argparse
import time
import warnings

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

def load_model(model_name="cognitivecomputations/dolphin-vision-72b"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    return model, tokenizer

def process_image(image_path, model, tokenizer, prompt="Describe this image in detail"):
    image = Image.open(image_path)
    
    tokenization_start = time.time()
    
    messages = [
        {"role": "user", "content": f'<image>\n{prompt}'}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
    input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1], dtype=torch.long).unsqueeze(0)
    
    image_tensor = model.process_images([image], model.config).to(dtype=model.dtype)
    
    tokenization_time = time.time() - tokenization_start

    inference_start = time.time()
    
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        max_new_tokens=2048,
        use_cache=True
    )[0]
    
    inference_time = time.time() - inference_start
    
    generated_text = tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    return generated_text, tokenization_time, inference_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, default="Describe this image in detail", help="Prompt for image description")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model()
    
    print(f"Processing image: {args.image_path}")
    generated_text, tokenization_time, inference_time = process_image(
        args.image_path, 
        model, 
        tokenizer, 
        args.prompt
    )
    
    print("\nResults:")
    print("-" * 50)
    print(f"Tokenization time: {tokenization_time:.2f} seconds")
    print(f"Inference time: {inference_time:.2f} seconds")
    print(f"Total time: {(tokenization_time + inference_time):.2f} seconds")
    print("-" * 50)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    torch.set_default_device('cuda')
    main() 