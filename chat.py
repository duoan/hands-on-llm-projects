import argparse
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description="Chat with LLM")
parser.add_argument(
    "--model_path",
    type=str,
    default="roneneldan/TinyStories-1M",
    help="Pretrained model name or path",
)

args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

print("This a simple command line chat box")
while True:
    text = input("user:")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(**model_inputs, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("assistant:", response)
