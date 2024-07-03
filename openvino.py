from torch.profiler import profile, ProfilerActivity
from transformers import AutoTokenizer, TextStreamer, AutoModelForCausalLM
from threading import Thread
import intel_npu_acceleration_library
import torch
import time
import sys

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True).eval()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
streamer = TextStreamer(tokenizer, skip_special_tokens=True)

# Compile for NPU (if available)
try:
    print("Compile model for the NPU")
    model = intel_npu_acceleration_library.compile(model)
except ImportError:
    print("Intel NPU acceleration library not found. Running on CPU/GPU.")

# Context Management
context_tokens = []
max_context_length = 4096  # Adjust based on model and memory

def get_user_input():
    return input("You: ")

while True:
    query = get_user_input()
    if query.lower() in ["exit", "quit"]:
        break

    new_tokens = tokenizer(query, return_tensors="pt")["input_ids"]
    context_tokens.extend(new_tokens[0].tolist())

    # Trim context to avoid exceeding model's max length
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[-max_context_length:]

    generation_kwargs = dict(
        input_ids=torch.tensor([context_tokens], device=model.device),
        streamer=streamer,
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )

    print("Assistant:")
    _ = model.generate(**generation_kwargs)

    # Update context with the model's response
    response_tokens = generation_kwargs["input_ids"][0].tolist()
    context_tokens = response_tokens
