from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Prepare dummy input for export
dummy_input = tokenizer("Hello, OpenVINO!", return_tensors="pt").input_ids

# Export to ONNX
torch.onnx.export(model, dummy_input, "tiny/tinyllama.onnx", 
                  input_names=["input_ids"], 
                  output_names=["logits"], 
                  dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence"}, "logits": {0: "batch_size", 1: "sequence"}})
