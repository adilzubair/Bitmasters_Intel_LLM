import gradio as gr
import numpy as np
import openvino.runtime as ov
from transformers import AutoTokenizer

# Initialize OpenVINO Runtime core
core = ov.Core()

# Load and compile the model
compiled_model = core.compile_model("model_ir/tinyllama.xml", "CPU")

# Create an inference request
infer_request = compiled_model.create_infer_request()

# Initialize the tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_default_system_prompt=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Context management
context_tokens = []
max_context_length = 4096  # Adjust based on model and memory

def infer(input_tokens):
    input_tensor = ov.Tensor(array=np.array([input_tokens], dtype=np.int64))
    infer_request.set_input_tensor(input_tensor)
    infer_request.start_async()
    infer_request.wait()
    outputs = [infer_request.get_tensor(output).data for output in compiled_model.outputs]
    return outputs

def chatbot(query):
    global context_tokens

    new_tokens = tokenizer(query, return_tensors="pt")["input_ids"].numpy().flatten().tolist()
    context_tokens.extend(new_tokens)

    # Trim context to avoid exceeding model's max length
    if len(context_tokens) > max_context_length:
        context_tokens = context_tokens[-max_context_length:]

    # Perform inference
    response = infer(context_tokens)
    
    # Ensure the output tokens are integers
    response_tokens = response[0].flatten().astype(int).tolist()

    # Filter out any invalid token IDs
    response_tokens = [token for token in response_tokens if token >= 0 and token < tokenizer.vocab_size]
    
    # Decode the tokens into text
    response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
    
    # Update context with the model's response
    context_tokens.extend(response_tokens)
    
    return response_text

# Log to check if the function is being executed
print("Launching Gradio Interface...")

iface = gr.Interface(fn=chatbot, inputs="text", outputs="text", title="TinyLlama Chatbot")

# Adding more logging to ensure this part is executed
print("Gradio Interface should now be available.")

iface.launch()
