# Intel_LLM
This repository contains the files of the project titled "Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO"

How to run?

First run convert_to_onnx.py. Model will be converted to onnx format in the folder named tiny.
Then run onnx_to_IR.py. it converts onnx to openvino intermediate representation.
Then run optimized model.py
