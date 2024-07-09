# Intel_LLM
This repository contains the files of the project titled "Running GenAI on Intel AI Laptops and Simple LLM Inference on CPU and fine-tuning of LLM Models using Intel® OpenVINO"

How to run?

1. Create an empty folder `tiny `
   
2. Run convert_to_onnx.py. Model will be converted to onnx format in the folder named tiny.
   
3. Run onnx_to_IR.py. it converts onnx to openvino intermediate representation.
4. Run optimized model.py
