import openvino as ov

# Convert the ONNX model to OpenVINO IR format
ov_model = ov.convert_model('tiny/tinyllama.onnx')

# Save the converted OpenVINO model
ov.save_model(ov_model, 'model_ir/tinyllama.xml')

compiled_model = ov.compile_model(ov_model)
