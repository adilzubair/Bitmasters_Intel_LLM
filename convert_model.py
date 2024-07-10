from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from optimum.intel import OVQuantizer, OVWeightQuantizationConfig
import openvino as ov
from pathlib import Path

def convert_model(model_id="togethercomputer/RedPajama-INCITE-Chat-3B-v1"):
    # Download and convert the model
    model = OVModelForCausalLM.from_pretrained(model_id, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Save the model in OpenVINO IR format
    model.save_pretrained("openvino_model")
    tokenizer.save_pretrained("openvino_model")

def compress_model():
    model = OVModelForCausalLM.from_pretrained("openvino_model")
    int8_model_dir = Path("openvino_model") / "INT8_compressed_weights"
    ov_config = OVWeightQuantizationConfig()

    # Compress weights to INT8
    quantizer = OVQuantizer.from_pretrained(model, ov_config=ov_config)
    quantizer.quantize(save_directory=int8_model_dir, weights_only=True)

if __name__ == "__main__":
    convert_model()
    compress_model()
