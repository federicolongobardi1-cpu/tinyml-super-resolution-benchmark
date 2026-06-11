from pathlib import Path

from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization import QuantType


INPUT_MODEL = Path("weights/espcn_x2_dynamic.onnx")
OUTPUT_MODEL = Path("weights/espcn_x2_dynamic_int8.onnx")


quantize_dynamic(
    model_input=str(INPUT_MODEL),
    model_output=str(OUTPUT_MODEL),
    weight_type=QuantType.QInt8,
)

print(f"Saved quantized model to {OUTPUT_MODEL}")