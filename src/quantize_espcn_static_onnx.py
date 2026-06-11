"""
Static INT8 quantization for ESPCN ONNX.

This follows the professor's approach:
1. ONNX pre-processing
2. calibration data reader
3. static quantization
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import quantization


INPUT_MODEL = Path("weights/espcn_x2_dynamic.onnx")
PREPROCESSED_MODEL = Path("weights/espcn_x2_preprocessed.onnx")
OUTPUT_MODEL = Path("weights/espcn_x2_static_int8.onnx")

CALIBRATION_IMAGE_DIR = Path("data/raw/DIV2K_valid_HR")

LR_SIZE = 224
LIMIT_SAMPLES = 50


class ESPCNCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, image_dir, input_name, limit_samples=50):
        self.input_name = input_name
        self.image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
        self.image_paths = self.image_paths[:limit_samples]
        self.index = 0

    def preprocess_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.resize(
            img,
            (LR_SIZE, LR_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        tensor = img.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, np.newaxis, :, :]

        return tensor

    def get_next(self):
        if self.index >= len(self.image_paths):
            return None

        tensor = self.preprocess_image(self.image_paths[self.index])
        self.index += 1

        return {self.input_name: tensor}

    def rewind(self):
        self.index = 0


def main():
    print("Pre-processing ONNX model...")

    quantization.shape_inference.quant_pre_process(
        str(INPUT_MODEL),
        str(PREPROCESSED_MODEL),
        skip_symbolic_shape=True,
        skip_onnx_shape=False,
    )

    session = ort.InferenceSession(
        str(PREPROCESSED_MODEL),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    print("Input name:", input_name)
    print("Preparing calibration data...")

    calibration_reader = ESPCNCalibrationDataReader(
        image_dir=CALIBRATION_IMAGE_DIR,
        input_name=input_name,
        limit_samples=LIMIT_SAMPLES,
    )

    print("Calibration samples:", len(calibration_reader.image_paths))
    print("Performing static quantization...")

    quantization.quantize_static(
        model_input=str(PREPROCESSED_MODEL),
        model_output=str(OUTPUT_MODEL),
        calibration_data_reader=calibration_reader,
        activation_type=quantization.QuantType.QInt8,
        weight_type=quantization.QuantType.QInt8,
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    print(f"Saved quantized model to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()