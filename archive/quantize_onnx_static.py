from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime import quantization

# CONFIG

INPUT_MODEL = Path(
    "weights/Finetuned/640x640/espcn_x2_finetuned_640_fixed.onnx"
)

PREPROCESSED_MODEL = Path(
    "weights/Finetuned/640x640/espcn_x2_finetuned_640_preprocessed.onnx"
)

OUTPUT_MODEL = Path(
    "weights/Finetuned/640x640/espcn_x2_finetuned_640_static_int8.onnx"
)

LR_SIZE = 640

# --------------------


CALIBRATION_IMAGE_DIR = Path("data/raw/DIV2K_valid_HR")
LIMIT_SAMPLES = 100


class ESPCNCalibrationDataReader(quantization.CalibrationDataReader):
    def __init__(self, image_dir, input_name, limit_samples):
        self.input_name = input_name
        self.image_paths = sorted(image_dir.glob("*.png"))[:limit_samples]
        self.index = 0

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No calibration images found in {image_dir}")

    def preprocess_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")

        img = cv2.resize(
            img,
            (LR_SIZE, LR_SIZE),
            interpolation=cv2.INTER_CUBIC,
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
    print("Input model:", INPUT_MODEL)
    print("Output model:", OUTPUT_MODEL)
    print("LR size:", LR_SIZE)

    PREPROCESSED_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    print("\nPre-processing ONNX model...")

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

    print(f"\nSaved quantized model to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()