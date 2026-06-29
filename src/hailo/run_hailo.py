import time
from pathlib import Path

import cv2
import numpy as np

from hailo_platform import (
    HEF,
    VDevice,
    ConfigureParams,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    FormatType,
)


HEF_PATH = Path("weights/espcn_x2_fixed_224_level2.hef")
IMAGE_PATH = Path("data/raw/test.png")
OUTPUT_PATH = Path("data/outputs/hailo_espcn_x2_224.png")

INPUT_SIZE = 224


def preprocess(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img[np.newaxis, :, :, np.newaxis]  # NHWC

    return img


def postprocess(output):
    output = np.squeeze(output)
    output = np.clip(output, 0.0, 1.0)
    output = (output * 255.0).astype(np.uint8)
    return output


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    hef = HEF(str(HEF_PATH))

    input_data = preprocess(IMAGE_PATH)

    with VDevice() as target:
        configure_params = ConfigureParams.create_from_hef(
            hef,
            interface=HailoStreamInterface.PCIe,
        )

        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = InputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        output_vstreams_params = OutputVStreamParams.make(
            network_group,
            quantized=False,
            format_type=FormatType.FLOAT32,
        )

        input_info = hef.get_input_vstream_infos()[0]
        output_info = hef.get_output_vstream_infos()[0]

        input_name = input_info.name
        output_name = output_info.name

        print("Input:", input_name, input_info.shape)
        print("Output:", output_name, output_info.shape)

        with InferVStreams(
            network_group,
            input_vstreams_params,
            output_vstreams_params,
        ) as infer_pipeline:
            with network_group.activate(network_group_params):
                start = time.perf_counter()
                results = infer_pipeline.infer({input_name: input_data})
                end = time.perf_counter()

    output = results[output_name]
    output_img = postprocess(output)

    cv2.imwrite(str(OUTPUT_PATH), output_img)

    latency_ms = (end - start) * 1000.0

    print(f"Saved: {OUTPUT_PATH}")
    print(f"End-to-end latency: {latency_ms:.3f} ms")


if __name__ == "__main__":
    main()