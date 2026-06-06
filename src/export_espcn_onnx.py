"""
Export ESPCN x2 to ONNX.

Exports:
1. Dynamic ONNX model (any resolution)
2. Fixed ONNX model (224x224 input)

The dynamic model is useful for ONNX Runtime.
The fixed model may be easier to compile for Hailo.
"""

import torch
from pathlib import Path

from models.espcn import ESPCN


WEIGHTS_PATH = Path("weights/ESPCN_x2.pth.tar")

DYNAMIC_ONNX_PATH = Path("weights/espcn_x2_dynamic.onnx")
FIXED_ONNX_PATH = Path("weights/espcn_x2_fixed_224.onnx")

SCALE = 2
INPUT_HEIGHT = 224
INPUT_WIDTH = 224


def load_model():
    """
    Load pretrained ESPCN model.
    """

    model = ESPCN(
        upscale_factor=SCALE,
        in_channels=1,
        out_channels=1
    )

    checkpoint = torch.load(
        str(WEIGHTS_PATH),
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    model.eval()

    return model


def export_dynamic(model):
    """
    Export ONNX model with dynamic input size.
    """

    dummy_input = torch.randn(
        1,
        1,
        INPUT_HEIGHT,
        INPUT_WIDTH
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(DYNAMIC_ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes={
            "input": {
                2: "height",
                3: "width",
            },
            "output": {
                2: "out_height",
                3: "out_width",
            },
        },
    )

    print(f"Saved: {DYNAMIC_ONNX_PATH}")


def export_fixed(model):
    """
    Export ONNX model with fixed 224x224 input.
    """

    dummy_input = torch.randn(
        1,
        1,
        INPUT_HEIGHT,
        INPUT_WIDTH
    )

    torch.onnx.export(
        model,
        dummy_input,
        str(FIXED_ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
    )

    print(f"Saved: {FIXED_ONNX_PATH}")


def main():
    model = load_model()

    export_dynamic(model)
    export_fixed(model)

    print("\nExport completed.")


if __name__ == "__main__":
    main()