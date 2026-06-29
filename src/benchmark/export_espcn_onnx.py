import argparse
from pathlib import Path

import torch

from models.espcn import ESPCN


SCALE = 2
OPSET_VERSION = 13


CONFIGS = {
    "224": {
        "weights": Path("weights/Finetuned/224x224/espcn_x2_finetuned_224.pth.tar"),
        "fixed_onnx": Path("weights/Finetuned/224x224/espcn_x2_finetuned_224_fixed.onnx"),
        "dynamic_onnx": Path("weights/Finetuned/224x224/espcn_x2_finetuned_224_dynamic.onnx"),
        "input_size": 224,
    },
    "640": {
        "weights": Path("weights/Finetuned/640x640/espcn_x2_finetuned_640_stage2.pth.tar"),
        "fixed_onnx": Path("weights/Finetuned/640x640/espcn_x2_finetuned_640_fixed.onnx"),
        "dynamic_onnx": Path("weights/Finetuned/640x640/espcn_x2_finetuned_640_dynamic.onnx"),
        "input_size": 640,
    },
}


def load_model(weights_path):
    model = ESPCN(
        upscale_factor=SCALE,
        in_channels=1,
        out_channels=1,
    )

    checkpoint = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model


def export_onnx(model, output_path, input_size, dynamic=False):
    dummy_input = torch.randn(1, 1, input_size, input_size)

    kwargs = {}

    if dynamic:
        kwargs["dynamic_axes"] = {
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=OPSET_VERSION,
        do_constant_folding=True,
        external_data=False,
        dynamo=False,
        **kwargs,
    )

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", default="all", choices=["224", "640", "all"])
    parser.add_argument("--dynamic", action="store_true")
    args = parser.parse_args()

    keys = CONFIGS.keys() if args.resolution == "all" else [args.resolution]

    for key in keys:
        cfg = CONFIGS[key]

        print(f"\nExporting {key}x{key}")
        print(f"Weights: {cfg['weights']}")

        model = load_model(cfg["weights"])

        export_onnx(
            model=model,
            output_path=cfg["fixed_onnx"],
            input_size=cfg["input_size"],
            dynamic=False,
        )

        if args.dynamic:
            export_onnx(
                model=model,
                output_path=cfg["dynamic_onnx"],
                input_size=cfg["input_size"],
                dynamic=True,
            )

    print("\nExport completed.")


if __name__ == "__main__":
    main()