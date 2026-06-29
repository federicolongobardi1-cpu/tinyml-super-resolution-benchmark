"""
Plot ESPCN PyTorch vs ONNX Runtime benchmark results.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("results/csv/benchmark_espcn_onnx_x2.csv")
PLOTS_DIR = Path("results/plots")

PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df = pd.read_csv(CSV_PATH)

    pivot_fps = df.pivot(
        index="lr_resolution",
        columns="method",
        values="fps"
    )

    pivot_latency = df.pivot(
        index="lr_resolution",
        columns="method",
        values="mean_time_ms"
    )

    resolution_order = [
    "64x64",
    "112x112",
    "224x224",
    "640x480",
    "640x640",
]

    pivot_fps = pivot_fps.reindex(resolution_order)
    pivot_latency = pivot_latency.reindex(resolution_order)

    pivot_fps.plot(marker="o")
    plt.xlabel("LR input resolution")
    plt.ylabel("FPS")
    plt.title("ESPCN PyTorch vs ONNX Runtime - FPS")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "espcn_onnx_fps.png", dpi=300)
    plt.close()

    pivot_latency.plot(marker="o")
    plt.xlabel("LR input resolution")
    plt.ylabel("Mean latency [ms]")
    plt.title("ESPCN PyTorch vs ONNX Runtime - Latency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "espcn_onnx_latency.png", dpi=300)
    plt.close()

    print("Saved plots to results/plots/")


if __name__ == "__main__":
    main()