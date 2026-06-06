import pandas as pd
import matplotlib.pyplot as plt

# carica CSV
df = pd.read_csv("results/benchmark.csv")

# separa i metodi
nearest = df[df["method"] == "nearest"]
bicubic = df[df["method"] == "bicubic"]

# grafico PSNR
plt.figure(figsize=(8,5))

plt.plot(
    nearest["scale"],
    nearest["psnr"],
    marker="o",
    label="Nearest"
)

plt.plot(
    bicubic["scale"],
    bicubic["psnr"],
    marker="o",
    label="Bicubic"
)

plt.xlabel("Scale factor")
plt.ylabel("PSNR (dB)")
plt.title("PSNR vs Scale Factor")

plt.legend()
plt.grid(True)

plt.savefig("results/psnr_vs_scale.png")

plt.show()

# grafico FPS
plt.figure(figsize=(8,5))

plt.plot(
    nearest["scale"],
    nearest["fps"],
    marker="o",
    label="Nearest"
)

plt.plot(
    bicubic["scale"],
    bicubic["fps"],
    marker="o",
    label="Bicubic"
)

plt.xlabel("Scale factor")
plt.ylabel("FPS")
plt.title("FPS vs Scale Factor")

plt.legend()
plt.grid(True)

plt.savefig("results/fps_vs_scale.png")

plt.show()