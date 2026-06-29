# TinyML Image Super-Resolution Benchmark

Implementation and evaluation of a lightweight image super-resolution pipeline for edge AI applications using the Efficient Sub-Pixel Convolutional Neural Network (ESPCN). The project benchmarks traditional interpolation methods against deep learning implementations on Raspberry Pi 5 and the Hailo-8L AI accelerator.

The project was developed as part of the **Internet of Things Laboratory** at the **University of Trento**.

---

## Overview

Image super-resolution reconstructs a high-resolution image from a low-resolution input. While traditional interpolation methods such as Nearest Neighbor and Bicubic are computationally efficient, they generally produce lower reconstruction quality than deep learning approaches.

This project evaluates the ESPCN architecture across multiple execution backends:

- Nearest Neighbor interpolation
- Bicubic interpolation
- ESPCN (PyTorch)
- ESPCN (ONNX Runtime)
- ESPCN (Hailo-8L)

The comparison considers both reconstruction quality and embedded deployment performance.

---

## Hardware

The experimental platform consists of:

- Raspberry Pi 5 (8 GB)
- Hailo-8L AI Accelerator
- Raspberry Pi Camera Module 3
- USB-C Power Meter

---

## Evaluation Metrics

The following metrics are evaluated:

- PSNR (Peak Signal-to-Noise Ratio)
- Latency
- Frames Per Second (FPS)
- Current Consumption
- Power Consumption
- Energy per Frame

---

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Input images
│   ├── calibration/         # Calibration datasets
│   └── outputs/             # Generated outputs
│
├── report/                  # Final report
│
├── results/
│   ├── csv/                 # Benchmark results
│   ├── logs/                # Execution logs
│   └── plots/               # Generated plots
│
├── src/
│   ├── benchmark/           # Training, export and benchmark scripts
│   ├── camera/              # Camera utilities
│   ├── hailo/               # Hailo benchmarks and live demo
│   ├── models/              # ESPCN architecture
│   └── utils/               # Metrics and helper functions
│
└── weights/                 # Trained models and compiled HEF files
```

---

## Main Scripts

### Traditional interpolation benchmark

```bash
python src/hailo/benchmark_traditional.py
```

### PyTorch benchmark

```bash
python src/hailo/benchmark_pytorch.py
```

### ONNX Runtime benchmark

```bash
python src/hailo/benchmark_onnx.py
```

### Hailo benchmark

```bash
python src/hailo/benchmark_hailo.py
```

### Live Hailo demonstration

```bash
python src/hailo/demo_hailo_mjpeg.py
```

The live demo captures frames from the Raspberry Pi Camera Module 3, performs ×2 image super-resolution using the Hailo-8L accelerator, and streams the output to a remote computer through an MJPEG server.

---

## Main Results

The experimental evaluation shows that:

- ESPCN consistently improves PSNR over traditional interpolation methods.
- ONNX Runtime significantly accelerates inference compared with native PyTorch.
- The Hailo-8L accelerator achieves the highest throughput while maintaining low power consumption.
- The complete live pipeline reaches real-time performance (>30 FPS at 224×224 input resolution), exceeding the project target of 15 FPS.

---

## Report

The complete technical report describing the methodology, experimental evaluation, and conclusions is available in:

```text
report/
```

---

## Author

**Federico Longobardi**

Master's Degree in Autonomous Systems and Intelligent Robots

University of Trento
