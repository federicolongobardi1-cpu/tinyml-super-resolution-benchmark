# TinyML Image Super-Resolution Benchmark

Benchmarking lightweight image super-resolution algorithms for Edge AI deployment on Raspberry Pi 5 and the Hailo-8L AI accelerator.

## Overview

This project evaluates the feasibility of deploying lightweight deep learning super-resolution models on embedded hardware.

The study compares traditional interpolation algorithms (Nearest Neighbor and Bicubic) with the Efficient Sub-Pixel Convolutional Neural Network (ESPCN) under different deployment backends:

- PyTorch
- ONNX Runtime
- Hailo-8L AI Accelerator

The evaluation considers:

- Reconstruction quality (PSNR and VIF)
- Inference latency
- Frames per second (FPS)
- Power consumption
- Energy per frame
- Real-time camera deployment

The complete methodology and experimental results are reported in the paper available in the `report/` directory.

---

# Repository Structure

```
.
├── archive/            # Auxiliary scripts used during development
├── data/
│   ├── calibration/    # Hailo calibration datasets
│   ├── outputs/        # Generated images and videos
│   └── raw/            # DIV2K validation images
├── report/             # IEEE paper
├── results/
│   ├── csv/
│   ├── logs/
│   └── plots/
├── src/
│   ├── benchmark/
│   ├── camera/
│   ├── hailo/
│   ├── models/
│   └── utils/
└── weights/
```

---

# Hardware

## Desktop

Used for:

- ESPCN fine-tuning
- ONNX export
- Calibration dataset generation
- Hailo model compilation

## Raspberry Pi 5

Used for:

- Traditional interpolation benchmark
- PyTorch benchmark
- ONNX Runtime benchmark
- Hailo benchmark
- Real-time camera demonstration

Hardware configuration:

- Raspberry Pi 5 (8 GB)
- Hailo-8L AI Accelerator
- Raspberry Pi Camera Module 3

---

# Installation

Clone the repository:

```bash
git clone <repository-url>
cd super-resolution-project
```

Create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

For Raspberry Pi, install the required HailoRT runtime together with the dependencies listed in `requirements.txt`.

---

# Experimental Workflow

The complete workflow adopted in this project is illustrated below.

```
Desktop
│
├── Fine-tuning ESPCN
├── Export to ONNX
├── Calibration dataset generation
└── HEF compilation
                │
                ▼
Raspberry Pi 5
│
├── Traditional benchmark
├── PyTorch benchmark
├── ONNX benchmark
├── Hailo benchmark
└── Live camera demonstration
```

All quantitative results reported in the paper were obtained directly on the Raspberry Pi 5 platform. Hailo measurements were performed on the same Raspberry Pi equipped with the Hailo-8L accelerator.

---

# Model Preparation

The adopted workflow is:

```
Pre-trained ESPCN
        │
        ▼
Fine-tuning (DIV2K)
        │
        ▼
PyTorch model
        │
        ▼
Export to ONNX
        │
        ▼
Calibration dataset generation
        │
        ▼
HEF compilation
```

The calibration datasets used during Hailo optimization are stored in:

```
data/calibration/
```

---

# Hailo Deployment

The Hailo models were generated using the Hailo AI Software Suite running inside the official Docker environment.

The deployment workflow consisted of:

1. Exporting the fine-tuned ESPCN model to ONNX.
2. Generating a calibration dataset from DIV2K.
3. Performing INT8 optimization.
4. Compiling the optimized model into a HEF executable.

The repository provides the final compiled models:

```
weights/espcn_x2_finetuned_224_fixed.hef
weights/espcn_x2_finetuned_640_fixed.hef
```

The intermediate HAR, ALLS and compiler configuration files generated during the Hailo compilation process are not included in this repository. Therefore, the repository provides the final HEF models together with the scripts used for their evaluation.

The 224×224 model was successfully compiled using Hailo optimization level 2.

The 640×640 model exceeded the optimization constraints of the Hailo compiler and could only be compiled using optimization level 0. Consequently, the resulting HEF provides a functional deployment but lower reconstruction quality than the optimized model.

---

# Running the Benchmarks

All benchmarks below were executed on Raspberry Pi 5.

## Traditional interpolation

```bash
python src/hailo/benchmark_traditional.py
```

## PyTorch

```bash
python src/hailo/benchmark_pytorch.py
```

## ONNX Runtime

```bash
python src/hailo/benchmark_onnx.py
```

## Hailo-8L

```bash
python src/hailo/benchmark_hailo.py
```

Benchmark results are automatically saved inside

```
results/csv/
```

---

# Real-Time Demonstration

The live demonstration uses the Raspberry Pi Camera Module 3 together with the Hailo-8L accelerator.

Pipeline:

```
Camera
    │
    ▼
Grayscale conversion
    │
    ▼
ESPCN (Hailo)
    │
    ▼
Super-resolved frame
    │
    ▼
MJPEG streaming
    │
    ▼
Remote visualization
```

Run:

```bash
python src/hailo/demo_hailo_mjpeg.py
```

The live demo continuously reports:

- FPS
- Acquisition time
- Pre-processing time
- Hailo inference time
- Post-processing time
- JPEG encoding time
- CPU utilization
- Memory utilization

---

# Main Results

The project evaluates:

- PSNR
- Visual Information Fidelity (VIF)
- Latency
- FPS
- Power consumption
- Energy per frame

Measurements were collected for:

- Nearest Neighbor
- Bicubic
- ESPCN (PyTorch)
- ESPCN (ONNX Runtime)
- ESPCN (Hailo-8L)

using the following input resolutions:

- 64×64
- 112×112
- 224×224
- 480×640
- 640×640

---

# Weights

Included models:

```
weights/
├── Original Weights/
├── Finetuned/
├── espcn_x2_finetuned_224_fixed.hef
└── espcn_x2_finetuned_640_fixed.hef
```

---

# Report

The complete technical report describing:

- methodology,
- implementation,
- benchmarking procedure,
- deployment,
- experimental evaluation,
- discussion,

is available in

```
report/
```

---

# Acknowledgments

This project was developed as part of the TinyML Image Super-Resolution Benchmark project at the University of Trento.

The ESPCN implementation is based on the open-source PyTorch implementation released by Lornatang and the original ESPCN architecture proposed by Shi et al.