# HumanStreets: Walkability Analysis & Interactive AI Dashboard

**HumanStreets** is a comprehensive project integrating geospatial analysis, deep learning for computer vision, and large language models (LLMs) to assess and visualize urban walkability. This repository houses the complete pipeline: from satellite imagery processing and dataset generation to model fine-tuning and a real-time interactive dashboard.

## 🌟 Project Overview

This project is divided into three main components:

1.  **Dashboard**: A full-stack application providing a real-time chat interface with a localized LLM and an interactive 3D map for walkability visualization.
2.  **Data Preparation**: Automated pipelines to process satellite imagery (Sentinel-2/High-Res), detect sidewalks using SAM3/YOLO, and generate walkability scores.
3.  **Fine-Tuning (PEFT)**: Research and implementation of Parameter-Efficient Fine-Tuning (QLoRA) to adapt the LiquidAI LFM2-1.2B model for specific dialects (Saudi Dialect).

---

## 📂 Project Structure

```text
HumanStreets/
├── dashboard/               # Full-stack Application (FastAPI + React)
│   ├── backend/             # Python API, Model Inference, SSE logic
│   ├── frontend/            # React + Vite + Deck.gl visualization
│   └── PROJECT_DOCUMENTATION.md
├── data_prep/               # Computer Vision & Geospatial Pipelines
│   ├── generate_yolo_dataset.py # Sidewalk segmentation using SAM3
│   ├── visualize_dataset_map.py # Validation tools
│   └── yolo_training/       # Training artifacts
├── fineTuning_PEFT/         # LLM Customization
│   ├── finetuning_QLora.py  # QLoRA training script
│   ├── fine_tuning_docs.md  # Detailed methodology
│   └── saudi-lora-test/     # Model checkpoints
└── README.md                # This file
```

---

## 1. Interactive Dashboard

The dashboard serves as the user interface for the project, combining a chat assistant with a high-performance geospatial map.

*   **Tech Stack**: React, Vite, Deck.gl, MapLibre (Frontend); FastAPI, Uvicorn, Transformers (Backend).
*   **Key Features**:
    *   Real-time streaming chat with `LiquidAI/LFM2-1.2B`.
    *   Interactive map centered on NYC (demo) with walkability overlays.
    *   Responsive, dark-themed UI.

### Architecture
![Architecture Diagram](https://mermaid.ink/img/Z3JhcGggVEQKICAgIFVzZXJbVXNlcl0gLS0+fEludGVyYWN0c3wgRnJvbnRlbmRbUmVhY3QgRnJvbnRlbmRdCgogICAgc3ViZ3JhcGggRnJvbnRFbmQgWyJGcm9udGVuZCBMb2dpYyJdCiAgICAgICAgQXBwW0FwcC5qc3hdIC0tPnxNYW5hZ2VzIFN0YXRlfCBDaGF0U2lkZWJhcgogICAgICAgIEFwcCAtLT58UmVuZGVyc3wgTWFwQ29udGFpbmVyCiAgICAgICAgQ2hhdFNpZGViYXJbQ2hhdFNpZGViYXIgQ29tcG9uZW50XSAtLT58RGlzcGxheXN8IENoYXRVSQogICAgICAgIE1hcENvbnRhaW5lcltNYXBDb250YWluZXIgQ29tcG9uZW50XSAtLT58UmVuZGVyc3wgRGVja0dML01hcExpYnJlCiAgICBlbmQKCiAgICBGcm9udGVuZCAtLT58SFRUUCBQT1NUIC9jaGF0fCBCYWNrZW5kW0Zhc3RBUEkgQmFja2VuZF0KICAgIEZyb250ZW5kIC0tPnxIVFRQIEdFVCAvaGVhbHRofCBCYWNrZW5kCgogICAgc3ViZ3JhcGggQmFja0VuZCBbIkJhY2tlbmQgTG9naWMiXQogICAgICAgIEJhY2tlbmQgLS0+fExvYWRzfCBNb2RlbFtMaXF1aWRBSS9MRk0yLTEuMkJdCiAgICAgICAgQmFja2VuZCAtLT58VG9rZW5pemVzfCBUb2tlbml6ZXIKICAgICAgICBNb2RlbCAtLT58U3RyZWFtcyBUb2tlbnN8IFN0cmVhbWVyW1RleHRJdGVyYXRvclN0cmVhbWVyXQogICAgICAgIFN0cmVhbWVyIC0tPnxZaWVsZHMgQ2h1bmtzfCBCYWNrZW5kCiAgICBlbmQKCiAgICBCYWNrZW5kIC0uLT58U1NFLWxpa2UgU3RyZWFtfCBGcm9udGVuZAo=)

> [!NOTE]
> For detailed architecture, API endpoints, and component breakdowns, please read the **[Dashboard Documentation](./dashboard/PROJECT_DOCUMENTATION.md)**.
> For frontend-specific setup, see the **[Frontend README](./dashboard/frontend/README.md)**.

### Quick Start
To run the demo (requires configured environment):
```bash
# Windows
.\dashboard\start_demo.bat
```

---

## 2. Data Preparation (Walkability Scoring)

This module handles the extraction of sidewalk feature data from raw satellite imagery to calculate walkability scores.

*   **Core Script**: `generate_yolo_dataset.py`
    *   **Method**: Uses **SAM3 (Segment Anything Model 3)** to generate high-quality segmentation masks for sidewalks from raw `.tif` satellite imagery.
    *   **Output**: Creates a localized dataset (images/labels) strictly meant for training YOLO models for efficient real-time inference later.
*   **Validation**: `visualize_dataset_map.py` allows identifying and visualising the generated datasets on a map to ensure ground truth alignement.

### Workflow
1.  Place raw GeoTIFF images in `data_prep/raw_images/`.
2.  Run `python data_prep/generate_yolo_dataset.py` to tile images and generate segmentation masks.
3.  (Optional) Train a YOLO model using the output in `datasets/sidewalk_segmentation`.

---

## 3. LLM Fine-Tuning (Saudi Dialect)

We address the need for localized AI assistants by fine-tuning the `LiquidAI/LFM2-1.2B` model on the **SauDial** dataset.

*   **Technique**: **QLoRA** (Quantized Low-Rank Adaptation).
*   **Why QLoRA?**: Enables fine-tuning a 1.2B parameter model on consumer-grade GPUs (e.g., 6GB VRAM) by using 4-bit quantization.
*   **Dataset**: `SauDial Dataset.xlsx` - English to Saudi Dialect translations.

> [!IMPORTANT]
> The complete training methodology, including quantization configs, LoRA parameters, and performance notes, is documented in **[Fine-Tuning Docs](./fineTuning_PEFT/fine_tuning_docs.md)**.

### How to Train
```bash
cd fineTuning_PEFT
uv run --with openpyxl --with pandas python finetuning_QLora.py
```
*Note: Requires `uv` package manager.*

---

## 🚀 Getting Started

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/HumanStreets.git
    cd HumanStreets
    ```

2.  **Environment Setup**:
    *   This project relies on Python 3.10+ and Node.js 18+.
    *   We recommend using `uv` or `venv` for Python dependency management.
    *   Install backend libs: `pip install -r requirements.txt` (and `requirements.txt` inside subfolders if present).
    *   Install frontend libs: `cd dashboard/frontend && npm install`.

3.  **Explore**:
    *   Start with the **Dashboard** to see the end-user experience.
    *   Dive into **Fine-Tuning** to see how we customized the model.
    *   Check **Data Prep** to understand how we turn pixels into walkability scores.

---


