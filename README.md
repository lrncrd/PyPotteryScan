# PyPotteryScan

<div align="center">

<img src="app/static/imgs/LogoScan.png" width="350"/>

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Open Source](https://img.shields.io/badge/Open%20Source-community--driven-green.svg)](https://lrncrd.github.io/PyPottery/community.html)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)](https://github.com/lrncrd/PyPotteryScan)
[![GPU Support](https://img.shields.io/badge/GPU-CUDA-green.svg)](https://github.com/lrncrd/PyPotteryScan)
[![Status](https://img.shields.io/badge/Status-Development-orange)](https://github.com/lrncrd/PyPotteryScan)

Digitize scanned pottery plates: extract drawings, read the text with OCR, export clean data

</div>

---

## Introduction

As part of the [**PyPottery**](https://github.com/lrncrd/PyPottery) toolkit, **PyPotteryScan** is a Flask-based web application for processing and digitizing archaeological pottery drawings with OCR (Optical Character Recognition). It provides a complete workflow for extracting individual drawings from scanned plates, recognizing their text annotations, and preparing clean images for digital cataloging.

## ✨ Features

- **Project Management**: every archaeological dataset gets its own workspace with dedicated folders and metadata tracking
- **Image Loading**: import scanned plates from a folder, with cached thumbnails for fast navigation
- **Interactive Annotation**: canvas tools to mark pottery profiles and text boxes on plates
- **OCR Processing**: local vision-language models — the lightweight **GLM-OCR** (CPU or GPU) or the heavier **OlmOCR-FP4** (NVIDIA GPU) — or skip OCR and type the text yourself
- **Automatic Cropping**: extract individual drawings from your annotations
- **Drawing Cleanup**: eraser with undo, plus a straighten tool with alignment grid
- **Text Review**: review and correct OCR results with zoom
- **Few-Shot Parsing**: structured parsing of recognized text into fields, using a small local language model
- **Full Persistence**: annotations, crops, cleaned drawings and OCR results are saved automatically
- **Export**: one ZIP with renamed drawings, an Excel/CSV catalogue with all metadata and OCR text, and the box coordinates for machine learning

## 🚀 Quick Start

### Option 1 — PyPottery Suite Launcher (recommended)

The easiest way to get started, no Python installation required.

<p align="center">
  <a href="https://github.com/lrncrd/PyPottery/releases/latest">
    <img src="https://img.shields.io/badge/Download-PyPottery%20Launcher-667eea?style=for-the-badge&logoColor=white" alt="Download Launcher">
  </a>
</p>

1. Grab the installer for your OS from [Releases](https://github.com/lrncrd/PyPottery/releases/latest)
2. Run it (Windows) or drag-to-Applications (macOS) — no Python install required
3. Launch PyPotteryScan from the suite launcher; updates are handled automatically

### Option 2 — Manual installation (from source)

For developers, or anyone who wants to run the app on its own:

```bash
# Clone repository
git clone https://github.com/lrncrd/PyPotteryScan.git
cd PyPotteryScan

# Install dependencies (includes PyTorch)
pip install -r requirements.txt

# Run the app
python app.py
# Then open http://127.0.0.1:5002 in your browser
```

On first launch you choose which OCR model to download from HuggingFace (**GLM-OCR**, **OlmOCR-7B-FP4** ~5GB for NVIDIA GPUs, or none). A small text model (**Qwen3.5-2B**) is always downloaded for few-shot parsing; if that download fails the app still starts, with only structured parsing disabled. Models are cached in `models/`. For CUDA-specific PyTorch builds and platform notes, see the [Getting Started guide](https://lrncrd.github.io/PyPottery/pypotteryscan/index.html). Installer scripts are also provided: `PyPotteryScan_WIN.bat` and `PyPotteryScan_UNIX.sh`.

## 📋 System Requirements

- **Python**: 3.12 (tested)
- **Operating System**: Windows/macOS/Linux
- **Memory**: 16GB RAM minimum (32GB recommended for OCR)
- **GPU** (optional): NVIDIA with CUDA for faster OCR; OlmOCR-FP4 requires an NVIDIA GPU

## 🎯 Usage

1. **Create a project** and select the folder with your scanned plates (JPEG, PNG, WebP, TIFF, BMP)
2. **Annotate**: draw rectangles around pottery profiles and around text boxes
3. **Process OCR** on all text boxes (or skip and type the text later)
4. **Generate crops** of the individual drawings
5. **Clean** the drawings: erase text, straighten tilted profiles
6. **Review** and correct the recognized texts
7. **Export** a ZIP with renamed drawings, an Excel/CSV catalogue and the box coordinates

For the full walkthrough, project folder structure and keyboard shortcuts, see the **[Usage Guide](https://lrncrd.github.io/PyPottery/pypotteryscan/usage.html)**.

## 📊 What's New

See the **[Version History](https://lrncrd.github.io/PyPottery/pypotteryscan/version_history.html)** for the full changelog.

## 🤝 Contributing

Contributions are welcome: report bugs with reproduction steps, suggest features, improve the documentation, test on different platforms, or open a pull request.

## 🙏 Acknowledgments

**OlmOCR** (Allen AI), **GLM-OCR** (Z.ai), **Qwen** (Alibaba Cloud), **HuggingFace** for model hosting and `transformers`, and the Flask community.

## 👥 Contributors

<a href="https://github.com/lrncrd/PyPotteryScan/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=lrncrd/PyPotteryScan" />
</a>

## ☕ Support This Project

If you find PyPotteryScan useful for your research, consider supporting its development:

[![Ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/lrncrd)

Your support helps maintain and improve this open-source tool for the archaeological community!

---

Developed with ❤️ by [Lorenzo Cardarelli](https://github.com/lrncrd)
