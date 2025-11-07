# Multimodal Techniques for Video-Based Violence Detection

A comprehensive framework for detecting violence in videos using multiple state-of-the-art AI models across different modalities (vision, audio, and text).

## Overview

This project implements a multimodal approach to violence detection in videos by leveraging:
- **Vision Models**: CLIP, X-CLIP for visual content analysis
- **Audio Models**: CLAP, BEATs for audio event detection
- **Multimodal LLMs**: Qwen-VL, Gemini for video understanding and transcription
- **Speech Recognition**: Whisper for audio transcription
- **Voice Activity Detection (VAD)**: For identifying speech segments

## Features

- Zero-shot video classification across multiple modalities
- Support for both local and cloud-based video processing
- Batch processing with rate limiting and resume capability
- Comprehensive metadata tracking (tokens, costs, errors)
- Async processing for efficient API calls

## 📁 Project Structure

```
├── actions/              # Dataset extraction and utility functions
├── data/                 # Data storage
│   ├── labels.json      # Violence category labels
│   ├── ontology.json    # Audio event ontology
│   ├── metadata/        # Video URLs and labels
│   ├── results/         # Model outputs (gitignored)
│   └── prompts/         # LLM prompts
├── external/            # External model checkpoints (BEATs)
├── models/              # Model implementations
│   ├── beats.py        # BEATs audio model
│   ├── clap.py         # CLAP audio-text model
│   ├── clip.py         # CLIP vision-text model
│   ├── gemini.py       # Google Gemini multimodal LLM
│   ├── qwen.py         # Qwen-VL multimodal LLM
│   ├── whisper.py      # Whisper speech recognition
│   ├── xclip.py        # X-CLIP video-text model
│   └── vad.py          # Voice activity detection
└── notebooks/           # Jupyter notebooks for experiments
    └── zero_shot_classification/
```

## Setup

### Prerequisites

- Python 3.11+
- Poetry (for dependency management)
- GPU with CUDA 12.6+ (optional, but recommended)
- API keys for cloud services (see below)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/annkou/Multimodal-Techniques-for-Video-Based-Violence-Detection.git
   cd Multimodal-Techniques-for-Video-Based-Violence-Detection
   ```

2. **Install dependencies using Poetry**
   ```bash
   # Install Poetry if you haven't already
   pip install poetry

   # Install project dependencies
   poetry install
   ```

3. **Install PyTorch with CUDA support** (if using GPU)
   
   Poetry method:
   ```bash
   # Add PyTorch source to pyproject.toml (already configured)
   poetry add torch torchvision torchaudio --source pytorch-gpu
   ```
   
   Or pip method (inside Poetry shell):
   ```bash
   poetry shell
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

4. **Download model checkpoints**
   
   Download the BEATs model checkpoint:
   - Download from: [BEATs GitHub](https://github.com/microsoft/unilm/tree/master/beats)
   - Place in: `external/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`

5. **Download the dataset** (optional)
   
   If you want to work with the XD-Violence dataset:
   ```bash
   # Use the provided notebook
   jupyter notebook notebooks/download_dataset.ipynb
   ```
   
   Or extract video URLs and labels only (no download):
   ```python
   from actions.extract_dataset import XDViolence, extract_video_links_and_labels_to_csv
   
   dataset = XDViolence()
   extract_video_links_and_labels_to_csv(dataset, "data/metadata/video_links_labels.csv")
   ```

## 🔑 API Keys & Services

### Required Services

1. **Alibaba DashScope** (for Qwen-VL models)
   - Sign up at: https://dashscope.aliyun.com/
   - Get API key and set as `DASHSCOPE_API_KEY`

2. **Google Gemini** (for Gemini models)
   - Sign up at: https://ai.google.dev/
   - Get API key and set as `GEMINI_API_KEY`

3. **Google Cloud Storage** (optional, for video hosting)

### Dataset Citation

This project uses the XD-Violence dataset. If you use it, please cite:

- Original dataset: [XD-Violence](https://roc-ng.github.io/XD-Violence/)
- Adapted from: [XD-Violence on Hugging Face](https://huggingface.co/datasets/jherng/xd-violence)
