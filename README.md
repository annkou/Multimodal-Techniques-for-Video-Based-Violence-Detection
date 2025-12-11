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
├── actions/                     # Dataset extraction and utility functions
│   ├── __init__.py
│   ├── evaluate.py              # Evaluation metrics and analysis
│   ├── extract_dataset.py       # XD-Violence dataset extraction
│   ├── extract_metadata.py      # Video metadata extraction
│   └── helpers.py               # Utility functions (GCS, label parsing)
├── data/                         # Data storage
│   ├── labels.json              # Violence category labels and taxonomy
│   ├── ontology.json            # AudioSet ontology for audio events
│   ├── metadata/                # Video metadata and links
│   │   └── video_links_labels.csv
│   ├── prompts/                 # LLM prompts for different tasks
│   │   ├── llm_judge.txt       # LLM-as-a-judge prompt
│   │   ├── video_understanding_gemini.txt
│   │   └── video_understanding_qwen_*.txt
│   ├── results/                 # Model outputs (gitignored)
│   │   └── zero_shot_classification/
│   └── videos/                  # Video files (gitignored)
├── external/                     # External model checkpoints
│   └── beats/                   # BEATs audio model files
│       ├── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
│       ├── Beats.py
│       └── ...
├── models/                       # Model implementations
│   ├── __init__.py
│   ├── beats.py                 # BEATs audio model wrapper
│   ├── clap.py                  # CLAP audio-text model
│   ├── clip.py                  # CLIP vision-text model
│   ├── gemini.py                # Google Gemini multimodal LLM
│   ├── gpt.py                   # OpenAI GPT models
│   ├── ollama.py                # Ollama local LLM wrapper
│   ├── qwen.py                  # Qwen-VL multimodal LLM
│   ├── qwen_judge.py            # Qwen LLM-as-a-judge
│   ├── whisper.py               # Whisper speech recognition
│   ├── xclip.py                 # X-CLIP video-text model
│   └── vad.py                   # Voice activity detection
├── notebooks/                    # Jupyter notebooks for experiments
│   ├── download_dataset.ipynb
│   ├── extract_metadata.ipynb
│   └── zero_shot_classification/
│       ├── run_vision_models.ipynb
│       ├── run_audio_models.ipynb
│       ├── get_gemini_transcripts.ipynb
│       ├── get_qwen_transcripts.ipynb
│       ├── get_whisper_transcripts.ipynb
│       ├── llm_as_a_judge.ipynb
│       └── evaluate_results.ipynb
├── .env                          # Environment variables (not in git)
├── .gitignore
├── pyproject.toml               # Poetry dependencies
├── poetry.lock
└── README.md
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

4. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   # API Keys (obtain from respective services)
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   DASHSCOPE_API_KEY=your_dashscope_api_key_here
   
   # Google Cloud (optional - only if using GCS)
   GOOGLE_APPLICATION_CREDENTIALS=path/to/service-account-key.json
   BUCKET_NAME=your_gcs_bucket_name
   
   # Local Paths
   LABELS_PATH=./data/labels.json
   ONTOLOGY_JSON_PATH=./data/ontology.json
   VIDEO_METADATA_PATH=./data/metadata/video_metadata.csv
   BEATS_MODEL_PATH=./external/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
   VIDEOS_PATH=./data/videos
   ```

5. **Download model checkpoints**
   
   Download the BEATs model checkpoint:
   - Download from: [BEATs GitHub](https://github.com/microsoft/unilm/tree/master/beats)
   - Place in: `external/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`

6. **Download the dataset** (optional)
   
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

The following services are needed depending on which models you use:

1. **Alibaba DashScope** (for Qwen-VL models)
   - Sign up at: https://dashscope.aliyun.com/
   - Get API key and add to `.env` as `DASHSCOPE_API_KEY`
   - Used for: Video understanding, transcription generation

2. **Google Gemini** (for Gemini models)
   - Sign up at: https://ai.google.dev/
   - Get API key and add to `.env` as `GEMINI_API_KEY`
   - Used for: Multimodal video analysis, transcript generation

3. **OpenAI** (for GPT models)
   - Sign up at: https://platform.openai.com/
   - Get API key and add to `.env` as `OPENAI_API_KEY`
   - Used for: LLM-as-a-judge evaluations

4. **Google Cloud Storage** (optional)
   - Required only if hosting videos on GCS
   - Create a service account with Storage permissions
   - Download credentials JSON and set path in `.env`

### Dataset Citation

This project uses the XD-Violence dataset. If you use it, please cite:

- Original dataset: [XD-Violence](https://roc-ng.github.io/XD-Violence/)
- Adapted from: [XD-Violence on Hugging Face](https://huggingface.co/datasets/jherng/xd-violence)

## Usage

### Running Experiments

The project provides Jupyter notebooks for different stages of the violence detection pipeline:

1. **Extract Metadata**
   ```bash
   jupyter notebook notebooks/extract_metadata.ipynb
   ```
   Extract video metadata including duration, resolution, fps, etc.

2. **Run Vision Models** (CLIP, X-CLIP)
   ```bash
   jupyter notebook notebooks/zero_shot_classification/run_vision_models.ipynb
   ```
   Perform zero-shot classification using vision-text models.

3. **Run Audio Models** (CLAP, BEATs)
   ```bash
   jupyter notebook notebooks/zero_shot_classification/run_audio_models.ipynb
   ```
   Analyze audio content for violence-related sounds.

4. **Generate Transcripts**
   - **Gemini:** `notebooks/zero_shot_classification/get_gemini_transcripts.ipynb`
   - **Qwen:** `notebooks/zero_shot_classification/get_qwen_transcripts.ipynb`
   - **Whisper:** `notebooks/zero_shot_classification/get_whisper_transcripts.ipynb`

5. **LLM-as-a-Judge Evaluation**
   ```bash
   jupyter notebook notebooks/zero_shot_classification/llm_as_a_judge.ipynb
   ```
   Use LLMs to judge violence detection results based on multimodal evidence.

6. **Evaluate Results**
   ```bash
   jupyter notebook notebooks/zero_shot_classification/evaluate_results.ipynb
   ```
   Analyze and compare performance across different models.


## 📊 Models & Capabilities

| Model Category | Model Name | Input Modality | Use Case |
|---------------|------------|----------------|----------|
| Vision | CLIP | Image/Video | Zero-shot image classification |
| Vision | X-CLIP | Video | Video-level classification |
| Audio | CLAP | Audio + Text | Audio-text similarity |
| Audio | BEATs | Audio | Audio event detection |
| Multimodal LLM | Gemini 2.0 | Video + Audio + Text | Video understanding |
| Multimodal LLM | Qwen-VL | Video + Audio + Text | Video analysis |
| Speech | Whisper | Audio | Speech-to-text |
| Judge | GPT-5 / Qwen | Text | Result evaluation |

