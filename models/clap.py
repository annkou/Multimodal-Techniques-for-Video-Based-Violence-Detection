import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import librosa
import numpy as np
import torch
from transformers import pipeline


class CLAPModel:
    """
    Wrapper for CLAP (Contrastive Language-Audio Pretraining) audio classification model.
    Handles model loading, audio preprocessing, chunking, inference, and batch processing.
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-fused",
        sr: int = 48000,
        window_s: float = 10.0,
        hop_s: float = 10.0,
    ):
        """
        Args:
            model_name: HuggingFace model name for CLAP.
            sr: Target sample rate for audio (CLAP expects 48kHz).
            window_s: Window size in seconds for chunking.
            hop_s: Hop size in seconds for chunking.
        """
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        self.model_name = model_name
        self.sr = sr
        self.window_s = window_s
        self.hop_s = hop_s
        self.model = self.load_model()

    def load_model(self):
        """
        Loads CLAP model from HuggingFace.
        Returns:
            CLAP pipeline object.
        """
        clap = pipeline(
            task="zero-shot-audio-classification",
            model=self.model_name,
            device=self.device,
        )
        print("Loaded CLAP model:", clap.feature_extractor)
        return clap

    def has_audio(self, video_path: str) -> bool:
        cmd = f'ffprobe -v error -print_format json -show_streams "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        info = json.loads(result.stdout)
        return any(s.get("codec_type") == "audio" for s in info["streams"])

    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Loads audio from file and resamples to target sample rate.
        Args:
            audio_path: Path to audio or video file.
        Returns:
            1D numpy array of audio samples.
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return audio

    def chunk_and_embed_audio(
        self, audio: np.ndarray, labels: list[str], top_k: int = 3
    ):
        """
        Chunks audio into windows and runs CLAP inference on each chunk.
        Args:
            audio: 1D numpy array of audio samples.
            labels: List of candidate label descriptions.
            top_k: Number of top labels.
        Returns:
            Dictionary mapping (start_sec, end_sec) to top 3 label scores.
        """
        win = int(self.window_s * self.sr)
        hop = int(self.hop_s * self.sr)
        if len(audio) == 0:
            audio = np.zeros(win, dtype=np.float32)

        probs_as_dict = {}
        for start in range(0, len(audio), hop):
            seg = audio[start : start + win]  # Slices
            span = (start / self.sr, (start + len(seg)) / self.sr)
            if len(seg) < win:
                seg = np.pad(
                    seg, (0, win - len(seg))
                )  # Pads the last segment with zeros so every chunk is exactly win samples (10 s).
            # CLAP expects raw waveform; processor handles features
            output = self.model(seg, candidate_labels=labels)
            # output: [{"label": ..., "score": ...}, ...]
            label_scores = [(x["label"], round(float(x["score"]), 3)) for x in output]
            label_scores_sorted = dict(
                sorted(label_scores, key=lambda x: x[1], reverse=True)[:top_k]
            )
            probs_as_dict[span] = label_scores_sorted
        return probs_as_dict

    def process_video(self, video_path: str, labels: list[str], top_k: int = 3):
        """
        Processes a video file, runs CLAP inference on audio chunks, and returns results.
        Args:
            video_path: Path to video file.
            labels: List of candidate label descriptions.
            top_k: Number of top labels.
        Returns:
             Dict with video_path, response list, and modality.
        """
        if not self.has_audio(video_path):
            print(f"No audio stream found in {video_path}")
            return {
                "video_path": os.path.basename(video_path),
                "response": [],
                "modality": "audio",
            }
        audio = self.load_audio(video_path)
        probs_as_dict = self.chunk_and_embed_audio(audio, labels, top_k)
        response = []
        for span, score_dict in probs_as_dict.items():
            start_time, end_time = span
            response.append(
                {
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "labels": score_dict,
                }
            )
        return {
            "video_path": os.path.basename(video_path),
            "response": response,
            "modality": "audio",
        }

    async def process_all_videos(
        self,
        video_paths: list[str],
        labels: list[str],
        output_json: str = "clap_results.json",
        overwrite: bool = False,
        top_k: int = 3,
    ):
        """
        Processes all videos asynchronously and saves results to JSON.
        Args:
            video_paths: List of video file paths.
            labels: List of candidate label descriptions.
            output_json: Output JSON file path.
            overwrite: If True, overwrite existing file.
            top_k: Number of top labels.
        Returns:
            List of all result dicts.
        """

        loop = asyncio.get_event_loop()
        results = []
        with ThreadPoolExecutor() as executor:
            tasks = [
                loop.run_in_executor(
                    executor, self.process_video, video_path, labels, top_k
                )
                for video_path in video_paths
            ]
            for video_result in await asyncio.gather(*tasks):
                results.append(video_result)
        # Save results to JSON
        if overwrite or not os.path.exists(output_json):
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        else:
            with open(output_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.extend(results)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        print(f"Processed {len(results)} videos")
