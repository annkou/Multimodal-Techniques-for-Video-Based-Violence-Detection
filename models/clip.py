import asyncio
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import torch
from PIL import Image
from transformers import pipeline


class CLIPModel:
    """
    Wrapper for OpenAI CLIP zero-shot image classification.
    Handles model loading, frame extraction, batch inference, and result formatting.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        sample_rate: int = 10,
        batch_size: int = 8,
    ):
        """
        Args:
            model_name: HuggingFace model name for CLIP.
            sample_rate: Number of frames to skip between samples (e.g., 10 = every 10th frame).
            batch_size: Number of frames to process in each batch.
        """
        self.model_name = model_name
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
        self.model = self.load_model()

    def load_model(self):
        """
        Loads CLIP model from HuggingFace.
        Returns:
            CLIP pipeline object.
        """
        clip = pipeline(
            task="zero-shot-image-classification",
            model=self.model_name,
            dtype=torch.bfloat16,
            device=self.device,
        )
        print(f"Loaded CLIP model: {self.model_name}")
        return clip

    def extract_frames_with_times(self, video_path: str):
        """
        Extracts frames and their timestamps from a video.
        Args:
            video_path: Path to video file.
        Returns:
            frames: List of frame arrays.
            frame_times: List of timestamps (seconds) for each frame.
        """
        frames = []
        frame_times = []
        cap = cv.VideoCapture(video_path)
        fps = cap.get(cv.CAP_PROP_FPS)
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        # print(f"Total frames: {total_frames}")
        for i in range(0, total_frames, self.sample_rate):
            cap.set(cv.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                frame_times.append(i / fps if fps else 0)
        cap.release()
        return frames, frame_times

    def analyze_frames(
        self, frames: list[any], candidate_labels: list[str], top_k: int = 3
    ) -> list[dict[str, float]]:
        """
        Runs CLIP zero-shot classification on frames in batches.
        Args:
            frames: List of frame arrays.
            candidate_labels: List of label descriptions.
            top_k: Number of top labels.
        Returns:
            List of dicts: {label: score, ...} for top 3 labels per frame.
        """
        results = []
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i : i + self.batch_size]
            for frame in batch:
                pil_image = Image.fromarray(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                result = self.model(image=pil_image, candidate_labels=candidate_labels)
                # Sort and keep top 3 labels
                label_probs = [
                    (item["label"], round(float(item["score"]), 3)) for item in result
                ]
                label_probs_sorted = dict(
                    sorted(label_probs, key=lambda x: x[1], reverse=True)[:top_k]
                )
                results.append(label_probs_sorted)
        return results

    def process_video(
        self, video_path: str, candidate_labels: list[str], top_k: int = 3
    ):
        """
        Processes a video file, runs CLIP inference on sampled frames, and returns results.
        Args:
            video_path: Path to video file.
            candidate_labels: List of label descriptions.
            top_k: Number of top labels.
        Returns:
            List of result dicts per frame.
        """
        frames, frame_times = self.extract_frames_with_times(video_path)
        clip_results = self.analyze_frames(frames, candidate_labels, top_k)
        output = []
        for idx, label_probs_sorted in enumerate(clip_results):
            start_time = round(frame_times[idx], 3)
            end_time = round(frame_times[idx], 3)  # single frame, so start == end
            output.append(
                {
                    "start_time": start_time,
                    "end_time": end_time,
                    "labels": label_probs_sorted,
                }
            )
        return {
            "video_path": os.path.basename(video_path),
            "response": output,
            "modality": "vision",
        }

    async def process_all_videos(
        self,
        video_paths: list[str],
        candidate_labels: list[str],
        output_json: str = "clip_results.json",
        overwrite: bool = False,
        top_k: int = 3,
    ):
        """
        Processes all videos asynchronously and saves results to JSON.
        Args:
            video_paths: List of video file paths.
            candidate_labels: List of label descriptions.
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
                    executor, self.process_video, video_path, candidate_labels, top_k
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
