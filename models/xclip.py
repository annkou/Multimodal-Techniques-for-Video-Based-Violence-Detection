import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from decord import VideoReader, cpu
from transformers import XCLIPModel, XCLIPProcessor


class XCLIPVideoClassifier:
    """
    Wrapper for XCLIP zero-shot video classification.
    Handles model loading, frame sampling, batch inference, and result formatting.
    """

    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch16-zero-shot",
        clip_len: int = 32,
        frame_sample_rate: int = 2,
    ):
        """
        Args:
            model_name: HuggingFace model name for XCLIP.
            clip_len: Number of frames per segment (default: 32).
            frame_sample_rate: Interval between sampled frames (default: 2).
        """
        self.model_name = model_name
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor, self.model = self.load_model()
        self.model = self.model.to(self.device)

    def load_model(self):
        """
        Loads XCLIP processor and model from HuggingFace.
        Returns:
            processor: XCLIPProcessor object.
            model: XCLIPModel object.
        """
        processor = XCLIPProcessor.from_pretrained(self.model_name)
        model = XCLIPModel.from_pretrained(self.model_name)
        print(f"Loaded XCLIP model: {self.model_name}")
        return processor, model

    def sample_all_segments_indices(self, seg_len: int) -> list[np.ndarray]:
        """
        Divide the video into non-overlapping segments and sample indices for each segment.

        Args:
        seg_len (int): Total number of frames in the video.

        Returns:
          segments (list of np.ndarray): Each array contains the indices of frames for one segment.

        Ensures all parts of a long video are sampled.
        """
        converted_len = int(self.clip_len * self.frame_sample_rate)
        segments = []
        for start in range(0, seg_len - converted_len + 1, converted_len):
            end = start + converted_len
            indices = np.linspace(start, end, num=self.clip_len)
            indices = np.clip(indices, start, end - 1).astype(np.int64)
            segments.append(indices)
        return segments

    def process_video(
        self,
        video_path: str,
        labels: list[str],
        top_k: int = 3,
    ):
        """
        Processes a video file, runs XCLIP inference on sampled segments, and returns results.
        Args:
            video_path: Path to video file.
            labels: List of candidate label descriptions.
            top_k: Number of top labels per chunk.
        Returns:
            List of result dicts per segment.
        """
        videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        fps = videoreader.get_avg_fps()
        seg_indices_list = self.sample_all_segments_indices(seg_len=len(videoreader))
        all_results = []
        for seg_num, indices in enumerate(seg_indices_list):
            video = videoreader.get_batch(indices).asnumpy()
            inputs = self.processor(
                text=labels, videos=list(video), return_tensors="pt", padding=True
            )
            # Move all tensor inputs to the correct device
            for k in inputs:
                if isinstance(inputs[k], torch.Tensor):
                    inputs[k] = inputs[k].to(self.device)
            # forward pass
            # with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_video.softmax(dim=1)
            # Sort labels by probability (descending) and keep top k
            label_probs = [
                (labels[i], round(float(probs[0][i].item()), 3))
                for i in range(len(labels))
            ]
            label_probs_sorted = dict(
                sorted(label_probs, key=lambda x: x[1], reverse=True)[:top_k]
            )
            start_frame = int(indices[0])
            end_frame = int(indices[-1])
            start_time = round(start_frame / fps, 3)
            end_time = round(end_frame / fps, 3)
            all_results.append(
                {
                    "video_path": os.path.basename(video_path),
                    "start_time": start_time,
                    "end_time": end_time,
                    "modality": "vision",
                    "labels": label_probs_sorted,
                }
            )
        return all_results

    async def process_all_videos(
        self,
        video_paths: list[str],
        labels: list[str],
        output_json: str = "xclip_results.json",
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
            top_k: Number of top labels per chunk.
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
                results.extend(video_result)
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
        return results
