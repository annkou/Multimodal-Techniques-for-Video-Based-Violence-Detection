import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import librosa
import torch

from external.beats.Beats import BEATs, BEATsConfig


class BeatsModel:
    """Wrapper for Beats audio classification model.
    Handles model loading, audio preprocessing, chunking, inference and batch processing."""

    def __init__(
        self,
        checkpoint_path: str = None,
        human_labels_path: str = None,
        sr: int = 16000,
        win_ms: int = 2000,
        hop_ms: int = 2000,
    ):
        """
        Args:
            checkpoint_path: Path to BEATs checkpoint file.
            human_labels_path: Path to ontology.json mapping label ids to human-readable names.
            sr: Target sample rate for audio.
            win_ms: Window size in milliseconds for chunking.
            hop_ms: Hop size in milliseconds for chunking.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sr = sr
        self.win_ms = win_ms
        self.hop_ms = hop_ms
        self.win_samples = int(
            self.sr * (self.win_ms / 1000.0)
        )  # number of audio samples in each window
        self.hop_samples = int(self.sr * (self.hop_ms / 1000.0))
        self.checkpoint_path = checkpoint_path
        self.human_labels_path = human_labels_path
        self.model, self.label_dict = self.load_model()

    def load_model(self):
        """
        Load BEATs fine-tuned checkpoint.
        Expects checkpoint to contain keys: 'cfg', 'model', 'label_dict'.

        Returns:
            model: Loaded BEATs model.
            label_dict: Mapping from class indices to label ids.
        """
        checkpoint = torch.load(self.checkpoint_path)
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        model.to(self.device).eval()
        label_dict = checkpoint["label_dict"]
        print("Loaded BEATs model")
        return model, label_dict

    def has_audio(self, video_path: str) -> bool:
        """
        Checks if a video file contains an audio stream.
        Args:
            video_path: Path to video file.
        Returns:
            True if audio stream exists, False otherwise.
        """
        cmd = f'ffprobe -v error -print_format json -show_streams "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        info = json.loads(result.stdout)
        return any(s.get("codec_type") == "audio" for s in info["streams"])

    def load_audio(self, audio_path: str) -> torch.Tensor:
        """
        Loads audio from file and resamples to target sample rate.
        Args:
            audio_path: Path to audio or video file.
        Returns:
            1D float tensor of audio samples.
        """
        audio, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        return torch.from_numpy(audio).float()

    def get_human_label_from_audio(self, audio_label: str) -> str:
        """
        Maps BEATs label id to human-readable label using ontology.json.
        Args:
            audio_label: Label id from BEATs output.
        Returns:
            Human-readable label name.
        """
        with open(self.human_labels_path, encoding="utf-8") as f:
            ontology = json.load(f)
        for obj in ontology:
            if obj["id"] == audio_label:
                return obj["name"]
        return "Label not found"

    def chunk_waveform_with_spans(
        self, wav: torch.Tensor, pad_last: bool = False
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[float, float]]]:
        """
        Splits waveform into fixed-length windows with optional padding.
        hop=win for no overlap
        hop<win for overlap
        If the last window is shorter than win and pad_last=True, zero-pads the tail to reach win.
        Args:
            wav: 1D tensor of audio samples.
            pad_last: If True, pad last chunk if shorter than window.
        Returns:
            chunks: List of waveform tensors (win_samples,)
            masks: List of bool tensors (win_samples,) (True=padded)
            spans: List of (start_time, end_time) tuples in seconds.
        """
        chunks, masks, spans = [], [], []
        T = wav.numel()  # Total number of samples
        win = self.win_samples
        hop = self.hop_samples
        sr = self.sr
        # If input is empty and padding is requested, return a single padded chunk
        if T == 0 and pad_last:
            pad = torch.zeros(win, dtype=wav.dtype)
            mask = torch.zeros(win, dtype=torch.bool)
            chunks.append(pad)
            masks.append(mask)
            spans.append((0.0, win / sr))
            return chunks, masks, spans
        # Main chunking loop
        for start in range(0, max(T - win, 0) + 1, hop):
            end = start + win
            seg = wav[start:end]
            # If segment is shorter than window, pad if requested
            if seg.numel() < win:
                if not pad_last:
                    continue
                pad = torch.zeros(win - seg.numel(), dtype=seg.dtype)
                seg = torch.cat([seg, pad], 0)
                mask = torch.zeros(win, dtype=torch.bool)
                mask[win - pad.numel() :] = True  # Mark padded tail
            else:
                mask = torch.zeros(win, dtype=torch.bool)
            chunks.append(seg)
            masks.append(mask)
            spans.append((start / sr, (start + win) / sr))
        # Handle very short audio (< win)
        if not chunks and pad_last and T < win:
            pad = torch.zeros(win - T, dtype=wav.dtype)
            seg = torch.cat([wav, pad], 0)
            mask = torch.zeros(win, dtype=torch.bool)
            mask[T:] = True
            chunks.append(seg)
            masks.append(mask)
            spans.append((0.0, win / sr))
        return chunks, masks, spans

    def collate(self, chunks: list[torch.Tensor], masks: list[torch.Tensor]):
        """
        Stacks all windows into a batch so the model can process many windows at once.
        Args:
            chunks: N tensors, each shape (win,) containing waveform samples
            masks: N tensors, each shape (win,) with False=real, True=padded
        If the lists are empty, it returns empty placeholders with shape (0, win) so the caller can detect 'nothing to run'.

        Returns:
            batch: Tensor (N, win_samples)
            pad_mask: Tensor (N, win_samples)
        """
        batch = (
            torch.stack(chunks, dim=0) if chunks else torch.empty(0, self.win_samples)
        )
        pad_mask = (
            torch.stack(masks, dim=0)
            if masks
            else torch.empty(0, self.win_samples, dtype=torch.bool)
        )
        return batch, pad_mask

    def process_video(self, video_path: str, top_k: int = 3):
        """
        Processes a video file, runs BEATs inference on audio chunks, and returns results.
        Args:
            video_path: Path to video file.
            top_k: Number of top labels to retain per chunk.
        Returns:
            Dict with video_path, response list, and modality.
        """
        if not self.has_audio(video_path):
            print(f"No audio stream found in {video_path}")
            return {
                "video_path": os.path.basename(video_path),
                "response": [],
                "modality": "audio",
                "error": "No audio stream found",
            }
        # Extract audio
        wav = self.load_audio(video_path)
        chunks, masks, spans = self.chunk_waveform_with_spans(wav)

        audio_input, padding_mask = self.collate(
            chunks, masks
        )  # This batching allows the model to process all chunks in parallel (efficient inference)
        if audio_input.shape[0] == 0:
            print(f"No valid audio chunks for {video_path}")
            return {
                "video_path": os.path.basename(video_path),
                "response": [],
                "modality": "audio",
                "error": "No valid audio chunks",
            }
        
        # Move tensors to the same device as the model
        audio_input = audio_input.to(self.device)
        padding_mask = padding_mask.to(self.device)
        # Run inference
        with torch.no_grad():
            # Pass padding_mask alongside audio_input_16khz
            # Internally, BEATs uses this mask to avoid attending to or aggregating padded positions, so the zero‑filled tail won’t affect predictions
            probs = self.model.extract_features(audio_input, padding_mask=padding_mask)[0]
            
        results = []
        for i, (top_label_prob, top_label_idx) in enumerate(zip(*probs.topk(k=top_k))):
            # Get human-readable labels
            top_labels = [
                self.get_human_label_from_audio(self.label_dict[label_idx.item()])
                for label_idx in top_label_idx
            ]
            top_probs = [
                round(top_label_prob[j].item(), 3) for j in range(len(top_label_prob))
            ]
            # Save results for each chunk
            start_time, end_time = spans[i]
            results.append(
                {
                    "start_time": round(start_time, 3),
                    "end_time": round(end_time, 3),
                    "labels": dict(
                        sorted(
                            zip(top_labels, top_probs), key=lambda x: x[1], reverse=True
                        )
                    ),
                }
            )
        return {
            "video_path": os.path.basename(video_path),
            "response": results,
            "modality": "audio",
        }

    async def process_all_videos(
        self,
        video_paths: list[str],
        output_json: str = "beats_results.json",
        overwrite: bool = False,
        top_k: int = 3,
    ):
        """
        Processes all videos asynchronously and saves results to JSON.
        Args:
            video_paths: List of video file paths.
            output_json: Output JSON file path.
            overwrite: If True, overwrite existing file.
            top_k: Number of top labels per chunk.
        Returns:
            List of all result dicts.
        """
        # Load existing results if not overwriting
        if overwrite or not os.path.exists(output_json):
            all_results = []
        else:
            with open(output_json, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"Loaded {len(all_results)} existing results")
        
        print(f"Processing {len(video_paths)} new videos...")
        # Process each video sequentially
        for i, video_path in enumerate(video_paths, 1):
            video_name = os.path.basename(video_path)
            print(f"\n[{i}/{len(video_paths)}] Processing: {video_name}")
            
            try:
                # Process single video
                result = self.process_video(video_path, top_k)
                all_results.append(result)
                
                # Save after each video
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)
                
                # Clear GPU cache after each video
                if self.device == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"Error processing {video_name}: {e}")
                # Save error result
                error_result = {
                    "video_path": video_name,
                    "response": [],
                    "modality": "audio",
                    "error": str(e)
                }
                all_results.append(error_result)
                
                # Save even on error
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)

        print(f"Processed {len(all_results)} videos")
