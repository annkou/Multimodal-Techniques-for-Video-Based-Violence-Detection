import csv
import json
import os
import subprocess

import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class WhisperSegmentTranscriber:
    """
    Transcribes only detected speech segments in videos using Whisper.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3-turbo",
    ):
        """
        Args:
            model_name: HuggingFace model name for Whisper.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16
            if "cuda" in self.device and torch.cuda.is_available()
            else torch.float32
        )
        self.model, self.processor, self.pipe = self.load_whisper(model_name)

    def load_whisper(self, model_name: str):
        """
        Loads Whisper model and pipeline.
        """
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=self.torch_dtype, low_cpu_mem_usage=True
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(model_name)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        return model, processor, pipe

    def extract_audio_segment(
        self, video_path: str, start: float, end: float, out_wav: str, sr: int = 16000
    ) -> str:
        """
        Extracts a segment of audio from a video using ffmpeg.
        """
        if os.path.exists(out_wav):
            return out_wav
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(start),
            "-to",
            str(end),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "wav",
            out_wav,
        ]
        subprocess.run(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True
        )
        return out_wav

    def transcribe_segments(
        self,
        video_path: str,
        segments,
        output_json: str,
        sr: int = 16000,
    ):
        """
        Transcribes only the detected speech segments for a video.

        Args:
            video_path: Path to video file.
            segments: List of dicts with 'start' and 'end' keys (seconds).
            output_json: Path to save results.
            sr: Audio sample rate.

        Returns:
            List of transcript dicts.
        """
        video_basename = os.path.basename(video_path)
        results = []
        for i, seg in enumerate(segments):
            start, end = seg["start"], seg["end"]
            seg_wav = f"{video_path}_seg_{i}_{start:.2f}_{end:.2f}.wav"
            self.extract_audio_segment(video_path, start, end, seg_wav, sr=sr)
            asr_result = self.pipe(seg_wav)
            transcript = (
                asr_result["text"] if isinstance(asr_result, dict) else asr_result
            )
            results.append(
                {
                    "video_path": video_basename,
                    "start_time": round(start, 3),
                    "end_time": round(end, 3),
                    "modality": "transcript",
                    "transcript": transcript,
                }
            )
        if output_json:
            if os.path.exists(output_json):
                with open(output_json, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                existing.extend(results)
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(existing, f, indent=2)
            else:
                with open(output_json, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)
        return results

    def process_all_videos(
        self,
        video_segments_dict: dict,
        output_json: str,
        sr: int = 16000,
    ):
        """
        Process all videos with VAD segments and save transcripts to JSON.

        Args:
            video_segments_dict: Dict mapping video URLs/paths to their VAD segments.
                                Example: {
                                    "video_path": [
                                        {"start": 0.0, "end": 0.5},
                                        {"start": 2.0, "end": 2.5},
                                        ...
                                    ]
                                }
            output_json: Path to output JSON file (will append if exists).
            sr: Audio sample rate.
        """
        # Load existing results if file exists
        if os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                try:
                    all_results = json.load(f)
                except Exception:
                    all_results = []
        else:
            all_results = []

        # Get already processed video URLs
        processed_urls = set(r["video_url"] for r in all_results)

        # Process each video
        for video_url, segments in tqdm(
            video_segments_dict.items(), desc="Processing videos"
        ):
            # Skip if already processed
            if video_url in processed_urls:
                continue

            video_result = {
                "video_url": video_url,
                "response": [],
                "modality": "transcript",
            }

            # Transcribe each segment
            for seg in segments:
                start, end = seg["start"], seg["end"]
                seg_wav = f"temp_seg_{start:.2f}_{end:.2f}.wav"

                try:
                    # Extract audio segment
                    self.extract_audio_segment(video_url, start, end, seg_wav, sr=sr)

                    # Transcribe
                    asr_result = self.pipe(seg_wav)
                    transcript = (
                        asr_result["text"]
                        if isinstance(asr_result, dict)
                        else asr_result
                    )

                    video_result["response"].append(
                        {
                            "start_time": round(start, 3),
                            "end_time": round(end, 3),
                            "audio_transcript": transcript.strip(),
                        }
                    )

                    # Clean up temp file
                    if os.path.exists(seg_wav):
                        os.remove(seg_wav)

                except Exception as e:
                    print(
                        f"Error processing segment {start}-{end} for {video_url}: {e}"
                    )
                    continue

            # Append result to all_results
            all_results.append(video_result)

            # Save after each video (incremental save)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        print(f"Transcribed {len(all_results)} segments")
