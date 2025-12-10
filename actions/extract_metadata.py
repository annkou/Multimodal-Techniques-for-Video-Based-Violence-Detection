import asyncio
import json
import os
import subprocess

import cv2 as cv
import librosa


class VideoMetadataExtractor:
    """Extracts metadata from video files asynchronously."""

    def __init__(self, video_paths):
        self.video_paths = video_paths

    async def extract_all_metadata(self):
        """
        Asynchronously extract metadata for all video files.

        Returns:
            list: List of metadata dictionaries for each video.
        """
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.extract_metadata, vp)
            for vp in self.video_paths
        ]
        return await asyncio.gather(*tasks)

    def extract_metadata(self, video_path):
        """
        Extract metadata for a single video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Metadata dictionary containing video path, name, audio presence, details, and label.
        """
        metadata = {
            "video_path": video_path,
            "video_name": os.path.splitext(os.path.basename(video_path))[0],
            "has_audio": self.has_audio(video_path),
        }
        details = self.get_video_details(video_path)
        metadata.update(details)
        # print(f"\nProcessed {os.path.basename(video_path)}")
        return metadata

    def has_audio(self, video_path: str) -> bool:
        """
        Check if a video file contains an audio stream.

        Args:
            video_path (str): Path to the video file.

        Returns:
            bool: True if audio stream exists, False otherwise.
        """
        cmd = f'ffprobe -v error -print_format json -show_streams "{video_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            return False
        info = json.loads(result.stdout)
        return any(s.get("codec_type") == "audio" for s in info.get("streams", []))

    def get_video_details(self, video_path: str):
        """
        Extract frame count, FPS, duration, and audio sampling rate from a video file.

        Args:
            video_path (str): Path to the video file.

        Returns:
            dict: Dictionary with frame count, FPS, duration, and audio sampling rate.
        """
        cap = cv.VideoCapture(video_path)
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        size = os.path.getsize(video_path)
        size_mb = round(size / (1024 * 1024), 2)  # Convert bytes to MB
        sr = None
        if self.has_audio(video_path):
            try:
                sr = librosa.get_samplerate(video_path)
            except Exception:
                sr = None
        cap.release()
        return {
            "frame_count": frame_count,
            "fps": fps,
            "duration": duration,
            "audio_sr": sr,
            "size_bytes": size,
            "size_mb": size_mb,
        }
