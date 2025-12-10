import asyncio
import json
import os
import time

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


class Transcripts(BaseModel):
    """Pydantic schema for the Gemini JSON response"""

    start_time: float = Field(..., description="Segment start in seconds (inclusive)")
    end_time: float
    audio_transcript: str
    visual_transcript: str


class GeminiAsyncRequester:
    """
    Async Gemini client wrapper to extract audio + visual transcripts with timestamps.
    Uses google-genai async client to send local videos and prompt, parses responses, and saves results to JSON.
    """

    def __init__(
        self,
        api_key: str,
        prompt: str,
        model: str = "models/gemini-2.5-pro",
        max_inline_bytes: int = 20
        * 1024
        * 1024,  # Videos of size <20Mb can be passed inline
    ):
        """
        Args:
            api_key: Gemini API key (from env if None).
            model: Gemini model name.
            prompt: Prompt for transcript extraction.
            max_inline_bytes: Inline upload threshold (bytes).
        """
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.max_inline_bytes = max_inline_bytes
        self.client = genai.Client(api_key=self.api_key)
        self._io_lock = asyncio.Lock()

    def upload_video(self, video_file_name: str):
        """
        Upload a video file to Gemini File API and wait for processing.
        
        Args:
            video_file_name: Path to the video file to upload.
            
        Returns:
            Uploaded video file object with URI.
            
        Raises:
            ValueError: If video processing fails.
        """
        video_file = self.client.files.upload(file=video_file_name)

        while video_file.state == "PROCESSING":
            print('Waiting for video to be processed.')
            time.sleep(10)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state}")
        
        print(f'Video processing complete: {video_file.uri}')
        return video_file
    
    def delete_video(self, video_file_name: str):
        """
        Delete an uploaded video file from Gemini File API.
        
        Args:
            video_file_name: Name of the uploaded file to delete.
        """
        self.client.files.delete(name=video_file_name)
        print(f'Deleted video file: {video_file_name}')
    
    def _append_response_to_json(
        self, path: str, response, overwrite=False, append=False
    ):
        """Append response from client to JSON file"""
        if overwrite or not os.path.exists(path):
            # Overwrite: start fresh
            with open(path, "w", encoding="utf-8") as f:
                json.dump([response], f, indent=2, ensure_ascii=False)
        elif append:
            # Append: extend existing
            with open(path, "r", encoding="utf-8") as file:
                try:
                    existing = json.load(file)
                except Exception:
                    existing = []
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(response)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        else:
            # Default: overwrite existing file
            with open(path, "w", encoding="utf-8") as f:
                json.dump([response], f, indent=2, ensure_ascii=False)

    def _build_contents(self, video_source, use_url=False):
        if use_url:
            return [
                video_source, # types.Part.from_uri(file_uri=video_source, mime_type="video/mp4"),
                self.prompt,
            ]

        # types.Content(
        #         parts=[
        #             types.Part.from_uri(file_uri=video_source, mime_type="video/mp4"),
        #             types.Part(text=self.prompt),
        #         ]
        #     )

        else:
            return types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_source, mime_type="video/mp4")
                    ),
                    types.Part(text=self.prompt),
                ]
            )

    def _parse_response(self, response):
        # print(response.parsed)
        data = response.parsed
        if data and not isinstance(data[0], dict):
            segments = [d.model_dump() for d in data]
        else:
            segments = data
        return segments

    async def gemini_response(
        self,
        video_path_or_url: str,
        output_json: str,
        metadata_json: str,
        overwrite: bool = False,
        append: bool = True,
        use_url: bool = False,
        original_path: str = None,
        
    ):
        """
        Make a single Gemini API call, using either a local file (inline) or a public URL.
        Args:
            video_path_or_url: Local file path or public URL.
            output_json: Path to output JSON.
            metadata_json: Path to gemini metadata JSON file.
            overwrite: If overwrite=True, deletes existing file before writing.
            append: If append=True, appends results to existing JSON file.
            use_url: If True, treat input as URL; else treat as local file.
            original_path: Original local path for saving results (if different from video_path_or_url).
        """
        # Use original_path for saving if provided, otherwise use video_path_or_url
        save_path = original_path if original_path else video_path_or_url
        if use_url:
            contents = self._build_contents(video_path_or_url, use_url=True)
        else:
            size = os.path.getsize(video_path_or_url)
            if size > self.max_inline_bytes:
                raise ValueError(
                    f"Video {video_path_or_url} exceeds inline upload limit ({self.max_inline_bytes} bytes)."
                )
            video_bytes = open(video_path_or_url, "rb").read()
            contents = self._build_contents(video_bytes, use_url=False)
        try:
            # Async Gemini request with schema
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": list[Transcripts],
                },
            )
            segments = self._parse_response(response)
            # Build response in new format
            video_result = {
                "video_path": os.path.basename(save_path),
                "response": [
                    {
                        "start_time": segment["start_time"],
                        "end_time": segment["end_time"],
                        "visual_transcript": segment["visual_transcript"],
                        "audio_transcript": segment["audio_transcript"],
                    }
                    for segment in segments
                ],
                "modality": "transcript",
            }
            # Save immediately (async lock for safety)
            async with self._io_lock:
                self._append_response_to_json(
                    output_json, video_result, overwrite, append
                )
            # Save metadata (call after response)
            self.save_gemini_metadata(video_path_or_url, response, metadata_json, save_path)
        except Exception as e:
            # Save error entry
            error_entry = {
                "video_path": os.path.basename(save_path),
                "start_time": 0.0,
                "end_time": 0.0,
                "transcripts": {"visual_transcript": "", "audio_transcript": ""},
                "error": str(e),
            }
            async with self._io_lock:
                self._append_response_to_json(
                    output_json, error_entry, overwrite, append
                )

    async def run_async(
        self,
        video_paths: list[str] | dict[str, str],
        output_json: str,
        metadata_json: str = "gemini_metadata.json",
        concurrency_limit: int = 4,
        overwrite: bool = False,
        append: bool = True,
        use_url: bool = False,
    ):
        """
        Run async requests over a list of videos.

        Args:
            video_paths: List of local video file paths OR dict of {local_path: uri_or_path}.
            local_video_paths: List of local video file paths for metadata saving.
            output_json: Path to transcripts JSON file.
            metadata_json: Path to gemini metadata JSON file.
            concurrency_limit: Max concurrent API requests.
            overwrite: If overwrite=True, deletes existing file before writing.
            append: If append=True, appends results to existing JSON file.
            use_url: If True, treat inputs as URLs; else treat as local files.
        """
        sem = asyncio.Semaphore(concurrency_limit)

        async def process_with_limit(video_path: str, original_path: str = None):
            async with sem:
                await self.gemini_response(
                    video_path,
                    output_json,
                    metadata_json,
                    overwrite,
                    append,
                    use_url=use_url,
                    original_path=original_path,
                )
        # Handle both list and dict inputs
        if isinstance(video_paths, dict):
            tasks = [
                process_with_limit(vf, original_path=lp)
                for lp, vf in video_paths.items()
            ]
        else:
            tasks = [process_with_limit(vp) for vp in video_paths]
        await asyncio.gather(*tasks)

    def save_gemini_metadata(self, video_path: str, response, output_json: str, original_path: str = None):
        """
        Save Gemini response metadata for a video to a separate JSON file.
        Appends if file exists, else creates new.
        """

        video_basename = os.path.basename(original_path) if original_path else os.path.basename(video_path)
        # If response is an object with .json(), call it; else assume dict
        if hasattr(response, "json"):
            data = response.json()
        else:
            data = response
        if isinstance(data, str):
            data = json.loads(data)

        meta = {
            "video_path": video_basename,
            "model_version": data.get("model_version"),
            "response_id": data.get("response_id"),
            "finish_reason": data.get("candidates", [{}])[0].get("finish_reason"),
            "usage_metadata": data.get("usage_metadata", {}),
        }

        if os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except Exception:
                    existing = []
            if not isinstance(existing, list):
                existing = [existing]
            existing.append(meta)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
        else:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump([meta], f, indent=2)
