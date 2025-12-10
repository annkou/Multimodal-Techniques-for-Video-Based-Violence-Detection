import asyncio
import json
import os
import re
import pandas as pd

import dashscope
from openai import OpenAI
from together import Together
from pydantic import BaseModel, Field

class ViolenceEval(BaseModel):
    violence_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)  # model’s self-reported certainty
    abstain: bool = False  # True if evidence insufficient
    rationale: str  # concise evidence-based reasoning
    primary_modalities: list[str]  # ordered by contribution
    primary_models: list[str] = Field(..., description="Models ranked by importance (e.g., ['clip', 'beats', 'whisper'])")


class QwenViolenceEvaluator:
    """
    Violence evaluator using Alibaba Cloud's Qwen model via DashScope API.
    Supports async batch processing with rate limiting.
    """
    def __init__(
        self,
        api_key: str,
        model_name: str = "qwen3-30b-a3b-thinking-2507",
        max_concurrent_requests: int = 10,
        requests_per_minute: int = 60,
        request_timeout: int = 60
    ):
        """
        Initialize Qwen evaluator.
        
        Args:
            api_key: DashScope API key
            model_name: Qwen model name
            max_concurrent_requests: Max concurrent API calls
            requests_per_minute: Rate limit per minute
            request_timeout: Timeout per request (seconds)
        """

        self.dashcope_api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(
            # The API keys for the Singapore and China (Beijing) regions are different. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
            api_key=self.dashcope_api_key,
            # The following is the URL for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        dashscope.api_key = self.dashcope_api_key

        self.MODEL_COSTS = {
            "qwen3-vl-235b-a22b-instruct": {
                "input": 0.700,
                "output": 2.800,
                "currency": "USD",
            },
            "qwen2.5-vl-72b-instruct": {"input": 2.8, "output": 8.4, "currency": "USD"},
            "qwen3-30b-a3b-thinking-2507": {"input": 0.2, "output": 2.4, "currency": "USD"},
            # Add more models here as needed
        }
        
        print(f"Initialized QwenViolenceEvaluator")
        print(f"   Model: {self.model_name}")

    def build_user_prompt_for_video(
        self, 
        video_name: str, 
        video_results: dict,
        models_to_include: list = None
    ) -> str:
        """
        Build user prompt from filtered model results (same as GPT version).
        
        Args:
            video_name: Name of the video file
            video_results: Dict with structure {model_name: {'response': [...], 'modality': '...'}}
            models_to_include: List of model names to include (if None, include all)
        
        Returns:
            str: JSON string of prompt data
        """
        prompt_data = {}
        
        # If no specific models requested, include all
        if models_to_include is None:
            models_to_include = list(video_results.keys())
        
        # Filter and format results for requested models
        for model_name in models_to_include:
            if model_name in video_results:
                model_data = video_results[model_name]
                prompt_data[model_name] = {
                    "response": model_data.get('response', []),
                    "modality": model_data.get('modality', 'unknown')
                }
        
        return json.dumps(prompt_data, ensure_ascii=False, indent=2)
    
    def _build_messages(
        self, system_prompt: str, user_prompt: str
    ):
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_prompt}],
            },
        ]
    
    async def evaluate_video(
        self,
        video_name: str,
        video_results: dict,
        system_prompt: str,
        models_to_include: list[str] = None
    ) -> dict:
        """
        Evaluate a single video asynchronously.
        
        Args:
            video_name: Name of the video
            video_results: Dict with structure {model_name: {'response': [...], 'modality': '...'}}
            system_prompt: System instructions
            models_to_include: List of models to include in prompt
        
        Returns:
            dict: Result with video_name, response, etc.
        """
        try:
            # Build user prompt (same as GPT)
            user_prompt = self.build_user_prompt_for_video(
                video_name,
                video_results[video_name],
                models_to_include
            )
            messages = self._build_messages(
                system_prompt, user_prompt
            )
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                # response_format={"type": "json_object"},
            )
            content = completion.choices[0].message.content
            
            # Extract token usage and cost
            usage = getattr(completion, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
            output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            total_tokens = input_tokens + output_tokens

            cost_info = self.MODEL_COSTS.get(
                self.model_name, {"input": 0, "output": 0, "currency": "USD"}
            )
            input_cost = (input_tokens / 1000000) * cost_info["input"]
            output_cost = (output_tokens / 1000000) * cost_info["output"]
            total_cost = input_cost + output_cost
            
            return {
                "video_name": video_name,
                "response": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "currency": cost_info["currency"],
                "model": self.model_name,
                "success": True
            }
            
        except Exception as e:
            return {
                "video_name": video_name,
                "response": {},
                "error": str(e),
                "model": self.model_name,
                "success": False
            }
        
    async def evaluate_videos_batch(
        self,
        video_results: dict,
        system_prompt: str,
        models_to_include: list = None,
        batch_size: int = 10,
        output_json: str = "qwen_dashscope_results.json",
        overwrite: bool = False,
        append: bool = False,
    ):
        """
        Process videos asynchronously in batches, limiting concurrency with a semaphore.
        """
        results = []

        # Handle appending to JSON
        if append and os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                try:
                    results = json.load(f)
                except Exception:
                    results = []
        elif overwrite and os.path.exists(output_json):
            os.remove(output_json)

        semaphore = asyncio.Semaphore(batch_size)

        async def sem_task(video_name: str):
            async with semaphore:
                result = await self.evaluate_video(
                    video_name, video_results, system_prompt, models_to_include
                )

                # await asyncio.sleep(20)  # <-- Add sleep here
                return result
            
        video_names = list(video_results.keys())
        total = len(video_names)
        completed = 0
        failed = 0
        
        print(f"Processing {total} videos in batches of {batch_size}...")
        print(f"Output: {output_json}\n")

        tasks = [sem_task(video_name) for video_name in video_names]
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)

            # Track progress
            completed += 1
            if not result.get('success', False):
                failed += 1
            # Print progress every 10 videos or at completion
            if completed % 10 == 0 or completed == total:
                success_rate = ((completed - failed) / completed * 100) if completed > 0 else 0
                print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                    f"Success: {completed - failed} | Failed: {failed} | "
                    f"Success Rate: {success_rate:.1f}%")

            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        return results
    

    def _parse_response(self, response_text: str, video_name: str = "") -> dict:
        """
        Parse LLM response to extract structured data (matching ollama.py logic).
        
        Expected format:
        {
            "violence_probability": 0.0-1.0,
            "confidence": 0.0-1.0,
            "abstain": false,
            "rationale": "...",
            "primary_modalities": ["vision", "audio", "transcripts"],
            "primary_models": ["clip", "beats", "whisper"]
        }
        
        Args:
            response_text: Raw text response from LLM
            video_name: Name of the video (for error context)
        
        Returns:
            dict: Parsed response with structure matching ViolenceEval
        """
        try:
            # Try to extract JSON from response (same regex as ollama.py)
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            
            else:
                # Fallback: Could not extract JSON (matching ollama.py)
                return {
                    "violence_probability": 0.5,
                    "confidence": 0.0,
                    "abstain": True,
                    "rationale": response_text[:200],  # Truncate to first 200 chars
                    "primary_modalities": [],
                    "primary_models": [],
                    "parse_error": "Could not extract JSON from response"
                }
        
        except Exception as e:
            # Any other error
            return {
                "violence_probability": 0.0,
                "confidence": 0.0,
                "abstain": True,
                "rationale": response_text[:200] if response_text else "Empty response",
                "primary_modalities": [],
                "primary_models": [],
                "parse_error": f"Unexpected error: {str(e)}"
            }
    
    def parse_and_save_results(
        self,
        input_json: str,
        output_json: str = "qwen_results.json",
        output_csv: str = "qwen_results.csv",
        metadata_json: str = "qwen_metadata.json",
        metadata_csv: str = "qwen_metadata.csv"
    ):
        """
        Parse raw results from JSON file and save to separate result and metadata files.
        
        Args:
            input_json: Path to raw results JSON file (from evaluate_videos_batch)
            output_json: Path to save parsed results (predictions only)
            output_csv: Path to save parsed results as CSV
            metadata_json: Path to save metadata (tokens, costs, errors)
            metadata_csv: Path to save metadata as CSV
        
        Returns:
            tuple: (parsed_results, metadata_results, results_df, metadata_df)
        """
        
        # Load raw results from file
        print(f"Loading raw results from: {input_json}")
        with open(input_json, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
        
        print(f"Found {len(raw_results)} results to parse...")
        
        parsed_results = []
        metadata_results = []
        
        parse_errors = 0
        failed = 0
        
        for result in raw_results:
            video_name = result.get('video_name', 'unknown')
            success = result.get('success', False)
            
            if success:
                # Parse the raw response
                raw_response = result.get('response', '')
                parsed_response = self._parse_response(raw_response, video_name)
                
                has_parse_error = 'parse_error' in parsed_response
                if has_parse_error:
                    parse_errors += 1
                
                # Store parsed result (predictions only)
                parsed_results.append({
                    'video_name': video_name,
                    **parsed_response
                })
                
                # Store metadata (tokens, costs, errors)
                metadata_entry = {
                    'video_name': video_name,
                    'input_tokens': result.get('input_tokens', 0),
                    'output_tokens': result.get('output_tokens', 0),
                    'total_tokens': result.get('total_tokens', 0),
                    'input_cost': result.get('input_cost', 0.0),
                    'output_cost': result.get('output_cost', 0.0),
                    'total_cost': result.get('total_cost', 0.0),
                    'currency': result.get('currency', 'USD'),
                    'model': result.get('model', self.model_name),
                    'success': True,
                    'parse_error': parsed_response.get('parse_error', '') if has_parse_error else ''
                }
                
                if has_parse_error:
                    # Include truncated raw response for debugging
                    metadata_entry['raw_response'] = raw_response[:500]
                
                metadata_results.append(metadata_entry)
            
            else:
                # API error
                failed += 1
                
                # Store failed result
                parsed_results.append({
                    'video_name': video_name,
                    'violence_probability': None,
                    'confidence': None,
                    'abstain': None,
                    'rationale': None,
                    'primary_modalities': None,
                    'primary_models': None
                })
                
                # Store error metadata
                metadata_results.append({
                    'video_name': video_name,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'total_tokens': 0,
                    'input_cost': 0.0,
                    'output_cost': 0.0,
                    'total_cost': 0.0,
                    'currency': 'USD',
                    'model': result.get('model', self.model_name),
                    'success': False,
                    'error': result.get('error', 'Unknown error')
                })
        print(parsed_results)
        # Save results JSON
        print(f"Saving parsed results...")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(parsed_results, f, indent=2, ensure_ascii=False)
        print("Saved parsed results to json.")
        # Save metadata JSON
        with open(metadata_json, "w", encoding="utf-8") as f:
            json.dump(metadata_results, f, indent=2, ensure_ascii=False)
        print("Saved metadata to json.")
        
        # Convert to DataFrames and save CSV
        results_df = pd.DataFrame(parsed_results)
        metadata_df = pd.DataFrame(metadata_results)
        print("Converted results and metadata to DataFrames.")
        
        # Format list columns for CSV
        if not results_df.empty:
            results_df['primary_modalities'] = results_df['primary_modalities'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else ''
            )
            results_df['primary_models'] = results_df['primary_models'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else ''
            )
        
        results_df.to_csv(output_csv, index=False, encoding='utf-8')
        metadata_df.to_csv(metadata_csv, index=False, encoding='utf-8')
        print(f"Saved parsed results to csv.")
        print(f"Saved metadata to csv.")