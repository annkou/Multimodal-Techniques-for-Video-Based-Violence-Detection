import asyncio
import json
import os
from typing import List

from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential


class ViolenceEval(BaseModel):
    video_name: str
    violence_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)  # model’s self-reported certainty
    abstain: bool = False  # True if evidence insufficient
    rationale: str  # concise evidence-based reasoning
    primary_modalities: list[str]  # ordered by contribution


class GPTViolenceEvaluator:
    """
    GPT-based violence evaluation that aggregates results from multiple models.
    Uses OpenAI's GPT models to analyze and synthesize predictions from vision, audio, and text models.
    """

    def __init__(self, api_key, model="gpt-5"):
        """
        Initialize the GPT Violence Evaluator.
        """
        # Async client for API calls
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.api_key = api_key
        self.MODEL_COSTS = {
            "gpt-4o": {
                "input": 2.50,
                "output": 10.00,
                "currency": "USD",
            },
            "gpt-4.1": {"input": 2.00, "output": 8.00, "currency": "USD"},
            "gpt-5": {"input": 1.25, "output": 10.00, "currency": "USD"},
            # Add more models here as needed
        }

    def get_cost(self, model_name, prompt_tokens, completion_tokens):
        prompt_cost = (prompt_tokens / 1000000) * self.MODEL_COSTS[model_name]["input"]
        completion_cost = (completion_tokens / 1000) * self.MODEL_COSTS[model_name][
            "output"
        ]
        return {
            "input_cost": prompt_cost,
            "output_cost": completion_cost,
            "total_cost": prompt_cost + completion_cost,
            "currency": self.MODEL_COSTS[model_name]["currency"],
        }

    def build_user_prompt_for_video(
            self, 
            video_name: str, 
            video_results: dict,
            models_to_include: list = None
        ):
        """
        Build user prompt from filtered model results for a specific video.

        Args:
            video_name: Name of the video file
            video_results: Dict with structure {model_name: {'response': [...], 'modality': '...'}}
            models_to_include: List of model names to include (if None, include all)

        Returns:
            dict: {model_name: [results_for_this_video]}
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
        
        return prompt_data

    def create_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        # file_id: str = None,
    ):
        """
        Create the full prompt structure for OpenAI API.

        Args:
            system_prompt: System/developer instructions

            user_prompt: Dict of model results for a video
            file_id: Optional file ID for uploaded context

        Returns:
            List of message dicts for the API
        """
        return [
            {
                "role": "developer",  # "system" or "developer" for newer models
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": str(user_prompt)},
                    # {"type": "file", "file": {"id": file.id}},
                ],
            },
        ]

    def save_results_to_json(self, results: list, output_path: str):
        """
        Save results to JSON, appending if file exists.

        Args:
            results: List of result dicts
            output_path: Path to output JSON file
        """
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                except json.JSONDecodeError:
                    existing = []
            existing.extend(results)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        else:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # ==================== STANDARD API MODE ====================
    @retry(
        wait=wait_random_exponential(multiplier=2, min=1, max=80),
        stop=stop_after_attempt(3),
        reraise=True,  # Raise the error after retries are exhausted
    )
    async def gpt_response(self, prompt):
        """
        Make a single GPT API call with retry logic.

        Args:
            prompt: List of message dicts

        Returns:
            OpenAI response object
        """
        # Convert Pydantic model to JSON Schema
        response_format_schema = ViolenceEval

        response = await self.client.chat.completions.parse(
            model=self.model,
            messages=prompt,
            # temperature=0,
            max_completion_tokens=30000,
            response_format=response_format_schema,
        )

        return response

    async def process_video(
        self,
        video_name: str,
        video_results: dict,
        system_prompt: str,
        models_to_include: list = None,
    ):
        """
        Process a single video using standard API calls.

        Args:
            video_name: Name of the video
            video_results: Dict with structure {model_name: {'response': [...], 'modality': '...'}}
            system_prompt: System instructions
            models_to_include: List of model names to include (if None, include all)

        Returns:
            dict: Result with video_name, response, cost info
        """
        # Build user prompt from filtered model results
        user_prompt_data = self.build_user_prompt_for_video(
            video_name,
            video_results,
            models_to_include
        )
        user_prompt = json.dumps(user_prompt_data, ensure_ascii=False, indent=2)
        prompt = self.create_prompt(system_prompt, user_prompt)

        # Call GPT API
        response = await self.gpt_response(prompt)

        # Extract result
        try:
            response_dict = response.to_dict()
            result_content = response_dict["choices"][0]["message"]["content"]
            result_dict = json.loads(result_content)
        except Exception as e:
            print(f"Error parsing response for {video_name}: {e}")
            result_dict = {"error": str(e)}

        # Calculate cost
        usage = response_dict["usage"]
        cost_info = self.get_cost(
            self.model, usage["prompt_tokens"], usage["completion_tokens"]
        )

        # Build final result
        final_result = {
            "video_name": video_name,
            "response": result_dict,
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            **cost_info,
        }

        return final_result

    async def process_all_videos(
        self,
        video_results: dict,
        system_prompt: str,
        output_json: str,
        models_to_include: list = None,
        batch_size: int = 10,
        overwrite: bool = False,
        append: bool = True,
    ):
        """
        Process all videos using standard API calls (async).

        Args:
            video_results: Dict mapping video_name -> {model_name: {'response': [...], 'modality': '...'}}
            system_prompt: System prompt
            output_json: Output JSON file path
            models_to_include: List of model names to include (if None, include all)
            batch_size: Number of concurrent API calls
            overwrite: If True, delete existing output file
            append: If True, append to existing results

        Returns:
            List of results
        """

        if overwrite and os.path.exists(output_json):
            os.remove(output_json)

        # Load existing results to skip already processed videos
        processed_videos = set()
        if append and os.path.exists(output_json):
            with open(output_json, "r", encoding="utf-8") as f:
                try:
                    existing = json.load(f)
                    processed_videos = {r["video_name"] for r in existing}
                except:
                    pass

        # Filter out already processed videos
        videos_to_process = [
            v for v in video_results.keys() if v not in processed_videos
        ]

        print(
            f"Processing {len(videos_to_process)} videos ({len(processed_videos)} already done)"
        )

        # Process with semaphore for rate limiting
        semaphore = asyncio.Semaphore(batch_size)

        async def process_with_limit(video_name):
            async with semaphore:
                result = await self.process_video(
                    video_name,
                    video_results[video_name],
                    system_prompt,
                    models_to_include,
                )
                # Save immediately after each video
                self.save_results_to_json([result], output_json)
                return result

        tasks = [process_with_limit(vn) for vn in videos_to_process]
        results = await asyncio.gather(*tasks)

        print(f"All videos processed. Results saved to {output_json}")
        return results

    # ==================== BATCH API MODE ====================
    def create_batch_requests(
        self,
        video_results: dict,
        system_prompt: str,
        models_to_include: list = None
    ):
        """
        Create batch API request objects for all videos.

        Args:
            video_results: Dict with {model_name: {'response': [...], 'modality': '...'}}
            system_prompt: System instructions
            models_to_include: List of models to include in prompt

        Returns:
            list: List of batch request dicts
        """
        batch_requests = []

        for idx, (video_name, models_data) in enumerate(video_results.items()):
            # Build user prompt with filtered models
            user_prompt_data = self.build_user_prompt_for_video(
                video_name, 
                models_data,
                models_to_include
            )
            user_prompt = json.dumps(user_prompt_data, ensure_ascii=False, indent=2)
            prompt = self.create_prompt(system_prompt, user_prompt)

            response_format_schema = (ViolenceEval).model_json_schema()
            # Create batch request format
            request = {
                "custom_id": f"video-{idx}-{video_name}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model,
                    "messages": prompt,
                    # "temperature": self.temperature,
                    "max_completion_tokens": 30000,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "violence-detection",
                            "schema": response_format_schema,
                        },
                    },
                },
            }
            batch_requests.append(request)
        return batch_requests

    def create_batch_input_file(
        self,
        video_results: dict,
        system_prompt: str,
        output_file: str = "batch_input.jsonl",
        models_to_include: list = None
    ):
        """
        Create a JSONL file for batch API input.

        Args:
            video_results: Dict mapping video_name -> {model_name: {'response': [...], 'modality': '...'}}
            system_prompt: System prompt
            output_file: Path to output JSONL file
            models_to_include: List of model names to include (if None, include all)

        Returns:
            str: Path to created JSONL file
        """

        # Create batch requests
        batch_requests = self.create_batch_requests(
            video_results,
            system_prompt,
            models_to_include
        )

        # Write to JSONL
        with open(output_file, "w", encoding="utf-8") as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + "\n")

        print(
            f"Created batch input file: {output_file} with {len(batch_requests)} requests"
        )
        return output_file

    @retry(
        wait=wait_random_exponential(multiplier=2, min=1, max=60),
        stop=stop_after_attempt(3),
    )
    async def upload_batch_file(self, file_path):
        """Upload a batch JSONL file to OpenAI API."""
        batch_file = await self.client.files.create(
            file=open(file_path, "rb"), purpose="batch"
        )
        return batch_file.id

    @retry(
        wait=wait_random_exponential(multiplier=2, min=1, max=60),
        stop=stop_after_attempt(3),
    )
    async def create_batch(self, batch_input_file_id, completion_window="24h"):
        """Create a batch job."""
        batch = await self.client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        return batch

    @retry(
        wait=wait_fixed(180),  # Wait 3 minutes between retries
        stop=stop_after_attempt(2),  # Retry 2 times on failure
        reraise=True,  # Raise the error after retries are exhausted
    )
    async def get_batch_status(self, batch_id):
        """Retrieve the status of a batch."""
        batch_status = (await self.client.batches.retrieve(batch_id)).status
        output_file_id = (await self.client.batches.retrieve(batch_id)).output_file_id
        return batch_status, output_file_id

    @retry(
        wait=wait_random_exponential(multiplier=2, min=1, max=60),
        stop=stop_after_attempt(3),
    )
    async def download_batch_output(self, output_file_id):
        """Download batch output using file ID."""
        file_response = await self.client.files.content(output_file_id)
        return file_response
