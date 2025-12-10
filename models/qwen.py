import asyncio
import json
import os

import dashscope
from openai import OpenAI
from together import Together


class QwenAsyncRequester:
    """
    Qwen2.5-VL-72B-Instruct via Alibaba DashScope. Together AI.
    Async client wrapper to extract visual transcripts with timestamps.
    Sends local videos and prompt, parses responses, and saves results to JSON.
    """

    def __init__(
        self,
        dashcope_api_key: str,
        together_api_key: str,
        model_name: str = "qwen2.5-vl-72b-instruct",
    ):
        """
        Args:
            api_key: Alibaba API key (from env if None).
            model: Qwen model name.
            prompt: Prompt for transcript extraction.
        """
        self.dashcope_api_key = dashcope_api_key
        self.together_api_key = together_api_key
        self.model_name = model_name
        self.client = OpenAI(
            # The API keys for the Singapore and China (Beijing) regions are different. To get an API key, see https://www.alibabacloud.com/help/en/model-studio/get-api-key
            api_key=self.dashcope_api_key,
            # The following is the URL for the Singapore region. If you use a model in the China (Beijing) region, replace the URL with: https://dashscope.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        dashscope.api_key = self.dashcope_api_key
        self.together_model_name = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.together_client = Together(api_key=self.together_api_key)
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

    def _build_messages(
        self, system_prompt: str, user_prompt: str, video: str, mode: str = "dashcope"
    ):
        if mode == "dashscope":
            return [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"video": f"file://{os.path.abspath(video)}"},
                        {"text": user_prompt},
                    ],
                },
            ]
        elif mode == "openai":
            # Doesn't work with local file paths
            return [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": video  # "file://" + os.path.abspath(video).replace("\\", "/")
                            },
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
        elif mode == "together":
            return [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "video_url",
                            "video_url": {"url": video},
                        },
                    ],
                },
            ]

    async def analyze_video_openai(
        self, video_url: str, system_prompt: str, user_prompt: str
    ) -> dict:
        """
        Send a single video request and return the parsed response.
        """
        try:
            messages = self._build_messages(
                system_prompt, user_prompt, video_url, mode="openai"
            )
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            # print(completion)
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
                "video_url": video_url,
                "response": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "currency": cost_info["currency"],
            }
        except Exception as e:
            return {
                "video_url": video_url,
                "error": str(e),
            }

    async def analyze_video_dashscope(
        self, video_path: str, system_prompt: str, user_prompt: str
    ) -> dict:
        """
        Send a single video request using DashScope SDK and return the parsed response.
        """
        try:
            messages = self._build_messages(
                system_prompt, user_prompt, video_path, mode="dashcope"
            )
            completion = await asyncio.to_thread(
                dashscope.MultiModalConversation.call,
                model=self.model_name,
                messages=messages,
            )
            # print(completion)
            content = completion["output"]["choices"][0]["message"].content[0]["text"]
            return {
                "video_path": video_path,
                "response": content or "",
            }
        except Exception as e:
            return {
                "video_path": video_path,
                "error": str(e),
            }

    async def analyze_video_together(
        self,
        video_path: str,
        system_prompt: str,
        user_prompt: str,
    ) -> dict:
        """
        Run Qwen2.5-VL via Together using multiple frame image URLs.
        Returns aggregated text (streaming or non-streaming).
        """
        try:
            messages = self._build_messages(
                system_prompt, user_prompt, video_path, mode="together"
            )

            completion = await asyncio.to_thread(
                self.together_client.chat.completions.create,
                model=self.together_model_name,
                messages=messages,
            )
            # print(completion)
            content = (
                completion.choices[0].message.content
                if completion and completion.choices
                else ""
            )
            return {
                "video_path": video_path,
                "response": content or "",
            }
        except Exception as e:
            return {
                "video_path": video_path,
                "error": str(e),
            }

    async def process_videos(
        self,
        video_urls: list,
        system_prompt: str,
        user_prompt: str,
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

        async def sem_task(url):
            async with semaphore:
                # return await self.analyze_video_openai(url, system_prompt, user_prompt)
                result = await self.analyze_video_openai(
                    url, system_prompt, user_prompt
                )

                # await asyncio.sleep(20)  # <-- Add sleep here
                return result

        tasks = [sem_task(url) for url in video_urls]
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        return results

    def parse_and_save_transcripts(
        self, input_json_path, output_json_path, metadata_json_path
    ):
        """
        Parses the 'response' field from each object in the input JSON,
        replacing it with a list of transcript dicts:
        [{"start_time": ..., "end_time": ..., "transcript": ..., "modality": "transcript"}, ...]

        Args:
            input_json_path (str): Path to the input JSON file.
            output_json_path (str): Path to the output JSON file.
            metadata_json_path (str): Path to the output JSON file for metadata.
        """
        with open(input_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        transcript_results = []
        metadata_results = []

        for obj in data:
            video_url = obj.get("video_url", "")
            response = obj.get("response", "")
            error = obj.get("error", "")
            # Extract metadata
            metadata = {
                "video_url": video_url,
                "input_tokens": obj.get("input_tokens", 0),
                "output_tokens": obj.get("output_tokens", 0),
                "total_tokens": obj.get("total_tokens", 0),
                "input_cost": obj.get("input_cost", 0.0),
                "output_cost": obj.get("output_cost", 0.0),
                "total_cost": obj.get("total_cost", 0.0),
                "currency": obj.get("currency", "USD"),
                "error": error,
            }
            metadata_results.append(metadata)
            # Handle errors - empty transcript
            if error or not response:
                transcript_results.append(
                    {
                        "video_url": video_url,
                        "response": [
                            {
                                "start_time": 0.0,
                                "end_time": 0.0,
                                "visual_transcript": "",
                            }
                        ],
                        "modality": "transcript",
                        "error": error,
                    }
                )
                continue

            # Parse transcripts
            try:
                segments = json.loads(response)
            except Exception:
                obj["response"] = []
                continue

            parsed_segments = []
            for seg in segments:
                parsed_segments.append(
                    {
                        "start_time": seg.get("start_time"),
                        "end_time": seg.get("end_time"),
                        "visual_transcript": seg.get("visual_transcript"),
                    }
                )
            transcript_results.append(
                {
                    "video_path": video_url,
                    "response": parsed_segments
                    if parsed_segments
                    else [
                        {"start_time": 0.0, "end_time": 0.0, "visual_transcript": ""}
                    ],
                    "modality": "transcript",
                }
            )

        # Save transcripts
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(transcript_results, f, indent=2, ensure_ascii=False)

        # Save metadata
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata_results, f, indent=2, ensure_ascii=False)
