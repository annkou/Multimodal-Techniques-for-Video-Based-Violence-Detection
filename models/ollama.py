import json
import re
from llama_index.llms.ollama import Ollama

class OllamaViolenceEvaluator:
    """
    Ollama-based violence detection evaluator using local LLM models.
    """
    
    def __init__(self, model: str = "qwen3:14b", base_url: str = "http://localhost:11434", request_timeout: int = 300):
        """
        Initialize Ollama evaluator.
        
        Args:
            model: Ollama model name (e.g., "qwen2.5:14b", "llama3.2")
            base_url: Ollama API base URL
            request_timeout: Timeout for API requests in seconds
        """
        self.model = model
        self.base_url = base_url
        self.request_timeout = request_timeout

        # Initialize llama-index Ollama client
        self.llm = Ollama(
            model=model,
            base_url=base_url,
            temperature=0.1,
            request_timeout=self.request_timeout
        )
        
        # Test connection
        self._test_connection()

    def _test_connection(self):
        """Test if Ollama is running and model responds."""
        try:
            response = self.llm.complete("Hello, are you working?")
            print(f"Connected to Ollama - Model: {self.model}")
            print(f"   Test response: {response.text[:50]}...")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   And model is pulled: ollama pull {self.model}")
    
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
    

    def evaluate_video(
        self,
        video_name: str,
        video_results: dict,
        system_prompt: str,
        models_to_include: list[str] = None
    ) -> dict:
        """
        Evaluate a single video using Ollama (same interface as GPT).
        
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
                video_results,
                models_to_include
            )
            
            # Combine system and user prompts
            full_prompt = f"{system_prompt}\n\nVideo Data:\n{user_prompt}"
            
            # Call Ollama using llama-index
            response = self.llm.complete(full_prompt)
            response_text = response.text
            
            # Parse response
            parsed = self._parse_response(response_text)
            
            return {
                "video_name": video_name,
                "response": parsed,
                "raw_response": response_text,
                "model": self.model,
                "success": True
            }
            
        except Exception as e:
            return {
                "video_name": video_name,
                "response": {},
                "error": str(e),
                "model": self.model,
                "success": False
            }
    
    def _parse_response(self, response_text: str) -> dict:
        """
        Parse LLM response to extract structured data (same as before).
        
        Expected format:
        {
            "video_name": "...",
            "violence_probability": 0.0-1.0,
            "confidence": 0.0-1.0,
            "abstain": false,
            "rationale": "...",
            "primary_modalities": ["clip", "beats"]
        }
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback: manual parsing
                return {
                    "video_name": "",
                    "violence_probability": 0.5,
                    "confidence": 0.0,
                    "abstain": True,
                    "rationale": response_text[:200],
                    "primary_modalities": [],
                    "parse_error": "Could not extract JSON"
                }
        except Exception as e:
            return {
                "video_name": "",
                "violence_probability": 0.0,
                "confidence": 0.0,
                "abstain": True,
                "rationale": response_text[:200],
                "primary_modalities": [],
                "parse_error": str(e)
            }