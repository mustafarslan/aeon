from abc import ABC, abstractmethod
from typing import Generator
import os
import requests
import json

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        """
        Generates a streaming response for the given prompt.
        
        Args:
            prompt: The user input or full prompt context.
            system_prompt: Optional system instruction (if supported separately).
            
        Yields:
            Chunks of the generated text response.
        """
        pass

class MockProvider(LLMProvider):
    """Echo provider for testing purposes."""
    
    def generate(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        # Mimic a stream
        response = f"[Mock Response] You said: {prompt}..."
        for word in response.split():
            yield word + " "

class OllamaProvider(LLMProvider):
    """
    Production provider using Ollama's REST API.
    
    Environment Variables:
        AEON_OLLAMA_HOST: Base URL (default: http://localhost:11434)
        AEON_LLM_MODEL: Model name (default: llama3)
    """
    
    def __init__(self):
        self.host = os.getenv("AEON_OLLAMA_HOST", "http://localhost:11434")
        self.model = os.getenv("AEON_LLM_MODEL", "llama3")
        self.api_url = f"{self.host}/api/generate"

    def generate(self, prompt: str, system_prompt: str = "") -> Generator[str, None, None]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": True
        }
        
        try:
            with requests.post(self.api_url, json=payload, stream=True, timeout=30) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if not line:
                        continue
                        
                    decoded_line = line.decode("utf-8")
                    try:
                        data = json.loads(decoded_line)
                        if "response" in data:
                            yield data["response"]
                        if data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            yield f"\n[System Error: Could not connect to LLM Provider at {self.host}. Details: {str(e)}]"
