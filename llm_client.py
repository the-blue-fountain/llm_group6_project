"""
Async OpenAI client wrapper for generating candidate solutions.
Requires OPENAI_API_KEY environment variable.
"""
import os
import asyncio
from typing import Optional
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class OpenAIClient:
    """Async wrapper for OpenAI API with rate limiting."""
    
    def __init__(self, model: str = "gpt-4o", concurrency: int = 10, max_tokens: Optional[int] = 4000, reasoning_effort: str = "medium"):
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._sem = asyncio.Semaphore(concurrency)
        
        # Check if model is a reasoning model (o1/o3 series)
        self.is_reasoning_model = any(x in model.lower() for x in ["o1", "o3"])
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize async OpenAI client
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    async def generate(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate a single response from OpenAI API."""
        async with self._sem:
            try:
                # Build request parameters based on model type
                params = {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
                
                if self.is_reasoning_model:
                    # Reasoning models (o1/o3) use reasoning_effort instead of temperature/max_tokens
                    params["reasoning_effort"] = self.reasoning_effort
                else:
                    # Standard models use system message, temperature, and max_tokens
                    params["messages"].insert(0, {"role": "system", "content": "You are an expert competitive programmer."})
                    params["temperature"] = temperature
                    if self.max_tokens:
                        params["max_tokens"] = self.max_tokens
                
                response = await self.client.chat.completions.create(**params)
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"OpenAI API error: {e}")
                return ""
    
    async def generate_multiple(self, prompt: str, n: int, temperature: float = 0.7) -> list[str]:
        """Generate multiple responses concurrently."""
        tasks = [self.generate(prompt, temperature) for _ in range(n)]
        return await asyncio.gather(*tasks)
