"""
Ollama client for LLM and embedding operations.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import httpx

from config.default import (EMBEDDING_MODEL, LLM_CONTEXT_WINDOW, LLM_MODEL,
                           OLLAMA_HOST, SYSTEM_PROMPT)

logger = logging.getLogger(__name__)

class OllamaClient:
    """
    Client for interacting with Ollama API for LLM and embedding operations.
    """
    
    def __init__(
        self, 
        base_url: str = OLLAMA_HOST,
        llm_model: str = LLM_MODEL,
        embedding_model: str = EMBEDDING_MODEL,
        timeout: int = 120
    ):
        """Initialize the Ollama client."""
        self.base_url = base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
    
    def check_status(self) -> bool:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            True if Ollama is running, False otherwise
        """
        try:
            response = self.client.get(f"{self.base_url}")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False
    
    def list_models(self) -> List[str]:
        """
        List available models in Ollama.
        
        Returns:
            List of model names
        """
        try:
            response = self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = []
                for model in response.json().get("models", []):
                    # Extract model name, handling different formats
                    model_name = model.get("name", "")
                    # If name contains a tag (e.g., llama3:latest), split and take the model name only
                    if ":" in model_name:
                        model_name = model_name.split(":")[0]
                    if model_name:
                        models.append(model_name)
                logger.info(f"Available models: {', '.join(models)}")
                return models
            else:
                logger.error(f"Failed to list models: {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for a text using Ollama.
        
        Args:
            text: Input text to embed
        
        Returns:
            List of floats representing the embedding vector
        """
        try:
            response = self.client.post(
                f"{self.base_url}/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("embedding", [])
            else:
                logger.error(f"Failed to generate embedding: {response.text}")
                raise Exception(f"Embedding generation failed: {response.text}")
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        system_prompt: str = SYSTEM_PROMPT,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Generate text using Ollama LLM.
        
        Args:
            prompt: Input prompt
            system_prompt: System prompt to guide the model's behavior
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stop_sequences: List of sequences that will stop generation
            stream: Whether to stream the response
        
        Returns:
            Generated text or stream
        """
        request_data = {
            "model": self.llm_model,
            "prompt": prompt,
            "system": system_prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        if stop_sequences:
            request_data["stop"] = stop_sequences
        
        try:
            if stream:
                # Return the stream for the caller to process
                return self.client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=self.timeout
                )
            else:
                # Process the response and return the generated text
                response = self.client.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                else:
                    logger.error(f"Text generation failed: {response.text}")
                    raise Exception(f"Text generation failed: {response.text}")
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def process_stream(self, stream):
        """
        Process a streaming response from Ollama.
        
        Args:
            stream: Streaming response
        
        Yields:
            Text chunks as they are generated
        """
        response_text = ""
        try:
            for chunk in stream.iter_lines():
                if chunk:
                    try:
                        json_chunk = json.loads(chunk.decode('utf-8'))
                        content = json_chunk.get("response", "")
                        response_text += content
                        yield content
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")
        finally:
            yield response_text
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False
    ) -> Union[str, Any]:
        """
        Generate a chat completion using Ollama.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
        
        Returns:
            Generated chat completion or stream
        """
        request_data = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": stream
        }
        
        try:
            if stream:
                return self.client.stream(
                    "POST",
                    f"{self.base_url}/api/chat",
                    json=request_data,
                    timeout=self.timeout
                )
            else:
                response = self.client.post(
                    f"{self.base_url}/api/chat",
                    json=request_data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json().get("message", {}).get("content", "")
                else:
                    logger.error(f"Chat completion failed: {response.text}")
                    raise Exception(f"Chat completion failed: {response.text}")
        except Exception as e:
            logger.error(f"Error generating chat completion: {str(e)}")
            raise 