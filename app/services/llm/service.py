"""
LLM Service

Handles LLM API calls with strict separation of concerns.
Only responsible for calling Gemini 2.5 Flash - no business logic.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

from google import genai
from google.genai.types import GenerateContentConfig, GenerateContentResponse
from loguru import logger

from app.settings import settings
from app.services.llm.response_handler import ResponseHandler


class LLMService:
    """
    Service for calling LLM APIs.

    Strict separation of concerns: Only handles LLM API calls.
    No business logic, no prompt building, no response processing.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.7,
        max_tokens: int = 8192,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM service.

        Args:
            api_key: Gemini API key (defaults to GEMINI_API_KEY from settings)
            model: Model name (default: gemini-2.5-flash)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Maximum output tokens (default: 8192)
        """
        self.api_key = api_key or getattr(settings, "gemini_api_key", None)
        if not self.api_key:
            logger.warning("Gemini API key not found. LLM calls will fail.")
            self.client = None
        else:
            self.client = genai.Client(api_key=self.api_key)

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.response_handler = ResponseHandler()

    def call(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, any]]]:
        """
        Call the LLM with a prompt.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text response (or tuple with metadata if return_metadata=True)

        Raises:
            ValueError: If API key is not configured or response is empty
        """
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")

        start_time = time.time()
        
        # Retry logic for transient failures
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self._call_with_retry(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    return_metadata=return_metadata,
                    attempt=attempt,
                    start_time=start_time,
                )
            except Exception as e:
                last_error = e
                error_type = self._classify_error(e)
                
                # Don't retry on certain errors
                if error_type in ["invalid_request", "empty_response"]:
                    logger.error(f"Non-retryable error: {e}")
                    break
                
                # Retry on transient errors
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")
        
        # If all retries failed, return error response
        error_response = self.response_handler.create_error_response(
            last_error,
            error_type=self._classify_error(last_error),
            retry_suggested=True,
        )
        
        if return_metadata:
            return error_response["text"], error_response["metadata"]
        return error_response["text"]

    def _call_with_retry(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_metadata: bool = False,
        attempt: int = 0,
        start_time: float = 0,
    ) -> Union[str, Tuple[str, Dict[str, any]]]:
        """
        Internal method to make LLM call with retry support.
        
        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            return_metadata: Whether to return metadata
            attempt: Current attempt number
            start_time: Start time for timing
            
        Returns:
            Generated text response (or tuple with metadata)
        """
        try:
            # Build messages
            messages = [{"role": "user", "parts": [{"text": prompt}]}]

            # Build config
            # Ensure max_tokens is reasonable (Gemini 2.5 Flash supports up to 8192)
            requested_max_tokens = max_tokens or self.max_tokens
            # Cap at 8192 (Gemini 2.5 Flash limit)
            actual_max_tokens = min(requested_max_tokens, 8192)
            
            if requested_max_tokens > 8192:
                logger.warning(
                    f"Requested max_tokens ({requested_max_tokens}) exceeds Gemini limit (8192). "
                    f"Using 8192 instead."
                )
            
            config = GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature or self.temperature,
                max_output_tokens=actual_max_tokens,
            )
            
            logger.debug(
                f"LLM call config: max_output_tokens={actual_max_tokens}, "
                f"temperature={temperature or self.temperature}"
            )

            # Call API
            response: GenerateContentResponse = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=config,
            )

            # Check if response was truncated
            finish_reason = None
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = getattr(response.candidates[0], "finish_reason", None)
                if finish_reason == "MAX_TOKENS":
                    logger.warning(
                        f"Response was truncated due to MAX_TOKENS limit. "
                        f"Requested max_tokens: {max_tokens or self.max_tokens}"
                    )

            # Extract text - try multiple methods to get full response
            text = response.text
            if not text:
                # Fallback: try to get text from candidates
                if hasattr(response, "candidates") and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, "content") and candidate.content:
                        if hasattr(candidate.content, "parts"):
                            text_parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, "text") and part.text:
                                    text_parts.append(part.text)
                            text = "".join(text_parts)
                
                if not text:
                    raise ValueError("Empty response from LLM")

            # Log response details
            response_length = len(text)
            
            # Get token usage if available
            output_tokens = None
            if hasattr(response, "usage_metadata"):
                output_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
            
            logger.info(
                f"LLM call successful: {response_length} chars, "
                f"output_tokens: {output_tokens}, "
                f"finish_reason: {finish_reason}, "
                f"max_tokens: {actual_max_tokens}"
            )
            
            # Validate and sanitize response
            is_truncated = finish_reason == "MAX_TOKENS"
            sanitized_text, response_metadata = self.response_handler.validate_and_sanitize(
                text=text,
                finish_reason=finish_reason,
                is_truncated=is_truncated,
            )
            
            # Add token usage to metadata
            response_metadata.update({
                "output_tokens": output_tokens,
                "max_tokens": actual_max_tokens,
                "processing_time": time.time() - start_time if start_time else None,
            })
            
            # Format response with warnings if needed
            formatted_text = self.response_handler.format_response(
                text=sanitized_text,
                metadata=response_metadata,
                add_warning=True,
            )
            
            # Log completion status
            if is_truncated:
                logger.warning(
                    f"Response TRUNCATED: {response_length} chars, "
                    f"{output_tokens}/{actual_max_tokens} tokens used"
                )
            elif not response_metadata.get("is_complete"):
                logger.warning(f"Response may be incomplete: {response_length} chars")
            else:
                logger.info(
                    f"Response complete: {response_length} chars, "
                    f"{output_tokens} tokens, "
                    f"{response_metadata.get('processing_time', 0):.2f}s"
                )
            
            if return_metadata:
                return formatted_text, response_metadata
            return formatted_text

        except Exception as e:
            logger.error(f"Error calling LLM (attempt {attempt + 1}): {e}")
            raise

    def _classify_error(self, error: Exception) -> str:
        """
        Classify error type for appropriate handling.
        
        Args:
            error: Exception to classify
            
        Returns:
            Error type string
        """
        error_str = str(error).lower()
        
        if "timeout" in error_str or "timed out" in error_str:
            return "timeout"
        elif "rate limit" in error_str or "quota" in error_str or "429" in error_str:
            return "rate_limit"
        elif "invalid" in error_str or "400" in error_str:
            return "invalid_request"
        elif "empty" in error_str or "no response" in error_str:
            return "empty_response"
        else:
            return "api_error"

    def call_with_messages(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Call the LLM with a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_instruction: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override

        Returns:
            Generated text response
        """
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")

        try:
            # Convert messages to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "model" if msg.get("role") == "assistant" else msg.get("role", "user")
                content = msg.get("content", "")
                gemini_messages.append({"role": role, "parts": [{"text": content}]})

            # Build config
            config = GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature or self.temperature,
                max_output_tokens=max_tokens or self.max_tokens,
            )

            # Call API
            response: GenerateContentResponse = self.client.models.generate_content(
                model=self.model,
                contents=gemini_messages,
                config=config,
            )

            # Extract text
            text = response.text
            if not text:
                raise ValueError("Empty response from LLM")

            logger.debug(f"LLM call with messages successful: {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"Error calling LLM with messages: {e}")
            raise
