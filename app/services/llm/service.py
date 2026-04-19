"""
LLM Service

Handles LLM API calls with strict separation of concerns.
Only responsible for calling Gemini 2.5 Flash - no business logic.
"""

import hashlib
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    CreateCachedContentConfig,
    GenerateContentConfig,
    GenerateContentResponse,
    ThinkingConfig,
)


# Applied to every call. Disables 2.5 Flash's implicit "dynamic thinking"
# (which burns wall-clock even when thoughts_token_count reports 0) and
# turns off automatic function-calling schema validation (not used here).
_FAST_MODE_CONFIG = {
    "thinking_config": ThinkingConfig(thinking_budget=0),
    "automatic_function_calling": AutomaticFunctionCallingConfig(disable=True),
}
from loguru import logger

from app.settings import settings
from app.services.llm.response_handler import ResponseHandler


# System-prompt cache: {(model, sha256(system_prompt)): (cache_name, created_at)}
# Shared across LLMService instances so every request hits the same cache.
_SYSTEM_CACHE: Dict[Tuple[str, str], Tuple[str, float]] = {}
_SYSTEM_CACHE_LOCK = threading.Lock()
_SYSTEM_CACHE_TTL_SECONDS = 3600  # 1h — matches cache TTL sent to Gemini
_SYSTEM_CACHE_MIN_TOKENS = 1024   # Gemini 2.5 Flash caching floor
_AVG_CHARS_PER_TOKEN = 3.5        # rough heuristic to skip tiny prompts


# Gemini pricing in USD per 1M tokens.
#   - "input"   — uncached prompt tokens
#   - "cached"  — prompt tokens served from a context cache (25% of input rate)
#   - "output"  — generated tokens (thoughts billed at this rate too)
#   - "storage_per_hour" — cost per 1M tokens held in a context cache per hour
# Source: https://ai.google.dev/gemini-api/docs/pricing (verify periodically)
_PRICE_PER_1M = {
    "gemini-2.5-flash": {
        "input": 0.30, "cached": 0.075, "output": 2.50, "thoughts": 2.50,
        "storage_per_hour": 1.00,
    },
    "gemini-2.5-pro": {
        "input": 1.25, "cached": 0.3125, "output": 10.00, "thoughts": 10.00,
        "storage_per_hour": 4.50,
    },
    "gemini-2.0-flash": {
        "input": 0.10, "cached": 0.025, "output": 0.40, "thoughts": 0.40,
        "storage_per_hour": 1.00,
    },
}


def _cost_usd(
    model: str,
    in_tok: int,
    out_tok: int,
    think_tok: int,
    cached_tok: int = 0,
) -> float:
    """Per-call cost in USD.

    ``in_tok`` is the API-reported ``prompt_token_count`` which INCLUDES the
    cached portion. We split it so cached tokens are billed at the cached rate
    (25% of input) and the rest at the standard input rate.
    """
    rates = _PRICE_PER_1M.get(model)
    if not rates:
        return 0.0
    cached_tok = max(0, min(cached_tok, in_tok))
    uncached_in = in_tok - cached_tok
    return (
        uncached_in * rates["input"]
        + cached_tok * rates["cached"]
        + out_tok * rates["output"]
        + think_tok * rates["thoughts"]
    ) / 1_000_000


def _cache_storage_cost_usd(model: str, cached_tok: int, ttl_seconds: int) -> float:
    """Storage cost for holding ``cached_tok`` tokens in a context cache for ``ttl_seconds``.

    This is charged once at cache-creation time and represents the full TTL.
    """
    rates = _PRICE_PER_1M.get(model)
    if not rates or cached_tok <= 0 or ttl_seconds <= 0:
        return 0.0
    hours = ttl_seconds / 3600.0
    return (cached_tok * rates["storage_per_hour"] * hours) / 1_000_000


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

    def _get_or_create_system_cache(self, system_prompt: str) -> Optional[str]:
        """Return a Gemini cache resource name for `system_prompt`, or None.

        Caches are keyed by (model, sha256(system_prompt)) and refreshed after
        TTL expiry. Returns None when the prompt is below Gemini's cache floor
        (~1024 tokens) or when cache creation fails — callers then fall back to
        inline system_instruction.
        """
        if not self.client or not system_prompt:
            return None
        # Skip caching for prompts that are clearly too small to qualify.
        if len(system_prompt) < _SYSTEM_CACHE_MIN_TOKENS * _AVG_CHARS_PER_TOKEN:
            return None

        key = (self.model, hashlib.sha256(system_prompt.encode("utf-8")).hexdigest())
        now = time.time()

        with _SYSTEM_CACHE_LOCK:
            entry = _SYSTEM_CACHE.get(key)
            if entry and (now - entry[1]) < (_SYSTEM_CACHE_TTL_SECONDS - 60):
                return entry[0]

            try:
                cache = self.client.caches.create(
                    model=self.model,
                    config=CreateCachedContentConfig(
                        system_instruction=system_prompt,
                        ttl=f"{_SYSTEM_CACHE_TTL_SECONDS}s",
                    ),
                )
                _SYSTEM_CACHE[key] = (cache.name, now)
                # Report the one-time storage cost for the full TTL. The
                # cache's usage_metadata carries the exact token count; fall
                # back to a char-based estimate if the field is absent.
                cache_tok = 0
                um = getattr(cache, "usage_metadata", None)
                if um is not None:
                    cache_tok = getattr(um, "total_token_count", 0) or 0
                if not cache_tok:
                    cache_tok = int(len(system_prompt) / _AVG_CHARS_PER_TOKEN)
                storage_cost = _cache_storage_cost_usd(
                    self.model, cache_tok, _SYSTEM_CACHE_TTL_SECONDS,
                )
                logger.info(
                    f"[cache] created system-prompt cache {cache.name} "
                    f"({len(system_prompt)}ch, {cache_tok} tok, model={self.model}) "
                    f"storage ${storage_cost:.6f} for {_SYSTEM_CACHE_TTL_SECONDS // 60}min TTL"
                )
                return cache.name
            except Exception as e:
                # Most common cause: prompt below the model's minimum cache size.
                logger.warning(f"[cache] create failed ({e}); falling back to inline system_instruction")
                # Negative-cache the failure briefly so we don't hammer the API.
                _SYSTEM_CACHE[key] = ("", now)
                return None

    def _invalidate_system_cache(self, system_prompt: str) -> None:
        """Drop a cached entry — call when Gemini reports the cache expired."""
        if not system_prompt:
            return
        key = (self.model, hashlib.sha256(system_prompt.encode("utf-8")).hexdigest())
        with _SYSTEM_CACHE_LOCK:
            _SYSTEM_CACHE.pop(key, None)

    def call(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        return_metadata: bool = False,
        add_warning: bool = True,
        response_mime_type: Optional[str] = None,
    ) -> Union[str, Tuple[str, Dict[str, any]]]:
        """
        Call the LLM with a prompt.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Optional temperature override
            max_tokens: Optional max tokens override
            add_warning: Whether to append a truncation warning to the response.
                         Set to False for structured JSON calls so the warning
                         text does not corrupt the JSON output.
            response_mime_type: Force the model's response format, e.g.
                                "application/json" for strict JSON output.
                                On Gemini 2.5 this substantially reduces
                                thinking-token consumption on structured tasks
                                because the model doesn't need to explore how
                                to format the answer.

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
                    add_warning=add_warning,
                    response_mime_type=response_mime_type,
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
        add_warning: bool = True,
        response_mime_type: Optional[str] = None,
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
            add_warning: Append truncation warning to response text (disable for JSON)
            attempt: Current attempt number
            start_time: Start time for timing

        Returns:
            Generated text response (or tuple with metadata)
        """
        try:
            # Build messages
            messages = [{"role": "user", "parts": [{"text": prompt}]}]

            # Build config
            # Gemini 2.5 Flash supports up to 65,536 output tokens
            _GEMINI_MAX_OUTPUT_TOKENS = 65_536
            requested_max_tokens = max_tokens or self.max_tokens
            actual_max_tokens = min(requested_max_tokens, _GEMINI_MAX_OUTPUT_TOKENS)

            cache_name = self._get_or_create_system_cache(system_instruction) if system_instruction else None

            config_kwargs = {
                "temperature": temperature or self.temperature,
                "max_output_tokens": actual_max_tokens,
                **_FAST_MODE_CONFIG,
            }
            if cache_name:
                config_kwargs["cached_content"] = cache_name
            elif system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            # When a structured mime type is requested (e.g. application/json),
            # pass it through. This dramatically reduces thinking-token
            # consumption on Gemini 2.5 for JSON tasks because the model is
            # no longer exploring how to format the output.
            if response_mime_type:
                config_kwargs["response_mime_type"] = response_mime_type
            config = GenerateContentConfig(**config_kwargs)

            # Call API
            try:
                response: GenerateContentResponse = self.client.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=config,
                )
            except Exception as api_err:
                if cache_name and "cache" in str(api_err).lower():
                    logger.warning(f"[cache] stale cache {cache_name}; retrying without: {api_err}")
                    self._invalidate_system_cache(system_instruction)
                    config_kwargs.pop("cached_content", None)
                    config_kwargs["system_instruction"] = system_instruction
                    config = GenerateContentConfig(**config_kwargs)
                    response = self.client.models.generate_content(
                        model=self.model,
                        contents=messages,
                        config=config,
                    )
                else:
                    raise

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
            prompt_tokens = None
            thoughts_tokens = None
            cached_tokens = None
            if hasattr(response, "usage_metadata"):
                um = response.usage_metadata
                output_tokens = getattr(um, "candidates_token_count", None)
                prompt_tokens = getattr(um, "prompt_token_count", None)
                thoughts_tokens = getattr(um, "thoughts_token_count", None)
                cached_tokens = getattr(um, "cached_content_token_count", None)

            # Compute estimated cost (treats None as 0 for missing fields)
            call_cost_usd = _cost_usd(
                self.model,
                prompt_tokens or 0,
                output_tokens or 0,
                thoughts_tokens or 0,
                cached_tok=cached_tokens or 0,
            )

            # Validate and sanitize response
            is_truncated = finish_reason == "MAX_TOKENS"
            sanitized_text, response_metadata = self.response_handler.validate_and_sanitize(
                text=text,
                finish_reason=finish_reason,
                is_truncated=is_truncated,
            )
            
            # Add token usage and cost to metadata
            response_metadata.update({
                "input_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thoughts_tokens,
                "max_tokens": actual_max_tokens,
                "cost_usd": call_cost_usd,
                "processing_time": time.time() - start_time if start_time else None,
            })
            
            # Format response with warnings if needed
            formatted_text = self.response_handler.format_response(
                text=sanitized_text,
                metadata=response_metadata,
                add_warning=add_warning,
            )
            
            elapsed = response_metadata.get("processing_time", 0) or 0.0
            if is_truncated:
                logger.warning(
                    f"LLM [{self.model}] {elapsed:.2f}s TRUNCATED "
                    f"{output_tokens}/{actual_max_tokens} tok, {response_length}ch, "
                    f"cost=${call_cost_usd:.6f}"
                )
            elif not response_metadata.get("is_complete"):
                logger.warning(
                    f"LLM [{self.model}] {elapsed:.2f}s incomplete "
                    f"{response_length}ch, cost=${call_cost_usd:.6f}"
                )
            else:
                logger.info(
                    f"LLM [{self.model}] {elapsed:.2f}s | {response_length}ch | "
                    f"in={prompt_tokens} out={output_tokens} "
                    f"cached={cached_tokens or 0} thoughts={thoughts_tokens or 0} | "
                    f"${call_cost_usd:.6f}"
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
                **_FAST_MODE_CONFIG,
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

    def _build_synthesis_prompt(
        self,
        query: str,
        chunks: list,
        language: str = "en",
    ) -> Tuple[str, str, int]:
        """Build the system + user prompt for answer synthesis.

        Returns (system_prompt, user_prompt, context_chars).
        """
        from app.prompts.factory import get_prompts

        context_parts = []
        for idx, chunk in enumerate(chunks, 1):
            source = chunk.source if hasattr(chunk, "source") else chunk.get("source", {})
            doc_name = source.get("document_name", "Unknown")
            doc_id = source.get("document_id", "")
            section = source.get("section", "")
            node_id = source.get("node_id", "")
            page_range = source.get("page_range", [])
            first_page = page_range[0] if page_range else 1
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")

            context_parts.append(
                f"[Section {idx}]\n"
                f"Act: {doc_name}\n"
                f"Document ID: {doc_id}\n"
                f"Node: {node_id} — {section}\n"
                f"Page: {first_page}\n"
                f"Content: {text}\n"
            )

        context = "\n\n".join(context_parts)
        act_name = "Finance Act"
        if chunks:
            first_source = chunks[0].source if hasattr(chunks[0], "source") else chunks[0].get("source", {})
            act_name = first_source.get("document_name", "Finance Act")

        system_prompt, user_prompt_template = get_prompts("page_index", "answer_synthesis", language)
        user_prompt = user_prompt_template.format(
            act_name=act_name,
            sections=context,
            query=query,
        )
        return system_prompt, user_prompt, len(context)

    def synthesize(
        self,
        query: str,
        chunks: list,
        language: str = "en",
    ) -> str:
        """
        Synthesize an answer from retrieved chunks.

        Args:
            query: User question.
            chunks: List of RetrievedChunk objects.
            language: Answer language.

        Returns:
            Synthesized answer text.
        """
        system_prompt, user_prompt, context_chars = self._build_synthesis_prompt(
            query, chunks, language
        )

        synth_start = time.time()
        result, metadata = self.call(
            prompt=user_prompt,
            system_instruction=system_prompt,
            temperature=0.2,
            max_tokens=8192,
            return_metadata=True,
        )
        logger.info(
            f"[latency] synthesis LLM call: {time.time() - synth_start:.2f}s "
            f"(chunks={len(chunks)}, context_chars={context_chars})"
        )

        # Attach cost info for caller to record
        text = result if isinstance(result, str) else result
        return text, metadata

    def stream_synthesize(
        self,
        query: str,
        chunks: list,
        language: str = "en",
    ):
        """Synchronous generator yielding synthesis text chunks as they arrive.

        Callers bridge this to async by pulling one chunk at a time via
        ``asyncio.to_thread(next, gen, sentinel)``.

        Yields text strings. The final item is a dict
        ``{"__done__": True, "full_text": ..., "metadata": {...}}`` with
        aggregated metadata (cost, tokens, latency) — letting the caller
        record usage without re-parsing the SDK response object.
        """
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")

        system_prompt, user_prompt, context_chars = self._build_synthesis_prompt(
            query, chunks, language
        )

        cache_name = self._get_or_create_system_cache(system_prompt)
        if cache_name:
            config = GenerateContentConfig(
                cached_content=cache_name,
                temperature=0.2,
                max_output_tokens=8192,
                **_FAST_MODE_CONFIG,
            )
        else:
            config = GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=8192,
                **_FAST_MODE_CONFIG,
            )
        messages = [{"role": "user", "parts": [{"text": user_prompt}]}]

        synth_start = time.time()
        parts: list[str] = []
        last_response = None

        def _open_stream(cfg):
            return self.client.models.generate_content_stream(
                model=self.model,
                contents=messages,
                config=cfg,
            )

        try:
            try:
                stream = _open_stream(config)
            except Exception as e:
                # Cache may have been evicted server-side before we used it.
                if cache_name and "cache" in str(e).lower():
                    logger.warning(f"[cache] stale cache {cache_name}; retrying without: {e}")
                    self._invalidate_system_cache(system_prompt)
                    config = GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        max_output_tokens=8192,
                        **_FAST_MODE_CONFIG,
                    )
                    stream = _open_stream(config)
                else:
                    raise
            for chunk in stream:
                last_response = chunk
                # ``chunk.text`` is lazy-parsed by the SDK and can raise on
                # malformed intermediate chunks (empty candidates, safety
                # filter, transient protocol hiccups). Skip the bad chunk
                # rather than abort the whole stream — the next chunk almost
                # always recovers, and at worst we still have the last_response
                # to extract aggregate text from.
                try:
                    text = getattr(chunk, "text", None)
                except Exception as chunk_err:
                    logger.debug(f"Skipping unreadable stream chunk: {chunk_err}")
                    continue
                if text:
                    parts.append(text)
                    yield text
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            raise

        full_text = "".join(parts)
        elapsed = time.time() - synth_start

        # Aggregate usage/cost from the final response object
        input_tokens = output_tokens = thinking_tokens = cached_tokens = 0
        if last_response is not None and getattr(last_response, "usage_metadata", None):
            um = last_response.usage_metadata
            input_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = getattr(um, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(um, "thoughts_token_count", 0) or 0
            cached_tokens = getattr(um, "cached_content_token_count", 0) or 0

        cost_usd = _cost_usd(
            self.model, input_tokens, output_tokens, thinking_tokens,
            cached_tok=cached_tokens,
        )

        logger.info(
            f"LLM [{self.model}] {elapsed:.2f}s stream | {len(full_text)}ch | "
            f"in={input_tokens} out={output_tokens} cached={cached_tokens} thoughts={thinking_tokens} | "
            f"${cost_usd:.6f} (chunks={len(chunks)}, ctx={context_chars}ch)"
        )

        yield {
            "__done__": True,
            "full_text": full_text,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "cost_usd": cost_usd,
                "elapsed_seconds": elapsed,
                "model": self.model,
            },
        }

    # ------------------------------------------------------------------
    # Native async paths — use client.aio over aiohttp. Bypass the SDK's
    # sync-in-a-thread wrapper that costs 5–10× on streaming throughput.
    # ------------------------------------------------------------------

    async def a_call(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_mime_type: Optional[str] = None,
        return_metadata: bool = False,
    ) -> Union[str, Tuple[str, Dict[str, Any]]]:
        """Async single-turn call.

        Returns the response text. When ``return_metadata=True``, returns
        ``(text, metadata)`` where metadata carries input/output/cached/thinking
        tokens and the per-call ``cost_usd`` — exactly what the orchestrator
        needs to credit the correct amount to the user's daily usage counter.
        """
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")

        _GEMINI_MAX_OUTPUT_TOKENS = 65_536
        actual_max_tokens = min(max_tokens or self.max_tokens, _GEMINI_MAX_OUTPUT_TOKENS)

        cache_name = self._get_or_create_system_cache(system_instruction) if system_instruction else None

        config_kwargs = {
            "temperature": temperature or self.temperature,
            "max_output_tokens": actual_max_tokens,
            **_FAST_MODE_CONFIG,
        }
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        elif system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_mime_type:
            config_kwargs["response_mime_type"] = response_mime_type

        messages = [{"role": "user", "parts": [{"text": prompt}]}]

        start = time.time()
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=messages,
                config=GenerateContentConfig(**config_kwargs),
            )
        except Exception as api_err:
            if cache_name and "cache" in str(api_err).lower():
                logger.warning(f"[cache] stale cache {cache_name}; retrying without: {api_err}")
                self._invalidate_system_cache(system_instruction)
                config_kwargs.pop("cached_content", None)
                config_kwargs["system_instruction"] = system_instruction
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=messages,
                    config=GenerateContentConfig(**config_kwargs),
                )
            else:
                raise

        text = getattr(response, "text", "") or ""
        elapsed = time.time() - start

        in_tok = out_tok = cached_tok = think_tok = 0
        cost = 0.0
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            in_tok = getattr(um, "prompt_token_count", 0) or 0
            out_tok = getattr(um, "candidates_token_count", 0) or 0
            cached_tok = getattr(um, "cached_content_token_count", 0) or 0
            think_tok = getattr(um, "thoughts_token_count", 0) or 0
            cost = _cost_usd(
                self.model, in_tok, out_tok, think_tok, cached_tok=cached_tok,
            )
            logger.info(
                f"LLM [{self.model}] {elapsed:.2f}s | {len(text)}ch | "
                f"in={in_tok} out={out_tok} cached={cached_tok} thoughts={think_tok} | ${cost:.6f}"
            )
        if return_metadata:
            return text, {
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "cached_tokens": cached_tok,
                "thinking_tokens": think_tok,
                "cost_usd": cost,
                "elapsed_seconds": elapsed,
                "model": self.model,
            }
        return text

    async def a_stream_synthesize(
        self,
        query: str,
        chunks: list,
        language: str = "en",
    ):
        """Native-async streaming synthesis. Yields text; final yield is the
        ``{"__done__": True, ...}`` metadata envelope, mirroring the sync variant.
        """
        if not self.client:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY in .env file.")

        system_prompt, user_prompt, context_chars = self._build_synthesis_prompt(
            query, chunks, language
        )

        cache_name = self._get_or_create_system_cache(system_prompt)
        config_kwargs = {
            "temperature": 0.2,
            "max_output_tokens": 8192,
            **_FAST_MODE_CONFIG,
        }
        if cache_name:
            config_kwargs["cached_content"] = cache_name
        else:
            config_kwargs["system_instruction"] = system_prompt
        messages = [{"role": "user", "parts": [{"text": user_prompt}]}]

        synth_start = time.time()
        parts: list[str] = []
        last_response = None

        async def _iter(cfg):
            return await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=messages,
                config=cfg,
            )

        try:
            try:
                stream = await _iter(GenerateContentConfig(**config_kwargs))
            except Exception as e:
                if cache_name and "cache" in str(e).lower():
                    logger.warning(f"[cache] stale cache {cache_name}; retrying without: {e}")
                    self._invalidate_system_cache(system_prompt)
                    config_kwargs.pop("cached_content", None)
                    config_kwargs["system_instruction"] = system_prompt
                    stream = await _iter(GenerateContentConfig(**config_kwargs))
                else:
                    raise

            async for chunk in stream:
                last_response = chunk
                try:
                    text = getattr(chunk, "text", None)
                except Exception as chunk_err:
                    logger.debug(f"Skipping unreadable stream chunk: {chunk_err}")
                    continue
                if text:
                    parts.append(text)
                    yield text
        except Exception as e:
            logger.error(f"Async streaming synthesis failed: {e}")
            raise

        full_text = "".join(parts)
        elapsed = time.time() - synth_start

        input_tokens = output_tokens = thinking_tokens = cached_tokens = 0
        if last_response is not None and getattr(last_response, "usage_metadata", None):
            um = last_response.usage_metadata
            input_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = getattr(um, "candidates_token_count", 0) or 0
            thinking_tokens = getattr(um, "thoughts_token_count", 0) or 0
            cached_tokens = getattr(um, "cached_content_token_count", 0) or 0

        cost_usd = _cost_usd(
            self.model, input_tokens, output_tokens, thinking_tokens,
            cached_tok=cached_tokens,
        )

        logger.info(
            f"LLM [{self.model}] {elapsed:.2f}s stream(async) | {len(full_text)}ch | "
            f"in={input_tokens} out={output_tokens} cached={cached_tokens} thoughts={thinking_tokens} | "
            f"${cost_usd:.6f} (chunks={len(chunks)}, ctx={context_chars}ch)"
        )

        yield {
            "__done__": True,
            "full_text": full_text,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thinking_tokens": thinking_tokens,
                "cost_usd": cost_usd,
                "elapsed_seconds": elapsed,
                "model": self.model,
            },
        }
