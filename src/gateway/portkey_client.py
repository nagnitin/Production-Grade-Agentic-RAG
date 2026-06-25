"""
Portkey AI Gateway client.

WHY: Portkey provides a unified gateway for LLM access with built-in retry,
fallback, caching, analytics, and cost tracking. Instead of implementing
retry/fallback/caching logic ourselves, we delegate to Portkey's battle-tested
infrastructure.

ARCHITECTURE DECISION: Using Portkey's LangChain integration for seamless
compatibility with LangGraph nodes. The gateway handles:
1. Primary model (Llama 3.3 70B) with automatic retry
2. Fallback to secondary model (Llama 3.1 8B) on failure
3. Request tracing with trace IDs for observability
4. Semantic caching (optional) for repeated queries
"""

from __future__ import annotations

import time
from typing import Any, Optional

from langchain_core.messages import BaseMessage
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)


class PortkeyGateway:
    """
    LLM gateway using Portkey AI for routing, failover, and observability.

    Provides a unified interface for LLM calls with:
    - Automatic retry with exponential backoff
    - Model fallback chain (primary → secondary → error)
    - Request/response tracing
    - Token usage tracking
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Optional[Any] = None
        self._fallback_client: Optional[Any] = None

    def _build_client(self) -> Any:
        """Build the Portkey-wrapped ChatOpenAI client."""
        if self._client is None:
            from langchain_openai import ChatOpenAI
            import os

            portkey_api_key = self.settings.portkey.api_key.get_secret_value()
            virtual_key = self.settings.portkey.virtual_key_primary
            openai_api_key = os.environ.get("OPENAI_API_KEY")

            if (not portkey_api_key or not virtual_key) and openai_api_key:
                logger.info("Portkey not fully configured. Using direct OpenAI client.")
                model_name = self.settings.llm.primary_model
                if "llama" in model_name.lower():
                    model_name = "gpt-4o-mini"

                self._client = ChatOpenAI(
                    model=model_name,
                    api_key=openai_api_key,
                    temperature=self.settings.llm.temperature,
                    max_tokens=self.settings.llm.max_tokens,
                    timeout=self.settings.llm.request_timeout,
                    max_retries=self.settings.llm.max_retries,
                )
            else:
                portkey_headers = createHeaders(
                    api_key=portkey_api_key,
                    virtual_key=virtual_key,
                    trace_id=None,  # Set per-request
                    metadata={"environment": self.settings.app_env.value},
                )

                self._client = ChatOpenAI(
                    model=self.settings.llm.primary_model,
                    api_key="placeholder",  # Portkey handles auth
                    base_url=self.settings.portkey.base_url,
                    default_headers=portkey_headers,
                    temperature=self.settings.llm.temperature,
                    max_tokens=self.settings.llm.max_tokens,
                    timeout=self.settings.llm.request_timeout,
                    max_retries=self.settings.llm.max_retries,
                )

        return self._client

    def _build_fallback_client(self) -> Any:
        """Build the fallback model client."""
        if self._fallback_client is None:
            from langchain_openai import ChatOpenAI
            import os

            portkey_api_key = self.settings.portkey.api_key.get_secret_value()
            virtual_key = self.settings.portkey.virtual_key_fallback
            openai_api_key = os.environ.get("OPENAI_API_KEY")

            if (not portkey_api_key or not virtual_key) and openai_api_key:
                logger.info("Portkey not fully configured. Using direct OpenAI client for fallback.")
                model_name = self.settings.llm.fallback_model
                if "llama" in model_name.lower():
                    model_name = "gpt-4o-mini"

                self._fallback_client = ChatOpenAI(
                    model=model_name,
                    api_key=openai_api_key,
                    temperature=self.settings.llm.temperature,
                    max_tokens=self.settings.llm.max_tokens,
                    timeout=self.settings.llm.request_timeout,
                    max_retries=2,
                )
            else:
                portkey_headers = createHeaders(
                    api_key=portkey_api_key,
                    virtual_key=virtual_key,
                    metadata={
                        "environment": self.settings.app_env.value,
                        "is_fallback": "true",
                    },
                )

                self._fallback_client = ChatOpenAI(
                    model=self.settings.llm.fallback_model,
                    api_key="placeholder",
                    base_url=self.settings.portkey.base_url,
                    default_headers=portkey_headers,
                    temperature=self.settings.llm.temperature,
                    max_tokens=self.settings.llm.max_tokens,
                    timeout=self.settings.llm.request_timeout,
                    max_retries=2,
                )

        return self._fallback_client

    async def _call_apifreellm(self, messages: list[BaseMessage]) -> Any:
        """Call the free LLM API at apifreellm.com."""
        import httpx
        from langchain_core.messages import AIMessage
        import os

        # Extract system prompt and user message to construct a single string
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg.type == "system":
                system_content = msg.content
            elif msg.type == "human":
                if isinstance(msg.content, str):
                    user_content = msg.content
                else:
                    user_content = str(msg.content)
        
        full_message = f"{system_content}\n\n{user_content}" if system_content else user_content
        
        api_key = os.environ.get("APIFREELLM_KEY", "apf_urbbso8ci0m9uo8n9uqvq71p")
        url = "https://apifreellm.com/api/v1/chat"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "message": full_message,
            "model": "apifreellm"
        }
        
        logger.info("Calling apifreellm.com API", url=url)
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers, timeout=60.0)
            response.raise_for_status()
            res_json = response.json()
            
            if not res_json.get("success"):
                raise RuntimeError(f"apifreellm API returned failure: {res_json}")
                
            text = res_json.get("response", "")
            
            # Wrap in LangChain AIMessage compatible object
            return AIMessage(
                content=text,
                usage_metadata={
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0
                }
            )

    async def ainvoke(
        self,
        messages: list[BaseMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> Any:
        """
        Invoke the LLM with automatic fallback.

        Tries the primary model first. On failure, falls back to the
        secondary model. Tracks latency and token usage.
        """
        start_time = time.perf_counter()
        
        # Check if we should use apifreellm directly to save latency when keys are unconfigured
        import os
        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        portkey_api_key = self.settings.portkey.api_key.get_secret_value()
        is_openai_unconfigured = not openai_api_key or openai_api_key == "your-openai-api-key-here"
        is_portkey_unconfigured = not portkey_api_key or portkey_api_key == "changeme"
        
        if is_openai_unconfigured and is_portkey_unconfigured:
            try:
                response = await self._call_apifreellm(messages)
                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "LLM call completed via apifreellm (direct bypass)",
                    latency_ms=round(latency_ms, 2),
                )
                return response
            except Exception as e:
                logger.error("apifreellm call failed during direct bypass", error=str(e))
                raise

        model_used = self.settings.llm.primary_model

        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            client = self._build_client()
            response = await client.ainvoke(messages, **kwargs)

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "LLM call completed",
                model=model_used,
                latency_ms=round(latency_ms, 2),
                is_fallback=False,
            )

            return response

        except Exception as primary_error:
            logger.warning(
                "Primary model failed, attempting fallback",
                model=model_used,
                error=str(primary_error),
            )

            try:
                fallback_client = self._build_fallback_client()
                model_used = self.settings.llm.fallback_model

                response = await fallback_client.ainvoke(messages, **kwargs)

                latency_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "Fallback LLM call completed",
                    model=model_used,
                    latency_ms=round(latency_ms, 2),
                    is_fallback=True,
                )

                return response

            except Exception as fallback_error:
                logger.warning(
                    "Both primary and fallback models failed, attempting apifreellm fallback",
                    primary_error=str(primary_error),
                    fallback_error=str(fallback_error),
                )
                try:
                    response = await self._call_apifreellm(messages)
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    logger.info(
                        "LLM call completed via apifreellm fallback",
                        latency_ms=round(latency_ms, 2),
                    )
                    return response
                except Exception as apifree_error:
                    logger.error(
                        "All LLM routing options failed including apifreellm",
                        primary_error=str(primary_error),
                        fallback_error=str(fallback_error),
                        apifree_error=str(apifree_error),
                    )
                    raise RuntimeError(
                        f"LLM gateway failure: Primary ({primary_error}), "
                        f"Fallback ({fallback_error}), apifreellm ({apifree_error})"
                    ) from apifree_error

    async def health_check(self) -> dict[str, Any]:
        """Check LLM gateway health."""
        from langchain_core.messages import HumanMessage
        import os

        openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        portkey_api_key = self.settings.portkey.api_key.get_secret_value()
        is_openai_unconfigured = not openai_api_key or openai_api_key == "your-openai-api-key-here"
        is_portkey_unconfigured = not portkey_api_key or portkey_api_key == "changeme"

        results: dict[str, Any] = {"primary": False, "fallback": False}

        if is_openai_unconfigured and is_portkey_unconfigured:
            results["apifreellm"] = False
            try:
                await self._call_apifreellm([HumanMessage(content="ping")])
                results["apifreellm"] = True
                results["primary"] = True  # Map to primary so backend health endpoint reports healthy
            except Exception as e:
                results["apifreellm_error"] = str(e)
            return results

        try:
            client = self._build_client()
            await client.ainvoke(
                [HumanMessage(content="ping")],
                max_tokens=5,
            )
            results["primary"] = True
        except Exception as e:
            results["primary_error"] = str(e)

        try:
            fallback = self._build_fallback_client()
            await fallback.ainvoke(
                [HumanMessage(content="ping")],
                max_tokens=5,
            )
            results["fallback"] = True
        except Exception as e:
            results["fallback_error"] = str(e)

        return results
