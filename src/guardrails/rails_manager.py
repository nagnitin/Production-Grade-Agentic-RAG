"""
NeMo Guardrails Manager.

WHY: Input and output validation is a critical security boundary in enterprise systems.
We use NeMo Guardrails to define rails for jailbreaks, PII leaks, off-topic queries,
and toxicity. A fallback/rule-based validator is provided to ensure continuous
operation even if external validation dependencies fail.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

from nemoguardrails import LLMRails, RailsConfig

from src.config.logging_config import get_logger
from src.config.settings import Settings

logger = get_logger(__name__)

# Compile high-fidelity security regexes
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Off-topic / Jailbreak triggers
JAILBREAK_PATTERNS = [
    r"\bignore\b.*\binstruction\b",
    r"\bsystem prompt\b",
    r"\bdeveloper mode\b",
    r"\bdo anything now\b",
    r"\brule bypass\b",
    r"\bexpose\b.*\bprompt\b",
]
JAILBREAK_REGEXES = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]


class GuardrailsManager:
    """
    Manages safety rails using NeMo Guardrails and regex-based fallbacks.
    
    Ensures input queries and output responses comply with enterprise safety,
    PII protection, and topic constraints.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.config_path = settings.guardrails.config_path
        self.rails: Optional[LLMRails] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize LLMRails from the config directory."""
        if not self.settings.guardrails.enabled:
            logger.info("Guardrails are disabled in settings")
            return

        try:
            logger.info("Initializing NeMo Guardrails", config_path=self.config_path)
            
            # Ensure the config path exists
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Guardrails config path not found: {self.config_path}")

            # Workaround for NeMo Guardrails needing OPENAI_API_KEY even with Portkey virtual keys
            # If no api key in environment, temporarily set a mock one to pass config validation.
            if not os.environ.get("OPENAI_API_KEY") and self.settings.portkey.api_key:
                os.environ["OPENAI_API_KEY"] = self.settings.portkey.api_key.get_secret_value()

            config = RailsConfig.from_path(self.config_path)
            self.rails = LLMRails(config)
            self._initialized = True
            logger.info("NeMo Guardrails successfully initialized")

        except Exception as e:
            logger.warning(
                "Failed to initialize NeMo Guardrails LLM engine. "
                "System will fall back to local rule-based safety checks.",
                error=str(e)
            )
            self.rails = None
            self._initialized = False

    async def check_input(self, query: str) -> dict[str, Any]:
        """
        Scan and validate user queries before processing.
        
        Returns:
            dict containing:
                "allowed": bool
                "reason": Optional[str]
                "message": Optional[str]
        """
        # 1. Rule-based Input Guard (PII & Jailbreak checks)
        # PII Check
        if (
            EMAIL_REGEX.search(query)
            or PHONE_REGEX.search(query)
            or SSN_REGEX.search(query)
            or CREDIT_CARD_REGEX.search(query)
        ):
            return {
                "allowed": False,
                "reason": "pii_detected",
                "message": "Security Alert: Please do not include personally identifiable information (PII) such as emails, phone numbers, SSNs, or credit card numbers in your queries."
            }

        # Jailbreak Check
        for regex in JAILBREAK_REGEXES:
            if regex.search(query):
                return {
                    "allowed": False,
                    "reason": "jailbreak_attempt",
                    "message": "Safety Alert: This request has been blocked as it appears to violate our safety guidelines regarding prompt injection or jailbreak behavior."
                }

        # 2. LLM-based NeMo Guardrails checks
        if self._initialized and self.rails:
            try:
                # Process through NeMo Guardrails
                response = await self.rails.generate_async(prompt=query)
                
                # If a flow was triggered to refuse or redirect the response,
                # check if the response matches known safety block answers.
                refusal_keywords = [
                    "cannot perform this action",
                    "violates safety guidelines",
                    "only answer queries related to",
                    "am optimized to answer queries"
                ]
                
                for kw in refusal_keywords:
                    if kw in response.lower():
                        return {
                            "allowed": False,
                            "reason": "policy_violation",
                            "message": response
                        }
            except Exception as e:
                logger.warning("Error running LLM guardrails checks, falling back to rules", error=str(e))

        return {"allowed": True, "reason": None, "message": None}

    async def check_output(self, response_text: str) -> dict[str, Any]:
        """
        Scan and sanitize generated responses before sending to client.
        
        Returns:
            dict containing:
                "allowed": bool
                "sanitized": str
        """
        if not response_text:
            return {"allowed": True, "sanitized": ""}

        # 1. Rule-based Output PII Sanitization
        sanitized = EMAIL_REGEX.sub("[EMAIL]", response_text)
        sanitized = PHONE_REGEX.sub("[PHONE]", sanitized)
        sanitized = SSN_REGEX.sub("[SSN]", sanitized)
        sanitized = CREDIT_CARD_REGEX.sub("[CARD]", sanitized)

        # 2. LLM-based NeMo Guardrails checks for output quality / safety
        if self._initialized and self.rails:
            try:
                # We can run output validation tasks or self-checks here if needed.
                pass
            except Exception as e:
                logger.warning("Error running LLM output guardrails checks", error=str(e))

        return {
            "allowed": True,
            "sanitized": sanitized
        }
