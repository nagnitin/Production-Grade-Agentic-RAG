import re
from typing import Optional
from nemoguardrails.actions.actions import action

# Compiled regex patterns for PII detection
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")

# Off-topic patterns (e.g. system commands, code injection)
OFF_TOPIC_REGEX = re.compile(r"\b(hack|crack|bypass|jailbreak|system prompt|ignore instruction)\b", re.IGNORECASE)


@action(name="check_pii_input")
async def check_pii_input(user_input: Optional[str] = None) -> bool:
    """Check if the user input contains sensitive PII."""
    if not user_input:
        return False
    
    if (
        EMAIL_REGEX.search(user_input)
        or PHONE_REGEX.search(user_input)
        or SSN_REGEX.search(user_input)
        or CREDIT_CARD_REGEX.search(user_input)
    ):
        return True
    return False


@action(name="check_pii_output")
async def check_pii_output(bot_response: Optional[str] = None) -> str:
    """Mask any PII detected in the bot output."""
    if not bot_response:
        return ""
    
    sanitized = EMAIL_REGEX.sub("[EMAIL]", bot_response)
    sanitized = PHONE_REGEX.sub("[PHONE]", sanitized)
    sanitized = SSN_REGEX.sub("[SSN]", sanitized)
    sanitized = CREDIT_CARD_REGEX.sub("[CARD]", sanitized)
    return sanitized


@action(name="check_off_topic_input")
async def check_off_topic_input(user_input: Optional[str] = None) -> bool:
    """Check if user input is malicious or contains jailbreak terms."""
    if not user_input:
        return False
    
    if OFF_TOPIC_REGEX.search(user_input):
        return True
    return False
