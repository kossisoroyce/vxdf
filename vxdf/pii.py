"""Lightweight regex-based PII detector used during ingestion.

The detector is deliberately simple and dependency-free so it can run on any
machine with the Python stdlib.  It searches for well-formatted patterns such as
email addresses, US Social-Security numbers, credit-card numbers, IPv4/IPv6
addresses and US phone numbers.

The goal is *good enough* recall for an initial tag-only pass; downstream tools
or stricter detectors can refine the result.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional

__all__ = [
    "contains_pii",
    "compile_patterns",
    "DEFAULT_PATTERNS",
]

# ---------------------------------------------------------------------------
# Default regexes (anchored with word boundaries where possible)
# ---------------------------------------------------------------------------
# NOTE: These are not perfect; they trade recall vs precision for raw speed.

_EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"  # RFC-lite
_SSN_RE = r"\b\d{3}-\d{2}-\d{4}\b"
_PHONE_RE = r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
_CREDIT_RE = r"\b(?:\d[ -]*?){13,16}\b"
_IPV4_RE = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
# ---------------------------------------------------------------------------
# API key token patterns (commonly leaked secrets)
# ---------------------------------------------------------------------------
# Matches OpenAI keys like "sk-..."
_OPENAI_RE = r"\bsk-[A-Za-z0-9]{32,}\b"
# Matches AWS Access Key IDs (16-char suffix)
_AWS_ACCESS_RE = r"\bAKIA[0-9A-Z]{16}\b"
# Matches GitHub personal/org/refresh tokens (ghp_/gho_/ghu_)
_GITHUB_RE = r"\bgh[opsu]_[A-Za-z0-9]{36}\b"
# Matches Google API keys starting with AIza followed by 35 chars
_GOOGLE_API_RE = r"\bAIza[0-9A-Za-z-_]{35}\b"
# Matches Stripe secret keys
_STRIPE_RE = r"\bsk_(live|test)_[0-9a-zA-Z]{24}\b"
# Matches Slack tokens (bot/user/app)
_SLACK_RE = r"\bxox[baprs]-[0-9a-zA-Z]{10,48}\b"
# Matches Notion integration secrets
_NOTION_RE = r"\bsecret_[0-9a-zA-Z]{32}\b"
# Matches SendGrid API keys
_SENDGRID_RE = r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"
# Matches Twilio auth tokens (32 hex)
_TWILIO_RE = r"\b[0-9a-fA-F]{32}\b"
# Matches Facebook Graph API tokens (EAAG...)
_FACEBOOK_RE = r"\bEAAG[A-Za-z0-9]{20,}\b"
# Matches Mailgun private keys
_MAILGUN_RE = r"\bkey-[0-9a-fA-F]{32}\b"
# Matches generic JWTs (header.payload.signature)
_JWT_RE = r"\b[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"
# Matches PEM private key headers
_PEM_RE = r"-----BEGIN (?:RSA |EC |DSA |)PRIVATE KEY-----"

DEFAULT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in (
        _EMAIL_RE,
        _SSN_RE,
        _PHONE_RE,
        _CREDIT_RE,
        _IPV4_RE,
        _OPENAI_RE,
        _AWS_ACCESS_RE,
        _GITHUB_RE,
        _GOOGLE_API_RE,
        _STRIPE_RE,
        _SLACK_RE,
        _NOTION_RE,
        _SENDGRID_RE,
        _TWILIO_RE,
        _FACEBOOK_RE,
        _MAILGUN_RE,
        _JWT_RE,
        _PEM_RE,
    )
]


def compile_patterns(patterns: Iterable[str]) -> List[re.Pattern[str]]:
    """Compile a list of string patterns into regex Pattern objects."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def contains_pii(text: str, *, patterns: Optional[List[re.Pattern[str]]] = None) -> bool:  # noqa: D401
    """Return *True* if *text* matches any of the PII regexes.

    This helper is intentionally fast; it stops at the first match.
    """
    pats = patterns or DEFAULT_PATTERNS
    return any(p.search(text) for p in pats)
