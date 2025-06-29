"""Centralised helpers for authentication credentials.

This module avoids importing heavy cloud SDKs at import-time; it only touches
`os.environ` and lightweight stdlib modules. Third-party cloud libraries still
perform their own credential resolution, but we surface friendly errors early
so users know what to configure.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import AuthenticationError, MissingDependencyError

_CONFIG_PATHS = [
    Path.home() / ".vxdf" / "config.toml",
    Path.home() / ".config" / "vxdf" / "config.toml",
]


def _load_config() -> Dict[str, Any]:
    """Return merged user config from the first TOML file found."""
    for cfg_path in _CONFIG_PATHS:
        if cfg_path.is_file():
            try:
                import toml  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise MissingDependencyError(
                    "Reading ~/.vxdf/config.toml requires the 'toml' package. Install via 'pip install toml'."
                ) from exc
            with open(cfg_path, encoding="utf-8") as f:
                return toml.load(f)
    return {}


# cache after first read so we don't re-parse the file on every call
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = _load_config()
    return _CONFIG_CACHE


# ---------------------------------------------------------------------------
# AWS helpers
# ---------------------------------------------------------------------------

def ensure_aws_credentials() -> None:
    """Raise ``AuthenticationError`` if AWS credentials cannot be resolved.

    We rely on ``boto3``'s default credential chain (env vars, AWS CLI config,
    IAM roles). For a fast-fail and nicer error UX we test resolution early so
    the user gets an actionable message before a download starts.
    """
    try:
        import boto3  # type: ignore
        import botocore.exceptions  # type: ignore
    except ImportError:
        # Dependency helper already handled elsewhere
        return

    session = boto3.session.Session()
    if session.get_credentials() is None:
        raise AuthenticationError(
            "AWS credentials not found. Configure them via environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY), \n"
            "AWS CLI (`aws configure`), or an IAM role."
        )


# ---------------------------------------------------------------------------
# GCP helpers
# ---------------------------------------------------------------------------

def ensure_gcp_credentials() -> None:
    """Raise ``AuthenticationError`` if Application Default Credentials are missing."""
    try:
        from google.auth import default as _google_default  # type: ignore
        from google.auth.exceptions import DefaultCredentialsError  # type: ignore
    except ImportError:
        # dependency missing – earlier checks cover this
        return

    try:
        _google_default()
    except DefaultCredentialsError as exc:
        raise AuthenticationError(
            "Google Cloud credentials not found. Run `gcloud auth application-default login` or set GOOGLE_APPLICATION_CREDENTIALS."
        ) from exc


# ---------------------------------------------------------------------------
# OpenAI helpers
# ---------------------------------------------------------------------------

def _write_config(cfg: Dict[str, Any]) -> None:
    """Persist *cfg* to the first config path in ``_CONFIG_PATHS``.

    Creates parent directories as needed. Overwrites the file (if it exists).
    Requires the optional ``toml`` dependency.
    """
    cfg_path = _CONFIG_PATHS[0]
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import toml  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise MissingDependencyError(
            "Saving ~/.vxdf/config.toml requires the 'toml' package. Install via 'pip install toml'."
        ) from exc

    with open(cfg_path, "w", encoding="utf-8") as f:
        toml.dump(cfg, f)

    # refresh cache so subsequent reads see the new value
    global _CONFIG_CACHE
    _CONFIG_CACHE = cfg

def prompt_and_save_openai_key() -> Optional[str]:
    """Interactively prompt the user for an OpenAI API key and save it.

    Returns the key if one was entered, otherwise ``None``. No prompt is shown
    if stdin is not a TTY (e.g., running in CI).
    """
    import getpass
    import sys

    if not sys.stdin.isatty():
        return None

    try:
        key = getpass.getpass(
            "No OpenAI API key found.\nEnter key now (leave blank to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        # User aborted input
        sys.stderr.write("\n")
        return None

    if not key:
        return None

    # Merge with existing config (if any) and persist
    cfg = _config()
    openai_cfg = cfg.setdefault("openai", {})  # type: ignore[arg-type]
    openai_cfg["api_key"] = key
    _write_config(cfg)

    print(f"Stored key in {_CONFIG_PATHS[0]} for future runs.")
    return key



def get_openai_api_key(cli_key: Optional[str] = None) -> str:
    """Return an OpenAI API key from CLI flag, env var, or user config.

    Lookup order (first win):
    1. *cli_key* argument passed by caller.
    2. Environment variable ``OPENAI_API_KEY``.
    3. ``openai.api_key`` field in ``~/.vxdf/config.toml``.

    Raises
    ------
    AuthenticationError
        If no key is found by any method.
    """
    if cli_key:
        return cli_key

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    cfg_key = _config().get("openai", {}).get("api_key")  # type: ignore[arg-type]
    if cfg_key:
        return str(cfg_key)

    raise AuthenticationError(
        "OpenAI API key not found. Pass --openai-key, set OPENAI_API_KEY env var, or add it to ~/.vxdf/config.toml under [openai]."
    )
