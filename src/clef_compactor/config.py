"""Configuration with environment-variable validation at startup.

Settings are loaded once and validated eagerly, so a misconfigured deployment
fails fast with one actionable message that lists **all** problems instead of
an obscure HTTP 401 later.

Environment variables:

* ``CLEF_ACCOUNT_ID`` (fallback: ``CLOUDFLARE_ACCOUNT_ID``) -- required.
* ``CLEF_API_TOKEN`` (fallback: ``CLOUDFLARE_API_TOKEN``) -- required.
* ``CLEF_MODEL`` -- ``clef`` or ``clef-flash`` (default: ``clef``).
* ``CLEF_TIMEOUT`` -- per-request timeout seconds (default: ``60``).
* ``CLEF_MAX_RETRIES`` -- retries per request (default: ``2``).
* ``CLEF_LOG_LEVEL`` -- DEBUG/INFO/WARNING/ERROR/CRITICAL (default: ``WARNING``).
* ``CLEF_BASE_URL`` -- API root override for tests or AI Gateway (optional).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass

from .exceptions import ConfigurationError

__all__ = [
    "Settings",
    "SUPPORTED_MODELS",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_LOG_LEVEL",
]

SUPPORTED_MODELS: tuple[str, ...] = ("clef", "clef-flash")
DEFAULT_BASE_URL = "https://api.cloudflare.com/client/v4"
DEFAULT_MODEL = "clef"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_LOG_LEVEL = "WARNING"

_ACCOUNT_ID_VARS = ("CLEF_ACCOUNT_ID", "CLOUDFLARE_ACCOUNT_ID")
_API_TOKEN_VARS = ("CLEF_API_TOKEN", "CLOUDFLARE_API_TOKEN")
_MODEL_VAR = "CLEF_MODEL"
_BASE_URL_VAR = "CLEF_BASE_URL"
_TIMEOUT_VAR = "CLEF_TIMEOUT"
_MAX_RETRIES_VAR = "CLEF_MAX_RETRIES"
_LOG_LEVEL_VAR = "CLEF_LOG_LEVEL"

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _resolve_credential(env: Mapping[str, str], vars_: tuple[str, ...]) -> str | None:
    """Return the first non-empty value among *vars_*, or ``None``."""
    for name in vars_:
        raw = env.get(name)
        if raw and raw.strip():
            return raw.strip()
    return None


@dataclass(frozen=True)
class Settings:
    """Validated connection settings for the Cloudflare Workers AI API.

    Attributes:
        account_id: Cloudflare account id.
        api_token: Bearer token with Workers AI run permission.
        model: Model selector, ``"clef"`` or ``"clef-flash"``.
        base_url: API root; override for tests or AI Gateway proxies.
        timeout: Per-request timeout in seconds.
        max_retries: Retries per request for transient failures, not counting
            the initial attempt.
        log_level: Stdlib logging level name for the ``clef_compactor`` logger.
    """

    account_id: str
    api_token: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    log_level: str = DEFAULT_LOG_LEVEL

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        account_id: str | None = None,
        api_token: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        log_level: str | None = None,
    ) -> Settings:
        """Build validated settings from the environment.

        Explicit keyword arguments win over environment variables, which makes
        overrides in tests and embeddings painless. Every problem is collected
        into a single :class:`ConfigurationError` message.
        """
        env = os.environ if environ is None else environ

        resolved_account_id = (
            account_id
            if account_id is not None
            else _resolve_credential(env, _ACCOUNT_ID_VARS)
        )
        resolved_token = (
            api_token if api_token is not None else _resolve_credential(env, _API_TOKEN_VARS)
        )
        resolved_model = model if model is not None else env.get(_MODEL_VAR, DEFAULT_MODEL)
        resolved_base_url = (
            base_url if base_url is not None else env.get(_BASE_URL_VAR, DEFAULT_BASE_URL)
        )
        resolved_log_level = (
            log_level if log_level is not None else env.get(_LOG_LEVEL_VAR, DEFAULT_LOG_LEVEL)
        )

        problems: list[str] = []
        if not (resolved_account_id or "").strip():
            problems.append(
                f"missing credentials: set {' or '.join(_ACCOUNT_ID_VARS)}"
            )
        if not (resolved_token or "").strip():
            problems.append(
                f"missing credentials: set {' or '.join(_API_TOKEN_VARS)}"
            )

        candidate = cls(
            account_id=(resolved_account_id or "").strip(),
            api_token=(resolved_token or "").strip(),
            model=(resolved_model or "").strip(),
            base_url=(resolved_base_url or "").strip().rstrip("/"),
            timeout=(
                timeout
                if timeout is not None
                else _parse_float(env.get(_TIMEOUT_VAR), _TIMEOUT_VAR, problems, DEFAULT_TIMEOUT_SECONDS)
            ),
            max_retries=(
                max_retries
                if max_retries is not None
                else _parse_int(env.get(_MAX_RETRIES_VAR), _MAX_RETRIES_VAR, problems, DEFAULT_MAX_RETRIES)
            ),
            log_level=(resolved_log_level or "").strip().upper(),
        )
        problems.extend(candidate._collect_problems())
        if problems:
            raise ConfigurationError(_format_problems(problems))
        return candidate

    def _collect_problems(self) -> list[str]:
        """Return every validation problem found on this instance."""
        problems: list[str] = []
        if self.model not in SUPPORTED_MODELS:
            problems.append(
                f"unsupported model {self.model!r}: choose one of "
                f"{', '.join(SUPPORTED_MODELS)} (set CLEF_MODEL)"
            )
        if not self.base_url.startswith(("http://", "https://")):
            problems.append(
                f"base_url must start with http:// or https://, got {self.base_url!r} "
                f"(set CLEF_BASE_URL)"
            )
        if self.timeout <= 0:
            problems.append(f"timeout must be positive, got {self.timeout} (set CLEF_TIMEOUT)")
        if self.max_retries < 0:
            problems.append(
                f"max_retries must be >= 0, got {self.max_retries} (set CLEF_MAX_RETRIES)"
            )
        if self.log_level not in _LOG_LEVELS:
            problems.append(
                f"log_level must be one of {', '.join(_LOG_LEVELS)}, got {self.log_level!r} "
                f"(set CLEF_LOG_LEVEL)"
            )
        return problems

    def validate(self) -> None:
        """Re-validate an existing instance.

        Raises:
            ConfigurationError: Listing all problems found on the instance.
        """
        problems = self._collect_problems()
        if problems:
            raise ConfigurationError(_format_problems(problems))

    @property
    def model_id(self) -> str:
        """Full Workers AI model identifier, e.g. ``@cf/cloudflare/clef``."""
        return f"@cf/cloudflare/{self.model}"

    @property
    def endpoint(self) -> str:
        """Full REST endpoint that runs this model."""
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model_id}"

    def configure_logging(self) -> logging.Logger:
        """Apply :attr:`log_level` to the ``clef_compactor`` logger and return it."""
        clef_logger = logging.getLogger("clef_compactor")
        clef_logger.setLevel(self.log_level)
        return clef_logger


def _parse_float(
    raw: str | None,
    env_var: str,
    problems: list[str],
    default: float,
) -> float:
    """Parse *raw* as a positive float, appending problems instead of raising."""
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        problems.append(f"{env_var} must be a number, got {raw!r}")
        return default
    if value <= 0:
        problems.append(f"{env_var} must be positive, got {value}")
        return default
    return value


def _parse_int(
    raw: str | None,
    env_var: str,
    problems: list[str],
    default: int,
) -> int:
    """Parse *raw* as a non-negative int, appending problems instead of raising."""
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        problems.append(f"{env_var} must be an integer, got {raw!r}")
        return default
    if value < 0:
        problems.append(f"{env_var} must be >= 0, got {value}")
        return default
    return value


def _format_problems(problems: list[str]) -> str:
    """Render every problem as one bulleted ConfigurationError message."""
    bullets = "\n".join(f"  - {problem}" for problem in problems)
    return f"Invalid clef-compactor configuration:\n{bullets}"
