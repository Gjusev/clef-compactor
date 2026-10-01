"""Tests for configuration: env fallbacks, precedence, all-problems errors."""

from __future__ import annotations

import pytest

from clef_compactor.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    Settings,
)
from clef_compactor.exceptions import ConfigurationError


def test_defaults_with_explicit_credentials() -> None:
    settings = Settings.from_env(account_id="acct", api_token="tok")
    assert settings.account_id == "acct"
    assert settings.api_token == "tok"
    assert settings.model == DEFAULT_MODEL == "clef"
    assert settings.base_url == DEFAULT_BASE_URL
    assert settings.timeout == DEFAULT_TIMEOUT_SECONDS == 60.0
    assert settings.max_retries == DEFAULT_MAX_RETRIES == 2
    assert settings.log_level == "WARNING"


def test_endpoint_and_model_id() -> None:
    settings = Settings.from_env(account_id="acct", api_token="tok", model="clef-flash")
    assert settings.model_id == "@cf/cloudflare/clef-flash"
    assert settings.endpoint == (
        "https://api.cloudflare.com/client/v4/accounts/acct/ai/run/@cf/cloudflare/clef-flash"
    )


def test_cloudflare_env_fallbacks() -> None:
    settings = Settings.from_env(
        environ={"CLOUDFLARE_ACCOUNT_ID": "cf-acct", "CLOUDFLARE_API_TOKEN": "cf-tok"}
    )
    assert settings.account_id == "cf-acct"
    assert settings.api_token == "cf-tok"


def test_clef_env_wins_over_cloudflare_env() -> None:
    settings = Settings.from_env(
        environ={
            "CLEF_ACCOUNT_ID": "clef-acct",
            "CLEF_API_TOKEN": "clef-tok",
            "CLOUDFLARE_ACCOUNT_ID": "cf-acct",
            "CLOUDFLARE_API_TOKEN": "cf-tok",
        }
    )
    assert settings.account_id == "clef-acct"
    assert settings.api_token == "clef-tok"


def test_explicit_kwargs_beat_env() -> None:
    settings = Settings.from_env(
        environ={"CLEF_ACCOUNT_ID": "env-acct", "CLEF_API_TOKEN": "env-tok"},
        account_id="kwarg-acct",
        api_token="kwarg-tok",
        model="clef-flash",
        timeout=12.5,
        max_retries=5,
        log_level="debug",
    )
    assert settings.account_id == "kwarg-acct"
    assert settings.api_token == "kwarg-tok"
    assert settings.model == "clef-flash"
    assert settings.timeout == 12.5
    assert settings.max_retries == 5
    assert settings.log_level == "DEBUG"


def test_env_var_parsing() -> None:
    settings = Settings.from_env(
        environ={
            "CLEF_ACCOUNT_ID": "acct",
            "CLEF_API_TOKEN": "tok",
            "CLEF_MODEL": "clef-flash",
            "CLEF_TIMEOUT": "12.5",
            "CLEF_MAX_RETRIES": "4",
            "CLEF_LOG_LEVEL": "info",
            "CLEF_BASE_URL": "https://gateway.example.com/client/v4/",
        }
    )
    assert settings.model == "clef-flash"
    assert settings.timeout == 12.5
    assert settings.max_retries == 4
    assert settings.log_level == "INFO"
    assert settings.base_url == "https://gateway.example.com/client/v4"


def test_missing_credentials_lists_all_problems_at_once() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings.from_env(environ={})
    message = str(excinfo.value)
    assert "CLEF_ACCOUNT_ID" in message
    assert "CLOUDFLARE_ACCOUNT_ID" in message
    assert "CLEF_API_TOKEN" in message
    assert "CLOUDFLARE_API_TOKEN" in message


def test_all_problems_reported_in_one_error() -> None:
    with pytest.raises(ConfigurationError) as excinfo:
        Settings.from_env(
            environ={
                "CLEF_MODEL": "gpt-4",
                "CLEF_TIMEOUT": "not-a-number",
                "CLEF_MAX_RETRIES": "-3",
                "CLEF_LOG_LEVEL": "chatty",
                "CLEF_BASE_URL": "ftp://bad",
            }
        )
    message = str(excinfo.value)
    assert "CLEF_ACCOUNT_ID" in message
    assert "CLEF_API_TOKEN" in message
    assert "gpt-4" in message
    assert "CLEF_TIMEOUT" in message
    assert "not-a-number" in message
    assert "CLEF_MAX_RETRIES" in message
    assert "CLEF_LOG_LEVEL" in message
    assert "CHATTY" in message
    assert "CLEF_BASE_URL" in message


def test_validate_collects_multiple_problems() -> None:
    settings = Settings(account_id="a", api_token="t", model="nope", timeout=-1.0)
    with pytest.raises(ConfigurationError) as excinfo:
        settings.validate()
    message = str(excinfo.value)
    assert "nope" in message
    assert "timeout must be positive" in message


def test_validate_passes_on_valid_settings() -> None:
    Settings.from_env(account_id="acct", api_token="tok").validate()


def test_empty_env_values_are_missing() -> None:
    with pytest.raises(ConfigurationError):
        Settings.from_env(environ={"CLEF_ACCOUNT_ID": "   ", "CLEF_API_TOKEN": ""})


def test_configure_logging_sets_level(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    settings = Settings.from_env(account_id="acct", api_token="tok", log_level="INFO")
    clef_logger = settings.configure_logging()
    assert clef_logger.level == logging.INFO
