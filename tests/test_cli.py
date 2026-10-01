"""Tests for the command-line interface (compactor is faked, no API calls)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

import clef_compactor.cli as cli_module
from clef_compactor import CompactResult, ScoredChunk
from clef_compactor.cli import main
from clef_compactor.exceptions import ClefAPIError
from clef_compactor.models import DropReason, TokenUsage


class FakeCompactor:
    """Stands in for ClefCompactor; records calls, returns a fixed result."""

    instances: list[FakeCompactor] = []
    next_result: CompactResult | None = None
    next_error: Exception | None = None

    def __init__(
        self,
        result: CompactResult | None = None,
        error: Exception | None = None,
        **kwargs: object,
    ) -> None:
        self.result = result if result is not None else type(self).next_result
        self.error = error if error is not None else type(self).next_error
        type(self).next_result = None
        type(self).next_error = None
        self.kwargs = kwargs
        self.calls: list[tuple[str, list[str], int]] = []
        FakeCompactor.instances.append(self)

    def compact(
        self, query: str, chunks: list[str], token_budget: int
    ) -> CompactResult:
        self.calls.append((query, chunks, token_budget))
        if self.error is not None:
            raise self.error
        assert self.result is not None
        if self.result.query != query:
            self.result = replace(self.result, query=query)
        return self.result

    def close(self) -> None:
        pass

    def __enter__(self) -> FakeCompactor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def sample_result() -> CompactResult:
    kept = (
        ScoredChunk(index=0, text="refund doc", score=0.93, tokens=4),
        ScoredChunk(index=2, text="terms doc", score=0.80, tokens=4),
    )
    dropped = (
        ScoredChunk(
            index=1,
            text="paris doc",
            score=0.05,
            tokens=4,
            drop_reason=DropReason.IRRELEVANT,
        ),
    )
    return CompactResult(
        kept=kept,
        dropped=dropped,
        scores=kept + dropped,
        usage=TokenUsage(input_tokens=412, output_tokens=96),
        cost_estimate=9.888e-05,
        token_budget=100,
        query="refund policy?",
        model="clef",
        latency_ms=21.0,
    )


@pytest.fixture
def fake_compactor(monkeypatch: pytest.MonkeyPatch) -> type[FakeCompactor]:
    monkeypatch.setattr(cli_module, "ClefCompactor", FakeCompactor)
    FakeCompactor.instances = []
    FakeCompactor.next_result = sample_result()
    FakeCompactor.next_error = None
    return FakeCompactor


def test_human_output(fake_compactor: type[FakeCompactor], capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["-q", "refund policy?", "-d", "doc one", "doc two", "--budget", "100"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "kept:          2 chunks" in out
    assert "dropped:       1 chunks" in out
    assert "saved:" in out
    assert "[irrelevant] paris doc" in out
    engine = fake_compactor.instances[0]
    assert engine.calls == [("refund policy?", ["doc one", "doc two"], 100)]


def test_json_output(fake_compactor: type[FakeCompactor], capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["-q", "q", "-d", "doc", "--json"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"] == "q"
    assert len(payload["kept"]) == 2
    assert payload["usage"]["input_tokens"] == 412
    assert payload["cost_estimate"] == pytest.approx(9.888e-05)


def test_at_file_documents(
    fake_compactor: type[FakeCompactor], capsys: pytest.CaptureFixture[str], tmp_path: object
) -> None:
    doc_file = tmp_path / "chunk.txt"  # type: ignore[attr-defined]
    doc_file.write_text("file content", encoding="utf-8")
    exit_code = main(["-q", "q", "-d", f"@{doc_file}"])
    assert exit_code == 0
    assert fake_compactor.instances[0].calls[0][1] == ["file content"]


def test_model_and_retry_flags_forwarded(fake_compactor: type[FakeCompactor]) -> None:
    exit_code = main(["-q", "q", "-d", "doc", "--model", "clef-flash", "--retries", "5", "--timeout", "9"])
    assert exit_code == 0
    assert fake_compactor.instances[0].kwargs == {"model": "clef-flash", "timeout": 9.0, "max_retries": 5}


def test_api_error_exits_nonzero(fake_compactor: type[FakeCompactor], capsys: pytest.CaptureFixture[str]) -> None:
    error = ClefAPIError("Cloudflare API rejected the request.", status_code=400)
    fake_compactor.next_error = error
    exit_code = main(["-q", "q", "-d", "doc"])
    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


def test_configuration_error_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Real ClefCompactor without credentials -> ConfigurationError -> exit 1.
    exit_code = main(["-q", "q", "-d", "doc"])
    assert exit_code == 1
    assert "CLEF_ACCOUNT_ID" in capsys.readouterr().err


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "clef-compact" in capsys.readouterr().out
