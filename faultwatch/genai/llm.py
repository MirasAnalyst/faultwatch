"""Pluggable LLM providers for the copilot.

Every provider returns an `LLMResponse` carrying the text plus tokens,
latency and an estimated cost, so the monitor can track spend per call.

    provider = get_provider()          # first configured: Azure -> OpenAI -> Anthropic -> offline

  * AzureOpenAIProvider (Azure AI Foundry / Azure OpenAI deployment)
        AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT,
        AZURE_OPENAI_API_VERSION (default 2024-10-21)
  * OpenAIProvider                      OPENAI_API_KEY, OPENAI_MODEL
  * AnthropicProvider                   ANTHROPIC_API_KEY (or an `ant auth login` profile), ANTHROPIC_MODEL
  * ExtractiveProvider (offline)        no key: answers by quoting retrieved evidence

The offline provider keeps tests, CI and the evals deterministic and free;
it never invents text, it only selects and quotes evidence sentences.

Prices are per 1M tokens and only used for cost estimates; set
FAULTWATCH_PRICE_IN / FAULTWATCH_PRICE_OUT to your contracted rates.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

PRICES = {  # USD per 1M input / output tokens (list prices; override with env vars)
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    meta: dict = field(default_factory=dict)

    def json(self) -> dict:
        """Parse a JSON answer (tolerates a fenced block)."""
        m = re.search(r"\{.*\}", self.text, re.S)
        return json.loads(m.group(0)) if m else {}


def _cost(model: str, tin: int, tout: int) -> float:
    pin, pout = PRICES.get(model, (float(os.environ.get("FAULTWATCH_PRICE_IN", 0)),
                                   float(os.environ.get("FAULTWATCH_PRICE_OUT", 0))))
    if os.environ.get("FAULTWATCH_PRICE_IN"):
        pin, pout = float(os.environ["FAULTWATCH_PRICE_IN"]), float(os.environ.get("FAULTWATCH_PRICE_OUT", 0))
    return (tin * pin + tout * pout) / 1e6


class AzureOpenAIProvider:
    name = "azure_openai"

    def __init__(self):
        from openai import AzureOpenAI
        self.model = os.environ["AZURE_OPENAI_DEPLOYMENT"]
        self.client = AzureOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                                  api_key=os.environ["AZURE_OPENAI_API_KEY"],
                                  api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))

    def complete(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 1200) -> LLMResponse:
        t0 = time.perf_counter()
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=self.model, max_tokens=max_tokens, temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], **kw)
        u = r.usage
        return LLMResponse(r.choices[0].message.content or "", self.name, self.model, u.prompt_tokens,
                           u.completion_tokens, time.perf_counter() - t0,
                           _cost(self.model, u.prompt_tokens, u.completion_tokens))


class OpenAIProvider(AzureOpenAIProvider):
    name = "openai"

    def __init__(self):
        from openai import OpenAI
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4.1")
        self.client = OpenAI()


class AnthropicProvider:
    name = "anthropic"

    def __init__(self):
        import anthropic
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5")
        self.client = anthropic.Anthropic()

    def complete(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 4000) -> LLMResponse:
        t0 = time.perf_counter()
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        # server-side refusal fallback: a declined request is re-run on a fallback model in the same call
        r = self.client.beta.messages.create(
            model=self.model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        text = "".join(b.text for b in r.content if b.type == "text")
        u = r.usage
        return LLMResponse(text, self.name, r.model, u.input_tokens, u.output_tokens,
                           time.perf_counter() - t0, _cost(self.model, u.input_tokens, u.output_tokens),
                           meta={"stop_reason": r.stop_reason})


class ExtractiveProvider:
    """Offline stand-in: builds the answer from the evidence passages in the
    prompt (sentences that share the most terms with the question), each with
    its citation tag. Deterministic, free, and incapable of hallucinating text
    that is not in the evidence."""
    name = "extractive"
    model = "extractive-v1"

    def complete(self, system: str, user: str, json_mode: bool = False, max_tokens: int = 1200) -> LLMResponse:
        t0 = time.perf_counter()
        q = user.split("EVIDENCE", 1)[0]
        qterms = set(re.findall(r"[a-z]{4,}", q.lower()))
        picks = []
        for tag, body in re.findall(r"\[(\w[\w#.\-]*)\]\s*(.*?)(?=\n\[\w[\w#.\-]*\]|\Z)", user, re.S):
            body = re.sub(r"^\([^)]*\)\s*", "", body.strip())          # drop the "(title - section)" header
            for s in re.split(r"(?<=[.!?])\s+", body):
                score = len(qterms & set(re.findall(r"[a-z]{4,}", s.lower())))
                if 60 < len(s) < 400 and score:
                    picks.append((score, tag, s.strip()))
        picks.sort(key=lambda x: -x[0])
        seen, lines = set(), []
        for _, tag, s in picks:
            if tag in seen:
                continue
            seen.add(tag)
            lines.append(f"- {s} [{tag}]")
            if len(lines) == 4:
                break
        if json_mode:
            text = json.dumps({"answer": "\n".join(lines), "insufficient_evidence": not lines})
        else:
            text = ("Similar past failures (quoted from the reports):\n" + "\n".join(lines)) if lines else \
                "INSUFFICIENT_EVIDENCE: no retrieved report addresses this."
        return LLMResponse(text, self.name, self.model, len(user) // 4, len(text) // 4, time.perf_counter() - t0, 0.0)


def load_dotenv(path: str | os.PathLike | None = None) -> None:
    """Read KEY=value lines from the repo's .env (gitignored) into the
    environment, without overriding variables that are already set."""
    from pathlib import Path
    p = Path(path) if path else Path(os.environ.get("FAULTWATCH_ROOT", Path(__file__).resolve().parents[2])) / ".env"
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def get_provider(name: str | None = None):
    """Explicit name, FAULTWATCH_LLM, or the first provider with credentials."""
    load_dotenv()
    name = name or os.environ.get("FAULTWATCH_LLM")
    table = {"azure_openai": AzureOpenAIProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider,
             "extractive": ExtractiveProvider}
    if name:
        return table[name]()
    if os.environ.get("AZURE_OPENAI_API_KEY") and os.environ.get("AZURE_OPENAI_ENDPOINT"):
        return AzureOpenAIProvider()
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIProvider()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicProvider()
    return ExtractiveProvider()
