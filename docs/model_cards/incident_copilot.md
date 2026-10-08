# Model card: machinery incident copilot (`faultwatch/genai`)

**Intended use:** given a FaultWatch alert or an engineer's question, retrieve similar
machinery casualties and summarise likely causes, checks and failed barriers, with a
citation on every sentence; triage free-text failure reports into structured fields.
Decision support only: it does not give operating instructions and never overrides a
safety system.

**System:** 146 NTSB / MAIB / NSIA machinery-casualty reports (4,731 chunks);
hybrid BM25 + all-MiniLM-L6-v2 retrieval with reciprocal-rank fusion; LLM via Azure
OpenAI (default when configured), OpenAI or Anthropic; offline extractive provider.
Prompt version in `genai/prompts.py`, exported as Prompty to `deploy/azure_foundry/`.

**Guardrails:** citations validated against retrieved passages; scope gate refuses
non-machinery questions; weak evidence returns INSUFFICIENT_EVIDENCE; e-mail / phone
scrubbing; answers framed for an engineer's judgement.

**Evaluation (offline provider):** hit@3 96% (technical wording) / 83% (colloquial);
citation validity 100%; faithfulness 99.6% (token overlap); off-topic refusals 6/6,
false refusals 0/57; triage macro-F1 0.37 (k-NN baseline, 103 NTSB reports).

**Limits:** gold questions written by the author (wording bias - hence the colloquial
set); the LLM path and the Foundry groundedness evaluation have not been run without a
key; the corpus is US / UK-centric and skewed to small commercial vessels and fires.

**Monitoring:** tokens, latency and cost per call (`/monitoring`); re-run
`python -m faultwatch.genai.evals` on every prompt or corpus change (CI enforces a
minimum hit@5).
