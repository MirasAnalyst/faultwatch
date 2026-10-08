"""Export the copilot to Azure AI Foundry artifacts.

    python deploy/azure_foundry/export.py            # writes *.prompty and eval_data.jsonl

  * copilot_answer.prompty / triage.prompty - the exact prompts in
    faultwatch/genai/prompts.py, in Foundry's Prompty format (so the prompt
    can be versioned, evaluated and deployed from the Foundry portal)
  * eval_data.jsonl - query / context / response / ground_truth rows from the
    gold question set, for Foundry's built-in evaluators (run_foundry_eval.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from faultwatch.genai import prompts  # noqa: E402
from faultwatch.safety import EFFECTS  # noqa: E402

HEADER = """---
name: {name}
description: {description}
version: {version}
model:
  api: chat
  configuration:
    type: azure_openai
    azure_endpoint: ${{env:AZURE_OPENAI_ENDPOINT}}
    azure_deployment: ${{env:AZURE_OPENAI_DEPLOYMENT}}
    api_version: "2024-10-21"
  parameters:
    temperature: 0
    max_tokens: {max_tokens}{json_mode}
inputs:
{inputs}
---
system:
{system}

user:
{user}
"""


def prompty(name, description, system, user, inputs, max_tokens, json_mode=False):
    jm = "\n    response_format: {type: json_object}" if json_mode else ""
    ins = "\n".join(f"  {k}:\n    type: string" for k in inputs)
    user = user
    for k in inputs:                      # python {x} -> prompty {{x}}
        user = user.replace("{" + k + "}", "{{" + k + "}}")
    return HEADER.format(name=name, description=description, version=prompts.PROMPT_VERSION,
                         max_tokens=max_tokens, json_mode=jm, inputs=ins, system=system, user=user)


def main():
    (HERE / "copilot_answer.prompty").write_text(prompty(
        "faultwatch-copilot-answer", "Cited lessons from machinery-casualty reports for a FaultWatch alert",
        prompts.ANSWER_SYSTEM, prompts.ANSWER_USER, ["question", "evidence"], 1200))
    triage_system = prompts.TRIAGE_SYSTEM.format(event_types=prompts.EVENT_TYPES, systems=prompts.SYSTEMS,
                                                 effects=list(EFFECTS))
    (HERE / "triage.prompty").write_text(prompty(
        "faultwatch-triage", "Structured classification of a machinery failure narrative",
        triage_system, prompts.TRIAGE_USER, ["narrative"], 400, json_mode=True))

    from faultwatch.genai.copilot import Copilot
    from faultwatch.genai.index import Retriever
    from faultwatch.genai.llm import get_provider
    cp = Copilot(Retriever(ROOT / "corpus", dense="lsa"), get_provider())
    rows = []
    for line in (ROOT / "evals" / "gold_questions.jsonl").read_text().splitlines():
        g = json.loads(line)
        a = cp.answer(g["question"])
        rows.append({"id": g["id"], "query": g["question"], "response": a["answer"],
                     "context": a.get("evidence", ""), "ground_truth_docs": g["gold"],
                     "provider": a["provider"]})
    with open(HERE / "eval_data.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote 2 prompty files and {len(rows)} eval rows (provider: {rows[0]['provider']})")


if __name__ == "__main__":
    main()
