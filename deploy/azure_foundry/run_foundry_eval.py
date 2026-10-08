"""Grade the copilot's answers with Azure AI Foundry's built-in evaluators.

    pip install azure-ai-evaluation
    export AZURE_OPENAI_ENDPOINT=... AZURE_OPENAI_API_KEY=... AZURE_OPENAI_DEPLOYMENT=...   # judge model
    export AZURE_AI_PROJECT=https://<resource>.services.ai.azure.com/api/projects/<project>   # optional: log to portal
    FAULTWATCH_LLM=azure_openai python deploy/azure_foundry/export.py     # answers from the real deployment
    python deploy/azure_foundry/run_foundry_eval.py

Groundedness (is every claim supported by the retrieved evidence?) and
relevance (does it answer the engineer's question?) are scored by the judge
model; results land in eval_results.json and, with AZURE_AI_PROJECT set, in
the Foundry project's Evaluation tab next to the prompt versions.

Not run from this repo (no Azure subscription); the offline equivalents are
in faultwatch/genai/evals.py.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    from azure.ai.evaluation import GroundednessEvaluator, RelevanceEvaluator, evaluate

    model_config = {"azure_endpoint": os.environ["AZURE_OPENAI_ENDPOINT"],
                    "api_key": os.environ["AZURE_OPENAI_API_KEY"],
                    "azure_deployment": os.environ["AZURE_OPENAI_DEPLOYMENT"]}
    # off-topic rows have no evidence to ground against; they are scored by refusal rate offline
    rows = [json.loads(l) for l in (HERE / "eval_data.jsonl").read_text().splitlines()]
    data = HERE / "eval_data_answerable.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows if r["ground_truth_docs"] and r["context"]))
    result = evaluate(
        data=str(data),
        evaluators={"groundedness": GroundednessEvaluator(model_config),
                    "relevance": RelevanceEvaluator(model_config)},
        evaluator_config={"default": {"column_mapping": {"query": "${data.query}",
                                                         "context": "${data.context}",
                                                         "response": "${data.response}"}}},
        azure_ai_project=os.environ.get("AZURE_AI_PROJECT"),
        evaluation_name="faultwatch-copilot",
        output_path=str(HERE / "eval_results.json"),
    )
    print(json.dumps(result.get("metrics", {}), indent=2))


if __name__ == "__main__":
    main()
