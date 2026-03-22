# MMLU Entity Correlation Pipeline

This repository provides a Python CLI pipeline for:

1. preparing MMLU questions and aligning them with per-question `lm-eval` results,
2. extracting entities with GPT-5.4 and Llama-3.1-8B-Instruct,
3. scoring per-question entity recall with one-to-one matching,
4. analyzing Pearson correlation between recall and per-question accuracy.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Run the stages in order:

```bash
mmlu-entity-corr prepare-dataset --lm-eval path/to/lm_eval_samples.jsonl
mmlu-entity-corr extract-entities --target both
mmlu-entity-corr score-recall
mmlu-entity-corr analyze-correlation
```

Default configuration lives in `configs/experiment.yaml`.

## Model backends

- `GPT-5.4` uses the OpenAI `responses.create(...)` API through the configured proxy base URL.
- `Llama-3.1-8B-Instruct` uses Hugging Face chat completions with `HF_TOKEN`.
- Long `extract-entities` runs show a `tqdm` progress bar with cached-hit, API-call, and error counts.
- `extract-entities` supports parallel API calls; default worker count is configured in `configs/experiment.yaml`, and you can override it with `--workers`.
