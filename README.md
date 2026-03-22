# MMLU Entity Correlation Pipeline

This repository provides a Python CLI pipeline for:

1. preparing MMLU questions and aligning them with per-question `lm-eval` results,
2. extracting entities with GPT-5.4 and Llama-3.1-8B-Instruct,
3. scoring per-question entity recall with one-to-one matching,
4. analyzing Pearson correlation between recall and per-question accuracy.

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[vllm]
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
- `Llama-3.1-8B-Instruct` defaults to in-process local `vLLM` inference via `vllm.LLM(...)`.
- Long `extract-entities` runs show a `tqdm` progress bar with cached-hit, API-call, and error counts.
- `extract-entities` supports parallel API calls for remote providers; for local `vLLM`, throughput is controlled by vLLM batching instead of `--workers`.

## Local vLLM for `pred`

The `pred` path loads `meta-llama/Llama-3.1-8B-Instruct` directly inside the extraction process. Set your Hugging Face token so vLLM can download the gated model:

```bash
export HF_TOKEN=...
```

Then run:

```bash
uv run --env-file .env mmlu-entity-corr extract-entities --target pred
```

Useful `pred` model knobs live in `configs/experiment.yaml`:

- `dtype`
- `tensor_parallel_size`
- `gpu_memory_utilization`
- `batch_size`

`HF_TOKEN` is used by the in-process vLLM runtime to pull the model from Hugging Face.
