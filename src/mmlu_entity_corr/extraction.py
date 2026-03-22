"""Entity extraction with OpenAI-compatible endpoints and JSON caching."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import hashlib
import os
from pathlib import Path
from typing import Any

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - exercised in environments without tqdm
    class _TqdmFallback:  # pragma: no cover - exercised in environments without tqdm
        def __init__(self, iterable=None, **_kwargs: Any):
            self.iterable = iterable

        def update(self, _n: int = 1) -> None:
            return None

        def set_postfix(self, **_kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):  # type: ignore[no-redef]
        return _TqdmFallback(iterable=iterable, **kwargs)

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
except ImportError:  # pragma: no cover - exercised in environments without tenacity
    def retry(*_args: Any, **_kwargs: Any):
        def decorator(function):
            return function

        return decorator

    def retry_if_exception_type(_exceptions: Any) -> Any:
        return _exceptions

    def stop_after_attempt(_attempts: int) -> int:
        return _attempts

    def wait_fixed(_seconds: int) -> int:
        return _seconds

from .config import get_path
from .io_utils import read_json, read_jsonl, write_json, write_jsonl
from .json_utils import loads


INSTRUCTION_BLOCK = """Please extract entity or concept spans from the question.
Return only a JSON list of strings.

Rules:
- Copy spans verbatim from the question text.
- Do not answer or complete the question.
- Do not paraphrase or infer missing text.
- If there are no valid spans, return [].

Q: Which education institution has a sports team named George Washington Colonials men's basketball?
A: ["George Washington Colonials men's basketball"]

Q: What year did Baltimore Ravens win the superbowl?
A: ["Baltimore Ravens", "superbowl"]

Q: What countries share borders with Spain?
A: ["Spain"]
"""

SUCCESS_STATUSES = {"ok", "empty"}


class ExtractionError(RuntimeError):
    """Raised when an extraction request cannot be parsed after retries."""


class ResponseParseError(ExtractionError):
    """Raised when the model returns text that cannot be parsed as the required JSON array."""

    def __init__(self, message: str, *, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


def build_cache_key(question_uid: str, model_name: str, prompt_version: str, provider: str) -> str:
    payload = "\n".join((question_uid, model_name, prompt_version, provider))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _extract_first_json_array(text: str) -> str | None:
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_entity_response(raw_response: str) -> list[str]:
    candidate = raw_response.strip()
    try:
        payload = loads(candidate)
    except Exception:
        bracketed = _extract_first_json_array(candidate)
        if bracketed is None:
            raise ValueError("No JSON array found in model response.")
        payload = loads(bracketed)
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array, got {type(payload)!r}")
    entities: list[str] = []
    for item in payload:
        if not isinstance(item, str):
            raise ValueError("Entity array must contain only strings.")
        entities.append(item)
    return entities


def _slugify_model_name(model_name: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model_name.lower()).strip("_")


def _should_reuse_cached_payload(payload: dict[str, Any]) -> bool:
    return payload.get("parse_status") in SUCCESS_STATUSES


def _build_success_payload(
    *,
    question_uid: str,
    model_name: str,
    raw_response: str,
    entities: list[str],
    prompt_version: str,
) -> dict[str, Any]:
    return {
        "question_uid": question_uid,
        "model_name": model_name,
        "raw_response": raw_response,
        "entities_raw": entities,
        "parse_status": "empty" if not entities else "ok",
        "prompt_version": prompt_version,
    }


def _build_error_payload(
    *,
    question_uid: str,
    model_name: str,
    prompt_version: str,
    error: str,
    raw_response: str = "",
) -> dict[str, Any]:
    return {
        "question_uid": question_uid,
        "model_name": model_name,
        "raw_response": raw_response,
        "entities_raw": [],
        "parse_status": "parse_error",
        "prompt_version": prompt_version,
        "error": error,
    }


def build_prompt_text(question: str) -> str:
    return f"{INSTRUCTION_BLOCK}\n\nQ: {question}\nA:"


def build_chat_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INSTRUCTION_BLOCK},
        {"role": "user", "content": f"Q: {question}\nA:"},
    ]


def _get_openai_client(model_config: dict[str, Any]):
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing at runtime
        raise RuntimeError("openai is required for entity extraction.") from exc
    base_url = os.environ.get(model_config.get("base_url_env", ""), model_config.get("base_url"))
    api_key = os.environ.get(model_config.get("api_key_env", ""), "") or model_config.get("api_key", "")
    if not api_key and not base_url:
        raise RuntimeError(
            f"Missing API key for model {model_config['model']}. Set {model_config.get('api_key_env')}."
        )
    if not api_key:
        api_key = "EMPTY"
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    if model_config.get("default_headers"):
        client_kwargs["default_headers"] = model_config["default_headers"]
    return OpenAI(**client_kwargs)


def _get_hf_client(model_config: dict[str, Any]):
    try:
        from huggingface_hub import InferenceClient  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing at runtime
        raise RuntimeError("huggingface_hub is required for Hugging Face entity extraction.") from exc
    api_key = os.environ.get(model_config.get("api_key_env", ""), "")
    if not api_key:
        raise RuntimeError(
            f"Missing Hugging Face token for model {model_config['model']}. Set {model_config.get('api_key_env')}."
        )
    return InferenceClient(api_key=api_key)


def _get_vllm_components():
    try:
        from vllm import LLM, SamplingParams  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing at runtime
        raise RuntimeError(
            "vllm is required for local pred extraction. Install it in the active environment before running `--target pred`."
        ) from exc
    return LLM, SamplingParams


def _request_entities_openai_responses(
    question: str,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
) -> tuple[str, list[str]]:
    client = _get_openai_client(model_config)
    response = client.responses.create(
        model=model_config["model"],
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": build_prompt_text(question)}],
            }
        ],
        max_output_tokens=extraction_config["max_tokens"],
        store=False,
    )
    raw_response = response.output_text or ""
    if not raw_response.strip():
        raise ExtractionError("Model returned empty response.")
    try:
        entities = parse_entity_response(raw_response)
    except Exception as exc:
        raise ResponseParseError(str(exc), raw_response=raw_response) from exc
    return raw_response, entities


def _request_entities_huggingface_chat(
    question: str,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
) -> tuple[str, list[str]]:
    client = _get_hf_client(model_config)
    request_kwargs: dict[str, Any] = {
        "model": model_config["model"],
        "messages": build_chat_messages(question),
        "temperature": extraction_config["temperature"],
        "top_p": extraction_config["top_p"],
        "max_tokens": extraction_config["max_tokens"],
    }
    if model_config.get("supports_seed"):
        request_kwargs["seed"] = extraction_config.get("seed")
    response = client.chat.completions.create(**request_kwargs)
    raw_response = response.choices[0].message.content or ""
    if not raw_response.strip():
        raise ExtractionError("Model returned empty response.")
    try:
        entities = parse_entity_response(raw_response)
    except Exception as exc:
        raise ResponseParseError(str(exc), raw_response=raw_response) from exc
    return raw_response, entities


def _request_entities_openai_chat(
    question: str,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
) -> tuple[str, list[str]]:
    client = _get_openai_client(model_config)
    request_kwargs: dict[str, Any] = {
        "model": model_config["model"],
        "messages": build_chat_messages(question),
        "temperature": extraction_config["temperature"],
        "top_p": extraction_config["top_p"],
        "max_tokens": extraction_config["max_tokens"],
    }
    if model_config.get("supports_seed"):
        request_kwargs["seed"] = extraction_config.get("seed")
    response = client.chat.completions.create(**request_kwargs)
    raw_response = response.choices[0].message.content or ""
    if not raw_response.strip():
        raise ExtractionError("Model returned empty response.")
    try:
        entities = parse_entity_response(raw_response)
    except Exception as exc:
        raise ResponseParseError(str(exc), raw_response=raw_response) from exc
    return raw_response, entities


def _build_vllm_prompt(question: str, tokenizer: Any) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        try:
            return apply_chat_template(build_chat_messages(question), tokenize=False, add_generation_prompt=True)
        except TypeError:
            return apply_chat_template(build_chat_messages(question), tokenize=False)
    return build_prompt_text(question)


def _build_vllm_sampling_params(SamplingParams: Any, extraction_config: dict[str, Any]) -> Any:
    base_kwargs: dict[str, Any] = {
        "temperature": extraction_config["temperature"],
        "top_p": extraction_config["top_p"],
        "max_tokens": extraction_config["max_tokens"],
    }
    schema = {
        "type": "array",
        "items": {"type": "string"},
    }
    try:
        from vllm.sampling_params import StructuredOutputsParams  # type: ignore

        return SamplingParams(
            **base_kwargs,
            structured_outputs=StructuredOutputsParams(json=schema),
        )
    except Exception:
        pass
    try:
        from vllm.sampling_params import GuidedDecodingParams  # type: ignore

        if hasattr(GuidedDecodingParams, "from_optional"):
            guided = GuidedDecodingParams.from_optional(json=schema)
        else:
            guided = GuidedDecodingParams(json=schema)
        return SamplingParams(
            **base_kwargs,
            guided_decoding=guided,
        )
    except Exception:
        return SamplingParams(**base_kwargs)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(1),
    retry=retry_if_exception_type((ExtractionError, ValueError)),
    reraise=True,
)
def _request_entities(question: str, model_config: dict[str, Any], extraction_config: dict[str, Any]) -> tuple[str, list[str]]:
    provider = model_config.get("provider", "openai_responses")
    if provider == "openai_responses":
        return _request_entities_openai_responses(question, model_config, extraction_config)
    if provider == "openai_chat":
        return _request_entities_openai_chat(question, model_config, extraction_config)
    if provider == "huggingface_chat":
        return _request_entities_huggingface_chat(question, model_config, extraction_config)
    raise ExtractionError(f"Unsupported extraction provider: {provider}")


def _extract_payload_for_record(
    record: dict[str, Any],
    *,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
    cache_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any]
    try:
        raw_response, entities = _request_entities(record["question"], model_config, extraction_config)
        payload = _build_success_payload(
            question_uid=record["question_uid"],
            model_name=model_config["model"],
            raw_response=raw_response,
            entities=entities,
            prompt_version=extraction_config["prompt_version"],
        )
    except Exception as exc:  # noqa: BLE001
        payload = _build_error_payload(
            question_uid=record["question_uid"],
            model_name=model_config["model"],
            prompt_version=extraction_config["prompt_version"],
            error=str(exc),
            raw_response=getattr(exc, "raw_response", ""),
        )
    write_json(cache_path, payload, pretty=True)
    return payload


def _collect_pending_jobs(
    records: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
    cache_dir: Path,
    force: bool,
    progress: Any,
) -> tuple[list[dict[str, Any] | None], list[tuple[int, dict[str, Any], Path]], int]:
    outputs: list[dict[str, Any] | None] = [None] * len(records)
    model_dir = cache_dir / _slugify_model_name(model_config["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    pending_jobs: list[tuple[int, dict[str, Any], Path]] = []
    for index, record in enumerate(records):
        cache_key = build_cache_key(
            record["question_uid"],
            model_config["model"],
            extraction_config["prompt_version"],
            model_config.get("provider", "openai_responses"),
        )
        cache_path = model_dir / f"{cache_key}.json"
        if cache_path.exists() and not force:
            cached = read_json(cache_path)
            if _should_reuse_cached_payload(cached):
                outputs[index] = cached
                cache_hits += 1
                if hasattr(progress, "update"):
                    progress.update(1)
                continue
        pending_jobs.append((index, record, cache_path))
    return outputs, pending_jobs, cache_hits


def _chunk_items(items: list[tuple[int, dict[str, Any], Path]], chunk_size: int) -> list[list[tuple[int, dict[str, Any], Path]]]:
    return [items[start : start + chunk_size] for start in range(0, len(items), chunk_size)]


def _extract_entities_for_records_vllm(
    records: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
    cache_dir: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    progress = tqdm(
        total=len(records),
        desc=f"Extracting {model_config['model']}",
        unit="question",
        dynamic_ncols=True,
    )
    outputs, pending_jobs, cache_hits = _collect_pending_jobs(
        records,
        model_config=model_config,
        extraction_config=extraction_config,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )
    api_calls = 0
    parse_errors = 0
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
    if not pending_jobs:
        if hasattr(progress, "close"):
            progress.close()
        return [payload for payload in outputs if payload is not None]

    LLM, SamplingParams = _get_vllm_components()
    llm_kwargs: dict[str, Any] = {"model": model_config["model"]}
    for key in (
        "tokenizer",
        "dtype",
        "tensor_parallel_size",
        "gpu_memory_utilization",
        "max_model_len",
        "trust_remote_code",
        "download_dir",
        "enforce_eager",
    ):
        if key in model_config:
            llm_kwargs[key] = model_config[key]
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = _build_vllm_sampling_params(SamplingParams, extraction_config)
    batch_size = int(model_config.get("batch_size", 128))
    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}.")

    try:
        for batch in _chunk_items(pending_jobs, batch_size):
            prompts = [_build_vllm_prompt(record["question"], tokenizer) for _, record, _ in batch]
            try:
                responses = llm.generate(prompts, sampling_params, use_tqdm=False)
                if len(responses) != len(batch):
                    raise ExtractionError(
                        f"vLLM returned {len(responses)} outputs for a batch of size {len(batch)}."
                    )
                batch_payloads: list[dict[str, Any]] = []
                for response, (_, record, _cache_path) in zip(responses, batch, strict=True):
                    raw_response = ""
                    if response.outputs:
                        raw_response = response.outputs[0].text or ""
                    if not raw_response.strip():
                        raise ExtractionError("Model returned empty response.")
                    try:
                        entities = parse_entity_response(raw_response)
                    except Exception as parse_exc:
                        raise ResponseParseError(str(parse_exc), raw_response=raw_response) from parse_exc
                    batch_payloads.append(
                        _build_success_payload(
                            question_uid=record["question_uid"],
                            model_name=model_config["model"],
                            raw_response=raw_response,
                            entities=entities,
                            prompt_version=extraction_config["prompt_version"],
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                batch_payloads = []
                for _, record, _cache_path in batch:
                    try:
                        prompt = _build_vllm_prompt(record["question"], tokenizer)
                        single_response = llm.generate([prompt], sampling_params, use_tqdm=False)[0]
                        raw_response = ""
                        if single_response.outputs:
                            raw_response = single_response.outputs[0].text or ""
                        if not raw_response.strip():
                            raise ExtractionError("Model returned empty response.")
                        try:
                            entities = parse_entity_response(raw_response)
                        except Exception as parse_exc:
                            raise ResponseParseError(str(parse_exc), raw_response=raw_response) from parse_exc
                        batch_payloads.append(
                            _build_success_payload(
                                question_uid=record["question_uid"],
                                model_name=model_config["model"],
                                raw_response=raw_response,
                                entities=entities,
                                prompt_version=extraction_config["prompt_version"],
                            )
                        )
                    except Exception as single_exc:  # noqa: BLE001
                        batch_payloads.append(
                            _build_error_payload(
                                question_uid=record["question_uid"],
                                model_name=model_config["model"],
                                prompt_version=extraction_config["prompt_version"],
                                error=str(single_exc if str(single_exc) else exc),
                                raw_response=getattr(single_exc, "raw_response", ""),
                            )
                        )

            for payload, (index, _record, cache_path) in zip(batch_payloads, batch, strict=True):
                outputs[index] = payload
                write_json(cache_path, payload, pretty=True)
                api_calls += 1
                if payload["parse_status"] == "parse_error":
                    parse_errors += 1
                if hasattr(progress, "update"):
                    progress.update(1)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
    finally:
        if hasattr(progress, "close"):
            progress.close()
    return [payload for payload in outputs if payload is not None]


def extract_entities_for_records(
    records: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
    cache_dir: Path,
    workers: int = 1,
    force: bool = False,
) -> list[dict[str, Any]]:
    if model_config.get("provider") == "vllm_local":
        return _extract_entities_for_records_vllm(
            records,
            model_config=model_config,
            extraction_config=extraction_config,
            cache_dir=cache_dir,
            force=force,
        )
    if workers <= 0:
        raise ValueError(f"`workers` must be positive, got {workers}.")
    outputs: list[dict[str, Any] | None]
    api_calls = 0
    parse_errors = 0
    progress = tqdm(
        total=len(records),
        desc=f"Extracting {model_config['model']}",
        unit="question",
        dynamic_ncols=True,
    )
    outputs, pending_jobs, cache_hits = _collect_pending_jobs(
        records,
        model_config=model_config,
        extraction_config=extraction_config,
        cache_dir=cache_dir,
        force=force,
        progress=progress,
    )
    if hasattr(progress, "set_postfix"):
        progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
    if workers == 1:
        for index, record, cache_path in pending_jobs:
            payload = _extract_payload_for_record(
                record,
                model_config=model_config,
                extraction_config=extraction_config,
                cache_path=cache_path,
            )
            outputs[index] = payload
            api_calls += 1
            if payload["parse_status"] == "parse_error":
                parse_errors += 1
            if hasattr(progress, "update"):
                progress.update(1)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index: dict[Future[dict[str, Any]], int] = {}
            for index, record, cache_path in pending_jobs:
                future = executor.submit(
                    _extract_payload_for_record,
                    record,
                    model_config=model_config,
                    extraction_config=extraction_config,
                    cache_path=cache_path,
                )
                future_to_index[future] = index
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                payload = future.result()
                outputs[index] = payload
                api_calls += 1
                if payload["parse_status"] == "parse_error":
                    parse_errors += 1
                if hasattr(progress, "update"):
                    progress.update(1)
                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
    if hasattr(progress, "close"):
        progress.close()
    return [payload for payload in outputs if payload is not None]


def extract_entities(
    config: dict[str, Any],
    *,
    target: str,
    input_path: str | Path | None = None,
    workers: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    dataset_path = Path(input_path) if input_path else get_path(config, "dataset_with_acc")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Missing dataset input at {dataset_path}. Run `prepare-dataset` first to create dataset_with_acc.jsonl."
        )
    records = read_jsonl(dataset_path)
    if not records:
        raise ValueError(
            f"Dataset input at {dataset_path} is empty. Run `prepare-dataset` and confirm alignment produced rows."
        )
    cache_dir = get_path(config, "cache_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    targets = ("gold", "pred") if target == "both" else (target,)
    effective_workers = int(workers if workers is not None else config["extraction"].get("workers", 1))
    for name in targets:
        extracted = extract_entities_for_records(
            records,
            model_config=config["models"][name],
            extraction_config=config["extraction"],
            cache_dir=cache_dir,
            workers=effective_workers,
            force=force,
        )
        output_path = get_path(config, f"{name}_entities")
        write_jsonl(output_path, extracted)
        outputs[f"{name}_output"] = str(output_path)
        outputs[f"{name}_count"] = len(extracted)
    outputs["workers"] = effective_workers
    return outputs
