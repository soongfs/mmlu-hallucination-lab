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
    def tqdm(iterable, **_kwargs):  # type: ignore[no-redef]
        return iterable

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


INSTRUCTION_BLOCK = """Please identify the entities in the given question.
Return the answer directly as a JSON list of strings.

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


def build_prompt_text(question: str) -> str:
    return f"{INSTRUCTION_BLOCK}\nQuestion:\n{question}\n\nReturn only a JSON array of strings."


def build_chat_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": INSTRUCTION_BLOCK},
        {"role": "user", "content": question},
    ]


def _get_openai_client(model_config: dict[str, Any]):
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when dependency missing at runtime
        raise RuntimeError("openai is required for entity extraction.") from exc
    base_url = os.environ.get(model_config.get("base_url_env", ""), model_config.get("base_url"))
    api_key = os.environ.get(model_config.get("api_key_env", ""), "")
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
    entities = parse_entity_response(raw_response)
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
    entities = parse_entity_response(raw_response)
    return raw_response, entities


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
        payload = {
            "question_uid": record["question_uid"],
            "model_name": model_config["model"],
            "raw_response": raw_response,
            "entities_raw": entities,
            "parse_status": "empty" if not entities else "ok",
            "prompt_version": extraction_config["prompt_version"],
        }
    except Exception as exc:  # noqa: BLE001
        payload = {
            "question_uid": record["question_uid"],
            "model_name": model_config["model"],
            "raw_response": "",
            "entities_raw": [],
            "parse_status": "parse_error",
            "prompt_version": extraction_config["prompt_version"],
            "error": str(exc),
        }
    write_json(cache_path, payload, pretty=True)
    return payload


def extract_entities_for_records(
    records: list[dict[str, Any]],
    *,
    model_config: dict[str, Any],
    extraction_config: dict[str, Any],
    cache_dir: Path,
    workers: int = 1,
    force: bool = False,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError(f"`workers` must be positive, got {workers}.")
    outputs: list[dict[str, Any] | None] = [None] * len(records)
    model_dir = cache_dir / _slugify_model_name(model_config["model"])
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_hits = 0
    api_calls = 0
    parse_errors = 0
    progress = tqdm(
        records,
        desc=f"Extracting {model_config['model']}",
        unit="question",
        dynamic_ncols=True,
    )
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
            outputs[index] = cached
            cache_hits += 1
            if hasattr(progress, "update"):
                progress.update(1)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(cached=cache_hits, api=api_calls, errors=parse_errors)
            continue
        pending_jobs.append((index, record, cache_path))
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
