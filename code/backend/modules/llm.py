from collections import defaultdict
from functools import lru_cache
import hashlib
import json
import re
from typing import Any, Callable, List

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)

from constants import LLM_JUDGE_PROMPT, RELEVANT_METADATA_PROMPT, SQL_REGENERATION_PROMPT
from modules.verifiers import (
    DEFAULT_VERIFIERS,
    run_external_verifiers,
    schema_consistency_verifier,
    syntax_verifier,
)

# Model state (populated by init_model before any generation calls)
_tokenizer = None
_model = None
_is_causal: bool = False  # True for decoder-only LMs (Llama, Qwen); False for seq2seq (Flan-T5)
_model_id: str | None = None

# Judge model state
_judge_tokenizer = None
_judge_model = None
_judge_is_causal: bool = False
_judge_model_id: str | None = None

_PHASE1_CACHE_SCHEMA_VERSION = 1
_PHASE1_INPUT_MAX_LENGTH = 1024
_PHASE1_NUM_BEAMS = 4
_PHASE1_MAX_OUTPUT_TOKENS = 512
_PHASE1_GENERATION_CONFIG = {
    "input_max_length": _PHASE1_INPUT_MAX_LENGTH,
    "num_beams": _PHASE1_NUM_BEAMS,
    "max_output_tokens": _PHASE1_MAX_OUTPUT_TOKENS,
    "do_sample": False,
}


def init_model(model_id: str) -> None:
    # Load tokenizer and model. Must be called once before any generation functions
    global _tokenizer, _model, _is_causal, _model_id

    _model_id = model_id

    config = AutoConfig.from_pretrained(model_id)
    _is_causal = not getattr(config, "is_encoder_decoder", False)

    _tokenizer = AutoTokenizer.from_pretrained(model_id)

    if _is_causal:
        # Left-pad so batched generation works correctly for decoder-only models.
        _tokenizer.padding_side = "left"
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
    else:
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    if torch.cuda.is_available():
        _model = _model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _model = _model.to("mps")


def init_judge_model(model_id: str | None) -> None:
    """
    Optionally load a separate model for the LLM-as-judge verifier. When
    model_id is None or matches the generator model, the judge reuses the
    generator (no extra GPU memory).
    """
    global _judge_tokenizer, _judge_model, _judge_is_causal, _judge_model_id

    if model_id is None or model_id == _model_id:
        _judge_tokenizer = None
        _judge_model = None
        _judge_is_causal = False
        _judge_model_id = None
        return

    config = AutoConfig.from_pretrained(model_id)
    _judge_is_causal = not getattr(config, "is_encoder_decoder", False)

    _judge_tokenizer = AutoTokenizer.from_pretrained(model_id)

    if _judge_is_causal:
        _judge_tokenizer.padding_side = "left"
        if _judge_tokenizer.pad_token is None:
            _judge_tokenizer.pad_token = _judge_tokenizer.eos_token
        _judge_model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
    else:
        _judge_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    if torch.cuda.is_available():
        _judge_model = _judge_model.to("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _judge_model = _judge_model.to("mps")

    _judge_model_id = model_id


def _resolve_judge_handles():
    # Return (tokenizer, model, is_causal) for the judge — generator if no separate judge.
    if _judge_model is not None:
        return _judge_tokenizer, _judge_model, _judge_is_causal
    return _tokenizer, _model, _is_causal


def get_judge_model_id() -> str:
    # Return the model id used for the judge (separate judge model if set, else generator)
    if _judge_model_id is not None:
        return _judge_model_id
    if _model_id is None:
        raise RuntimeError("Model has not been initialized yet.")
    return _model_id


def _format_prompt(prompt: str, *, tokenizer=None, is_causal: bool | None = None) -> str:
    # Wrap a plain-text prompt in a chat template for causal LMs; no-op for seq2seq.
    if tokenizer is None:
        tokenizer = _tokenizer
    if is_causal is None:
        is_causal = _is_causal
    if not is_causal:
        return prompt
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert SQL and database assistant. "
                "Follow all instructions precisely and output only what is requested."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    # Fallback for tokenizers without apply_chat_template
    return f"[INST] {prompt} [/INST]"


def get_initialized_model_id() -> str:
    if _model_id is None:
        raise RuntimeError("Model has not been initialized yet.")
    return _model_id


def _generate_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    num_beams: int = 4,
    num_return_sequences: int = 1,
    max_output_tokens: int = 256,
    *,
    tokenizer=None,
    model=None,
    is_causal: bool | None = None,
) -> torch.Tensor:
    """
    Unified generation for seq2seq and causal LMs.
    Returns new-token tensors only (input tokens are stripped for causal models).
    """
    if tokenizer is None:
        tokenizer = _tokenizer
    if model is None:
        model = _model
    if is_causal is None:
        is_causal = _is_causal

    generate_kwargs: dict[str, Any] = dict(
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        do_sample=False,
    )
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask

    if is_causal:
        generate_kwargs["max_new_tokens"] = max_output_tokens
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        outputs = model.generate(input_ids=input_ids, **generate_kwargs)
        # With left-padding, new tokens always start at input_ids.shape[1].
        return outputs[:, input_ids.shape[1]:]
    else:
        generate_kwargs["max_length"] = max_output_tokens
        return model.generate(inputs=input_ids, **generate_kwargs)


def _extract_sql(text: str) -> str:
    # Strip markdown code fences from causal LM output if present.
    block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if block:
        return block.group(1).strip()
    return text.strip()


# Utility helpers 

def extract_first_json_block(text: str):
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for idx, ch in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def get_query_translation_prompt(question: str, tables: str | None = None) -> str:
    if tables:
        return f"""convert question and table into SQL query. tables: {tables}. question: {question}"""
    return f"""convert question into SQL query. question: {question}"""


def serialize_tables_for_prompt(tables: dict[str, List[str]] | None) -> str:
    if not tables:
        return ""
    tables = [f"""{table_name}({",".join(tables[table_name])})""" for table_name in tables]
    return ", ".join(tables)


def _dedupe_sql_candidates(sql_candidates: list[str]) -> list[str]:
    unique = []
    seen = set()
    for candidate in sql_candidates:
        normalized = candidate.strip().lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate.strip())
    return unique


# Single-sample generation

def generate_sql_candidates(
    question: str,
    tables: dict[str, List[str]] | None,
    num_candidates: int,
    feedback: str | None = None,
    previous_candidates: list[str] | None = None,
) -> list[str]:
    table_text = serialize_tables_for_prompt(tables)
    base_prompt = get_query_translation_prompt(question=question, tables=table_text or None)

    if feedback:
        prompt = SQL_REGENERATION_PROMPT.format(
            base_prompt=base_prompt,
            previous_candidates_json=json.dumps(previous_candidates or []),
            feedback=feedback,
        )
    else:
        prompt = base_prompt

    prompt = _format_prompt(prompt)
    enc = _tokenizer(prompt, max_length=1024, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(_model.device)

    new_tokens = _generate_tokens(
        input_ids=input_ids,
        num_beams=max(num_candidates, 4),
        num_return_sequences=max(num_candidates, 1),
        max_output_tokens=512,
    )
    decoded = [
        _tokenizer.decode(t, skip_special_tokens=True).strip() for t in new_tokens
    ]
    if _is_causal:
        decoded = [_extract_sql(d) for d in decoded]
    return _dedupe_sql_candidates(decoded)[:num_candidates]


def translate_query(question: str, tables: dict[str, List[str]] | None = None) -> str:
    candidates = generate_sql_candidates(question=question, tables=tables, num_candidates=1)
    return candidates[0] if candidates else ""


# Schema / metadata

@lru_cache(maxsize=4)
def _load_spider_table_rows(tables_path: str) -> list[dict[str, Any]]:
    with open(tables_path, "r", encoding="utf-8") as table_file:
        payload = json.load(table_file)

    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported Spider tables payload format in {tables_path}")


def _schema_row_to_metadata(row: dict[str, Any]) -> dict[str, Any]:
    table_names = row.get("table_names_original", [])
    column_names = row.get("column_names_original", [])
    column_types = row.get("column_types", [])

    table_dict: dict[str, dict[str, Any]] = {
        table_name: {"columns": [], "primary_keys": []}
        for table_name in table_names
    }

    for col_entry, col_type in zip(column_names, column_types):
        if not isinstance(col_entry, (list, tuple)) or len(col_entry) != 2:
            continue
        table_idx, column_name = col_entry
        if not isinstance(table_idx, int) or table_idx < 0 or table_idx >= len(table_names):
            continue
        table_name = table_names[table_idx]
        table_dict[table_name]["columns"].append((column_name, col_type))

    for pk_idx in row.get("primary_keys", []):
        if not isinstance(pk_idx, int) or pk_idx < 0 or pk_idx >= len(column_names):
            continue
        col_entry = column_names[pk_idx]
        if not isinstance(col_entry, (list, tuple)) or len(col_entry) != 2:
            continue
        table_idx, column_name = col_entry
        if not isinstance(table_idx, int) or table_idx < 0 or table_idx >= len(table_names):
            continue
        table_name = table_names[table_idx]
        table_dict[table_name]["primary_keys"].append(column_name)

    foreign_key_dict = defaultdict(list)
    for relation in row.get("foreign_keys", []):
        if not isinstance(relation, (list, tuple)) or len(relation) != 2:
            continue
        source_col_idx, target_col_idx = relation
        if not isinstance(source_col_idx, int) or not isinstance(target_col_idx, int):
            continue
        if source_col_idx < 0 or source_col_idx >= len(column_names):
            continue
        if target_col_idx < 0 or target_col_idx >= len(column_names):
            continue

        source_col_entry = column_names[source_col_idx]
        target_col_entry = column_names[target_col_idx]
        if not isinstance(source_col_entry, (list, tuple)) or len(source_col_entry) != 2:
            continue
        if not isinstance(target_col_entry, (list, tuple)) or len(target_col_entry) != 2:
            continue

        source_table_idx, source_col_name = source_col_entry
        target_table_idx, target_col_name = target_col_entry
        if not isinstance(source_table_idx, int) or not isinstance(target_table_idx, int):
            continue
        if source_table_idx < 0 or source_table_idx >= len(table_names):
            continue
        if target_table_idx < 0 or target_table_idx >= len(table_names):
            continue

        source_table_name = table_names[source_table_idx]
        target_table_name = table_names[target_table_idx]
        foreign_key_dict[source_table_name].append(
            {
                "source_column": source_col_name,
                "target_table": target_table_name,
                "target_column": target_col_name,
            }
        )

    return {"tables": table_dict, "foreign_keys": dict(foreign_key_dict)}


@lru_cache(maxsize=256)
def build_table_metadata(db_id: str, tables_path: str = "datasets/spider/tables.json") -> dict[str, Any]:
    rows = _load_spider_table_rows(tables_path)
    for row in rows:
        if row.get("db_id") == db_id:
            return _schema_row_to_metadata(row)
    raise ValueError(f"db_id '{db_id}' not found in {tables_path}")


def _build_relevant_metadata_prompt(raw_metadata: dict, question: str) -> str:
    return _format_prompt(
        RELEVANT_METADATA_PROMPT.format(
            metadata_json=json.dumps(raw_metadata),
            question=question,
        )
    )


def get_phase1_cache_schema_version() -> int:
    return _PHASE1_CACHE_SCHEMA_VERSION


def build_phase1_cache_key(raw_metadata: dict, question: str) -> str:
    prompt = _build_relevant_metadata_prompt(raw_metadata, question)
    payload = {
        "cache_schema_version": _PHASE1_CACHE_SCHEMA_VERSION,
        "generation_config": _PHASE1_GENERATION_CONFIG,
        "model_id": get_initialized_model_id(),
        "prompt": prompt,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_metadata_filter(raw_metadata: dict, question: str, decoded: str) -> dict:
    parsed = None
    try:
        parsed = json.loads(decoded)
    except Exception:
        json_block = extract_first_json_block(decoded)
        if json_block:
            try:
                parsed = json.loads(json_block)
            except Exception:
                parsed = None

    selected_tables: set[str] = set()
    selected_columns: defaultdict[str, set[str]] = defaultdict(set)

    if isinstance(parsed, dict):
        for table_name in parsed.get("tables", []):
            if isinstance(table_name, str):
                selected_tables.add(table_name)
        columns_obj = parsed.get("columns", {})
        if isinstance(columns_obj, dict):
            for table_name, columns in columns_obj.items():
                if isinstance(table_name, str) and isinstance(columns, list):
                    for col in columns:
                        if isinstance(col, str):
                            selected_columns[table_name].add(col)

    if not selected_tables and not selected_columns:
        question_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower()))
        for table_name, table_info in raw_metadata.get("tables", {}).items():
            table_match = table_name.lower() in question_tokens
            for column_name, _ in table_info.get("columns", []):
                if column_name.lower() in question_tokens:
                    selected_columns[table_name].add(column_name)
                    table_match = True
            if table_match:
                selected_tables.add(table_name)

    if not selected_tables and not selected_columns:
        return raw_metadata

    result_tables = {}
    for table_name, table_info in raw_metadata.get("tables", {}).items():
        if table_name not in selected_tables and table_name not in selected_columns:
            continue
        selected_for_table = selected_columns.get(table_name, set())
        if selected_for_table:
            filtered_columns = [
                (col_name, col_type)
                for col_name, col_type in table_info.get("columns", [])
                if col_name in selected_for_table
            ] or table_info.get("columns", [])
        else:
            filtered_columns = table_info.get("columns", [])
        result_tables[table_name] = {
            "columns": filtered_columns,
            "primary_keys": table_info.get("primary_keys", []),
        }

    result_foreign_keys = {
        src: [r for r in rels if r.get("target_table") in result_tables]
        for src, rels in raw_metadata.get("foreign_keys", {}).items()
        if src in result_tables
    }
    result_foreign_keys = {k: v for k, v in result_foreign_keys.items() if v}
    return {"tables": result_tables, "foreign_keys": result_foreign_keys}


def fetch_relevant_metadata(raw_metadata: dict, question: str) -> dict:
    if not raw_metadata or not question.strip():
        return raw_metadata

    prompt = _build_relevant_metadata_prompt(raw_metadata, question)
    enc = _tokenizer(
        prompt,
        max_length=_PHASE1_INPUT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(_model.device)
    new_tokens = _generate_tokens(
        input_ids=input_ids,
        num_beams=_PHASE1_NUM_BEAMS,
        max_output_tokens=_PHASE1_MAX_OUTPUT_TOKENS,
    )
    decoded = _tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    return _apply_metadata_filter(raw_metadata, question, decoded)


def fetch_relevant_metadata_batch(
    raw_metadatas: list[dict],
    questions: list[str],
) -> list[dict]:
    assert len(raw_metadatas) == len(questions)

    results: list[dict | None] = [None] * len(raw_metadatas)
    active_indices = []
    for idx, (meta, q) in enumerate(zip(raw_metadatas, questions)):
        if not meta or not q.strip():
            results[idx] = meta
        else:
            active_indices.append(idx)

    if not active_indices:
        return results  # type: ignore[return-value]

    prompts = [
        _build_relevant_metadata_prompt(raw_metadatas[i], questions[i])
        for i in active_indices
    ]
    enc = _tokenizer(
        prompts,
        max_length=_PHASE1_INPUT_MAX_LENGTH,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(_model.device)
    attention_mask = enc.attention_mask.to(_model.device)

    new_tokens = _generate_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=_PHASE1_NUM_BEAMS,
        max_output_tokens=_PHASE1_MAX_OUTPUT_TOKENS,
    )
    for out, idx in zip(new_tokens, active_indices):
        decoded = _tokenizer.decode(out, skip_special_tokens=True).strip()
        results[idx] = _apply_metadata_filter(raw_metadatas[idx], questions[idx], decoded)

    return results  # type: ignore[return-value]


# Batch SQL generation

# Generate SQL candidates for multiple samples in one model.generate() call.
def generate_sql_candidates_batch(
    items: list[dict[str, Any]],
) -> list[list[str]]:
    """
    Each item must have:
        question            (str)
        tables              (dict[str, list[str]])
        num_candidates      (int)
        feedback            (str | None)
        previous_candidates (list[str] | None)

    Returns a list of candidate lists, one per input item.
    """
    if not items:
        return []

    num_candidates = items[0]["num_candidates"]
    prompts = []
    for item in items:
        table_text = serialize_tables_for_prompt(item.get("tables"))
        base_prompt = get_query_translation_prompt(
            question=item["question"],
            tables=table_text or None,
        )
        feedback = item.get("feedback")
        if feedback:
            prompt = SQL_REGENERATION_PROMPT.format(
                base_prompt=base_prompt,
                previous_candidates_json=json.dumps(item.get("previous_candidates") or []),
                feedback=feedback,
            )
        else:
            prompt = base_prompt
        prompts.append(_format_prompt(prompt))

    num_beams = max(num_candidates, 4)
    num_return = max(num_candidates, 1)

    enc = _tokenizer(
        prompts, max_length=1024, truncation=True, padding=True, return_tensors="pt"
    )
    input_ids = enc.input_ids.to(_model.device)
    attention_mask = enc.attention_mask.to(_model.device)

    new_tokens = _generate_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=num_beams,
        num_return_sequences=num_return,
        max_output_tokens=512,
    )

    B = len(items)
    all_candidates: list[list[str]] = []
    for b in range(B):
        seqs = new_tokens[b * num_return : (b + 1) * num_return]
        decoded = [
            _tokenizer.decode(s, skip_special_tokens=True).strip() for s in seqs
        ]
        if _is_causal:
            decoded = [_extract_sql(d) for d in decoded]
        all_candidates.append(_dedupe_sql_candidates(decoded)[:num_candidates])
    return all_candidates


# LLM judge

def judge_sql_with_llm(question: str, candidate_sql: str, metadata: dict[str, Any]) -> dict[str, Any]:
    judge_tokenizer, judge_model, judge_is_causal = _resolve_judge_handles()
    prompt = _format_prompt(
        LLM_JUDGE_PROMPT.format(
            question=question,
            candidate_sql=candidate_sql,
            metadata_json=json.dumps(metadata),
        ),
        tokenizer=judge_tokenizer,
        is_causal=judge_is_causal,
    )
    enc = judge_tokenizer(prompt, max_length=1024, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(judge_model.device)
    new_tokens = _generate_tokens(
        input_ids=input_ids,
        num_beams=1,
        max_output_tokens=256,
        tokenizer=judge_tokenizer,
        model=judge_model,
        is_causal=judge_is_causal,
    )
    decoded = judge_tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    parsed = None
    try:
        parsed = json.loads(decoded)
    except Exception:
        json_block = extract_first_json_block(decoded)
        if json_block:
            try:
                parsed = json.loads(json_block)
            except Exception:
                parsed = None

    if isinstance(parsed, dict):
        return {
            "passed": bool(parsed.get("passed", False)),
            "confidence": float(parsed.get("confidence", 0.0)),
            "rationale": str(parsed.get("rationale", "No rationale provided.")),
            "raw": decoded,
        }

    heuristic_pass = bool(re.search(r"\bselect\b", candidate_sql, flags=re.IGNORECASE)) and bool(
        re.search(r"\bfrom\b", candidate_sql, flags=re.IGNORECASE)
    )
    return {
        "passed": heuristic_pass,
        "confidence": 0.3,
        "rationale": "LLM judge output was not valid JSON; fallback heuristic was used.",
        "raw": decoded,
    }


def judge_sql_batch(
    items: list[tuple[str, str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Batch version of judge_sql_with_llm.
    items: list of (question, candidate_sql, metadata) tuples.
    """
    if not items:
        return []

    judge_tokenizer, judge_model, judge_is_causal = _resolve_judge_handles()

    prompts = [
        _format_prompt(
            LLM_JUDGE_PROMPT.format(
                question=q,
                candidate_sql=sql,
                metadata_json=json.dumps(meta),
            ),
            tokenizer=judge_tokenizer,
            is_causal=judge_is_causal,
        )
        for q, sql, meta in items
    ]

    enc = judge_tokenizer(
        prompts, max_length=1024, truncation=True, padding=True, return_tensors="pt"
    )
    input_ids = enc.input_ids.to(judge_model.device)
    attention_mask = enc.attention_mask.to(judge_model.device)

    new_tokens = _generate_tokens(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=1,
        max_output_tokens=256,
        tokenizer=judge_tokenizer,
        model=judge_model,
        is_causal=judge_is_causal,
    )

    results = []
    for output, (_, candidate_sql, _) in zip(new_tokens, items):
        decoded = judge_tokenizer.decode(output, skip_special_tokens=True).strip()

        parsed = None
        try:
            parsed = json.loads(decoded)
        except Exception:
            json_block = extract_first_json_block(decoded)
            if json_block:
                try:
                    parsed = json.loads(json_block)
                except Exception:
                    parsed = None

        if isinstance(parsed, dict):
            results.append({
                "passed": bool(parsed.get("passed", False)),
                "confidence": float(parsed.get("confidence", 0.0)),
                "rationale": str(parsed.get("rationale", "No rationale provided.")),
                "raw": decoded,
            })
        else:
            heuristic_pass = bool(re.search(r"\bselect\b", candidate_sql, re.IGNORECASE)) and bool(
                re.search(r"\bfrom\b", candidate_sql, re.IGNORECASE)
            )
            results.append({
                "passed": heuristic_pass,
                "confidence": 0.3,
                "rationale": "LLM judge output was not valid JSON; fallback heuristic was used.",
                "raw": decoded,
            })

    return results


# ── Metadata helpers ──────────────────────────────────────────────────────────

def metadata_to_prompt_tables(metadata: dict[str, Any]) -> dict[str, list[str]]:
    tables = {}
    for table_name, table_info in metadata.get("tables", {}).items():
        columns = []
        for col_entry in table_info.get("columns", []):
            if isinstance(col_entry, (list, tuple)) and col_entry:
                columns.append(str(col_entry[0]))
            else:
                columns.append(str(col_entry))
        tables[table_name] = columns
    return tables


def metadata_from_prompt_tables(tables: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "tables": {
            table_name: {
                "columns": [(column_name, "unknown") for column_name in columns],
                "primary_keys": [],
            }
            for table_name, columns in tables.items()
        },
        "foreign_keys": {},
    }


# ── Verifier feedback ─────────────────────────────────────────────────────────

def build_verifier_feedback(candidate_results: list[dict[str, Any]]) -> str:
    lines = []
    for idx, result in enumerate(candidate_results, start=1):
        sql = result.get("sql", "")
        verifiers = result.get("verifiers", {})
        if verifiers.get("all_passed"):
            continue

        lines.append(f"Candidate {idx}: {sql}")

        syntax_result = verifiers.get("syntax")
        if syntax_result is not None and not syntax_result.get("passed", False):
            error = syntax_result.get("details", {}).get("error", "syntax parse failed")
            lines.append(f"- syntax failed: {error}")

        schema_result = verifiers.get("schema_consistency")
        if schema_result is not None and not schema_result.get("passed", False):
            lines.append(
                "- schema_consistency failed: "
                + json.dumps(schema_result.get("details", {}), ensure_ascii=True)
            )

        judge_result = verifiers.get("llm_judge")
        if judge_result is not None and not judge_result.get("passed", False):
            lines.append("- llm_judge failed: " + str(judge_result.get("rationale", "")))

    return "\n".join(lines) if lines else "No failing candidates."


# ── Modulo loop ───────────────────────────────────────────────────────────────

def run_modulo_loop(
    question: str,
    tables: dict[str, list[str]],
    metadata: dict[str, Any] | None = None,
    iter_limit: int = 3,
    candidates_per_iter: int = 3,
    enabled_verifiers: set[str] | None = None,
) -> dict[str, Any]:
    if enabled_verifiers is None:
        enabled_verifiers = set(DEFAULT_VERIFIERS)
    metadata_payload = metadata if metadata is not None else metadata_from_prompt_tables(tables)
    iteration_trace = []
    feedback = None
    previous_candidates: list[str] = []

    for iteration_idx in range(1, iter_limit + 1):
        candidates = generate_sql_candidates(
            question=question,
            tables=tables,
            num_candidates=candidates_per_iter,
            feedback=feedback,
            previous_candidates=previous_candidates,
        )
        previous_candidates = candidates

        syntax_results = (
            [syntax_verifier(sql) for sql in candidates]
            if "syntax" in enabled_verifiers
            else None
        )
        schema_results = (
            [schema_consistency_verifier(sql=sql, metadata=metadata_payload) for sql in candidates]
            if "schema" in enabled_verifiers
            else None
        )
        judge_verdicts: list[dict] | None = None
        if "judge" in enabled_verifiers:
            judge_inputs = [(question, sql, metadata_payload) for sql in candidates]
            judge_verdicts = judge_sql_batch(judge_inputs)

        candidate_results = []
        passing_queries = []
        for cand_idx, sql in enumerate(candidates):
            verification: dict[str, Any] = {}
            passes: list[bool] = []

            if syntax_results is not None:
                verification["syntax"] = syntax_results[cand_idx]
                passes.append(verification["syntax"]["passed"])

            if schema_results is not None:
                verification["schema_consistency"] = schema_results[cand_idx]
                passes.append(verification["schema_consistency"]["passed"])

            if judge_verdicts is not None:
                jv = judge_verdicts[cand_idx]
                verification["llm_judge"] = {
                    "name": "llm_judge",
                    "passed": jv["passed"],
                    "confidence": jv["confidence"],
                    "rationale": jv["rationale"],
                    "details": {"raw": jv["raw"]},
                }
                passes.append(verification["llm_judge"]["passed"])

            all_passed = all(passes) if passes else True
            verification["all_passed"] = all_passed
            record = {"sql": sql, "verifiers": verification}
            candidate_results.append(record)
            if all_passed:
                passing_queries.append(record)

        iteration_trace.append(
            {
                "iteration": iteration_idx,
                "feedback_used": feedback,
                "candidates": candidate_results,
            }
        )

        if passing_queries:
            return {
                "status": "success",
                "question": question,
                "iterations_used": iteration_idx,
                "passing_queries": passing_queries,
                "iteration_trace": iteration_trace,
            }

        feedback = build_verifier_feedback(candidate_results)

    return {
        "status": "failure",
        "reason": "max_retries_reached",
        "question": question,
        "iterations_used": iter_limit,
        "last_feedback": feedback,
        "iteration_trace": iteration_trace,
    }


def run_modulo_loop_batch(
    samples_data: list[dict[str, Any]],
    iter_limit: int,
    candidates_per_iter: int,
    inference_batch_size: int = 16,
    progress_callback: Callable[[str], None] | None = None,
    enabled_verifiers: set[str] | None = None,
) -> list[dict[str, Any]]:
    if enabled_verifiers is None:
        enabled_verifiers = set(DEFAULT_VERIFIERS)

    def log_progress(message: str) -> None:
        if progress_callback is not None:
            progress_callback(message)

    n = len(samples_data)
    active: list[int] = list(range(n))
    feedbacks: list[str | None] = [None] * n
    previous_candidates: list[list[str]] = [[] for _ in range(n)]
    iteration_traces: list[list[dict]] = [[] for _ in range(n)]
    final_results: list[dict[str, Any] | None] = [None] * n

    for iteration_idx in range(1, iter_limit + 1):
        if not active:
            break
        log_progress(
            f"  modulo iteration {iteration_idx}/{iter_limit}: "
            f"{len(active)}/{n} active samples"
        )

        all_candidates: dict[int, list[str]] = {}
        for chunk_start in range(0, len(active), inference_batch_size):
            chunk = active[chunk_start : chunk_start + inference_batch_size]
            batch_items = [
                {
                    "question": samples_data[i]["question"],
                    "tables": samples_data[i]["tables"],
                    "num_candidates": candidates_per_iter,
                    "feedback": feedbacks[i],
                    "previous_candidates": previous_candidates[i],
                }
                for i in chunk
            ]
            batch_results = generate_sql_candidates_batch(batch_items)
            for i, cands in zip(chunk, batch_results):
                all_candidates[i] = cands
                previous_candidates[i] = cands
            log_progress(
                f"    candidate generation: "
                f"{min(chunk_start + inference_batch_size, len(active))}/{len(active)} "
                "active samples"
            )

        # Syntax verifier (sqlglot, cheap, in-process)
        syntax_results: dict[int, list[dict]] = {}
        if "syntax" in enabled_verifiers:
            for i in active:
                syntax_results[i] = [syntax_verifier(sql) for sql in all_candidates[i]]

        # Schema consistency verifier (regex, in-process)
        schema_results: dict[int, list[dict]] = {}
        if "schema" in enabled_verifiers:
            for i in active:
                meta = samples_data[i]["metadata"]
                schema_results[i] = [
                    schema_consistency_verifier(sql=sql, metadata=meta)
                    for sql in all_candidates[i]
                ]

        # LLM judge verifier (batched GPU call)
        sample_judge_verdicts: dict[int, dict[int, dict]] = {i: {} for i in active}
        if "judge" in enabled_verifiers:
            judge_inputs: list[tuple[str, str, dict]] = []
            judge_index_map: list[tuple[int, int]] = []
            for i in active:
                q = samples_data[i]["question"]
                meta = samples_data[i]["metadata"]
                for cand_idx, sql in enumerate(all_candidates[i]):
                    judge_inputs.append((q, sql, meta))
                    judge_index_map.append((i, cand_idx))

            judge_verdicts_flat: list[dict] = []
            judge_batch_size = inference_batch_size * candidates_per_iter
            for chunk_start in range(0, len(judge_inputs), judge_batch_size):
                chunk = judge_inputs[chunk_start : chunk_start + inference_batch_size * candidates_per_iter]
                judge_verdicts_flat.extend(judge_sql_batch(chunk))
                log_progress(
                    f"    llm judge: "
                    f"{min(chunk_start + judge_batch_size, len(judge_inputs))}/{len(judge_inputs)} "
                    "candidate checks"
                )

            for (i, cand_idx), verdict in zip(judge_index_map, judge_verdicts_flat):
                sample_judge_verdicts[i][cand_idx] = verdict

        still_active: list[int] = []
        for i in active:
            candidates = all_candidates[i]
            question = samples_data[i]["question"]

            candidate_results: list[dict] = []
            passing_queries: list[dict] = []
            for cand_idx, sql in enumerate(candidates):
                verification: dict[str, Any] = {}
                passes: list[bool] = []

                if "syntax" in enabled_verifiers:
                    verification["syntax"] = syntax_results[i][cand_idx]
                    passes.append(verification["syntax"]["passed"])

                if "schema" in enabled_verifiers:
                    verification["schema_consistency"] = schema_results[i][cand_idx]
                    passes.append(verification["schema_consistency"]["passed"])

                if "judge" in enabled_verifiers:
                    jv = sample_judge_verdicts[i][cand_idx]
                    verification["llm_judge"] = {
                        "name": "llm_judge",
                        "passed": jv["passed"],
                        "confidence": jv["confidence"],
                        "rationale": jv["rationale"],
                        "details": {"raw": jv["raw"]},
                    }
                    passes.append(verification["llm_judge"]["passed"])

                all_passed = all(passes) if passes else True
                verification["all_passed"] = all_passed
                record = {"sql": sql, "verifiers": verification}
                candidate_results.append(record)
                if all_passed:
                    passing_queries.append(record)

            iteration_traces[i].append(
                {
                    "iteration": iteration_idx,
                    "feedback_used": feedbacks[i],
                    "candidates": candidate_results,
                }
            )

            if passing_queries:
                final_results[i] = {
                    "status": "success",
                    "question": question,
                    "iterations_used": iteration_idx,
                    "passing_queries": passing_queries,
                    "iteration_trace": iteration_traces[i],
                }
            else:
                feedbacks[i] = build_verifier_feedback(candidate_results)
                still_active.append(i)

        resolved_this_iter = len(active) - len(still_active)
        log_progress(
            f"  modulo iteration {iteration_idx}/{iter_limit} complete: "
            f"{resolved_this_iter} resolved, {len(still_active)} still active"
        )
        active = still_active

    for i in active:
        final_results[i] = {
            "status": "failure",
            "reason": "max_retries_reached",
            "question": samples_data[i]["question"],
            "iterations_used": iter_limit,
            "last_feedback": feedbacks[i],
            "iteration_trace": iteration_traces[i],
        }

    if active:
        log_progress(
            f"  modulo loop exhausted after {iter_limit} iterations: "
            f"{len(active)} unresolved samples"
        )
    else:
        log_progress("  modulo loop finished: all samples resolved")

    return final_results  # type: ignore[return-value]


def translate_and_verify(
    question: str,
    tables: dict[str, list[str]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    loop_result = run_modulo_loop(
        question=question,
        tables=tables,
        metadata=metadata,
        iter_limit=1,
        candidates_per_iter=1,
    )
    if loop_result.get("status") == "success":
        best = loop_result["passing_queries"][0]
        sql = best["sql"]
        verification = best["verifiers"]
    else:
        fallback_candidate = loop_result["iteration_trace"][0]["candidates"][0]
        sql = fallback_candidate["sql"]
        verification = fallback_candidate["verifiers"]
    return {
        "question": question,
        "predicted_sql": sql,
        "verifiers": verification,
    }


if __name__ == "__main__":
    from constants import DEFAULT_MODEL
    init_model(DEFAULT_MODEL)
    raw_metadata = build_table_metadata(db_id="department_management")
    question = "What is the number of employees in department '1'?"
    relevant_metadata = fetch_relevant_metadata(raw_metadata, question)

    print("Raw Metadata:")
    print(json.dumps(raw_metadata))
    print("Relevant Metadata:")
    print(json.dumps(relevant_metadata))
