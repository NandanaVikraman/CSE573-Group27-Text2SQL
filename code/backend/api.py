from __future__ import annotations

import os
import re
from collections import defaultdict
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from constants import DEFAULT_CANDIDATES_PER_ITER, DEFAULT_ITER_LIMIT, DEFAULT_MODEL, MODEL_ALIASES
from modules.llm import init_model, metadata_to_prompt_tables, run_modulo_loop, translate_query

app = FastAPI(title="Text2SQL API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL_LOCK = Lock()
_MODEL_READY = False

_DIALECT_MAP = {
    "postgresql": "postgres",
    "mysql": "mysql",
    "sqlite": "sqlite",
    "sql server": "tsql",
    "oracle": "oracle",
}


class GenerateSQLRequest(BaseModel):
    question: str = Field(min_length=1)
    schema_ddl: str = Field(min_length=1)
    db_type: str = Field(default="PostgreSQL")
    use_modulo: bool = Field(default=False)
    iter_limit: int = Field(default=DEFAULT_ITER_LIMIT, ge=1, le=8)
    candidates_per_iter: int = Field(default=DEFAULT_CANDIDATES_PER_ITER, ge=1, le=8)


class GenerateSQLResponse(BaseModel):
    status: str
    sql: str
    used_modulo: bool
    verifier_summary: dict[str, Any] | None = None
    iterations_used: int = 1


def _ensure_model_initialized() -> None:
    global _MODEL_READY
    if _MODEL_READY:
        return

    with _MODEL_LOCK:
        if _MODEL_READY:
            return
        model_alias_or_id = os.getenv("TEXT2SQL_MODEL", DEFAULT_MODEL)
        model_id = MODEL_ALIASES.get(model_alias_or_id, model_alias_or_id)
        init_model(model_id)
        _MODEL_READY = True


def _clean_identifier(name: str) -> str:
    return name.strip().strip('`"[]')


def _split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    in_single_quote = False
    in_double_quote = False
    in_backtick_quote = False
    i = 0

    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""

        if not in_single_quote and not in_double_quote and not in_backtick_quote:
            if ch == "-" and nxt == "-":
                while i < len(text) and text[i] != "\n":
                    i += 1
                continue
            if ch == "/" and nxt == "*":
                i += 2
                while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                    i += 1
                i += 2
                continue

        if ch == "'" and not in_double_quote and not in_backtick_quote:
            in_single_quote = not in_single_quote
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single_quote and not in_backtick_quote:
            in_double_quote = not in_double_quote
            current.append(ch)
            i += 1
            continue
        if ch == "`" and not in_single_quote and not in_double_quote:
            in_backtick_quote = not in_backtick_quote
            current.append(ch)
            i += 1
            continue

        if not in_single_quote and not in_double_quote and not in_backtick_quote:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)
            elif ch == separator and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                i += 1
                continue

        current.append(ch)
        i += 1

    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_paren_groups(text: str) -> list[str]:
    groups: list[str] = []
    depth = 0
    start = -1
    in_single_quote = False
    in_double_quote = False
    in_backtick_quote = False

    for idx, ch in enumerate(text):
        if ch == "'" and not in_double_quote and not in_backtick_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote and not in_backtick_quote:
            in_double_quote = not in_double_quote
        elif ch == "`" and not in_single_quote and not in_double_quote:
            in_backtick_quote = not in_backtick_quote

        if in_single_quote or in_double_quote or in_backtick_quote:
            continue

        if ch == "(":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                groups.append(text[start:idx])
                start = -1
    return groups


def _parse_foreign_key_line(line: str, table_name: str) -> list[dict[str, str]]:
    compact = " ".join(line.strip().split())
    match = re.search(
        r"FOREIGN\s+KEY\s*\(([^)]*)\)\s*REFERENCES\s+([`\"[\]\w\.]+)\s*\(([^)]*)\)",
        compact,
        flags=re.IGNORECASE,
    )
    if not match:
        return []

    source_columns = [_clean_identifier(col) for col in match.group(1).split(",") if col.strip()]
    target_table = _clean_identifier(match.group(2).split(".")[-1])
    target_columns = [_clean_identifier(col) for col in match.group(3).split(",") if col.strip()]

    relations: list[dict[str, str]] = []
    for source_col, target_col in zip(source_columns, target_columns):
        relations.append(
            {
                "source_column": source_col,
                "target_table": target_table,
                "target_column": target_col,
            }
        )
    return relations


def _parse_column_definition(line: str) -> tuple[str, str, bool, list[dict[str, str]]] | None:
    compact = line.strip()
    if not compact:
        return None

    tokens = compact.split()
    if not tokens:
        return None

    column_name = _clean_identifier(tokens[0])
    if not column_name:
        return None

    upper = compact.upper()
    stop_keywords = {"PRIMARY", "NOT", "NULL", "DEFAULT", "UNIQUE", "CHECK", "REFERENCES", "CONSTRAINT"}
    type_tokens: list[str] = []
    for token in tokens[1:]:
        if token.upper() in stop_keywords:
            break
        type_tokens.append(token)
    column_type = " ".join(type_tokens).strip() or "unknown"

    inline_pk = "PRIMARY KEY" in upper
    inline_fk: list[dict[str, str]] = []
    refs = re.search(r"REFERENCES\s+([`\"[\]\w\.]+)\s*\(([^)]*)\)", compact, flags=re.IGNORECASE)
    if refs:
        target_table = _clean_identifier(refs.group(1).split(".")[-1])
        target_columns = [_clean_identifier(col) for col in refs.group(2).split(",") if col.strip()]
        for target_col in target_columns:
            inline_fk.append(
                {
                    "source_column": column_name,
                    "target_table": target_table,
                    "target_column": target_col,
                }
            )

    return column_name, column_type, inline_pk, inline_fk


def _extract_create_table_blocks(schema_ddl: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?", flags=re.IGNORECASE)
    pos = 0

    while True:
        match = pattern.search(schema_ddl, pos)
        if not match:
            break

        i = match.end()
        while i < len(schema_ddl) and schema_ddl[i].isspace():
            i += 1

        name_chars: list[str] = []
        in_bracket = False
        in_quote = False
        quote_char = ""
        while i < len(schema_ddl):
            ch = schema_ddl[i]
            if in_quote:
                name_chars.append(ch)
                if ch == quote_char:
                    in_quote = False
                i += 1
                continue
            if ch in ('"', "`"):
                in_quote = True
                quote_char = ch
                name_chars.append(ch)
                i += 1
                continue
            if ch == "[":
                in_bracket = True
                name_chars.append(ch)
                i += 1
                continue
            if ch == "]":
                in_bracket = False
                name_chars.append(ch)
                i += 1
                continue
            if ch == "(" and not in_bracket:
                break
            name_chars.append(ch)
            i += 1

        table_name_raw = "".join(name_chars).strip()
        if i >= len(schema_ddl) or schema_ddl[i] != "(":
            pos = match.end()
            continue

        depth = 0
        body_start = i + 1
        i += 1
        in_single_quote = False
        in_double_quote = False
        in_backtick_quote = False
        while i < len(schema_ddl):
            ch = schema_ddl[i]
            if ch == "'" and not in_double_quote and not in_backtick_quote:
                in_single_quote = not in_single_quote
            elif ch == '"' and not in_single_quote and not in_backtick_quote:
                in_double_quote = not in_double_quote
            elif ch == "`" and not in_single_quote and not in_double_quote:
                in_backtick_quote = not in_backtick_quote

            if not in_single_quote and not in_double_quote and not in_backtick_quote:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    if depth == 0:
                        break
                    depth -= 1
            i += 1

        body_end = i
        table_name = _clean_identifier(table_name_raw.split(".")[-1])
        if table_name and body_end > body_start:
            blocks.append((table_name, schema_ddl[body_start:body_end]))
        pos = i + 1

    return blocks


def parse_schema_ddl_to_metadata(schema_ddl: str) -> dict[str, Any]:
    tables: dict[str, dict[str, Any]] = {}
    foreign_keys: defaultdict[str, list[dict[str, str]]] = defaultdict(list)

    blocks = _extract_create_table_blocks(schema_ddl)
    if not blocks:
        raise ValueError("No CREATE TABLE definitions were found in the schema.")

    for table_name, body in blocks:
        columns: list[tuple[str, str]] = []
        primary_keys: list[str] = []
        for line in _split_top_level(body):
            compact = " ".join(line.strip().split())
            if not compact:
                continue

            upper = compact.upper()
            if upper.startswith("PRIMARY KEY") or (
                upper.startswith("CONSTRAINT") and "PRIMARY KEY" in upper
            ):
                groups = _extract_paren_groups(compact)
                if groups:
                    for col in groups[0].split(","):
                        cleaned = _clean_identifier(col)
                        if cleaned and cleaned not in primary_keys:
                            primary_keys.append(cleaned)
                continue

            if "FOREIGN KEY" in upper:
                for relation in _parse_foreign_key_line(compact, table_name):
                    foreign_keys[table_name].append(relation)
                continue

            parsed_column = _parse_column_definition(compact)
            if not parsed_column:
                continue

            column_name, column_type, inline_pk, inline_fk = parsed_column
            columns.append((column_name, column_type))
            if inline_pk and column_name not in primary_keys:
                primary_keys.append(column_name)
            for relation in inline_fk:
                foreign_keys[table_name].append(relation)

        tables[table_name] = {"columns": columns, "primary_keys": primary_keys}

    return {"tables": tables, "foreign_keys": dict(foreign_keys)}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-sql", response_model=GenerateSQLResponse)
def generate_sql(payload: GenerateSQLRequest) -> GenerateSQLResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    schema_ddl = payload.schema_ddl.strip()
    if not schema_ddl:
        raise HTTPException(status_code=400, detail="Schema DDL cannot be empty.")

    _ensure_model_initialized()

    try:
        metadata = parse_schema_ddl_to_metadata(schema_ddl)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse schema DDL: {exc}") from exc

    prompt_tables = metadata_to_prompt_tables(metadata)
    if payload.use_modulo:
        result = run_modulo_loop(
            question=question,
            tables=prompt_tables,
            metadata=metadata,
            iter_limit=payload.iter_limit,
            candidates_per_iter=payload.candidates_per_iter,
        )
        if result.get("status") == "success":
            best = result["passing_queries"][0]
            return GenerateSQLResponse(
                status="success",
                sql=best["sql"],
                used_modulo=True,
                verifier_summary=best["verifiers"],
                iterations_used=int(result.get("iterations_used") or 1),
            )

        trace = result.get("iteration_trace") or []
        fallback_sql = ""
        if trace and trace[-1].get("candidates"):
            fallback_sql = trace[-1]["candidates"][0].get("sql", "")

        return GenerateSQLResponse(
            status="failure",
            sql=fallback_sql,
            used_modulo=True,
            verifier_summary=None,
            iterations_used=int(result.get("iterations_used") or payload.iter_limit),
        )

    sql = translate_query(question=question, tables=prompt_tables)
    return GenerateSQLResponse(
        status="success" if sql else "failure",
        sql=sql,
        used_modulo=False,
        verifier_summary=None,
        iterations_used=1,
    )
