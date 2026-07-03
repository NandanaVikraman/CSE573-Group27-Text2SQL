import re
from typing import Any, Callable

from .syntax_verifier import verify_syntax

VERIFIER_NAMES: set[str] = {"syntax", "schema", "judge"}
DEFAULT_VERIFIERS: set[str] = {"schema", "judge"}


def syntax_verifier(sql: str) -> dict[str, Any]:
    # Wrap verify_syntax into the standard verifier-dict shape 
    result = verify_syntax(sql)
    return {
        "name": "syntax",
        "passed": result.passed,
        "confidence": 1.0 if result.passed else 0.0,
        "rationale": result.error or "Syntax parse OK.",
        "details": {"error": result.error},
    }


_SQL_KEYWORDS = {
    "select",
    "from",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "on",
    "group",
    "by",
    "order",
    "having",
    "limit",
    "union",
    "intersect",
    "except",
    "insert",
    "into",
    "update",
    "delete",
    "as",
    "and",
    "or",
    "not",
    "in",
    "like",
    "count",
    "sum",
    "avg",
    "min",
    "max",
    "distinct",
}


def _normalize_identifier(identifier: str) -> str:
    cleaned = identifier.strip().strip('`"[]')
    return cleaned.lower()


def _extract_table_references(sql: str) -> set[str]:
    pattern = re.compile(
        r"\b(?:from|join|update|into)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        re.IGNORECASE,
    )
    return {_normalize_identifier(match) for match in pattern.findall(sql)}


def _extract_column_references(sql: str) -> tuple[set[tuple[str, str]], set[str]]:
    dotted_pattern = re.compile(
        r"\b([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)\b",
        re.IGNORECASE,
    )
    dotted = {
        (_normalize_identifier(table), _normalize_identifier(column))
        for table, column in dotted_pattern.findall(sql)
    }

    token_pattern = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b")
    all_tokens = {_normalize_identifier(token) for token in token_pattern.findall(sql)}
    bare = {
        token
        for token in all_tokens
        if token not in _SQL_KEYWORDS and not token.isdigit()
    }

    for table_name, column_name in dotted:
        bare.discard(table_name)
        bare.discard(column_name)

    return dotted, bare


def schema_consistency_verifier(sql: str, metadata: dict[str, Any]) -> dict[str, Any]:
    metadata_tables = metadata.get("tables", {}) if isinstance(metadata, dict) else {}
    known_tables = {_normalize_identifier(table_name) for table_name in metadata_tables}

    known_columns_by_table: dict[str, set[str]] = {}
    known_columns_global: set[str] = set()
    for table_name, table_info in metadata_tables.items():
        table_key = _normalize_identifier(table_name)
        columns = table_info.get("columns", []) if isinstance(table_info, dict) else []
        table_columns = set()
        for column_entry in columns:
            if isinstance(column_entry, (list, tuple)) and len(column_entry) > 0:
                column_name = str(column_entry[0])
            else:
                column_name = str(column_entry)
            column_key = _normalize_identifier(column_name)
            table_columns.add(column_key)
            known_columns_global.add(column_key)
        known_columns_by_table[table_key] = table_columns

    referenced_tables = _extract_table_references(sql)
    dotted_columns, bare_columns = _extract_column_references(sql)

    unknown_tables = sorted(table for table in referenced_tables if table not in known_tables)

    unknown_dotted_columns = []
    for table_name, column_name in dotted_columns:
        if table_name not in known_columns_by_table:
            unknown_dotted_columns.append(f"{table_name}.{column_name}")
            continue
        if column_name not in known_columns_by_table[table_name]:
            unknown_dotted_columns.append(f"{table_name}.{column_name}")

    ignored_bare_tokens = set(known_tables) | set(referenced_tables)
    unknown_bare_columns = sorted(
        column
        for column in bare_columns
        if column not in known_columns_global and column not in ignored_bare_tokens
    )

    passed = not unknown_tables and not unknown_dotted_columns and not unknown_bare_columns

    return {
        "name": "schema_consistency",
        "passed": passed,
        "confidence": 1.0 if passed else 0.0,
        "rationale": "Schema validation completed.",
        "details": {
            "unknown_tables": unknown_tables,
            "unknown_dotted_columns": sorted(unknown_dotted_columns),
            "unknown_bare_columns": unknown_bare_columns,
        },
    }


def llm_as_judge_verifier(
    question: str,
    sql: str,
    metadata: dict[str, Any],
    judge_callable: Callable[[str, str, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    verdict = judge_callable(question, sql, metadata)
    return {
        "name": "llm_judge",
        "passed": bool(verdict.get("passed", False)),
        "confidence": float(verdict.get("confidence", 0.0)),
        "rationale": str(verdict.get("rationale", "No rationale provided.")),
        "details": {"raw": verdict.get("raw")},
    }


def run_external_verifiers(
    question: str,
    sql: str,
    metadata: dict[str, Any],
    judge_callable: Callable[[str, str, dict[str, Any]], dict[str, Any]],
    enabled_verifiers: set[str] | None = None,
) -> dict[str, Any]:
    """
    Runs the specified verifiers and returns their results.

    enabled_verifiers: Set of verifiers to run (subset of {"syntax", "schema", "judge"}).
    Defaults to {"schema", "judge"} if None. Only enabled verifiers are executed.
    Returns a dict with results and 'all_passed' (AND of all enabled verifier passes).
    """
    if enabled_verifiers is None:
        enabled_verifiers = DEFAULT_VERIFIERS

    result: dict[str, Any] = {}

    if "syntax" in enabled_verifiers:
        result["syntax"] = syntax_verifier(sql)

    if "schema" in enabled_verifiers:
        result["schema_consistency"] = schema_consistency_verifier(sql=sql, metadata=metadata)

    if "judge" in enabled_verifiers:
        result["llm_judge"] = llm_as_judge_verifier(
            question=question,
            sql=sql,
            metadata=metadata,
            judge_callable=judge_callable,
        )

    enabled_passes = [v.get("passed", False) for v in result.values()]
    result["all_passed"] = all(enabled_passes) if enabled_passes else True

    return result
