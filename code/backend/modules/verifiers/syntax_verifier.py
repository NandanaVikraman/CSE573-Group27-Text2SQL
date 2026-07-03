from dataclasses import dataclass
from typing import List, Optional

import sqlglot
from sqlglot.errors import ParseError, SqlglotError

@dataclass
class SyntaxVerificationResult:
    query: str
    passed: bool
    error: Optional[str] = None

def verify_syntax(
    query: str,
    dialect: Optional[str] = None,
) -> SyntaxVerificationResult:

    if not query or not query.strip():
        return SyntaxVerificationResult(
            query=query,
            passed=False,
            error="Query is empty.",
        )

    try:
        sqlglot.parse_one(query, dialect=dialect, error_level=sqlglot.ErrorLevel.RAISE)
        return SyntaxVerificationResult(query=query, passed=True)

    except ParseError as exc:
        messages = [e.get("description", str(e)) for e in exc.errors]
        return SyntaxVerificationResult(
            query=query,
            passed=False,
            error="; ".join(messages) if messages else str(exc),
        )

    except SqlglotError as exc:
        return SyntaxVerificationResult(
            query=query,
            passed=False,
            error=f"Unexpected sqlglot error: {exc}",
        )


def verify_syntax_batch(
    queries: List[str],
    dialect: Optional[str] = None,
) -> List[SyntaxVerificationResult]:

    return [verify_syntax(q, dialect=dialect) for q in queries]
