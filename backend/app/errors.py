from __future__ import annotations

from typing import Any

from fastapi import HTTPException


def build_error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
        }
    }


def api_error(status: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
    raise HTTPException(status_code=status, detail=build_error(code=code, message=message, details=details))
