"""Supported-input refusal shared by the runner modules."""

from __future__ import annotations


class Refusal(Exception):
    """A supported-input refusal, not a programming error."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(message)
