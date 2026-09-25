"""Unified error model and error codes for ZhiDa.

Error code convention (see docs/architecture.md):
  0xxx  OK
  1xxx  input / validation
  2xxx  ingestion / index
  3xxx  retrieval / generation
  4xxx  configuration
"""
from enum import IntEnum
from typing import Any, Dict


class ZDCode(IntEnum):
    OK = 0
    E_INPUT = 1001
    E_LOAD = 2001
    E_EMBED = 2002
    E_INDEX = 2003
    E_RETRIEVE = 3001
    E_GEN = 3002
    E_CFG = 4001


class ZDError(Exception):
    """Structured, machine-readable error."""

    def __init__(self, code: ZDCode, msg: str, detail: Any = None):
        self.code = int(code)
        self.msg = msg
        self.detail = detail
        super().__init__(f"[{self.code}] {msg}")

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "msg": self.msg, "detail": self.detail}
