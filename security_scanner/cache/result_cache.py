"""Result cache for incremental scanning."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from security_scanner.models import SeverityLevel, Vulnerability, VulnerabilityType

logger = logging.getLogger(__name__)

CACHE_VERSION = "1"


def _vuln_to_dict(v: Vulnerability) -> Dict[str, Any]:
    return {
        "id": v.id,
        "title": v.title,
        "description": v.description,
        "severity": v.severity.value,
        "file_path": v.file_path,
        "line_number": v.line_number,
        "column": v.column,
        "code_snippet": v.code_snippet,
        "recommendation": v.recommendation,
        "vulnerability_type": v.vulnerability_type.value if v.vulnerability_type else None,
        "cwe_id": v.cwe_id,
        "confidence": v.confidence,
        "context": v.context,
    }


def _vuln_from_dict(d: Dict[str, Any]) -> Vulnerability:
    vtype = None
    if d.get("vulnerability_type"):
        try:
            vtype = VulnerabilityType(d["vulnerability_type"])
        except ValueError:
            pass
    return Vulnerability(
        id=d.get("id", ""),
        title=d.get("title", ""),
        description=d.get("description", ""),
        severity=SeverityLevel(d.get("severity", "info")),
        file_path=d.get("file_path", ""),
        line_number=d.get("line_number", 0),
        column=d.get("column", 0),
        code_snippet=d.get("code_snippet", ""),
        recommendation=d.get("recommendation", ""),
        vulnerability_type=vtype,
        cwe_id=d.get("cwe_id"),
        confidence=d.get("confidence", 1.0),
        context=d.get("context", {}),
    )


class ResultCache:
    """Caches scan results keyed by file path + SHA256 hash."""

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(".security_scanner_cache")
        self.cache_file = self.cache_dir / "scan_cache.json"
        self._data: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                if raw.get("version") == CACHE_VERSION:
                    self._data = raw.get("entries", {})
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not load cache from %s: %s", self.cache_file, e)
                self._data = {}

    def _save(self) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump({"version": CACHE_VERSION, "entries": self._data}, f, indent=2)
        except (OSError, IOError) as e:
            logger.warning("Could not save cache to %s: %s", self.cache_file, e)

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        h = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except (OSError, IOError):
            return ""
        return h.hexdigest()

    def get_cached_result(self, file_path: str, file_hash: str) -> Optional[List[Vulnerability]]:
        """Return cached vulnerabilities if the file hash matches."""
        self._ensure_loaded()
        entry = self._data.get(file_path)
        if entry and entry.get("hash") == file_hash:
            try:
                return [_vuln_from_dict(v) for v in entry.get("vulnerabilities", [])]
            except Exception as e:  # noqa: BLE001
                logger.warning("Corrupted cache entry for %s: %s", file_path, e)
                return None
        return None

    def store_result(self, file_path: str, file_hash: str, vulnerabilities: List[Vulnerability]) -> None:
        """Store scan results in the cache."""
        self._ensure_loaded()
        self._data[file_path] = {
            "hash": file_hash,
            "vulnerabilities": [_vuln_to_dict(v) for v in vulnerabilities],
        }
        self._save()

    def invalidate(self, file_path: str) -> None:
        """Remove a cache entry for the given file."""
        self._ensure_loaded()
        if file_path in self._data:
            del self._data[file_path]
            self._save()

    def clear(self) -> None:
        """Clear all cache entries."""
        self._data = {}
        self._loaded = True
        self._save()
