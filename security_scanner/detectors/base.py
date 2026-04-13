"""Base detector abstract class."""
from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    SeverityLevel,
    Vulnerability,
)


class BaseDetector(ABC):
    """Abstract base class for all vulnerability detectors."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self.name = self.__class__.__name__

    @abstractmethod
    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        """Detect vulnerabilities in the given file."""

    def get_severity(self, context: Optional[Dict[str, Any]] = None, base_severity: SeverityLevel = SeverityLevel.MEDIUM) -> SeverityLevel:
        """Determine final severity, applying any configured overrides."""
        if context is None:
            context = {}
        key = context.get("function_name", "")
        if key and key in self.config.severity_overrides:
            return self.config.severity_overrides[key]
        return base_severity

    def should_suppress(self, vuln: Vulnerability) -> bool:
        """Check if a vulnerability should be suppressed based on config."""
        for pattern in self.config.exclude_patterns:
            if pattern in vuln.file_path:
                return True
        return False

    def is_enabled(self) -> bool:
        return self.config.enabled
