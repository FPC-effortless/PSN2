"""False positive filter for the security scanner."""
from __future__ import annotations

import fnmatch
import json
import logging
from typing import Any, Dict, List, Optional

from security_scanner.models import Vulnerability

logger = logging.getLogger(__name__)

# Comments that indicate intentional/safe usage
SECURITY_COMMENT_KEYWORDS = {
    "nosec", "noqa", "safe", "trusted", "validated", "sanitized",
    "security-ok", "intentional", "allowlisted",
}

# Wrappers that indicate safe usage of dangerous functions
SAFE_WRAPPERS = {
    "ast.literal_eval",
    "json.loads",
    "json.load",
    "yaml.safe_load",
    "yaml.safe_dump",
}


class FalsePositiveFilter:
    """Filters out false positives from vulnerability lists."""

    def __init__(self, suppression_config: Optional[Dict[str, Any]] = None):
        self.suppression_rules: List[Dict[str, Any]] = []
        self.global_suppressions: Dict[str, List[str]] = {}

        if suppression_config:
            self.suppression_rules = suppression_config.get("suppressions", [])
            self.global_suppressions = suppression_config.get("global_suppressions", {})

    @classmethod
    def from_file(cls, config_path: str) -> "FalsePositiveFilter":
        """Load suppression config from a JSON file."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return cls(config)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load suppression config from %s: %s", config_path, e)
            return cls()

    def filter(self, vulnerabilities: List[Vulnerability]) -> List[Vulnerability]:
        """Filter out false positives from the vulnerability list."""
        return [v for v in vulnerabilities if not self._is_false_positive(v)]

    def _is_false_positive(self, vuln: Vulnerability) -> bool:
        return (
            self.is_suppressed(vuln)
            or self.has_security_comment(vuln)
            or self.has_safe_wrapper(vuln)
        )

    def is_suppressed(self, vuln: Vulnerability) -> bool:
        """Check if vulnerability matches a suppression rule."""
        # Check specific suppressions
        for rule in self.suppression_rules:
            rule_id = rule.get("id", "")
            rule_file = rule.get("file", "")
            rule_line = rule.get("line")

            id_match = not rule_id or rule_id == vuln.id
            file_match = not rule_file or rule_file in vuln.file_path or fnmatch.fnmatch(vuln.file_path, rule_file)
            line_match = rule_line is None or rule_line == vuln.line_number

            if id_match and file_match and line_match:
                return True

        # Check global suppressions (file pattern → list of vuln IDs)
        for file_pattern, suppressed_ids in self.global_suppressions.items():
            if fnmatch.fnmatch(vuln.file_path, file_pattern):
                if not suppressed_ids or vuln.id in suppressed_ids:
                    return True

        return False

    def has_security_comment(self, vuln: Vulnerability) -> bool:
        """Check if the code snippet contains a security-related comment."""
        snippet_lower = vuln.code_snippet.lower()
        return any(kw in snippet_lower for kw in SECURITY_COMMENT_KEYWORDS)

    def has_safe_wrapper(self, vuln: Vulnerability) -> bool:
        """Check if the code snippet uses a safe wrapper around a dangerous function."""
        snippet = vuln.code_snippet
        return any(wrapper in snippet for wrapper in SAFE_WRAPPERS)
