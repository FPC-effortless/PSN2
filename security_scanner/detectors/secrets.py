"""Secrets and credentials exposure detector."""
from __future__ import annotations

import ast
import math
import re
from typing import List, Optional

from security_scanner.ast_analyzer import ASTAnalyzer
from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)

# Regex patterns for credential detection
CREDENTIAL_PATTERNS = {
    "aws_access_key": (r"AKIA[0-9A-Z]{16}", SeverityLevel.CRITICAL),
    "private_key": (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", SeverityLevel.CRITICAL),
    "generic_api_key": (r"['\"]([A-Za-z0-9_\-]{32,})['\"]", SeverityLevel.HIGH),
    "password_assignment": (r"(?i)(password|passwd|pwd)\s*=\s*['\"][^'\"]{4,}['\"]", SeverityLevel.HIGH),
    "token_assignment": (r"(?i)(token|secret|api_key|apikey)\s*=\s*['\"][^'\"]{8,}['\"]", SeverityLevel.HIGH),
}

# Variable names that suggest credential storage
SENSITIVE_VAR_NAMES = {
    "password", "passwd", "pwd", "secret", "api_key", "apikey",
    "token", "auth_token", "access_token", "private_key", "credentials",
}


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: dict = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


class SecretsDetector(BaseDetector):
    """Detects hardcoded credentials, API keys, and sensitive data."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)

    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        if not self.is_enabled():
            return []
        vulns: List[Vulnerability] = []

        # Read raw source for regex-based detection
        try:
            with open(file_info.path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (OSError, IOError):
            source = ""

        vulns.extend(self._detect_patterns(source, file_info))

        # AST-based detection for high-entropy strings and sensitive assignments
        if ast_tree is not None:
            vulns.extend(self._detect_ast(ast_tree, file_info))

        return vulns

    def detect_in_code(self, source_code: str, file_path: str = "<string>") -> List[Vulnerability]:
        """Convenience method to detect in a source string."""
        from security_scanner.models import FileType
        file_info = FileInfo(path=file_path, file_type=FileType.PYTHON)
        vulns = self._detect_patterns(source_code, file_info)
        try:
            tree = ast.parse(source_code)
            vulns.extend(self._detect_ast(tree, file_info))
        except SyntaxError:
            pass
        return vulns

    def _detect_patterns(self, source: str, file_info: FileInfo) -> List[Vulnerability]:
        vulns = []
        lines = source.splitlines()
        for pattern_name, (pattern, severity) in CREDENTIAL_PATTERNS.items():
            for match in re.finditer(pattern, source, re.MULTILINE):
                line_num = source[: match.start()].count("\n") + 1
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""
                # Skip obvious test/placeholder values
                matched_text = match.group(0)
                if any(p in matched_text.lower() for p in ["example", "placeholder", "your_", "xxx", "test"]):
                    continue
                vuln = Vulnerability(
                    title=f"Hardcoded credential detected: {pattern_name}",
                    description=f"A potential hardcoded credential was found matching pattern '{pattern_name}'.",
                    severity=severity,
                    file_path=file_info.path,
                    line_number=line_num,
                    column=match.start() - source.rfind("\n", 0, match.start()) - 1,
                    code_snippet=snippet,
                    recommendation="Remove hardcoded credentials. Use environment variables or a secrets manager.",
                    vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                    confidence=0.85,
                )
                if not self.should_suppress(vuln):
                    vulns.append(vuln)
        return vulns

    def _detect_ast(self, tree: ast.AST, file_info: FileInfo) -> List[Vulnerability]:
        vulns = []
        for node in ast.walk(tree):
            # Check assignments: var_name = "some_string"
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(s in var_name for s in SENSITIVE_VAR_NAMES):
                            val = ASTAnalyzer.get_string_value(node.value)
                            if val and len(val) >= 4:
                                snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                                vuln = Vulnerability(
                                    title=f"Sensitive variable assignment: {target.id}",
                                    description=f"Variable '{target.id}' appears to store a credential with a hardcoded value.",
                                    severity=SeverityLevel.HIGH,
                                    file_path=file_info.path,
                                    line_number=node.lineno,
                                    column=node.col_offset,
                                    code_snippet=snippet,
                                    recommendation="Use environment variables or a secrets manager instead of hardcoded values.",
                                    vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                                    confidence=0.8,
                                )
                                if not self.should_suppress(vuln):
                                    vulns.append(vuln)

            # High-entropy string detection
            if isinstance(node, (ast.Constant, ast.Str)):
                val = ASTAnalyzer.get_string_value(node)
                if val and len(val) > 16 and _shannon_entropy(val) > 4.5:
                    line_num = getattr(node, "lineno", 0)
                    snippet = ASTAnalyzer.get_code_snippet(file_info.path, line_num)
                    vuln = Vulnerability(
                        title="High-entropy string (potential secret)",
                        description="A high-entropy string was detected that may be a hardcoded secret or key.",
                        severity=SeverityLevel.MEDIUM,
                        file_path=file_info.path,
                        line_number=line_num,
                        column=getattr(node, "col_offset", 0),
                        code_snippet=snippet,
                        recommendation="Verify this is not a hardcoded secret. Use environment variables for credentials.",
                        vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
                        confidence=0.6,
                    )
                    if not self.should_suppress(vuln):
                        vulns.append(vuln)
        return vulns
