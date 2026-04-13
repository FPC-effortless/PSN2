"""Checkpoint security detector — detects unsafe model checkpoint loading."""
from __future__ import annotations

import ast
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


class CheckpointDetector(BaseDetector):
    """Detects insecure model checkpoint loading patterns."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)

    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        if not self.is_enabled() or ast_tree is None:
            return []
        vulns: List[Vulnerability] = []

        for node in ast.walk(ast_tree):
            if not isinstance(node, ast.Call):
                continue
            name = ASTAnalyzer.get_function_name(node)
            if not name:
                continue

            # torch.load() without weights_only=True
            if name in {"torch.load", "torch.jit.load"}:
                has_weights_only = any(
                    kw.arg == "weights_only"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value is True
                    for kw in node.keywords
                )
                if not has_weights_only:
                    snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                    vuln = Vulnerability(
                        title=f"Unsafe checkpoint loading: {name}() without weights_only=True",
                        description=(
                            f"{name}() without `weights_only=True` can execute arbitrary code "
                            "embedded in a malicious checkpoint file."
                        ),
                        severity=SeverityLevel.HIGH,
                        file_path=file_info.path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        recommendation=(
                            "Use torch.load(path, weights_only=True) to safely load checkpoints. "
                            "Implement checkpoint signature verification for production use."
                        ),
                        vulnerability_type=VulnerabilityType.CHECKPOINT_SECURITY,
                        cwe_id="CWE-502",
                        confidence=0.95,
                    )
                    if not self.should_suppress(vuln):
                        vulns.append(vuln)

            # Checkpoint loaded from user-specified path without validation
            if name in {"torch.load", "torch.jit.load", "load_checkpoint", "load_model"}:
                if node.args and ASTAnalyzer.is_user_controlled(node.args[0]):
                    snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                    vuln = Vulnerability(
                        title=f"Checkpoint loaded from user-specified path: {name}()",
                        description=(
                            f"{name}() is called with a user-controlled path argument, "
                            "which may allow loading of malicious checkpoints."
                        ),
                        severity=SeverityLevel.HIGH,
                        file_path=file_info.path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        recommendation=(
                            "Validate checkpoint paths against an allowlist of trusted directories. "
                            "Verify checkpoint integrity with a cryptographic signature before loading."
                        ),
                        vulnerability_type=VulnerabilityType.CHECKPOINT_SECURITY,
                        cwe_id="CWE-73",
                        confidence=0.8,
                    )
                    if not self.should_suppress(vuln):
                        vulns.append(vuln)

        return vulns

    def detect_in_code(self, source_code: str, file_path: str = "<string>") -> List[Vulnerability]:
        from security_scanner.models import FileType
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        file_info = FileInfo(path=file_path, file_type=FileType.PYTHON)
        return self.detect(file_info, tree)
