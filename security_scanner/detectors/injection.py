"""Code injection vulnerability detector."""
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


class InjectionDetector(BaseDetector):
    """Detects code injection vulnerabilities: eval, exec, compile, __import__, pickle.loads."""

    DANGEROUS_FUNCTIONS = {
        "eval": SeverityLevel.HIGH,
        "exec": SeverityLevel.HIGH,
        "compile": SeverityLevel.HIGH,
        "__import__": SeverityLevel.MEDIUM,
        "pickle.loads": SeverityLevel.CRITICAL,
        "pickle.load": SeverityLevel.CRITICAL,
    }

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)

    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        if not self.is_enabled() or ast_tree is None:
            return []
        vulns: List[Vulnerability] = []
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                name = ASTAnalyzer.get_function_name(node)
                if name and name in self.DANGEROUS_FUNCTIONS:
                    severity = self.DANGEROUS_FUNCTIONS[name]
                    snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                    vuln = Vulnerability(
                        title=f"Dangerous function call: {name}()",
                        description=(
                            f"Use of `{name}()` can allow arbitrary code execution "
                            "if the input is user-controlled."
                        ),
                        severity=severity,
                        file_path=file_info.path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        recommendation=self._get_recommendation(name),
                        vulnerability_type=VulnerabilityType.CODE_INJECTION,
                        confidence=0.9,
                    )
                    if not self.should_suppress(vuln):
                        vulns.append(vuln)
        return vulns

    def detect_in_code(self, source_code: str, file_path: str = "<string>") -> List[Vulnerability]:
        """Convenience method to detect in a source string."""
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        file_info = FileInfo(path=file_path, file_type=__import__("security_scanner.models", fromlist=["FileType"]).FileType.PYTHON)
        return self.detect(file_info, tree)

    @staticmethod
    def _get_recommendation(func_name: str) -> str:
        recs = {
            "eval": "Avoid eval(). Use ast.literal_eval() for safe expression parsing.",
            "exec": "Avoid exec(). Refactor to use explicit function calls.",
            "compile": "Avoid compile() with user input. Validate and sanitize all inputs.",
            "__import__": "Avoid dynamic imports with __import__(). Use importlib with a whitelist.",
            "pickle.loads": "Never unpickle untrusted data. Use json or a safe serialization format.",
            "pickle.load": "Never unpickle untrusted data. Use json or a safe serialization format.",
        }
        return recs.get(func_name, "Avoid using this function with user-controlled input.")
