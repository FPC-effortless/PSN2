"""Path traversal vulnerability detector."""
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

FILE_OPEN_FUNCS = {"open"}
PATH_JOIN_FUNCS = {"os.path.join", "os.path.abspath", "os.path.realpath"}
PATH_COPY_FUNCS = {"shutil.copy", "shutil.copy2", "shutil.move", "shutil.copytree"}
PATH_CLASSES = {"Path", "PurePath", "PosixPath", "WindowsPath"}


class PathTraversalDetector(BaseDetector):
    """Detects path traversal vulnerabilities in file operations."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)

    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        if not self.is_enabled() or ast_tree is None:
            return []
        vulns: List[Vulnerability] = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                name = ASTAnalyzer.get_function_name(node)
                if not name:
                    continue

                # open() with user-controlled path
                if name in FILE_OPEN_FUNCS and node.args:
                    path_arg = node.args[0]
                    if self._is_unsafe_path(path_arg):
                        vulns.append(self._make_vuln(
                            file_info, node,
                            "Unsafe open() with user-controlled path",
                            f"open() called with a potentially user-controlled path argument.",
                            SeverityLevel.HIGH,
                            "Validate and sanitize file paths. Use pathlib.Path.resolve() and check the result is within an allowed directory.",
                        ))

                # os.path.join without validation
                if name in PATH_JOIN_FUNCS and node.args:
                    for arg in node.args[1:]:  # skip base path
                        if ASTAnalyzer.is_user_controlled(arg):
                            vulns.append(self._make_vuln(
                                file_info, node,
                                f"Unsafe path construction with {name}()",
                                f"{name}() called with user-controlled path component without validation.",
                                SeverityLevel.MEDIUM,
                                "Validate path components before joining. Use pathlib.Path.resolve() to prevent traversal.",
                            ))
                            break

                # shutil operations with user-controlled paths
                if name in PATH_COPY_FUNCS and node.args:
                    for arg in node.args[:2]:
                        if ASTAnalyzer.is_user_controlled(arg):
                            vulns.append(self._make_vuln(
                                file_info, node,
                                f"Unsafe file operation: {name}()",
                                f"{name}() called with a potentially user-controlled path.",
                                SeverityLevel.HIGH,
                                "Validate all path arguments before file operations.",
                            ))
                            break

            # Detect string concatenation used as a path (BinOp with + involving path-like names)
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
                if self._looks_like_path_concat(node):
                    vulns.append(self._make_vuln(
                        file_info, node,
                        "Path construction via string concatenation",
                        "File path constructed using string concatenation, which may allow path traversal.",
                        SeverityLevel.HIGH,
                        "Use pathlib.Path for path construction and call .resolve() to prevent traversal.",
                    ))

        # Deduplicate by (line_number, title)
        seen = set()
        unique = []
        for v in vulns:
            key = (v.line_number, v.title)
            if key not in seen:
                seen.add(key)
                if not self.should_suppress(v):
                    unique.append(v)
        return unique

    def detect_in_code(self, source_code: str, file_path: str = "<string>") -> List[Vulnerability]:
        from security_scanner.models import FileType
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return []
        file_info = FileInfo(path=file_path, file_type=FileType.PYTHON)
        return self.detect(file_info, tree)

    def _is_unsafe_path(self, node: ast.AST) -> bool:
        """Check if a path argument is potentially unsafe."""
        if ASTAnalyzer.is_user_controlled(node):
            return True
        # String concatenation
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return True
        # f-string with variables
        if isinstance(node, ast.JoinedStr):
            return True
        return False

    def _looks_like_path_concat(self, node: ast.BinOp) -> bool:
        """Heuristic: detect path-like string concatenation."""
        def has_path_name(n: ast.AST) -> bool:
            if isinstance(n, ast.Name):
                return any(p in n.id.lower() for p in ["path", "dir", "file", "folder", "base"])
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                return "/" in n.value or "\\" in n.value or ".." in n.value
            return False

        return has_path_name(node.left) or has_path_name(node.right)

    def _make_vuln(self, file_info: FileInfo, node: ast.AST, title: str,
                   description: str, severity: SeverityLevel, recommendation: str) -> Vulnerability:
        line = getattr(node, "lineno", 0)
        col = getattr(node, "col_offset", 0)
        snippet = ASTAnalyzer.get_code_snippet(file_info.path, line)
        return Vulnerability(
            title=title,
            description=description,
            severity=severity,
            file_path=file_info.path,
            line_number=line,
            column=col,
            code_snippet=snippet,
            recommendation=recommendation,
            vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
            confidence=0.8,
        )
