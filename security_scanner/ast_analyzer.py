"""AST analysis helpers for the security scanner."""
from __future__ import annotations

import ast
from typing import List, Optional, Set


# Names that indicate user-controlled input
USER_INPUT_FUNCTIONS: Set[str] = {
    "input", "sys.stdin.read", "sys.stdin.readline",
    "request.args.get", "request.form.get", "request.json",
    "request.data", "request.get_json",
    "os.environ.get", "os.getenv",
    "open",  # reading from files can be user-controlled
}

USER_INPUT_ATTRS: Set[str] = {
    "args", "form", "json", "data", "params", "query_string",
    "environ", "cookies",
}

VALIDATION_FUNCTIONS: Set[str] = {
    "isinstance", "type", "hasattr", "getattr",
    "re.match", "re.fullmatch", "re.search",
    "validate", "sanitize", "escape", "quote",
    "pathlib.Path.resolve",
}


class ASTAnalyzer:
    """Helper class for AST-based code analysis."""

    @staticmethod
    def get_function_name(node: ast.Call) -> Optional[str]:
        """Extract fully qualified function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            parts = []
            current = node.func
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            return ".".join(reversed(parts))
        return None

    @staticmethod
    def is_user_controlled(node: ast.AST, tree: Optional[ast.AST] = None) -> bool:
        """Determine if a node's value likely comes from user input."""
        # Check if the node itself is a call to a user-input function
        if isinstance(node, ast.Call):
            name = ASTAnalyzer.get_function_name(node)
            if name and any(name.endswith(f) or f.endswith(name) for f in USER_INPUT_FUNCTIONS):
                return True

        # Check if it's a Name that looks like a parameter or user-input variable
        if isinstance(node, ast.Name):
            suspicious = {"user_input", "user_data", "request", "query", "param",
                          "args", "kwargs", "data", "payload", "body", "content",
                          "cmd", "command", "path", "filename", "filepath"}
            if node.id.lower() in suspicious or any(s in node.id.lower() for s in suspicious):
                return True

        # Check attribute access on request-like objects
        if isinstance(node, ast.Attribute):
            if node.attr in USER_INPUT_ATTRS:
                return True
            if isinstance(node.value, ast.Name) and node.value.id in {"request", "req", "environ"}:
                return True

        # Check for subscript on dict-like user input
        if isinstance(node, ast.Subscript):
            return ASTAnalyzer.is_user_controlled(node.value, tree)

        return False

    @staticmethod
    def get_code_snippet(file_path: str, line_number: int, context_lines: int = 3) -> str:
        """Extract code snippet with surrounding context lines."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            snippet_lines = []
            for i, line in enumerate(lines[start:end], start=start + 1):
                marker = ">>>" if i == line_number else "   "
                snippet_lines.append(f"{marker} {i:4d}: {line.rstrip()}")
            return "\n".join(snippet_lines)
        except (OSError, IOError):
            return f"<line {line_number}>"

    @staticmethod
    def has_validation(node: ast.AST, context: ast.AST) -> bool:
        """Check if there is input validation before the given node."""
        # Walk the context tree looking for validation calls
        for n in ast.walk(context):
            if isinstance(n, ast.Call):
                name = ASTAnalyzer.get_function_name(n)
                if name and any(v in name for v in VALIDATION_FUNCTIONS):
                    return True
            # Check for isinstance checks
            if isinstance(n, ast.If):
                for child in ast.walk(n.test):
                    if isinstance(child, ast.Call):
                        name = ASTAnalyzer.get_function_name(child)
                        if name in {"isinstance", "type", "hasattr"}:
                            return True
        return False

    @staticmethod
    def get_all_calls(tree: ast.AST) -> List[ast.Call]:
        """Return all Call nodes in the AST tree."""
        return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    @staticmethod
    def get_string_value(node: ast.AST) -> Optional[str]:
        """Extract string value from a Constant or Str node."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        # Python < 3.8 compatibility
        if isinstance(node, ast.Str):
            return node.s
        return None
