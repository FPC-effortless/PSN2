"""Dependency security detector."""
from __future__ import annotations

import ast
import re
from typing import List, Optional

from security_scanner.ast_analyzer import ASTAnalyzer
from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)

# Libraries known to be deprecated or have security issues
INSECURE_LIBRARIES = {
    "md5": "Use hashlib.sha256 or stronger instead of MD5.",
    "sha": "Use hashlib.sha256 or stronger instead of SHA1.",
    "crypt": "The crypt module is deprecated. Use bcrypt or argon2 instead.",
    "telnetlib": "telnetlib is deprecated and insecure. Use paramiko (SSH) instead.",
    "ftplib": "FTP transmits credentials in plaintext. Use SFTP/FTPS instead.",
    "imaplib": "Consider using secure IMAP with TLS.",
    "poplib": "Consider using secure POP3 with TLS.",
    "xmlrpc": "XML-RPC is vulnerable to XXE. Consider REST/JSON APIs.",
}

# Functions with known security issues
INSECURE_FUNCTIONS = {
    "os.system": (SeverityLevel.HIGH, "Use subprocess with a list of arguments instead of os.system()."),
    "commands.getoutput": (SeverityLevel.HIGH, "commands module is deprecated. Use subprocess instead."),
    "commands.getstatusoutput": (SeverityLevel.HIGH, "commands module is deprecated. Use subprocess instead."),
    "subprocess.call": (SeverityLevel.MEDIUM, "Prefer subprocess.run() with shell=False and a list of arguments."),
    "subprocess.Popen": (SeverityLevel.MEDIUM, "Ensure shell=False and use a list of arguments to prevent injection."),
    "hashlib.md5": (SeverityLevel.MEDIUM, "MD5 is cryptographically broken. Use SHA-256 or stronger."),
    "hashlib.sha1": (SeverityLevel.MEDIUM, "SHA-1 is weak. Use SHA-256 or stronger."),
    "random.random": (SeverityLevel.LOW, "Use secrets module for cryptographic randomness."),
    "random.randint": (SeverityLevel.LOW, "Use secrets.randbelow() for cryptographic randomness."),
    "random.choice": (SeverityLevel.LOW, "Use secrets.choice() for cryptographic randomness."),
}

# Regex for version pinning in requirements.txt
VERSION_PINNED_RE = re.compile(r"^[A-Za-z0-9_\-\.\[\]]+\s*(==|>=|~=|<=|!=|>|<)")
COMMENT_OR_BLANK_RE = re.compile(r"^\s*(#.*)?$")


class DependencyDetector(BaseDetector):
    """Detects insecure dependencies and usage of deprecated/insecure functions."""

    def __init__(self, config: Optional[DetectorConfig] = None):
        super().__init__(config)

    def detect(self, file_info: FileInfo, ast_tree: Optional[ast.AST] = None) -> List[Vulnerability]:
        if not self.is_enabled():
            return []

        vulns: List[Vulnerability] = []

        if file_info.file_type == FileType.REQUIREMENTS:
            vulns.extend(self._check_requirements_file(file_info))
        elif ast_tree is not None:
            vulns.extend(self._check_python_file(file_info, ast_tree))

        return vulns

    def _check_requirements_file(self, file_info: FileInfo) -> List[Vulnerability]:
        vulns = []
        try:
            with open(file_info.path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except (OSError, IOError):
            return []

        for line_num, line in enumerate(lines, start=1):
            stripped = line.strip()
            if COMMENT_OR_BLANK_RE.match(stripped):
                continue
            # Skip options like -r, --index-url, etc.
            if stripped.startswith("-"):
                continue
            if not VERSION_PINNED_RE.match(stripped):
                vuln = Vulnerability(
                    title=f"Unpinned dependency: {stripped.split()[0] if stripped.split() else stripped}",
                    description=f"Dependency '{stripped}' lacks a pinned version specifier.",
                    severity=SeverityLevel.LOW,
                    file_path=file_info.path,
                    line_number=line_num,
                    column=0,
                    code_snippet=stripped,
                    recommendation=(
                        "Pin dependency versions (e.g., package==1.2.3) to ensure reproducible builds "
                        "and prevent supply chain attacks."
                    ),
                    vulnerability_type=VulnerabilityType.DEPENDENCY,
                    confidence=0.9,
                )
                if not self.should_suppress(vuln):
                    vulns.append(vuln)
        return vulns

    def _check_python_file(self, file_info: FileInfo, ast_tree: ast.AST) -> List[Vulnerability]:
        vulns = []
        for node in ast.walk(ast_tree):
            # Check imports of insecure libraries
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module = alias.name.split(".")[0]
                        if module in INSECURE_LIBRARIES:
                            snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                            vuln = Vulnerability(
                                title=f"Import of deprecated/insecure library: {module}",
                                description=f"The library '{module}' is deprecated or has known security issues.",
                                severity=SeverityLevel.MEDIUM,
                                file_path=file_info.path,
                                line_number=node.lineno,
                                column=node.col_offset,
                                code_snippet=snippet,
                                recommendation=INSECURE_LIBRARIES[module],
                                vulnerability_type=VulnerabilityType.DEPENDENCY,
                                confidence=0.85,
                            )
                            if not self.should_suppress(vuln):
                                vulns.append(vuln)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module = node.module.split(".")[0]
                    if module in INSECURE_LIBRARIES:
                        snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                        vuln = Vulnerability(
                            title=f"Import of deprecated/insecure library: {module}",
                            description=f"The library '{module}' is deprecated or has known security issues.",
                            severity=SeverityLevel.MEDIUM,
                            file_path=file_info.path,
                            line_number=node.lineno,
                            column=node.col_offset,
                            code_snippet=snippet,
                            recommendation=INSECURE_LIBRARIES[module],
                            vulnerability_type=VulnerabilityType.DEPENDENCY,
                            confidence=0.85,
                        )
                        if not self.should_suppress(vuln):
                            vulns.append(vuln)

            # Check calls to insecure functions
            if isinstance(node, ast.Call):
                name = ASTAnalyzer.get_function_name(node)
                if name and name in INSECURE_FUNCTIONS:
                    severity, recommendation = INSECURE_FUNCTIONS[name]
                    snippet = ASTAnalyzer.get_code_snippet(file_info.path, node.lineno)
                    vuln = Vulnerability(
                        title=f"Use of insecure function: {name}()",
                        description=f"The function '{name}()' has known security issues.",
                        severity=severity,
                        file_path=file_info.path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=snippet,
                        recommendation=recommendation,
                        vulnerability_type=VulnerabilityType.DEPENDENCY,
                        confidence=0.8,
                    )
                    if not self.should_suppress(vuln):
                        vulns.append(vuln)
        return vulns

    def detect_in_requirements(self, content: str, file_path: str = "requirements.txt") -> List[Vulnerability]:
        """Convenience method to detect in requirements.txt content."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(content)
            tmp_path = f.name
        try:
            file_info = FileInfo(path=tmp_path, file_type=FileType.REQUIREMENTS)
            vulns = self._check_requirements_file(file_info)
            for v in vulns:
                v.file_path = file_path
            return vulns
        finally:
            os.unlink(tmp_path)
