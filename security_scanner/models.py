"""Core data models for the security scanner."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SeverityLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    def __lt__(self, other: "SeverityLevel") -> bool:
        order = [SeverityLevel.INFO, SeverityLevel.LOW, SeverityLevel.MEDIUM,
                 SeverityLevel.HIGH, SeverityLevel.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "SeverityLevel") -> bool:
        return self == other or self < other

    def __gt__(self, other: "SeverityLevel") -> bool:
        return not self <= other

    def __ge__(self, other: "SeverityLevel") -> bool:
        return not self < other


class VulnerabilityType(str, Enum):
    CODE_INJECTION = "code_injection"
    SECRETS_EXPOSURE = "secrets_exposure"
    PATH_TRAVERSAL = "path_traversal"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    INPUT_VALIDATION = "input_validation"
    FILE_OPERATIONS = "file_operations"
    CONFIGURATION = "configuration"
    CHECKPOINT_SECURITY = "checkpoint_security"
    DATA_PIPELINE = "data_pipeline"
    DEPENDENCY = "dependency"


class FileType(str, Enum):
    PYTHON = "python"
    NOTEBOOK = "notebook"
    CONFIG_JSON = "config_json"
    CONFIG_YAML = "config_yaml"
    REQUIREMENTS = "requirements"


@dataclass
class Vulnerability:
    title: str
    description: str
    severity: SeverityLevel
    file_path: str
    line_number: int
    column: int
    code_snippet: str
    recommendation: str
    vulnerability_type: Optional[VulnerabilityType] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8].upper())
    cwe_id: Optional[str] = None
    confidence: float = 1.0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def vuln_type(self) -> Optional[VulnerabilityType]:
        return self.vulnerability_type


@dataclass
class FileInfo:
    path: str
    file_type: FileType
    last_modified: float = 0.0
    size: int = 0


@dataclass
class ScanConfig:
    target_path: str = "."
    exclude_patterns: List[str] = field(default_factory=list)
    min_severity: SeverityLevel = SeverityLevel.INFO
    incremental: bool = False
    cache_dir: Optional[str] = None
    suppression_rules: Dict[str, Any] = field(default_factory=dict)
    fail_on: List[SeverityLevel] = field(default_factory=lambda: [SeverityLevel.CRITICAL, SeverityLevel.HIGH])
    fail_on_severity: List[SeverityLevel] = field(default_factory=list)  # alias for fail_on


@dataclass
class ScanResult:
    vulnerabilities: List[Vulnerability] = field(default_factory=list)
    files_scanned: int = 0
    scan_duration: float = 0.0
    timestamp: str = ""
    target_path: str = ""


@dataclass
class DetectorConfig:
    enabled: bool = True
    severity_overrides: Dict[str, SeverityLevel] = field(default_factory=dict)
    custom_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    exclusions: List[str] = field(default_factory=list)  # alias for exclude_patterns
