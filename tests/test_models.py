"""Tests for core data models"""

import pytest
from security_scanner.models import (
    SeverityLevel,
    VulnerabilityType,
    FileType,
    Vulnerability,
    FileInfo,
    ScanResult,
    ScanConfig,
    DetectorConfig,
)


def test_severity_level_enum():
    """Test SeverityLevel enum values"""
    assert SeverityLevel.CRITICAL.value == "critical"
    assert SeverityLevel.HIGH.value == "high"
    assert SeverityLevel.MEDIUM.value == "medium"
    assert SeverityLevel.LOW.value == "low"
    assert SeverityLevel.INFO.value == "info"


def test_vulnerability_type_enum():
    """Test VulnerabilityType enum values"""
    assert VulnerabilityType.CODE_INJECTION.value == "code_injection"
    assert VulnerabilityType.SECRETS_EXPOSURE.value == "secrets_exposure"
    assert VulnerabilityType.PATH_TRAVERSAL.value == "path_traversal"


def test_file_type_enum():
    """Test FileType enum values"""
    assert FileType.PYTHON.value == "python"
    assert FileType.NOTEBOOK.value == "notebook"
    assert FileType.CONFIG_JSON.value == "config_json"


def test_vulnerability_creation():
    """Test Vulnerability dataclass creation"""
    vuln = Vulnerability(
        id="INJ001",
        title="Code Injection",
        description="Unsafe eval() call",
        severity=SeverityLevel.HIGH,
        file_path="test.py",
        line_number=10,
        column=5,
        code_snippet="eval(user_input)",
        recommendation="Use ast.literal_eval() instead",
    )
    
    assert vuln.id == "INJ001"
    assert vuln.severity == SeverityLevel.HIGH
    assert vuln.confidence == 1.0  # Default value
    assert vuln.context == {}  # Default empty dict


def test_file_info_creation():
    """Test FileInfo dataclass creation"""
    file_info = FileInfo(
        path="test.py",
        file_type=FileType.PYTHON,
        last_modified=1234567890.0,
        size=1024,
    )
    
    assert file_info.path == "test.py"
    assert file_info.file_type == FileType.PYTHON
    assert file_info.size == 1024


def test_scan_result_creation():
    """Test ScanResult dataclass creation"""
    vuln = Vulnerability(
        id="TEST001",
        title="Test",
        description="Test vulnerability",
        severity=SeverityLevel.LOW,
        file_path="test.py",
        line_number=1,
        column=0,
        code_snippet="test",
        recommendation="Fix it",
    )
    
    result = ScanResult(
        vulnerabilities=[vuln],
        files_scanned=5,
        scan_duration=1.5,
        timestamp="2024-01-01T00:00:00",
    )
    
    assert len(result.vulnerabilities) == 1
    assert result.files_scanned == 5
    assert result.scan_duration == 1.5


def test_scan_config_defaults():
    """Test ScanConfig default values"""
    config = ScanConfig(target_path="/path/to/scan")
    
    assert config.target_path == "/path/to/scan"
    assert config.exclude_patterns == []
    assert config.min_severity == SeverityLevel.INFO
    assert config.incremental is False
    assert config.cache_dir is None
    assert config.suppression_rules == {}


def test_detector_config_defaults():
    """Test DetectorConfig default values"""
    config = DetectorConfig()
    
    assert config.enabled is True
    assert config.severity_overrides == {}
    assert config.custom_patterns == []
    assert config.exclude_patterns == []
