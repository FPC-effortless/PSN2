"""Unit tests for SecretsDetector

Tests the secrets detector's ability to identify:
- AWS access keys
- Generic API keys
- Private keys
- Password assignments
- High-entropy strings
- Secret variable names
"""

import ast
import pytest

from security_scanner.detectors.secrets import SecretsDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    VulnerabilityType,
)


@pytest.fixture
def detector():
    """Create a SecretsDetector instance for testing"""
    config = DetectorConfig(enabled=True)
    return SecretsDetector(config)


@pytest.fixture
def file_info():
    """Create a FileInfo instance for testing"""
    return FileInfo(
        path="test_file.py",
        file_type=FileType.PYTHON,
        last_modified=0.0,
        size=100,
    )


def test_aws_access_key_detection(detector, file_info):
    """Test detection of AWS access keys"""
    code = '''
aws_key = "AKIAIOSFODNN7EXAMPLE"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) > 0
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE
    # Severity is reduced from CRITICAL to HIGH for test files by BaseDetector
    assert vuln.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    assert "AWS" in vuln.title


def test_private_key_detection(detector, file_info):
    """Test detection of private keys"""
    code = '''
key = "-----BEGIN PRIVATE KEY-----\\nMIIEvQIBADANBg..."
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) > 0
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE
    # Severity is reduced from CRITICAL to HIGH for test files by BaseDetector
    assert vuln.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    assert "Private Key" in vuln.title


def test_password_assignment_detection(detector, file_info):
    """Test detection of hardcoded passwords"""
    code = '''
password = "super_secret_password_123"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) > 0
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE
    # Severity is reduced from HIGH to MEDIUM for test files by BaseDetector
    assert vuln.severity in [SeverityLevel.HIGH, SeverityLevel.MEDIUM]
    assert "password" in vuln.title.lower()


def test_high_entropy_string_detection(detector, file_info):
    """Test detection of high-entropy strings"""
    code = '''
token = "aB3dE5fG7hI9jK1lM2nO4pQ6rS8tU0vW"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect either as high-entropy or as a token variable
    assert len(vulnerabilities) > 0
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE


def test_secret_variable_names(detector, file_info):
    """Test detection of secret-related variable names"""
    code = '''
api_key = "my_secret_api_key_value"
auth_token = "token_value_here"
secret_key = "secret123"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect at least 2 (api_key might be filtered if it matches generic_api_key pattern)
    assert len(vulnerabilities) >= 2
    for vuln in vulnerabilities:
        assert vuln.vulnerability_type == VulnerabilityType.SECRETS_EXPOSURE


def test_shannon_entropy_calculation(detector):
    """Test Shannon entropy calculation"""
    # Low entropy (repeated characters)
    low_entropy = detector._calculate_shannon_entropy("aaaaaaaaaa")
    assert low_entropy < 1.0
    
    # High entropy (random-looking string)
    high_entropy = detector._calculate_shannon_entropy("aB3dE5fG7hI9jK1lM2nO4pQ6rS8tU0vW")
    assert high_entropy > 4.0
    
    # Medium entropy (natural language)
    medium_entropy = detector._calculate_shannon_entropy("hello world this is a test")
    assert 2.0 < medium_entropy < 5.0


def test_placeholder_detection(detector):
    """Test placeholder value detection"""
    assert detector._is_placeholder("your_api_key_here")
    assert detector._is_placeholder("example_password")
    assert detector._is_placeholder("test_value")
    assert detector._is_placeholder("<your-key>")
    assert not detector._is_placeholder("AKIAIOSFODNN7EXAMPLE")


def test_no_false_positives_on_short_strings(detector, file_info):
    """Test that short strings don't trigger false positives"""
    code = '''
x = "abc"
y = "test"
z = "ok"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect short strings as secrets
    assert len(vulnerabilities) == 0


def test_no_false_positives_on_placeholders(detector, file_info):
    """Test that placeholder values don't trigger false positives"""
    code = '''
api_key = "your_api_key_here"
password = "changeme"
token = "example_token"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect placeholders as real secrets
    assert len(vulnerabilities) == 0


def test_confidence_scores(detector, file_info):
    """Test that confidence scores are appropriate"""
    code = '''
aws_key = "AKIAIOSFODNN7EXAMPLE"
maybe_secret = "aB3dE5fG7hI9jK1lM2nO4pQ6rS8tU0vW"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # AWS key should have high confidence
    aws_vuln = [v for v in vulnerabilities if "AWS" in v.title][0]
    assert aws_vuln.confidence >= 0.8
    
    # High-entropy string should have lower confidence
    entropy_vulns = [v for v in vulnerabilities if "Entropy" in v.title]
    if entropy_vulns:
        assert entropy_vulns[0].confidence < 0.8


def test_cwe_id_assignment(detector, file_info):
    """Test that CWE IDs are correctly assigned"""
    code = '''
password = "hardcoded_password"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) > 0
    assert vulnerabilities[0].cwe_id == "CWE-798"


def test_multiple_secrets_in_file(detector, file_info):
    """Test detection of multiple secrets in one file"""
    code = '''
aws_key = "AKIAIOSFODNN7EXAMPLE"
password = "my_password_123"
api_key = "sk_live_1234567890abcdef"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect at least 2 secrets (some may be filtered by pattern priority)
    assert len(vulnerabilities) >= 2


def test_disabled_detector(file_info):
    """Test that disabled detector returns no vulnerabilities"""
    config = DetectorConfig(enabled=False)
    detector = SecretsDetector(config)
    
    code = '''
aws_key = "AKIAIOSFODNN7EXAMPLE"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
