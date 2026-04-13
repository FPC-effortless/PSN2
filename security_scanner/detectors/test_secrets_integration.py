"""Integration tests for SecretsDetector

Tests the secrets detector with realistic code examples to ensure
it works correctly in real-world scenarios.
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
        path="app/config.py",  # Not a test file
        file_type=FileType.PYTHON,
        last_modified=0.0,
        size=100,
    )


def test_realistic_aws_configuration(detector, file_info):
    """Test detection in realistic AWS configuration code"""
    code = '''
import os

# BAD: Hardcoded AWS credentials
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"

# GOOD: Load from environment
aws_key = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect the hardcoded AWS key
    assert len(vulnerabilities) >= 1
    aws_vulns = [v for v in vulnerabilities if "AWS" in v.title]
    assert len(aws_vulns) >= 1
    assert aws_vulns[0].severity == SeverityLevel.CRITICAL


def test_realistic_database_configuration(detector, file_info):
    """Test detection in database configuration"""
    code = '''
# Database configuration
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "myapp",
    "user": "admin",
    "password": "super_secret_password_123",  # BAD: Hardcoded password
}

# Better approach (but still detected)
db_password = "hardcoded_pass"  # BAD

# Good approach
import os
db_password_secure = os.getenv("DB_PASSWORD")
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect both hardcoded passwords
    assert len(vulnerabilities) >= 1
    password_vulns = [v for v in vulnerabilities if "password" in v.title.lower()]
    assert len(password_vulns) >= 1


def test_realistic_api_client(detector, file_info):
    """Test detection in API client code"""
    code = '''
import requests

class APIClient:
    def __init__(self):
        # BAD: Hardcoded API key
        self.api_key = "sk_live_51H8xyzABCDEF123456789"
        self.base_url = "https://api.example.com"
    
    def make_request(self, endpoint):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return requests.get(f"{self.base_url}/{endpoint}", headers=headers)

# GOOD: Load from environment
import os
api_key = os.getenv("API_KEY")
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect the hardcoded API key (either as pattern match or high-entropy string)
    assert len(vulnerabilities) >= 1


def test_realistic_private_key(detector, file_info):
    """Test detection of private keys in code"""
    code = '''
# BAD: Hardcoded private key
PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA1234567890abcdefghijklmnopqrstuvwxyz
-----END RSA PRIVATE KEY-----"""

# GOOD: Load from file
with open("/secure/path/private_key.pem", "r") as f:
    private_key = f.read()
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect the hardcoded private key
    assert len(vulnerabilities) >= 1
    key_vulns = [v for v in vulnerabilities if "Private Key" in v.title]
    assert len(key_vulns) >= 1
    assert key_vulns[0].severity == SeverityLevel.CRITICAL


def test_no_false_positives_on_safe_code(detector, file_info):
    """Test that safe code doesn't trigger false positives"""
    code = '''
import os

# All good practices - should not trigger
api_key = os.getenv("API_KEY")
password = os.getenv("PASSWORD")
secret_key = os.getenv("SECRET_KEY")

# Configuration with placeholders
config = {
    "api_key": "your_api_key_here",
    "password": "changeme",
    "token": "example_token",
}

# Short strings
x = "test"
y = "ok"
z = "abc123"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect any secrets in this safe code
    assert len(vulnerabilities) == 0


def test_entropy_detection_on_realistic_tokens(detector, file_info):
    """Test entropy-based detection on realistic token-like strings"""
    code = '''
# High-entropy strings that look like tokens
jwt_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
session_id = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6"
random_secret = "xK9mP2nQ5rT8wY3zA6bC1dE4fG7hJ0"

# Natural language (should not trigger)
description = "This is a normal sentence with regular words"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect high-entropy strings
    assert len(vulnerabilities) >= 1
    entropy_vulns = [v for v in vulnerabilities if "Entropy" in v.title or "token" in v.title.lower()]
    assert len(entropy_vulns) >= 1


def test_mixed_good_and_bad_practices(detector, file_info):
    """Test detection in code with mixed good and bad practices"""
    code = '''
import os

class Config:
    # BAD: Hardcoded secrets
    AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
    DB_PASSWORD = "hardcoded_password"
    
    # GOOD: Environment variables
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    SECRET_KEY = os.getenv("SECRET_KEY")
    
    # BAD: High-entropy string
    ENCRYPTION_KEY = "aB3dE5fG7hI9jK1lM2nO4pQ6rS8tU0vW"
    
    # GOOD: Placeholder
    API_KEY = "your_api_key_here"
'''
    ast_tree = ast.parse(code)
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect the bad practices but not the good ones
    assert len(vulnerabilities) >= 2
    
    # Check that AWS key was detected
    aws_vulns = [v for v in vulnerabilities if "AWS" in v.title]
    assert len(aws_vulns) >= 1
    
    # Check that password was detected
    password_vulns = [v for v in vulnerabilities if "password" in v.title.lower()]
    assert len(password_vulns) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
