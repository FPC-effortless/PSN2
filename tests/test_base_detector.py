"""Tests for BaseDetector class"""

import ast
import pytest
from hypothesis import given, strategies as st

from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    Vulnerability,
    SeverityLevel,
)


class TestDetector(BaseDetector):
    """Concrete implementation of BaseDetector for testing"""
    
    def detect(self, file_info, ast_tree):
        return []


def test_base_detector_initialization():
    """Test BaseDetector initialization"""
    config = DetectorConfig()
    detector = TestDetector(config)
    
    assert detector.config == config
    assert detector.name == "TestDetector"


def test_base_detector_get_severity():
    """Test default severity level"""
    config = DetectorConfig()
    detector = TestDetector(config)
    
    severity = detector.get_severity({})
    assert severity == SeverityLevel.MEDIUM


def test_base_detector_should_suppress():
    """Test default suppression behavior"""
    config = DetectorConfig()
    detector = TestDetector(config)
    
    vuln = Vulnerability(
        id="TEST001",
        title="Test",
        description="Test",
        severity=SeverityLevel.LOW,
        file_path="test.py",
        line_number=1,
        column=0,
        code_snippet="test",
        recommendation="Fix",
    )
    
    assert detector.should_suppress(vuln) is False


# Feature: security-review-codebase, Property 16: Confidence Score Validity
@given(confidence=st.floats(min_value=0.0, max_value=1.0))
def test_vulnerability_confidence_score_range(confidence):
    """Property: Any vulnerability confidence score must be between 0.0 and 1.0"""
    vuln = Vulnerability(
        id="TEST001",
        title="Test",
        description="Test",
        severity=SeverityLevel.LOW,
        file_path="test.py",
        line_number=1,
        column=0,
        code_snippet="test",
        recommendation="Fix",
        confidence=confidence,
    )
    
    assert 0.0 <= vuln.confidence <= 1.0
    assert isinstance(vuln.confidence, float)
