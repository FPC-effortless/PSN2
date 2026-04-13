"""Unit tests for BaseDetector abstract class"""

import ast
from security_scanner.detectors.base import BaseDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    Vulnerability,
    VulnerabilityType,
)


class ConcreteDetector(BaseDetector):
    """Concrete implementation of BaseDetector for testing"""
    
    def detect(self, file_info: FileInfo, ast_tree=None):
        """Simple implementation that returns an empty list"""
        return []


def test_detector_initialization():
    """Test that detector initializes with correct name and config"""
    config = DetectorConfig(enabled=True)
    detector = ConcreteDetector(config)
    
    assert detector.name == "ConcreteDetector"
    assert detector.config == config
    assert detector.config.enabled is True


def test_get_severity_base():
    """Test basic severity determination"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    # Test base severity without context
    context = {'base_severity': SeverityLevel.HIGH}
    assert detector.get_severity(context) == SeverityLevel.HIGH


def test_get_severity_with_validation():
    """Test severity reduction when validation is present"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    # Critical -> High when validation present
    context = {
        'base_severity': SeverityLevel.CRITICAL,
        'has_validation': True
    }
    assert detector.get_severity(context) == SeverityLevel.HIGH
    
    # High -> Medium when validation present
    context = {
        'base_severity': SeverityLevel.HIGH,
        'has_validation': True
    }
    assert detector.get_severity(context) == SeverityLevel.MEDIUM


def test_get_severity_user_controlled():
    """Test severity increase when input is user-controlled"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    # Medium -> High when user-controlled
    context = {
        'base_severity': SeverityLevel.MEDIUM,
        'user_controlled': True
    }
    assert detector.get_severity(context) == SeverityLevel.HIGH
    
    # Low -> Medium when user-controlled
    context = {
        'base_severity': SeverityLevel.LOW,
        'user_controlled': True
    }
    assert detector.get_severity(context) == SeverityLevel.MEDIUM


def test_get_severity_test_files():
    """Test severity reduction for test files"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    # Critical -> High in test files
    context = {
        'base_severity': SeverityLevel.CRITICAL,
        'file_path': 'tests/test_module.py'
    }
    assert detector.get_severity(context) == SeverityLevel.HIGH
    
    # High -> Medium in test files
    context = {
        'base_severity': SeverityLevel.HIGH,
        'file_path': 'test_something.py'
    }
    assert detector.get_severity(context) == SeverityLevel.MEDIUM


def test_get_severity_overrides():
    """Test severity overrides from configuration"""
    config = DetectorConfig(
        severity_overrides={'PATTERN001': SeverityLevel.LOW}
    )
    detector = ConcreteDetector(config)
    
    # Override should take precedence
    context = {
        'pattern': 'PATTERN001',
        'base_severity': SeverityLevel.CRITICAL
    }
    assert detector.get_severity(context) == SeverityLevel.LOW


def test_should_suppress_by_id():
    """Test suppression by vulnerability ID"""
    config = DetectorConfig(exclusions=['INJ001', 'SEC002'])
    detector = ConcreteDetector(config)
    
    vuln = Vulnerability(
        id='INJ001',
        title='Test Vulnerability',
        description='Test',
        severity=SeverityLevel.HIGH,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='test.py',
        line_number=10,
        column=0,
        code_snippet='eval(user_input)',
        recommendation='Do not use eval'
    )
    
    assert detector.should_suppress(vuln) is True


def test_should_suppress_by_path():
    """Test suppression by file path pattern"""
    config = DetectorConfig(exclusions=['legacy/'])
    detector = ConcreteDetector(config)
    
    vuln = Vulnerability(
        id='INJ001',
        title='Test Vulnerability',
        description='Test',
        severity=SeverityLevel.HIGH,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='legacy/old_code.py',
        line_number=10,
        column=0,
        code_snippet='eval(user_input)',
        recommendation='Do not use eval'
    )
    
    assert detector.should_suppress(vuln) is True


def test_should_suppress_nosec_comment():
    """Test suppression by nosec comment"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    vuln = Vulnerability(
        id='INJ001',
        title='Test Vulnerability',
        description='Test',
        severity=SeverityLevel.HIGH,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='test.py',
        line_number=10,
        column=0,
        code_snippet='eval(user_input)  # nosec: INJ001',
        recommendation='Do not use eval'
    )
    
    assert detector.should_suppress(vuln) is True


def test_should_not_suppress():
    """Test that vulnerabilities without suppression rules are not suppressed"""
    config = DetectorConfig(exclusions=['OTHER001'])
    detector = ConcreteDetector(config)
    
    vuln = Vulnerability(
        id='INJ001',
        title='Test Vulnerability',
        description='Test',
        severity=SeverityLevel.HIGH,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='src/main.py',
        line_number=10,
        column=0,
        code_snippet='eval(user_input)',
        recommendation='Do not use eval'
    )
    
    assert detector.should_suppress(vuln) is False


def test_security_comment_reduces_confidence():
    """Test that security comments reduce confidence but don't suppress"""
    config = DetectorConfig()
    detector = ConcreteDetector(config)
    
    vuln = Vulnerability(
        id='INJ001',
        title='Test Vulnerability',
        description='Test',
        severity=SeverityLevel.HIGH,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='test.py',
        line_number=10,
        column=0,
        code_snippet='eval(user_input)  # security: validated input',
        recommendation='Do not use eval',
        confidence=1.0
    )
    
    result = detector.should_suppress(vuln)
    assert result is False
    assert vuln.confidence == 0.5


if __name__ == '__main__':
    # Run tests
    test_detector_initialization()
    test_get_severity_base()
    test_get_severity_with_validation()
    test_get_severity_user_controlled()
    test_get_severity_test_files()
    test_get_severity_overrides()
    test_should_suppress_by_id()
    test_should_suppress_by_path()
    test_should_suppress_nosec_comment()
    test_should_not_suppress()
    test_security_comment_reduces_confidence()
    print("All tests passed!")
