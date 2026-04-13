"""Integration tests demonstrating BaseDetector usage in concrete detectors"""

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


class SimpleInjectionDetector(BaseDetector):
    """Example concrete detector that detects eval() calls"""
    
    def detect(self, file_info: FileInfo, ast_tree=None):
        """Detect eval() calls in Python code"""
        vulnerabilities = []
        
        if ast_tree is None:
            return vulnerabilities
        
        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                    # Create vulnerability with context
                    context = {
                        'pattern': 'eval',
                        'base_severity': SeverityLevel.HIGH,
                        'user_controlled': True,
                        'file_path': file_info.path
                    }
                    
                    # Use get_severity to determine final severity
                    severity = self.get_severity(context)
                    
                    vuln = Vulnerability(
                        id='INJ001',
                        title='Dangerous eval() usage',
                        description='Use of eval() with potentially user-controlled input',
                        severity=severity,
                        vulnerability_type=VulnerabilityType.CODE_INJECTION,
                        file_path=file_info.path,
                        line_number=node.lineno,
                        column=node.col_offset,
                        code_snippet=f'eval(...) at line {node.lineno}',
                        recommendation='Avoid using eval(). Use ast.literal_eval() for safe evaluation.',
                        confidence=0.9
                    )
                    
                    # Check if should be suppressed
                    if not self.should_suppress(vuln):
                        vulnerabilities.append(vuln)
        
        return vulnerabilities


def test_concrete_detector_with_base_functionality():
    """Test that a concrete detector can use BaseDetector functionality"""
    config = DetectorConfig(enabled=True)
    detector = SimpleInjectionDetector(config)
    
    # Create test code with eval()
    code = """
def process_input(user_input):
    result = eval(user_input)  # Dangerous!
    return result
"""
    
    # Parse code to AST
    tree = ast.parse(code)
    
    # Create FileInfo (use a non-test filename)
    file_info = FileInfo(
        path='src/main.py',
        file_type=FileType.PYTHON,
        last_modified=0.0,
        size=100
    )
    
    # Run detection
    vulnerabilities = detector.detect(file_info, tree)
    
    # Verify detection
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].id == 'INJ001'
    # Severity should be HIGH (base HIGH + user_controlled = HIGH)
    assert vulnerabilities[0].severity == SeverityLevel.HIGH
    assert vulnerabilities[0].line_number == 3


def test_concrete_detector_with_suppression():
    """Test that suppression rules work in concrete detectors"""
    config = DetectorConfig(
        enabled=True,
        exclusions=['INJ001']
    )
    detector = SimpleInjectionDetector(config)
    
    code = """
def process_input(user_input):
    result = eval(user_input)
    return result
"""
    
    tree = ast.parse(code)
    file_info = FileInfo(
        path='src/main.py',
        file_type=FileType.PYTHON,
        last_modified=0.0,
        size=100
    )
    
    # Run detection - should be suppressed
    vulnerabilities = detector.detect(file_info, tree)
    
    # Verify suppression
    assert len(vulnerabilities) == 0


def test_concrete_detector_severity_adjustment():
    """Test that severity is adjusted based on context"""
    config = DetectorConfig(enabled=True)
    detector = SimpleInjectionDetector(config)
    
    # Test code in a test file (should reduce severity)
    code = """
def test_eval():
    result = eval("1 + 1")
    assert result == 2
"""
    
    tree = ast.parse(code)
    file_info = FileInfo(
        path='tests/test_module.py',
        file_type=FileType.PYTHON,
        last_modified=0.0,
        size=100
    )
    
    vulnerabilities = detector.detect(file_info, tree)
    
    # Severity should be reduced from HIGH to MEDIUM for test files
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].severity == SeverityLevel.MEDIUM


if __name__ == '__main__':
    test_concrete_detector_with_base_functionality()
    test_concrete_detector_with_suppression()
    test_concrete_detector_severity_adjustment()
    print("All integration tests passed!")
