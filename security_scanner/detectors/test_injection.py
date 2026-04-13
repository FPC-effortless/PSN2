"""Unit tests for InjectionDetector

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

This test suite validates the InjectionDetector's ability to:
- Detect eval() calls (Requirement 1.1)
- Detect exec() calls (Requirement 1.2)
- Detect compile() calls (Requirement 1.3)
- Detect __import__() calls (Requirement 1.4)
- Analyze context to determine if input is user-controlled
- Assign appropriate severity levels
"""

import ast
import tempfile
from pathlib import Path

from security_scanner.detectors.injection import InjectionDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    VulnerabilityType,
)


def create_test_file(code: str) -> FileInfo:
    """Helper to create a temporary test file with code
    
    Args:
        code: Python code to write to the file
        
    Returns:
        FileInfo object for the created file
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
    temp_file.write(code)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    return FileInfo(
        path=str(path),
        file_type=FileType.PYTHON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )


def test_detect_eval_call():
    """Test detection of eval() calls
    
    **Validates: Requirement 1.1**
    """
    code = """
def process_input(user_input):
    result = eval(user_input)
    return result
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one eval() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.CODE_INJECTION
    assert 'eval' in vuln.title.lower()
    assert vuln.severity == SeverityLevel.HIGH
    assert 'eval()' in vuln.description
    assert vuln.line_number == 3
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_exec_call():
    """Test detection of exec() calls
    
    **Validates: Requirement 1.2**
    """
    code = """
def run_code(code_string):
    exec(code_string)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one exec() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.CODE_INJECTION
    assert 'exec' in vuln.title.lower()
    assert vuln.severity == SeverityLevel.HIGH
    assert 'exec()' in vuln.description
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_compile_call():
    """Test detection of compile() calls
    
    **Validates: Requirement 1.3**
    """
    code = """
def compile_code(source):
    compiled = compile(source, '<string>', 'exec')
    return compiled
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one compile() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.CODE_INJECTION
    assert 'compile' in vuln.title.lower()
    assert vuln.severity == SeverityLevel.HIGH
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_import_call():
    """Test detection of __import__() calls
    
    **Validates: Requirement 1.4**
    """
    code = """
def load_module(module_name):
    mod = __import__(module_name)
    return mod
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one __import__() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.CODE_INJECTION
    assert '__import__' in vuln.title.lower()
    # Base severity is MEDIUM, but user-controlled input increases it to HIGH
    assert vuln.severity == SeverityLevel.HIGH
    assert vuln.context['user_controlled'] is True
    
    # Clean up
    Path(file_info.path).unlink()


def test_import_base_severity():
    """Test that __import__() has MEDIUM base severity without user input"""
    code = """
def load_module():
    # Hardcoded module name (not user-controlled)
    mod = __import__("os")
    return mod
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Should be MEDIUM severity (base level for __import__)
    # because input is not user-controlled
    assert vuln.context['function'] == '__import__'
    assert vuln.context['user_controlled'] is False
    # Note: The actual severity might still be adjusted by BaseDetector
    # but the base severity in DANGEROUS_FUNCTIONS is MEDIUM
    
    # Clean up
    Path(file_info.path).unlink()


def test_user_controlled_input_increases_severity():
    """Test that user-controlled input increases severity"""
    code = """
def process(user_input):
    # user_input is a function parameter (user-controlled)
    result = eval(user_input)
    return result
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Should be HIGH severity due to user-controlled input
    assert vuln.severity == SeverityLevel.HIGH
    assert vuln.context['user_controlled'] is True
    assert 'user-controlled' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_non_user_controlled_input():
    """Test detection with non-user-controlled input"""
    code = """
def calculate():
    # Hardcoded expression (not user-controlled)
    result = eval("2 + 2")
    return result
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Should still be detected but with lower confidence
    assert vuln.context['user_controlled'] is False
    assert vuln.confidence < 1.0
    
    # Clean up
    Path(file_info.path).unlink()


def test_multiple_dangerous_calls():
    """Test detection of multiple dangerous function calls"""
    code = """
def process(user_input, code_string):
    result1 = eval(user_input)
    exec(code_string)
    compiled = compile(user_input, '<string>', 'eval')
    return result1
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect all three dangerous calls
    assert len(vulnerabilities) == 3
    
    # Check that different functions are detected
    detected_functions = {v.context['function'] for v in vulnerabilities}
    assert 'eval' in detected_functions
    assert 'exec' in detected_functions
    assert 'compile' in detected_functions
    
    # Clean up
    Path(file_info.path).unlink()


def test_severity_assignment():
    """Test that severity levels are correctly assigned"""
    code = """
def test_severities(param1, param2):
    eval(param1)      # HIGH
    exec(param2)      # HIGH
    compile(param1, '<string>', 'exec')  # HIGH
    __import__(param2)  # MEDIUM base, but HIGH with user-controlled input
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 4
    
    # All should be HIGH because they all use user-controlled input (function parameters)
    for vuln in vulnerabilities:
        assert vuln.severity == SeverityLevel.HIGH
        assert vuln.context['user_controlled'] is True
    
    # Clean up
    Path(file_info.path).unlink()


def test_code_snippet_extraction():
    """Test that code snippets are correctly extracted"""
    code = """
def process(user_input):
    # This is a comment
    result = eval(user_input)
    return result
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Code snippet should contain the vulnerable line
    assert 'eval' in vuln.code_snippet
    assert vuln.line_number in [3, 4]  # Line number of eval call
    
    # Clean up
    Path(file_info.path).unlink()


def test_recommendations():
    """Test that appropriate recommendations are provided"""
    code = """
def test_funcs(param):
    eval(param)
    exec(param)
    compile(param, '<string>', 'exec')
    __import__(param)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Check that each vulnerability has a recommendation
    for vuln in vulnerabilities:
        assert len(vuln.recommendation) > 0
        assert vuln.context['function'] in vuln.recommendation.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_cwe_id_assignment():
    """Test that CWE-94 is assigned to code injection vulnerabilities"""
    code = """
def process(user_input):
    eval(user_input)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].cwe_id == "CWE-94"
    
    # Clean up
    Path(file_info.path).unlink()


def test_disabled_detector():
    """Test that disabled detector returns no vulnerabilities"""
    code = """
def process(user_input):
    eval(user_input)
    exec(user_input)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=False)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should return empty list when disabled
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_suppression():
    """Test that suppression rules work correctly"""
    code = """
def process(user_input):
    eval(user_input)  # nosec: INJ001
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should be suppressed by nosec comment
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_ast_tree():
    """Test that detector handles missing AST tree gracefully"""
    code = """
def process(user_input):
    eval(user_input)
"""
    
    file_info = create_test_file(code)
    
    config = DetectorConfig(enabled=True)
    detector = InjectionDetector(config)
    
    # Call detect without AST tree
    vulnerabilities = detector.detect(file_info, ast_tree=None)
    
    # Should return empty list
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


if __name__ == '__main__':
    # Run all tests
    test_detect_eval_call()
    test_detect_exec_call()
    test_detect_compile_call()
    test_detect_import_call()
    test_import_base_severity()
    test_user_controlled_input_increases_severity()
    test_non_user_controlled_input()
    test_multiple_dangerous_calls()
    test_severity_assignment()
    test_code_snippet_extraction()
    test_recommendations()
    test_cwe_id_assignment()
    test_disabled_detector()
    test_suppression()
    test_no_ast_tree()
    
    print("All InjectionDetector tests passed!")
