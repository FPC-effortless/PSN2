"""Unit tests for ValidationDetector

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

This test suite validates the ValidationDetector's ability to:
- Detect functions accepting external input without type checking (Requirement 5.1)
- Detect file path parameters without directory validation (Requirement 5.2)
- Detect numeric inputs without range validation (Requirement 5.3)
- Detect string inputs without length limits (Requirement 5.4)
- Detect dataset loaders without file existence checks (Requirement 5.5)
- Assign appropriate severity levels
"""

import ast
import tempfile
from pathlib import Path

from security_scanner.detectors.validation import ValidationDetector
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


def test_detect_path_without_directory_validation():
    """Test detection of file path parameters without directory validation
    
    **Validates: Requirement 5.2**
    """
    code = """
def load_file(file_path):
    with open(file_path, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect missing directory validation
    assert len(vulnerabilities) >= 1
    
    # Find the path directory validation vulnerability
    path_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'path_directory']
    assert len(path_vulns) == 1
    
    vuln = path_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
    assert vuln.severity == SeverityLevel.HIGH
    assert 'file_path' in vuln.context['parameter']
    assert 'directory' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_path_with_directory_validation():
    """Test that paths with directory validation are not flagged"""
    code = """
from pathlib import Path

def load_file(file_path):
    allowed_dir = Path('/safe/directory').resolve()
    file_path = Path(file_path).resolve()
    if not str(file_path).startswith(str(allowed_dir)):
        raise ValueError('Path outside allowed directory')
    with open(file_path, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because validation is present
    path_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'path_directory']
    assert len(path_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_numeric_without_range_validation():
    """Test detection of numeric parameters without range validation
    
    **Validates: Requirement 5.3**
    """
    code = """
def process_data(batch_size):
    # No range validation for batch_size
    data = [0] * batch_size
    return data
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect missing range validation
    numeric_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'numeric_range']
    assert len(numeric_vulns) == 1
    
    vuln = numeric_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
    assert vuln.severity == SeverityLevel.LOW
    assert 'batch_size' in vuln.context['parameter']
    assert 'range' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_numeric_with_range_validation():
    """Test that numeric parameters with range validation are not flagged"""
    code = """
def process_data(batch_size):
    if not (1 <= batch_size <= 1000):
        raise ValueError('batch_size must be between 1 and 1000')
    data = [0] * batch_size
    return data
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because validation is present
    numeric_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'numeric_range']
    assert len(numeric_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_string_without_length_validation():
    """Test detection of string parameters without length limits
    
    **Validates: Requirement 5.4**
    """
    code = """
def process_message(message):
    # No length validation for message
    return message.upper()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect missing length validation
    string_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'string_length']
    assert len(string_vulns) == 1
    
    vuln = string_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
    assert vuln.severity == SeverityLevel.MEDIUM
    assert 'message' in vuln.context['parameter']
    assert 'length' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_string_with_length_validation():
    """Test that string parameters with length validation are not flagged"""
    code = """
def process_message(message):
    MAX_LENGTH = 1000
    if len(message) > MAX_LENGTH:
        raise ValueError(f'Message too long: {len(message)} > {MAX_LENGTH}')
    return message.upper()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because validation is present
    string_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'string_length']
    assert len(string_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_dataset_loader_without_existence_check():
    """Test detection of dataset loaders without file existence checks
    
    **Validates: Requirement 5.5**
    """
    code = """
class CustomDataset:
    def __init__(self, data_path):
        # No existence check for data_path
        self.data_path = data_path
    
    def load(self):
        with open(self.data_path, 'r') as f:
            return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect missing existence check
    existence_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'path_existence']
    assert len(existence_vulns) >= 1
    
    vuln = existence_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
    assert vuln.severity == SeverityLevel.LOW
    assert 'data_path' in vuln.context['parameter']
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_dataset_loader_with_existence_check():
    """Test that dataset loaders with existence checks are not flagged"""
    code = """
from pathlib import Path

class CustomDataset:
    def __init__(self, data_path):
        file_path = Path(data_path)
        if not file_path.exists():
            raise FileNotFoundError(f'File not found: {data_path}')
        self.data_path = data_path
    
    def load(self):
        with open(self.data_path, 'r') as f:
            return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because existence check is present
    existence_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'path_existence']
    assert len(existence_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_external_input_without_type_checking():
    """Test detection of external input without type checking
    
    **Validates: Requirement 5.1**
    """
    code = """
def load_data(input_data):
    # No type checking for input_data
    return input_data.process()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect missing type checking
    type_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'type_checking']
    assert len(type_vulns) >= 1
    
    vuln = type_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.INPUT_VALIDATION
    assert vuln.severity == SeverityLevel.MEDIUM
    assert 'input_data' in vuln.context['parameter']
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_external_input_with_type_checking():
    """Test that external input with type checking is not flagged"""
    code = """
def load_data(input_data):
    if not isinstance(input_data, dict):
        raise TypeError(f'Expected dict, got {type(input_data).__name__}')
    return input_data.get('key')
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because type checking is present
    type_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'type_checking']
    assert len(type_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_external_input_with_type_annotation():
    """Test that type annotations count as type checking"""
    code = """
def load_data(input_data: dict):
    return input_data.get('key')
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect vulnerability because type annotation is present
    type_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'type_checking']
    assert len(type_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_multiple_validation_issues():
    """Test detection of multiple validation issues in one function"""
    code = """
def process_file(file_path, batch_size, message):
    # file_path: no directory validation
    # batch_size: no range validation
    # message: no length validation
    with open(file_path, 'r') as f:
        data = f.read()
    
    result = [data] * batch_size
    return message + str(result)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect all three validation issues
    assert len(vulnerabilities) >= 3
    
    # Check that different issue types are detected
    issue_types = {v.context.get('issue_type') for v in vulnerabilities}
    assert 'path_directory' in issue_types
    assert 'numeric_range' in issue_types
    assert 'string_length' in issue_types
    
    # Clean up
    Path(file_info.path).unlink()


def test_severity_levels():
    """Test that correct severity levels are assigned"""
    code = """
def process(file_path, size, text):
    # file_path: HIGH severity
    # size: LOW severity
    # text: MEDIUM severity
    with open(file_path, 'r') as f:
        data = f.read()
    return data[:size] + text
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Check severity levels
    for vuln in vulnerabilities:
        issue_type = vuln.context.get('issue_type')
        if issue_type == 'path_directory':
            assert vuln.severity == SeverityLevel.HIGH
        elif issue_type == 'numeric_range':
            assert vuln.severity == SeverityLevel.LOW
        elif issue_type == 'string_length':
            assert vuln.severity == SeverityLevel.MEDIUM
    
    # Clean up
    Path(file_info.path).unlink()


def test_recommendations():
    """Test that appropriate recommendations are provided"""
    code = """
def process(file_path, size, text):
    with open(file_path, 'r') as f:
        data = f.read()
    return data[:size] + text
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Check that each vulnerability has a recommendation
    for vuln in vulnerabilities:
        assert len(vuln.recommendation) > 0
        assert vuln.context['parameter'] in vuln.recommendation
    
    # Clean up
    Path(file_info.path).unlink()


def test_cwe_id_assignment():
    """Test that appropriate CWE IDs are assigned"""
    code = """
def process(file_path, size, text):
    with open(file_path, 'r') as f:
        data = f.read()
    return data[:size] + text
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Check CWE IDs
    for vuln in vulnerabilities:
        assert vuln.cwe_id is not None
        assert vuln.cwe_id.startswith('CWE-')
        
        issue_type = vuln.context.get('issue_type')
        if issue_type == 'path_directory':
            assert vuln.cwe_id == 'CWE-22'  # Path Traversal
        elif issue_type == 'numeric_range':
            assert vuln.cwe_id == 'CWE-190'  # Integer Overflow
        elif issue_type == 'string_length':
            assert vuln.cwe_id == 'CWE-120'  # Buffer Copy without Checking Size
        elif issue_type == 'type_checking':
            assert vuln.cwe_id == 'CWE-20'  # Improper Input Validation
    
    # Clean up
    Path(file_info.path).unlink()


def test_disabled_detector():
    """Test that disabled detector returns no vulnerabilities"""
    code = """
def process(file_path, size, text):
    with open(file_path, 'r') as f:
        data = f.read()
    return data[:size] + text
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=False)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should return empty list when disabled
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_ast_tree():
    """Test that detector handles missing AST tree gracefully"""
    code = """
def process(file_path):
    with open(file_path, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    # Call detect without AST tree
    vulnerabilities = detector.detect(file_info, ast_tree=None)
    
    # Should return empty list
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_function_with_no_parameters():
    """Test that functions with no parameters are not flagged"""
    code = """
def get_data():
    return [1, 2, 3]
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should not detect any vulnerabilities
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_function_with_self_parameter():
    """Test that self/cls parameters are ignored"""
    code = """
class MyClass:
    def method(self, data):
        return data
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should only check 'data' parameter, not 'self'
    for vuln in vulnerabilities:
        assert vuln.context['parameter'] != 'self'
    
    # Clean up
    Path(file_info.path).unlink()


if __name__ == '__main__':
    # Run all tests
    test_detect_path_without_directory_validation()
    test_detect_path_with_directory_validation()
    test_detect_numeric_without_range_validation()
    test_detect_numeric_with_range_validation()
    test_detect_string_without_length_validation()
    test_detect_string_with_length_validation()
    test_detect_dataset_loader_without_existence_check()
    test_detect_dataset_loader_with_existence_check()
    test_detect_external_input_without_type_checking()
    test_detect_external_input_with_type_checking()
    test_detect_external_input_with_type_annotation()
    test_multiple_validation_issues()
    test_severity_levels()
    test_recommendations()
    test_cwe_id_assignment()
    test_disabled_detector()
    test_no_ast_tree()
    test_function_with_no_parameters()
    test_function_with_self_parameter()
    
    print("All ValidationDetector tests passed!")

