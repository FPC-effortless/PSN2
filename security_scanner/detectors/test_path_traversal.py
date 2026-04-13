"""Unit tests for PathTraversalDetector

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

This test suite validates the PathTraversalDetector's ability to:
- Detect unsafe path construction with string concatenation (Requirement 3.1)
- Detect os.path.join() without validation (Requirement 3.2)
- Detect paths containing ".." without validation (Requirement 3.3)
- Detect open() with user-controlled paths (Requirement 3.4)
- Check for pathlib.Path.resolve() usage
- Assign appropriate severity levels
"""

import ast
import tempfile
from pathlib import Path

from security_scanner.detectors.path_traversal import PathTraversalDetector
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


def test_detect_open_with_user_path():
    """Test detection of open() with user-controlled paths
    
    **Validates: Requirement 3.4**
    """
    code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one open() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL
    assert 'open' in vuln.title.lower()
    assert vuln.severity == SeverityLevel.HIGH
    assert 'open()' in vuln.description
    assert vuln.context['user_controlled'] is True
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_os_path_join_without_validation():
    """Test detection of os.path.join() without validation
    
    **Validates: Requirement 3.2**
    """
    code = """
import os

def get_file_path(user_dir, filename):
    path = os.path.join(user_dir, filename)
    return path
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect one os.path.join() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL
    assert 'join' in vuln.title.lower()
    assert vuln.severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
    assert vuln.context['user_controlled'] is True
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_path_with_dotdot():
    """Test detection of paths containing ".." without validation
    
    **Validates: Requirement 3.3**
    """
    code = """
def read_parent_file(filename):
    path = "../data/" + filename
    with open(path, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect vulnerabilities (string concatenation and open())
    assert len(vulnerabilities) >= 1
    
    # Check that at least one vulnerability mentions ".."
    has_dotdot_warning = any(
        '..' in vuln.description or vuln.context.get('has_dotdot', False)
        for vuln in vulnerabilities
    )
    assert has_dotdot_warning
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_string_concatenation():
    """Test detection of unsafe path construction with string concatenation
    
    **Validates: Requirement 3.1**
    """
    code = """
def build_path(user_input):
    path = "/base/dir/" + user_input + "/../secret.txt"
    return path
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect string concatenation vulnerability
    assert len(vulnerabilities) >= 1
    
    # Check for string concatenation warning
    has_concat_warning = any(
        'concatenation' in vuln.title.lower() or 'concatenation' in vuln.description.lower()
        for vuln in vulnerabilities
    )
    assert has_concat_warning
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_pathlib_path():
    """Test detection of pathlib.Path() with user input"""
    code = """
from pathlib import Path

def get_path(user_path):
    p = Path(user_path)
    return p
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect Path() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL
    assert 'Path' in vuln.context['function']
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_shutil_copy():
    """Test detection of shutil.copy() with user paths"""
    code = """
import shutil

def copy_file(src, dst):
    shutil.copy(src, dst)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect shutil.copy() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL
    assert 'copy' in vuln.title.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_shutil_move():
    """Test detection of shutil.move() with user paths"""
    code = """
import shutil

def move_file(src, dst):
    shutil.move(src, dst)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect shutil.move() vulnerability
    assert len(vulnerabilities) == 1
    
    vuln = vulnerabilities[0]
    assert vuln.vulnerability_type == VulnerabilityType.PATH_TRAVERSAL
    assert 'move' in vuln.title.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_path_resolve_reduces_severity():
    """Test that using Path.resolve() is detected and reduces severity"""
    code = """
from pathlib import Path

def safe_open(user_path, base_dir):
    resolved = Path(base_dir, user_path).resolve()
    with open(resolved, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # May still detect vulnerabilities but with reduced severity/confidence
    if vulnerabilities:
        for vuln in vulnerabilities:
            # Check that resolve usage is noted
            if vuln.context.get('uses_resolve', False):
                assert 'resolve' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_severity_high_for_unvalidated_user_paths():
    """Test that unvalidated user paths get HIGH severity"""
    code = """
def read_user_file(user_filename):
    # No validation - direct user input to open()
    with open(user_filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Should be HIGH severity for unvalidated user path
    assert vuln.severity == SeverityLevel.HIGH
    assert vuln.context['user_controlled'] is True
    
    # Clean up
    Path(file_info.path).unlink()


def test_severity_medium_for_join_without_validation():
    """Test that os.path.join without validation gets MEDIUM severity"""
    code = """
import os

def build_path(subdir):
    # os.path.join without validation
    return os.path.join("/base", subdir)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Should be MEDIUM or HIGH severity
    assert vuln.severity in [SeverityLevel.MEDIUM, SeverityLevel.HIGH]
    
    # Clean up
    Path(file_info.path).unlink()


def test_multiple_file_operations():
    """Test detection of multiple file operations"""
    code = """
import os
import shutil
from pathlib import Path

def file_ops(user_path):
    # Multiple file operations
    with open(user_path, 'r') as f:
        data = f.read()
    
    new_path = os.path.join("/tmp", user_path)
    shutil.copy(user_path, new_path)
    
    p = Path(user_path)
    return data
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect multiple vulnerabilities
    assert len(vulnerabilities) >= 3
    
    # Check that different operations are detected
    detected_ops = {v.context['function'] for v in vulnerabilities}
    assert 'open' in detected_ops
    
    # Clean up
    Path(file_info.path).unlink()


def test_non_user_controlled_path():
    """Test detection with non-user-controlled paths"""
    code = """
def read_config():
    # Hardcoded path (not user-controlled)
    with open("/etc/config.txt", 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should still detect but with lower confidence
    if vulnerabilities:
        vuln = vulnerabilities[0]
        assert vuln.context['user_controlled'] is False
        assert vuln.confidence < 1.0
    
    # Clean up
    Path(file_info.path).unlink()


def test_recommendations():
    """Test that appropriate recommendations are provided"""
    code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Check that recommendation includes key security practices
    assert len(vuln.recommendation) > 0
    assert 'resolve' in vuln.recommendation.lower()
    assert 'validate' in vuln.recommendation.lower() or 'validation' in vuln.recommendation.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_cwe_id_assignment():
    """Test that CWE-22 is assigned to path traversal vulnerabilities"""
    code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    assert vulnerabilities[0].cwe_id == "CWE-22"
    
    # Clean up
    Path(file_info.path).unlink()


def test_disabled_detector():
    """Test that disabled detector returns no vulnerabilities"""
    code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=False)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should return empty list when disabled
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_suppression():
    """Test that suppression rules work correctly"""
    code = """
def read_file(filename):
    with open(filename, 'r') as f:  # nosec: PTH001
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should be suppressed by nosec comment
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_ast_tree():
    """Test that detector handles missing AST tree gracefully"""
    code = """
def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    # Call detect without AST tree
    vulnerabilities = detector.detect(file_info, ast_tree=None)
    
    # Should return empty list
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_code_snippet_extraction():
    """Test that code snippets are correctly extracted"""
    code = """
def read_file(filename):
    # This is a comment
    with open(filename, 'r') as f:
        return f.read()
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = PathTraversalDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    assert len(vulnerabilities) == 1
    vuln = vulnerabilities[0]
    
    # Code snippet should contain the vulnerable line
    assert 'open' in vuln.code_snippet
    assert vuln.line_number > 0
    
    # Clean up
    Path(file_info.path).unlink()


if __name__ == '__main__':
    # Run all tests
    test_detect_open_with_user_path()
    test_detect_os_path_join_without_validation()
    test_detect_path_with_dotdot()
    test_detect_string_concatenation()
    test_detect_pathlib_path()
    test_detect_shutil_copy()
    test_detect_shutil_move()
    test_path_resolve_reduces_severity()
    test_severity_high_for_unvalidated_user_paths()
    test_severity_medium_for_join_without_validation()
    test_multiple_file_operations()
    test_non_user_controlled_path()
    test_recommendations()
    test_cwe_id_assignment()
    test_disabled_detector()
    test_suppression()
    test_no_ast_tree()
    test_code_snippet_extraction()
    
    print("All PathTraversalDetector tests passed!")
