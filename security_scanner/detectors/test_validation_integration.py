"""Integration tests for ValidationDetector

**Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5**

This test suite validates the ValidationDetector's integration with real-world code patterns.
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
    """Helper to create a temporary test file with code"""
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


def test_real_world_data_loader():
    """Test detection in a realistic data loader implementation"""
    code = """
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, max_size=1000):
        # Missing: directory validation for data_path
        # Missing: existence check for data_path
        # Missing: range validation for max_size
        self.data_path = data_path
        self.max_size = max_size
        self.data = self.load_data()
    
    def load_data(self):
        with open(self.data_path, 'r') as f:
            return f.read()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect multiple validation issues
    assert len(vulnerabilities) >= 2
    
    # Check for path validation issues
    path_vulns = [v for v in vulnerabilities if 'path' in v.context.get('issue_type', '')]
    assert len(path_vulns) >= 1
    
    # Check for numeric validation issues
    numeric_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'numeric_range']
    assert len(numeric_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


def test_checkpoint_loader_pattern():
    """Test detection in checkpoint loading code"""
    code = """
import torch

def load_checkpoint(checkpoint_path, device='cpu'):
    # Missing: directory validation for checkpoint_path
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def save_checkpoint(model, checkpoint_path):
    # Missing: directory validation for checkpoint_path
    torch.save(model.state_dict(), checkpoint_path)
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect path validation issues in both functions
    path_vulns = [v for v in vulnerabilities if v.context.get('issue_type') == 'path_directory']
    assert len(path_vulns) >= 2
    
    # All should be HIGH severity
    for vuln in path_vulns:
        assert vuln.severity == SeverityLevel.HIGH
    
    # Clean up
    Path(file_info.path).unlink()


def test_config_loader_pattern():
    """Test detection in configuration loading code"""
    code = """
import json

def load_config(config_path):
    # Missing: directory validation
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def process_config(config_name, max_workers):
    # Missing: length validation for config_name
    # Missing: range validation for max_workers
    config_path = f"/configs/{config_name}.json"
    config = load_config(config_path)
    config['max_workers'] = max_workers
    return config
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should detect multiple validation issues
    assert len(vulnerabilities) >= 3
    
    # Check for different issue types
    issue_types = {v.context.get('issue_type') for v in vulnerabilities}
    assert 'path_directory' in issue_types
    assert 'string_length' in issue_types or 'numeric_range' in issue_types
    
    # Clean up
    Path(file_info.path).unlink()


def test_safe_implementation_pattern():
    """Test that properly validated code is not flagged"""
    code = """
from pathlib import Path

def safe_load_file(file_path, max_size=1000):
    # Proper validation
    allowed_dir = Path('/safe/data').resolve()
    file_path = Path(file_path).resolve()
    
    # Directory validation
    if not str(file_path).startswith(str(allowed_dir)):
        raise ValueError('Path outside allowed directory')
    
    # Existence check
    if not file_path.exists():
        raise FileNotFoundError(f'File not found: {file_path}')
    
    # Range validation
    if not (1 <= max_size <= 10000):
        raise ValueError('max_size must be between 1 and 10000')
    
    with open(file_path, 'r') as f:
        return f.read()[:max_size]
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


def test_mixed_validation_patterns():
    """Test detection with mixed validated and unvalidated parameters"""
    code = """
def process_data(safe_path, unsafe_path, validated_size, unvalidated_size):
    # safe_path has validation
    from pathlib import Path
    safe_path = Path(safe_path).resolve()
    if not safe_path.exists():
        raise FileNotFoundError()
    
    # unsafe_path has no validation
    with open(unsafe_path, 'r') as f:
        data = f.read()
    
    # validated_size has validation
    if validated_size < 0 or validated_size > 1000:
        raise ValueError()
    
    # unvalidated_size has no validation
    result = data[:unvalidated_size]
    
    return result
"""
    
    file_info = create_test_file(code)
    ast_tree = ast.parse(code)
    
    config = DetectorConfig(enabled=True)
    detector = ValidationDetector(config)
    
    vulnerabilities = detector.detect(file_info, ast_tree)
    
    # Should only detect issues for unvalidated parameters
    assert len(vulnerabilities) >= 2
    
    # Check that unsafe_path is flagged
    unsafe_path_vulns = [v for v in vulnerabilities if v.context.get('parameter') == 'unsafe_path']
    assert len(unsafe_path_vulns) >= 1
    
    # Check that unvalidated_size is flagged
    unvalidated_size_vulns = [v for v in vulnerabilities if v.context.get('parameter') == 'unvalidated_size']
    assert len(unvalidated_size_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


if __name__ == '__main__':
    # Run all tests
    test_real_world_data_loader()
    test_checkpoint_loader_pattern()
    test_config_loader_pattern()
    test_safe_implementation_pattern()
    test_mixed_validation_patterns()
    
    print("All ValidationDetector integration tests passed!")
