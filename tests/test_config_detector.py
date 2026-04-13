"""Unit tests for ConfigDetector

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

This test suite validates the ConfigDetector's ability to:
- Detect debug mode enabled in configuration (Requirement 7.1)
- Detect insecure default values (Requirement 7.2)
- Detect absolute paths to sensitive directories (Requirement 7.3)
- Detect missing required security settings (Requirement 7.4)
- Detect world-readable files with sensitive data (Requirement 7.5)
"""

import json
import os
import stat
import tempfile
from pathlib import Path

import yaml

from security_scanner.detectors.config import ConfigDetector
from security_scanner.models import (
    DetectorConfig,
    FileInfo,
    FileType,
    SeverityLevel,
    VulnerabilityType,
)


def create_test_json_file(config_data: dict, permissions: int = 0o644) -> FileInfo:
    """Helper to create a temporary JSON config file
    
    Args:
        config_data: Configuration data to write
        permissions: File permissions (octal)
        
    Returns:
        FileInfo object for the created file
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, temp_file, indent=2)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    # Set file permissions
    os.chmod(str(path), permissions)
    
    return FileInfo(
        path=str(path),
        file_type=FileType.CONFIG_JSON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )


def create_test_yaml_file(config_data: dict, permissions: int = 0o644) -> FileInfo:
    """Helper to create a temporary YAML config file
    
    Args:
        config_data: Configuration data to write
        permissions: File permissions (octal)
        
    Returns:
        FileInfo object for the created file
    """
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(config_data, temp_file, default_flow_style=False)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    # Set file permissions
    os.chmod(str(path), permissions)
    
    return FileInfo(
        path=str(path),
        file_type=FileType.CONFIG_YAML,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )


def test_detect_debug_mode_enabled_json():
    """Test detection of debug mode enabled in JSON configuration
    
    **Validates: Requirement 7.1**
    """
    config_data = {
        "app_name": "test_app",
        "debug": True,
        "port": 8080
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect debug mode enabled
    assert len(vulnerabilities) >= 1
    
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) == 1
    
    vuln = debug_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.CONFIGURATION
    assert vuln.severity == SeverityLevel.MEDIUM
    assert 'debug' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_debug_mode_uppercase():
    """Test detection of DEBUG mode with uppercase key
    
    **Validates: Requirement 7.1**
    """
    config_data = {
        "DEBUG": 1,
        "server": "localhost"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect DEBUG=1
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_debug_mode_yaml():
    """Test detection of debug mode in YAML configuration
    
    **Validates: Requirement 7.1**
    """
    config_data = {
        "application": {
            "name": "test_app",
            "debug_mode": True
        }
    }
    
    file_info = create_test_yaml_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect debug_mode enabled
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_insecure_default_password():
    """Test detection of insecure default password
    
    **Validates: Requirement 7.2**
    """
    config_data = {
        "database": {
            "host": "localhost",
            "password": "password"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect insecure default password
    default_vulns = [v for v in vulnerabilities if 'default' in v.title.lower()]
    assert len(default_vulns) >= 1
    
    vuln = default_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.CONFIGURATION
    assert vuln.severity == SeverityLevel.LOW
    assert 'password' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_insecure_default_secret():
    """Test detection of insecure default secret
    
    **Validates: Requirement 7.2**
    """
    config_data = {
        "api": {
            "secret": "secret",
            "endpoint": "https://api.example.com"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect insecure default secret
    default_vulns = [v for v in vulnerabilities if 'default' in v.title.lower()]
    assert len(default_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_sensitive_path_unix():
    """Test detection of absolute path to sensitive directory (Unix)
    
    **Validates: Requirement 7.3**
    """
    config_data = {
        "paths": {
            "config_dir": "/etc/myapp",
            "data_dir": "./data"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect /etc path
    path_vulns = [v for v in vulnerabilities if 'path' in v.title.lower()]
    assert len(path_vulns) >= 1
    
    vuln = path_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.CONFIGURATION
    assert vuln.severity == SeverityLevel.LOW
    assert '/etc' in vuln.description
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_sensitive_path_root():
    """Test detection of /root path
    
    **Validates: Requirement 7.3**
    """
    config_data = {
        "backup_dir": "/root/backups"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect /root path
    path_vulns = [v for v in vulnerabilities if 'path' in v.title.lower()]
    assert len(path_vulns) >= 1
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_missing_security_settings():
    """Test detection of missing security settings
    
    **Validates: Requirement 7.4**
    """
    config_data = {
        "app_name": "test_app",
        "port": 8080,
        "database": {
            "host": "localhost",
            "port": 5432
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect missing security settings
    missing_vulns = [v for v in vulnerabilities if 'missing' in v.title.lower()]
    assert len(missing_vulns) >= 1
    
    vuln = missing_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.CONFIGURATION
    assert vuln.severity == SeverityLevel.MEDIUM
    assert 'security' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_missing_security_settings_when_present():
    """Test that security settings are not flagged when present"""
    config_data = {
        "app_name": "test_app",
        "ssl": True,
        "authentication": "required"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should not detect missing security settings
    missing_vulns = [v for v in vulnerabilities if 'missing' in v.title.lower()]
    assert len(missing_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_detect_world_readable_sensitive_file():
    """Test detection of world-readable file with sensitive data
    
    **Validates: Requirement 7.5**
    """
    config_data = {
        "database": {
            "password": "secret123",
            "api_key": "key123"
        }
    }
    
    # Create file with world-readable permissions (0o644)
    file_info = create_test_json_file(config_data, permissions=0o644)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect world-readable sensitive file
    readable_vulns = [v for v in vulnerabilities if 'world-readable' in v.title.lower() or 'readable' in v.title.lower()]
    assert len(readable_vulns) >= 1
    
    vuln = readable_vulns[0]
    assert vuln.vulnerability_type == VulnerabilityType.CONFIGURATION
    assert vuln.severity == SeverityLevel.HIGH
    assert 'permission' in vuln.description.lower() or 'readable' in vuln.description.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_world_readable_vuln_when_restricted():
    """Test that restricted permissions don't trigger world-readable vulnerability
    
    Note: On Windows, file permissions work differently than Unix.
    This test may not work as expected on Windows systems.
    """
    import platform
    
    config_data = {
        "database": {
            "password": "secret123"
        }
    }
    
    # Create file with restricted permissions (0o600)
    file_info = create_test_json_file(config_data, permissions=0o600)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should not detect world-readable issue on Unix systems
    # On Windows, permissions work differently and this test may fail
    readable_vulns = [v for v in vulnerabilities if 'world-readable' in v.title.lower() or 'readable' in v.title.lower()]
    
    if platform.system() != 'Windows':
        assert len(readable_vulns) == 0
    # On Windows, we skip this assertion as permissions work differently
    
    # Clean up
    Path(file_info.path).unlink()


def test_multiple_issues_in_one_file():
    """Test detection of multiple configuration issues in one file"""
    config_data = {
        "debug": True,
        "database": {
            "password": "password",
            "config_path": "/etc/db/config"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect multiple issues
    assert len(vulnerabilities) >= 3
    
    # Check for different issue types
    issue_types = [v.context['issue_type'] for v in vulnerabilities]
    assert 'debug_mode_enabled' in issue_types
    assert 'insecure_default_value' in issue_types
    assert 'sensitive_path' in issue_types
    
    # Clean up
    Path(file_info.path).unlink()


def test_nested_configuration():
    """Test detection in deeply nested configuration"""
    config_data = {
        "app": {
            "settings": {
                "environment": {
                    "debug": True
                }
            }
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect debug mode in nested structure
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) >= 1
    
    # Check that key path is correct
    vuln = debug_vulns[0]
    assert 'app.settings.environment.debug' in vuln.context['key_path']
    
    # Clean up
    Path(file_info.path).unlink()


def test_confidence_scores():
    """Test that confidence scores are appropriate"""
    config_data = {
        "debug": True,
        "password": "password"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Check confidence scores are valid
    for vuln in vulnerabilities:
        assert 0.0 <= vuln.confidence <= 1.0
    
    # Clean up
    Path(file_info.path).unlink()


def test_cwe_id_assignment():
    """Test that CWE IDs are correctly assigned"""
    config_data = {
        "debug": True,
        "password": "password",
        "config_path": "/etc/config"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # All vulnerabilities should have CWE IDs
    for vuln in vulnerabilities:
        assert vuln.cwe_id is not None
        assert vuln.cwe_id.startswith('CWE-')
    
    # Clean up
    Path(file_info.path).unlink()


def test_recommendations():
    """Test that recommendations are provided"""
    config_data = {
        "debug": True,
        "password": "password"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # All vulnerabilities should have recommendations
    for vuln in vulnerabilities:
        assert vuln.recommendation is not None
        assert len(vuln.recommendation) > 0
    
    # Check specific recommendations
    debug_vuln = [v for v in vulnerabilities if 'debug' in v.title.lower()][0]
    assert 'false' in debug_vuln.recommendation.lower() or 'disable' in debug_vuln.recommendation.lower()
    
    # Clean up
    Path(file_info.path).unlink()


def test_disabled_detector():
    """Test that disabled detector returns no vulnerabilities"""
    config_data = {
        "debug": True,
        "password": "password"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=False)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should return no vulnerabilities when disabled
    assert len(vulnerabilities) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_invalid_json_file():
    """Test that detector handles invalid JSON gracefully"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    temp_file.write("{ invalid json }")
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.CONFIG_JSON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should return no vulnerabilities for invalid JSON
    assert len(vulnerabilities) == 0
    
    # Clean up
    path.unlink()


def test_invalid_yaml_file():
    """Test that detector handles invalid YAML gracefully"""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    temp_file.write("invalid: yaml: content:")
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.CONFIG_YAML,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should return no vulnerabilities for invalid YAML
    assert len(vulnerabilities) == 0
    
    # Clean up
    path.unlink()


def test_python_file_type_ignored():
    """Test that detector ignores non-config file types"""
    config_data = {
        "debug": True
    }
    
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, temp_file)
    temp_file.flush()
    temp_file.close()
    
    path = Path(temp_file.name)
    
    # Create FileInfo with PYTHON type (not CONFIG_JSON)
    file_info = FileInfo(
        path=str(path),
        file_type=FileType.PYTHON,
        last_modified=path.stat().st_mtime,
        size=path.stat().st_size
    )
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should return no vulnerabilities for non-config files
    assert len(vulnerabilities) == 0
    
    # Clean up
    path.unlink()


def test_debug_string_values():
    """Test detection of debug mode with string values"""
    config_data = {
        "debug": "true",
        "DEBUG": "1"
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should detect both debug flags
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) >= 2
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_false_positive_on_debug_false():
    """Test that debug=false doesn't trigger false positive"""
    config_data = {
        "debug": False,
        "DEBUG": 0
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should not detect debug mode when disabled
    debug_vulns = [v for v in vulnerabilities if 'debug' in v.title.lower()]
    assert len(debug_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_false_positive_on_secure_password():
    """Test that secure passwords don't trigger false positive"""
    config_data = {
        "database": {
            "password": "secure_random_password_123!@#"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should not detect insecure default for secure password
    default_vulns = [v for v in vulnerabilities if 'default' in v.title.lower()]
    assert len(default_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


def test_no_false_positive_on_relative_path():
    """Test that relative paths don't trigger false positive"""
    config_data = {
        "paths": {
            "data_dir": "./data",
            "config_dir": "../config"
        }
    }
    
    file_info = create_test_json_file(config_data)
    
    config = DetectorConfig(enabled=True)
    detector = ConfigDetector(config)
    
    vulnerabilities = detector.detect(file_info, None)
    
    # Should not detect sensitive paths for relative paths
    path_vulns = [v for v in vulnerabilities if 'path' in v.title.lower() and 'sensitive' in v.title.lower()]
    assert len(path_vulns) == 0
    
    # Clean up
    Path(file_info.path).unlink()


if __name__ == '__main__':
    # Run all tests
    test_detect_debug_mode_enabled_json()
    test_detect_debug_mode_uppercase()
    test_detect_debug_mode_yaml()
    test_detect_insecure_default_password()
    test_detect_insecure_default_secret()
    test_detect_sensitive_path_unix()
    test_detect_sensitive_path_root()
    test_detect_missing_security_settings()
    test_no_missing_security_settings_when_present()
    test_detect_world_readable_sensitive_file()
    test_no_world_readable_vuln_when_restricted()
    test_multiple_issues_in_one_file()
    test_nested_configuration()
    test_confidence_scores()
    test_cwe_id_assignment()
    test_recommendations()
    test_disabled_detector()
    test_invalid_json_file()
    test_invalid_yaml_file()
    test_python_file_type_ignored()
    test_debug_string_values()
    test_no_false_positive_on_debug_false()
    test_no_false_positive_on_secure_password()
    test_no_false_positive_on_relative_path()
    
    print("All tests passed!")
