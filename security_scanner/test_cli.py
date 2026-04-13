"""Unit tests for CLI interface

This module tests the SecurityCLI class functionality including:
- Argument parsing
- Configuration file loading
- Configuration merging
- Exit code logic
"""

import json
import tempfile
from pathlib import Path

from security_scanner.cli import SecurityCLI
from security_scanner.models import ScanResult, Vulnerability, SeverityLevel, VulnerabilityType


def test_argument_parsing():
    """Test that CLI arguments are parsed correctly"""
    cli = SecurityCLI()
    
    # Test default arguments
    args = cli.parse_args([])
    assert args.path == '.'
    assert args.format == 'markdown'
    assert args.severity == 'info'
    assert args.incremental is False
    assert args.exclude == []
    assert args.fail_on == []
    
    # Test custom arguments
    args = cli.parse_args([
        '--path', '/test/path',
        '--format', 'json',
        '--severity', 'high',
        '--incremental',
        '--exclude', 'test/*', '*.pyc',
        '--fail-on', 'critical', 'high'
    ])
    assert args.path == '/test/path'
    assert args.format == 'json'
    assert args.severity == 'high'
    assert args.incremental is True
    assert args.exclude == ['test/*', '*.pyc']
    assert args.fail_on == ['critical', 'high']
    
    print("✓ Argument parsing test passed")


def test_config_file_loading():
    """Test loading configuration from JSON file"""
    cli = SecurityCLI()
    
    # Create temporary config file
    config_data = {
        'target_path': '/custom/path',
        'exclude_patterns': ['*.log', 'temp/*'],
        'min_severity': 'medium',
        'suppression_rules': {
            'INJ001': 'Known false positive'
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    try:
        # Load config
        loaded_config = cli.load_config_file(config_path)
        
        assert loaded_config['target_path'] == '/custom/path'
        assert loaded_config['exclude_patterns'] == ['*.log', 'temp/*']
        assert loaded_config['min_severity'] == 'medium'
        assert 'INJ001' in loaded_config['suppression_rules']
        
        print("✓ Config file loading test passed")
    finally:
        Path(config_path).unlink()


def test_config_merging():
    """Test merging command-line args with file config"""
    cli = SecurityCLI()
    
    # File config
    file_config = {
        'target_path': '/file/path',
        'min_severity': 'low',
        'suppression_rules': {'INJ001': 'test'}
    }
    
    # Command-line args (override some settings)
    args = cli.parse_args([
        '--path', '/cli/path',
        '--severity', 'high',
        '--exclude', 'test/*'
    ])
    
    # Merge configs
    merged = cli.merge_config(args, file_config)
    
    # CLI args should override file config
    assert merged['target_path'] == '/cli/path'
    assert merged['min_severity'] == 'high'
    assert merged['exclude_patterns'] == ['test/*']
    
    # File config values should be preserved
    assert merged['suppression_rules'] == {'INJ001': 'test'}
    
    print("✓ Config merging test passed")


def test_scan_config_creation():
    """Test creating ScanConfig from dictionary"""
    cli = SecurityCLI()
    
    config_dict = {
        'target_path': '/test',
        'exclude_patterns': ['*.pyc'],
        'min_severity': 'medium',
        'incremental': True,
        'cache_dir': '/cache',
        'fail_on_severity': ['critical', 'high'],
        'suppression_rules': {}
    }
    
    scan_config = cli.create_scan_config(config_dict)
    
    assert scan_config.target_path == '/test'
    assert scan_config.exclude_patterns == ['*.pyc']
    assert scan_config.min_severity == SeverityLevel.MEDIUM
    assert scan_config.incremental is True
    assert scan_config.cache_dir == '/cache'
    assert SeverityLevel.CRITICAL in scan_config.fail_on_severity
    assert SeverityLevel.HIGH in scan_config.fail_on_severity
    
    print("✓ Scan config creation test passed")


def test_should_fail_logic():
    """Test exit code determination based on vulnerabilities"""
    cli = SecurityCLI()
    
    # Create mock vulnerabilities
    critical_vuln = Vulnerability(
        id='TEST001',
        title='Critical Issue',
        description='Test',
        severity=SeverityLevel.CRITICAL,
        vulnerability_type=VulnerabilityType.CODE_INJECTION,
        file_path='test.py',
        line_number=1,
        column=0,
        code_snippet='test',
        recommendation='Fix it'
    )
    
    low_vuln = Vulnerability(
        id='TEST002',
        title='Low Issue',
        description='Test',
        severity=SeverityLevel.LOW,
        vulnerability_type=VulnerabilityType.CONFIGURATION,
        file_path='test.py',
        line_number=2,
        column=0,
        code_snippet='test',
        recommendation='Fix it'
    )
    
    # Create mock scan result
    scan_result = ScanResult(
        vulnerabilities=[critical_vuln, low_vuln],
        files_scanned=1,
        scan_duration=1.0,
        timestamp='2024-01-01T00:00:00'
    )
    
    # Test: should fail on critical
    fail_on = [SeverityLevel.CRITICAL]
    assert cli.should_fail(scan_result, fail_on) is True
    
    # Test: should not fail on high (no high vulns present)
    fail_on = [SeverityLevel.HIGH]
    assert cli.should_fail(scan_result, fail_on) is False
    
    # Test: should fail on low
    fail_on = [SeverityLevel.LOW]
    assert cli.should_fail(scan_result, fail_on) is True
    
    # Test: empty fail_on list
    assert cli.should_fail(scan_result, []) is False
    
    print("✓ Should fail logic test passed")


def test_invalid_config_file():
    """Test handling of invalid config file"""
    cli = SecurityCLI()
    
    # Test non-existent file
    try:
        cli.load_config_file('/nonexistent/config.json')
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    
    # Test invalid JSON
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write('{ invalid json }')
        config_path = f.name
    
    try:
        cli.load_config_file(config_path)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    finally:
        Path(config_path).unlink()
    
    print("✓ Invalid config file test passed")


if __name__ == '__main__':
    test_argument_parsing()
    test_config_file_loading()
    test_config_merging()
    test_scan_config_creation()
    test_should_fail_logic()
    test_invalid_config_file()
    print("\n✅ All CLI tests passed!")
