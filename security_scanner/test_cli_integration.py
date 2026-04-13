"""Integration tests for CLI

This module tests the CLI end-to-end functionality by:
- Running actual scans
- Testing output formats
- Verifying exit codes
"""

import json
import tempfile
from pathlib import Path

from security_scanner.cli import SecurityCLI


def test_cli_basic_scan():
    """Test basic CLI scan execution"""
    cli = SecurityCLI()
    
    # Create a temporary Python file with a vulnerability
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Run scan
        args = cli.parse_args(['--path', tmpdir, '--format', 'json'])
        exit_code = cli.run(args)
        
        # Should succeed (no fail-on criteria)
        assert exit_code == 0
        
    print("✓ Basic scan test passed")


def test_cli_with_fail_on():
    """Test CLI with fail-on criteria"""
    cli = SecurityCLI()
    
    # Create a temporary Python file with a medium severity vulnerability
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Run scan with fail-on medium
        args = cli.parse_args([
            '--path', tmpdir,
            '--fail-on', 'medium', 'high', 'critical'
        ])
        exit_code = cli.run(args)
        
        # Should fail (eval without user-controlled input is medium severity)
        assert exit_code == 1
        
    print("✓ Fail-on test passed")


def test_cli_json_output():
    """Test CLI JSON output format"""
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Create output file
        output_file = Path(tmpdir) / 'report.json'
        
        # Run scan
        args = cli.parse_args([
            '--path', tmpdir,
            '--format', 'json',
            '--output', str(output_file)
        ])
        exit_code = cli.run(args)
        
        # Verify output file exists and is valid JSON
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            report = json.load(f)
        
        assert 'summary' in report
        assert 'vulnerabilities' in report
        assert report['summary']['total_vulnerabilities'] > 0
        
    print("✓ JSON output test passed")


def test_cli_with_config_file():
    """Test CLI with configuration file"""
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Create config file
        config_file = Path(tmpdir) / 'config.json'
        config_data = {
            'target_path': tmpdir,
            'min_severity': 'high',
            'fail_on_severity': ['critical']
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f)
        
        # Run scan with config
        args = cli.parse_args(['--config', str(config_file)])
        exit_code = cli.run(args)
        
        # Should succeed (eval is high, not critical)
        assert exit_code == 0
        
    print("✓ Config file test passed")


def test_cli_severity_filtering():
    """Test CLI severity filtering"""
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file with low severity issue
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('''
def process(value):
    # Some code
    pass
''')
        
        # Run scan with high severity filter
        output_file = Path(tmpdir) / 'report.json'
        args = cli.parse_args([
            '--path', tmpdir,
            '--severity', 'high',
            '--format', 'json',
            '--output', str(output_file)
        ])
        exit_code = cli.run(args)
        
        # Should succeed with no high severity issues
        assert exit_code == 0
        
        # Verify report shows filtered results
        with open(output_file, 'r') as f:
            report = json.load(f)
        
        # All reported vulnerabilities should be high or critical
        for severity_level, vulns in report['vulnerabilities'].items():
            assert severity_level in ['high', 'critical']
        
    print("✓ Severity filtering test passed")


if __name__ == '__main__':
    test_cli_basic_scan()
    test_cli_with_fail_on()
    test_cli_json_output()
    test_cli_with_config_file()
    test_cli_severity_filtering()
    print("\n✅ All CLI integration tests passed!")
