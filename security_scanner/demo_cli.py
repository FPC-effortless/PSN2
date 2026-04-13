"""Demo script for CLI functionality

This script demonstrates the SecurityCLI in action by:
1. Creating a sample Python file with vulnerabilities
2. Running the scanner with different configurations
3. Showing different output formats
"""

import tempfile
from pathlib import Path

from security_scanner.cli import SecurityCLI


def demo_basic_scan():
    """Demo: Basic scan with markdown output"""
    print("=" * 70)
    print("DEMO 1: Basic Scan with Markdown Output")
    print("=" * 70)
    
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample file with vulnerabilities
        test_file = Path(tmpdir) / 'vulnerable.py'
        test_file.write_text('''
import pickle

def process_data(user_input):
    # Dangerous: eval with user input
    result = eval(user_input)
    return result

def load_model(model_path):
    # Dangerous: pickle.load without validation
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

API_KEY = "sk-1234567890abcdef1234567890abcdef"
''')
        
        # Run scan
        args = cli.parse_args(['--path', tmpdir])
        cli.run(args)
    
    print()


def demo_json_output():
    """Demo: Scan with JSON output to file"""
    print("=" * 70)
    print("DEMO 2: Scan with JSON Output to File")
    print("=" * 70)
    
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample file
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Output file
        output_file = Path(tmpdir) / 'report.json'
        
        # Run scan
        args = cli.parse_args([
            '--path', tmpdir,
            '--format', 'json',
            '--output', str(output_file)
        ])
        cli.run(args)
        
        # Show the output file content
        print("\nGenerated report content:")
        print(output_file.read_text())
    
    print()


def demo_severity_filtering():
    """Demo: Scan with severity filtering"""
    print("=" * 70)
    print("DEMO 3: Scan with Severity Filtering (High and Critical only)")
    print("=" * 70)
    
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample file with mixed severity issues
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('''
import pickle

# High severity: pickle.load
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Medium severity: eval without user-controlled input
result = eval("2 + 2")
''')
        
        # Run scan with high severity filter
        args = cli.parse_args([
            '--path', tmpdir,
            '--severity', 'high'
        ])
        cli.run(args)
    
    print()


def demo_fail_on():
    """Demo: Scan with fail-on criteria"""
    print("=" * 70)
    print("DEMO 4: Scan with Fail-On Criteria")
    print("=" * 70)
    
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample file
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('result = eval(user_input)')
        
        # Run scan with fail-on
        args = cli.parse_args([
            '--path', tmpdir,
            '--fail-on', 'medium', 'high', 'critical'
        ])
        exit_code = cli.run(args)
        
        print(f"\nExit code: {exit_code}")
        print("(Exit code 1 means vulnerabilities were found that match fail-on criteria)")
    
    print()


def demo_config_file():
    """Demo: Scan with configuration file"""
    print("=" * 70)
    print("DEMO 5: Scan with Configuration File")
    print("=" * 70)
    
    import json
    
    cli = SecurityCLI()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a sample file
        test_file = Path(tmpdir) / 'test.py'
        test_file.write_text('''
result = eval(user_input)
password = "hardcoded123"
''')
        
        # Create config file
        config_file = Path(tmpdir) / 'security-config.json'
        config_data = {
            'target_path': tmpdir,
            'min_severity': 'high',
            'fail_on_severity': ['critical'],
            'suppression_rules': {}
        }
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"Configuration file content:")
        print(config_file.read_text())
        print()
        
        # Run scan with config
        args = cli.parse_args(['--config', str(config_file)])
        cli.run(args)
    
    print()


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Security Scanner CLI Demo")
    print("=" * 70 + "\n")
    
    demo_basic_scan()
    demo_json_output()
    demo_severity_filtering()
    demo_fail_on()
    demo_config_file()
    
    print("=" * 70)
    print("Demo Complete!")
    print("=" * 70)
