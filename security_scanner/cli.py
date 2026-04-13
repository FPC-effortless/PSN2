"""Command-line interface for security scanner

This module provides the SecurityCLI class that handles:
- Command-line argument parsing
- Configuration file loading
- Scan execution orchestration
- Report generation and output
- Exit code logic based on severity thresholds
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from security_scanner.models import ScanConfig, SeverityLevel
from security_scanner.scanner import SecurityScanner
from security_scanner.reporting.generator import ReportGenerator


class SecurityCLI:
    """Command-line interface for the security scanner
    
    The SecurityCLI class orchestrates the security scanning workflow:
    1. Parse command-line arguments
    2. Load configuration from file if provided
    3. Initialize scanner with configuration
    4. Execute scan
    5. Generate report in requested format
    6. Return appropriate exit code based on findings
    
    Exit codes:
    - 0: Success (no vulnerabilities or below fail-on threshold)
    - 1: Vulnerabilities found that meet fail-on criteria
    - 2: Error during execution
    """
    
    def __init__(self):
        """Initialize the CLI with argument parser"""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            prog='security-scanner',
            description='Security vulnerability scanner for Python codebases',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Scan current directory
  security-scanner
  
  # Scan specific path with JSON output
  security-scanner --path /path/to/code --format json
  
  # Scan with configuration file
  security-scanner --config security-config.json
  
  # Fail on high or critical vulnerabilities
  security-scanner --fail-on critical high
  
  # Incremental scan with caching
  security-scanner --incremental --cache-dir .security-cache
            """
        )
        
        # Target path
        parser.add_argument(
            '--path',
            type=str,
            default='.',
            help='Target directory or file to scan (default: current directory)'
        )
        
        # Configuration file
        parser.add_argument(
            '--config',
            type=str,
            help='Path to configuration file for suppression rules and settings'
        )
        
        # Output options
        parser.add_argument(
            '--output',
            type=str,
            help='Output file path (default: stdout)'
        )
        
        parser.add_argument(
            '--format',
            type=str,
            choices=['json', 'markdown', 'html'],
            default='markdown',
            help='Output format (default: markdown)'
        )
        
        # Filtering options
        parser.add_argument(
            '--severity',
            type=str,
            choices=['critical', 'high', 'medium', 'low', 'info'],
            default='info',
            help='Minimum severity level to report (default: info)'
        )
        
        parser.add_argument(
            '--exclude',
            type=str,
            nargs='+',
            default=[],
            help='Glob patterns for files to exclude'
        )
        
        # Performance options
        parser.add_argument(
            '--incremental',
            action='store_true',
            help='Scan only changed files (requires git)'
        )
        
        parser.add_argument(
            '--cache-dir',
            type=str,
            help='Directory for caching scan results'
        )
        
        # Exit code control
        parser.add_argument(
            '--fail-on',
            type=str,
            nargs='+',
            choices=['critical', 'high', 'medium', 'low', 'info'],
            default=[],
            help='Exit with non-zero code on specified severity levels'
        )
        
        return parser
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments
        
        Args:
            args: List of arguments to parse (None to use sys.argv)
            
        Returns:
            Parsed arguments namespace
        """
        return self.parser.parse_args(args)
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dictionary containing configuration settings
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
    
    def merge_config(self, args: argparse.Namespace, file_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge command-line arguments with file configuration
        
        Command-line arguments take precedence over file configuration.
        
        Args:
            args: Parsed command-line arguments
            file_config: Configuration loaded from file
            
        Returns:
            Merged configuration dictionary
        """
        # Start with file config
        merged = file_config.copy()
        
        # Override with command-line arguments (if provided)
        if args.path != '.':
            merged['target_path'] = args.path
        elif 'target_path' not in merged:
            merged['target_path'] = '.'
        
        if args.exclude:
            merged['exclude_patterns'] = args.exclude
        elif 'exclude_patterns' not in merged:
            merged['exclude_patterns'] = []
        
        if args.severity:
            merged['min_severity'] = args.severity
        elif 'min_severity' not in merged:
            merged['min_severity'] = 'info'
        
        if args.incremental:
            merged['incremental'] = True
        elif 'incremental' not in merged:
            merged['incremental'] = False
        
        if args.cache_dir:
            merged['cache_dir'] = args.cache_dir
        elif 'cache_dir' not in merged:
            merged['cache_dir'] = None
        
        if args.fail_on:
            merged['fail_on_severity'] = args.fail_on
        elif 'fail_on_severity' not in merged:
            merged['fail_on_severity'] = []
        
        # Preserve suppression_rules from file config
        if 'suppression_rules' not in merged:
            merged['suppression_rules'] = {}
        
        return merged
    
    def create_scan_config(self, config_dict: Dict[str, Any]) -> ScanConfig:
        """Create ScanConfig object from configuration dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ScanConfig instance
        """
        # Convert severity strings to SeverityLevel enums
        min_severity = SeverityLevel(config_dict.get('min_severity', 'info'))
        
        fail_on_severity = [
            SeverityLevel(s) for s in config_dict.get('fail_on_severity', [])
        ]
        
        return ScanConfig(
            target_path=config_dict.get('target_path', '.'),
            exclude_patterns=config_dict.get('exclude_patterns', []),
            min_severity=min_severity,
            incremental=config_dict.get('incremental', False),
            cache_dir=config_dict.get('cache_dir'),
            suppression_rules=config_dict.get('suppression_rules', {}),
            fail_on_severity=fail_on_severity
        )
    
    def should_fail(self, scan_result, fail_on_severity: List[SeverityLevel]) -> bool:
        """Determine if scan should exit with error code
        
        Args:
            scan_result: ScanResult from scanner
            fail_on_severity: List of severity levels that trigger failure
            
        Returns:
            True if any vulnerability meets fail-on criteria, False otherwise
        """
        if not fail_on_severity:
            return False
        
        for vuln in scan_result.vulnerabilities:
            if vuln.severity in fail_on_severity:
                return True
        
        return False
    
    def run(self, args: Optional[argparse.Namespace] = None) -> int:
        """Execute security scan and return exit code
        
        This is the main entry point for CLI execution. It:
        1. Loads configuration
        2. Initializes scanner
        3. Executes scan
        4. Generates report
        5. Returns appropriate exit code
        
        Args:
            args: Parsed arguments (None to parse from sys.argv)
            
        Returns:
            Exit code (0 for success, 1 for vulnerabilities found, 2 for error)
        """
        try:
            # Parse arguments if not provided
            if args is None:
                args = self.parse_args()
            
            # Load configuration
            config_dict = {}
            if args.config:
                try:
                    file_config = self.load_config_file(args.config)
                    config_dict = self.merge_config(args, file_config)
                except (FileNotFoundError, ValueError) as e:
                    print(f"Error loading configuration: {e}", file=sys.stderr)
                    return 2
            else:
                # Use only command-line arguments
                config_dict = {
                    'target_path': args.path,
                    'exclude_patterns': args.exclude,
                    'min_severity': args.severity,
                    'incremental': args.incremental,
                    'cache_dir': args.cache_dir,
                    'fail_on_severity': args.fail_on,
                    'suppression_rules': {}
                }
            
            # Create scan configuration
            scan_config = self.create_scan_config(config_dict)
            
            # Initialize scanner
            scanner = SecurityScanner(scan_config)
            
            # Execute scan
            print(f"Scanning: {scan_config.target_path}", file=sys.stderr)
            scan_result = scanner.scan(scan_config.target_path)
            
            # Filter vulnerabilities by minimum severity
            filtered_vulns = [
                v for v in scan_result.vulnerabilities
                if v.severity.value != 'info' or scan_config.min_severity == SeverityLevel.INFO
            ]
            
            # Apply minimum severity filter
            severity_order = {
                SeverityLevel.INFO: 0,
                SeverityLevel.LOW: 1,
                SeverityLevel.MEDIUM: 2,
                SeverityLevel.HIGH: 3,
                SeverityLevel.CRITICAL: 4
            }
            min_level = severity_order[scan_config.min_severity]
            filtered_vulns = [
                v for v in filtered_vulns
                if severity_order[v.severity] >= min_level
            ]
            
            # Update scan result with filtered vulnerabilities
            scan_result.vulnerabilities = filtered_vulns
            
            # Generate report
            generator = ReportGenerator()
            
            if args.format == 'json':
                report = generator.generate_json(scan_result)
            elif args.format == 'html':
                report = generator.generate_html(scan_result)
            else:  # markdown
                report = generator.generate_markdown(scan_result)
            
            # Output report
            if args.output:
                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"Report written to: {args.output}", file=sys.stderr)
            else:
                print(report)
            
            # Determine exit code
            if self.should_fail(scan_result, scan_config.fail_on_severity):
                print(f"\nScan failed: Found vulnerabilities matching fail-on criteria", file=sys.stderr)
                return 1
            
            print(f"\nScan complete: {len(scan_result.vulnerabilities)} vulnerabilities found", file=sys.stderr)
            return 0
            
        except Exception as e:
            print(f"Error during scan: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            return 2


def main():
    """Main entry point for command-line execution"""
    cli = SecurityCLI()
    exit_code = cli.run()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
