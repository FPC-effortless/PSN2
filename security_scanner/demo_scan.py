"""Demo script to show SecurityScanner in action"""

from security_scanner.models import ScanConfig, SeverityLevel
from security_scanner.scanner import SecurityScanner


def main():
    """Run a demo scan on the psn2 directory"""
    
    print("=" * 70)
    print("Security Scanner Demo")
    print("=" * 70)
    print()
    
    # Configure the scanner
    config = ScanConfig(
        target_path="psn2",  # Scan the psn2 directory
        exclude_patterns=["test", "__pycache__", ".git"],
        min_severity=SeverityLevel.INFO,
        incremental=False,
        cache_dir=None,
        suppression_rules={},
        fail_on_severity=[]
    )
    
    # Initialize scanner
    print("Initializing SecurityScanner...")
    scanner = SecurityScanner(config)
    print(f"✓ Initialized with {len(scanner.detectors)} detectors")
    print()
    
    # Run scan
    print("Starting scan of psn2 directory...")
    result = scanner.scan("psn2")
    print()
    
    # Display results
    print("=" * 70)
    print("Scan Results")
    print("=" * 70)
    print(f"Files scanned: {result.files_scanned}")
    print(f"Scan duration: {result.scan_duration:.2f} seconds")
    print(f"Total vulnerabilities: {len(result.vulnerabilities)}")
    print()
    
    # Group by severity
    by_severity = {}
    for vuln in result.vulnerabilities:
        severity = vuln.severity.value
        if severity not in by_severity:
            by_severity[severity] = []
        by_severity[severity].append(vuln)
    
    # Display summary by severity
    print("Vulnerabilities by Severity:")
    for severity in ["critical", "high", "medium", "low", "info"]:
        count = len(by_severity.get(severity, []))
        if count > 0:
            print(f"  {severity.upper()}: {count}")
    print()
    
    # Show first 5 vulnerabilities as examples
    if result.vulnerabilities:
        print("Sample Vulnerabilities (first 5):")
        print("-" * 70)
        for i, vuln in enumerate(result.vulnerabilities[:5], 1):
            print(f"\n{i}. {vuln.title}")
            print(f"   Severity: {vuln.severity.value.upper()}")
            print(f"   File: {vuln.file_path}:{vuln.line_number}")
            print(f"   Description: {vuln.description[:100]}...")
    
    print()
    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
