"""Demonstration script for ReportGenerator

This script demonstrates the ReportGenerator functionality by creating
sample vulnerabilities and generating reports in all supported formats.
"""

from security_scanner.reporting import ReportGenerator
from security_scanner.models import (
    ScanResult,
    Vulnerability,
    SeverityLevel,
    VulnerabilityType
)


def create_sample_data():
    """Create sample vulnerabilities for demonstration"""
    vulnerabilities = [
        Vulnerability(
            id="INJ001",
            title="Use of eval() function",
            description="Dangerous eval() call detected with user-controlled input",
            severity=SeverityLevel.HIGH,
            vulnerability_type=VulnerabilityType.CODE_INJECTION,
            file_path="app/processor.py",
            line_number=45,
            column=8,
            code_snippet="result = eval(user_input)",
            recommendation="Replace eval() with ast.literal_eval() for safe evaluation of literals, or use a proper parser",
            cwe_id="CWE-95",
            confidence=0.95
        ),
        Vulnerability(
            id="SEC001",
            title="Hardcoded API key",
            description="API key found hardcoded in source code",
            severity=SeverityLevel.CRITICAL,
            vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
            file_path="config/settings.py",
            line_number=12,
            column=0,
            code_snippet='API_KEY = "sk_live_1234567890abcdef"',
            recommendation="Move credentials to environment variables or use a secure secrets management system",
            cwe_id="CWE-798",
            confidence=1.0
        ),
        Vulnerability(
            id="PATH001",
            title="Path traversal vulnerability",
            description="Unsafe path construction allows directory traversal",
            severity=SeverityLevel.MEDIUM,
            vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
            file_path="utils/file_handler.py",
            line_number=78,
            column=12,
            code_snippet='file_path = base_dir + "/" + user_filename',
            recommendation="Use pathlib.Path.resolve() with validation to ensure paths stay within allowed directories",
            cwe_id="CWE-22",
            confidence=0.85
        ),
        Vulnerability(
            id="DESER001",
            title="Unsafe deserialization",
            description="torch.load() used without weights_only parameter",
            severity=SeverityLevel.HIGH,
            vulnerability_type=VulnerabilityType.UNSAFE_DESERIALIZATION,
            file_path="models/checkpoint_loader.py",
            line_number=34,
            column=16,
            code_snippet='model_state = torch.load(checkpoint_path)',
            recommendation="Use torch.load(checkpoint_path, weights_only=True) to prevent arbitrary code execution",
            cwe_id="CWE-502",
            confidence=0.90
        )
    ]
    
    scan_result = ScanResult(
        vulnerabilities=vulnerabilities,
        files_scanned=25,
        scan_duration=3.7,
        timestamp="2024-01-15T14:30:00Z",
        target_path="/home/user/project"
    )
    
    return scan_result


def main():
    """Generate and display sample reports"""
    print("=" * 70)
    print("ReportGenerator Demonstration")
    print("=" * 70)
    print()
    
    # Create sample data
    scan_result = create_sample_data()
    generator = ReportGenerator()
    
    # Generate JSON report
    print("Generating JSON report...")
    json_report = generator.generate_json(scan_result)
    print(f"JSON report size: {len(json_report)} bytes")
    print()
    
    # Generate Markdown report
    print("Generating Markdown report...")
    markdown_report = generator.generate_markdown(scan_result)
    print(f"Markdown report size: {len(markdown_report)} bytes")
    print()
    
    # Generate HTML report
    print("Generating HTML report...")
    html_report = generator.generate_html(scan_result)
    print(f"HTML report size: {len(html_report)} bytes")
    print()
    
    # Display summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total vulnerabilities: {len(scan_result.vulnerabilities)}")
    print(f"Files scanned: {scan_result.files_scanned}")
    print(f"Scan duration: {scan_result.scan_duration}s")
    print()
    
    # Display severity breakdown
    grouped = generator.group_by_severity(scan_result.vulnerabilities)
    print("Vulnerabilities by severity:")
    for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM,
                    SeverityLevel.LOW, SeverityLevel.INFO]:
        count = len(grouped.get(severity, []))
        if count > 0:
            print(f"  {severity.value.upper()}: {count}")
    print()
    
    print("All reports generated successfully!")
    print()
    
    # Optionally save reports to files
    save = input("Save reports to files? (y/n): ").strip().lower()
    if save == 'y':
        with open("security_report.json", "w") as f:
            f.write(json_report)
        print("Saved: security_report.json")
        
        with open("security_report.md", "w") as f:
            f.write(markdown_report)
        print("Saved: security_report.md")
        
        with open("security_report.html", "w") as f:
            f.write(html_report)
        print("Saved: security_report.html")


if __name__ == "__main__":
    main()
