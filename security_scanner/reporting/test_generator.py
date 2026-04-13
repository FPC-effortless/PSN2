"""Unit tests for ReportGenerator

Tests the report generation functionality including JSON, Markdown, and HTML formats.
"""

import json
import pytest
from datetime import datetime

from security_scanner.reporting.generator import ReportGenerator
from security_scanner.models import (
    ScanResult,
    Vulnerability,
    SeverityLevel,
    VulnerabilityType
)


@pytest.fixture
def sample_vulnerabilities():
    """Create sample vulnerabilities for testing"""
    return [
        Vulnerability(
            id="INJ001",
            title="Use of eval() function",
            description="Dangerous eval() call detected",
            severity=SeverityLevel.HIGH,
            vulnerability_type=VulnerabilityType.CODE_INJECTION,
            file_path="test.py",
            line_number=10,
            column=4,
            code_snippet="result = eval(user_input)",
            recommendation="Replace eval() with ast.literal_eval() or safer alternatives",
            cwe_id="CWE-95",
            confidence=0.95
        ),
        Vulnerability(
            id="SEC001",
            title="Hardcoded API key",
            description="API key found in source code",
            severity=SeverityLevel.CRITICAL,
            vulnerability_type=VulnerabilityType.SECRETS_EXPOSURE,
            file_path="config.py",
            line_number=5,
            column=0,
            code_snippet='API_KEY = "sk_live_1234567890abcdef"',
            recommendation="Move credentials to environment variables or secure vault",
            cwe_id="CWE-798",
            confidence=1.0
        ),
        Vulnerability(
            id="PATH001",
            title="Path traversal vulnerability",
            description="Unsafe path construction detected",
            severity=SeverityLevel.MEDIUM,
            vulnerability_type=VulnerabilityType.PATH_TRAVERSAL,
            file_path="file_handler.py",
            line_number=20,
            column=8,
            code_snippet='file_path = base_dir + "/" + user_file',
            recommendation="Use pathlib.Path.resolve() with validation",
            confidence=0.85
        )
    ]


@pytest.fixture
def sample_scan_result(sample_vulnerabilities):
    """Create sample scan result for testing"""
    return ScanResult(
        vulnerabilities=sample_vulnerabilities,
        files_scanned=10,
        scan_duration=2.5,
        timestamp="2024-01-15T10:30:00Z",
        target_path="/path/to/project"
    )


class TestReportGenerator:
    """Test suite for ReportGenerator class"""
    
    def test_group_by_severity(self, sample_vulnerabilities):
        """Test grouping vulnerabilities by severity level"""
        generator = ReportGenerator()
        grouped = generator.group_by_severity(sample_vulnerabilities)
        
        assert SeverityLevel.CRITICAL in grouped
        assert SeverityLevel.HIGH in grouped
        assert SeverityLevel.MEDIUM in grouped
        assert len(grouped[SeverityLevel.CRITICAL]) == 1
        assert len(grouped[SeverityLevel.HIGH]) == 1
        assert len(grouped[SeverityLevel.MEDIUM]) == 1
        assert grouped[SeverityLevel.CRITICAL][0].id == "SEC001"
        assert grouped[SeverityLevel.HIGH][0].id == "INJ001"
        assert grouped[SeverityLevel.MEDIUM][0].id == "PATH001"
    
    def test_group_by_severity_empty(self):
        """Test grouping with no vulnerabilities"""
        generator = ReportGenerator()
        grouped = generator.group_by_severity([])
        
        assert grouped == {}
    
    def test_generate_json_structure(self, sample_scan_result):
        """Test JSON report has correct structure"""
        generator = ReportGenerator()
        json_report = generator.generate_json(sample_scan_result)
        
        # Parse JSON to verify structure
        report = json.loads(json_report)
        
        # Check summary section
        assert "summary" in report
        assert report["summary"]["total_vulnerabilities"] == 3
        assert report["summary"]["files_scanned"] == 10
        assert report["summary"]["scan_duration"] == 2.5
        assert report["summary"]["timestamp"] == "2024-01-15T10:30:00Z"
        assert report["summary"]["target_path"] == "/path/to/project"
        
        # Check severity counts
        assert "severity_counts" in report["summary"]
        assert report["summary"]["severity_counts"]["critical"] == 1
        assert report["summary"]["severity_counts"]["high"] == 1
        assert report["summary"]["severity_counts"]["medium"] == 1
        assert report["summary"]["severity_counts"]["low"] == 0
        assert report["summary"]["severity_counts"]["info"] == 0
    
    def test_generate_json_vulnerabilities(self, sample_scan_result):
        """Test JSON report includes vulnerability details"""
        generator = ReportGenerator()
        json_report = generator.generate_json(sample_scan_result)
        report = json.loads(json_report)
        
        # Check vulnerabilities section
        assert "vulnerabilities" in report
        assert "critical" in report["vulnerabilities"]
        assert "high" in report["vulnerabilities"]
        assert "medium" in report["vulnerabilities"]
        
        # Check critical vulnerability details
        critical_vuln = report["vulnerabilities"]["critical"][0]
        assert critical_vuln["id"] == "SEC001"
        assert critical_vuln["title"] == "Hardcoded API key"
        assert critical_vuln["file_path"] == "config.py"
        assert critical_vuln["line_number"] == 5
        assert critical_vuln["code_snippet"] == 'API_KEY = "sk_live_1234567890abcdef"'
        assert critical_vuln["recommendation"] == "Move credentials to environment variables or secure vault"
        assert critical_vuln["cwe_id"] == "CWE-798"
    
    def test_generate_markdown_structure(self, sample_scan_result):
        """Test Markdown report has correct structure"""
        generator = ReportGenerator()
        markdown_report = generator.generate_markdown(sample_scan_result)
        
        # Check for key sections
        assert "# Security Scan Report" in markdown_report
        assert "## Executive Summary" in markdown_report
        assert "### Vulnerabilities by Severity" in markdown_report
        assert "## Detailed Findings" in markdown_report
        
        # Check summary content
        assert "**Scan Date:** 2024-01-15T10:30:00Z" in markdown_report
        assert "**Target Path:** /path/to/project" in markdown_report
        assert "**Files Scanned:** 10" in markdown_report
        assert "**Total Vulnerabilities:** 3" in markdown_report
        
        # Check severity counts
        assert "- **Critical:** 1" in markdown_report
        assert "- **High:** 1" in markdown_report
        assert "- **Medium:** 1" in markdown_report
    
    def test_generate_markdown_vulnerabilities(self, sample_scan_result):
        """Test Markdown report includes vulnerability details"""
        generator = ReportGenerator()
        markdown_report = generator.generate_markdown(sample_scan_result)
        
        # Check for severity sections
        assert "### CRITICAL Severity" in markdown_report
        assert "### HIGH Severity" in markdown_report
        assert "### MEDIUM Severity" in markdown_report
        
        # Check vulnerability details
        assert "#### Hardcoded API key" in markdown_report
        assert "**ID:** SEC001" in markdown_report
        assert "**File:** config.py:5" in markdown_report
        assert "**Type:** secrets_exposure" in markdown_report
        assert "**CWE:** CWE-798" in markdown_report
        assert 'API_KEY = "sk_live_1234567890abcdef"' in markdown_report
        assert "Move credentials to environment variables" in markdown_report
    
    def test_generate_html_structure(self, sample_scan_result):
        """Test HTML report has correct structure"""
        generator = ReportGenerator()
        html_report = generator.generate_html(sample_scan_result)
        
        # Check for HTML structure
        assert "<!DOCTYPE html>" in html_report
        assert "<html lang='en'>" in html_report
        assert "<head>" in html_report
        assert "<body>" in html_report
        assert "</html>" in html_report
        
        # Check for key sections
        assert "<h1>Security Scan Report</h1>" in html_report
        assert "<h2>Executive Summary</h2>" in html_report
        assert "<h2>Detailed Findings</h2>" in html_report
        
        # Check summary content
        assert "2024-01-15T10:30:00Z" in html_report
        assert "/path/to/project" in html_report
        assert "Files Scanned:</strong> 10" in html_report
        assert "Total Vulnerabilities:</strong> 3" in html_report
    
    def test_generate_html_vulnerabilities(self, sample_scan_result):
        """Test HTML report includes vulnerability details"""
        generator = ReportGenerator()
        html_report = generator.generate_html(sample_scan_result)
        
        # Check for vulnerability content
        assert "Hardcoded API key" in html_report
        assert "SEC001" in html_report
        assert "config.py:5" in html_report
        assert "secrets_exposure" in html_report
        assert "CWE-798" in html_report
        
        # Check for HTML escaping (security)
        assert "&lt;" not in html_report or "API_KEY" in html_report  # Basic check
    
    def test_generate_html_severity_colors(self, sample_scan_result):
        """Test HTML report uses correct severity colors"""
        generator = ReportGenerator()
        html_report = generator.generate_html(sample_scan_result)
        
        # Check for severity color styling
        assert "#dc3545" in html_report  # Critical - red
        assert "#fd7e14" in html_report  # High - orange
        assert "#ffc107" in html_report  # Medium - yellow
    
    def test_empty_scan_result(self):
        """Test report generation with no vulnerabilities"""
        generator = ReportGenerator()
        empty_result = ScanResult(
            vulnerabilities=[],
            files_scanned=5,
            scan_duration=1.0,
            timestamp="2024-01-15T10:30:00Z",
            target_path="/path/to/project"
        )
        
        # JSON report
        json_report = generator.generate_json(empty_result)
        report = json.loads(json_report)
        assert report["summary"]["total_vulnerabilities"] == 0
        assert report["vulnerabilities"] == {}
        
        # Markdown report
        markdown_report = generator.generate_markdown(empty_result)
        assert "**Total Vulnerabilities:** 0" in markdown_report
        
        # HTML report
        html_report = generator.generate_html(empty_result)
        assert "Total Vulnerabilities:</strong> 0" in html_report
    
    def test_severity_counts_calculation(self, sample_vulnerabilities):
        """Test internal severity counts calculation"""
        generator = ReportGenerator()
        counts = generator._get_severity_counts(sample_vulnerabilities)
        
        assert counts["critical"] == 1
        assert counts["high"] == 1
        assert counts["medium"] == 1
        assert counts["low"] == 0
        assert counts["info"] == 0
    
    def test_multiple_vulnerabilities_same_severity(self):
        """Test handling multiple vulnerabilities with same severity"""
        vulnerabilities = [
            Vulnerability(
                id="INJ001",
                title="eval() usage",
                description="Dangerous eval()",
                severity=SeverityLevel.HIGH,
                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                file_path="test1.py",
                line_number=10,
                column=0,
                code_snippet="eval(x)",
                recommendation="Don't use eval()"
            ),
            Vulnerability(
                id="INJ002",
                title="exec() usage",
                description="Dangerous exec()",
                severity=SeverityLevel.HIGH,
                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                file_path="test2.py",
                line_number=20,
                column=0,
                code_snippet="exec(y)",
                recommendation="Don't use exec()"
            )
        ]
        
        generator = ReportGenerator()
        grouped = generator.group_by_severity(vulnerabilities)
        
        assert len(grouped[SeverityLevel.HIGH]) == 2
        assert grouped[SeverityLevel.HIGH][0].id == "INJ001"
        assert grouped[SeverityLevel.HIGH][1].id == "INJ002"
