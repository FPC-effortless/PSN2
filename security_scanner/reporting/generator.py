"""Report generator for security scan results

This module provides the ReportGenerator class for generating vulnerability
reports in multiple formats (JSON, Markdown, HTML).
"""

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
from html import escape

from security_scanner.models import ScanResult, Vulnerability, SeverityLevel


class ReportGenerator:
    """Generates vulnerability reports in multiple formats
    
    The ReportGenerator takes scan results and produces formatted reports
    in JSON, Markdown, or HTML format. Reports include:
    - Executive summary with vulnerability counts by severity
    - Detailed vulnerability listings grouped by severity
    - File path, line number, code snippet, and recommendations
    - Scan metadata (timestamp, duration, files scanned)
    """
    
    def group_by_severity(self, vulnerabilities: List[Vulnerability]) -> Dict[SeverityLevel, List[Vulnerability]]:
        """Group vulnerabilities by severity level
        
        Args:
            vulnerabilities: List of detected vulnerabilities
            
        Returns:
            Dictionary mapping severity levels to lists of vulnerabilities
        """
        grouped: Dict[SeverityLevel, List[Vulnerability]] = defaultdict(list)
        for vuln in vulnerabilities:
            grouped[vuln.severity].append(vuln)
        return dict(grouped)
    
    def _get_severity_counts(self, vulnerabilities: List[Vulnerability]) -> Dict[str, int]:
        """Calculate vulnerability counts by severity
        
        Args:
            vulnerabilities: List of detected vulnerabilities
            
        Returns:
            Dictionary mapping severity level names to counts
        """
        counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0
        }
        for vuln in vulnerabilities:
            counts[vuln.severity.value] += 1
        return counts
    
    def generate_json(self, scan_result: ScanResult) -> str:
        """Generate JSON format report
        
        Args:
            scan_result: Scan results to format
            
        Returns:
            JSON string containing formatted report
        """
        severity_counts = self._get_severity_counts(scan_result.vulnerabilities)
        grouped = self.group_by_severity(scan_result.vulnerabilities)
        
        # Build JSON structure
        report = {
            "summary": {
                "total_vulnerabilities": len(scan_result.vulnerabilities),
                "files_scanned": scan_result.files_scanned,
                "scan_duration": scan_result.scan_duration,
                "timestamp": scan_result.timestamp,
                "target_path": scan_result.target_path,
                "severity_counts": severity_counts
            },
            "vulnerabilities": {}
        }
        
        # Add vulnerabilities grouped by severity
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, 
                        SeverityLevel.LOW, SeverityLevel.INFO]:
            if severity in grouped:
                report["vulnerabilities"][severity.value] = [
                    {
                        "id": vuln.id,
                        "title": vuln.title,
                        "description": vuln.description,
                        "file_path": vuln.file_path,
                        "line_number": vuln.line_number,
                        "column": vuln.column,
                        "code_snippet": vuln.code_snippet,
                        "recommendation": vuln.recommendation,
                        "vulnerability_type": vuln.vulnerability_type.value,
                        "cwe_id": vuln.cwe_id,
                        "confidence": vuln.confidence
                    }
                    for vuln in grouped[severity]
                ]
        
        return json.dumps(report, indent=2)
    
    def generate_markdown(self, scan_result: ScanResult) -> str:
        """Generate Markdown format report
        
        Args:
            scan_result: Scan results to format
            
        Returns:
            Markdown string containing formatted report
        """
        severity_counts = self._get_severity_counts(scan_result.vulnerabilities)
        grouped = self.group_by_severity(scan_result.vulnerabilities)
        
        lines = []
        lines.append("# Security Scan Report")
        lines.append("")
        
        # Summary section
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"**Scan Date:** {scan_result.timestamp}")
        lines.append(f"**Target Path:** {scan_result.target_path}")
        lines.append(f"**Files Scanned:** {scan_result.files_scanned}")
        lines.append(f"**Scan Duration:** {scan_result.scan_duration:.2f}s")
        lines.append(f"**Total Vulnerabilities:** {len(scan_result.vulnerabilities)}")
        lines.append("")
        
        # Severity breakdown
        lines.append("### Vulnerabilities by Severity")
        lines.append("")
        lines.append(f"- **Critical:** {severity_counts['critical']}")
        lines.append(f"- **High:** {severity_counts['high']}")
        lines.append(f"- **Medium:** {severity_counts['medium']}")
        lines.append(f"- **Low:** {severity_counts['low']}")
        lines.append(f"- **Info:** {severity_counts['info']}")
        lines.append("")
        
        # Detailed vulnerabilities by severity
        lines.append("## Detailed Findings")
        lines.append("")
        
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM,
                        SeverityLevel.LOW, SeverityLevel.INFO]:
            if severity in grouped:
                lines.append(f"### {severity.value.upper()} Severity")
                lines.append("")
                
                for vuln in grouped[severity]:
                    lines.append(f"#### {vuln.title}")
                    lines.append("")
                    lines.append(f"**ID:** {vuln.id}")
                    lines.append(f"**File:** {vuln.file_path}:{vuln.line_number}")
                    lines.append(f"**Type:** {vuln.vulnerability_type.value}")
                    if vuln.cwe_id:
                        lines.append(f"**CWE:** {vuln.cwe_id}")
                    lines.append("")
                    lines.append(f"**Description:** {vuln.description}")
                    lines.append("")
                    lines.append("**Code Snippet:**")
                    lines.append("```")
                    lines.append(vuln.code_snippet)
                    lines.append("```")
                    lines.append("")
                    lines.append(f"**Recommendation:** {vuln.recommendation}")
                    lines.append("")
                    lines.append("---")
                    lines.append("")
        
        return "\n".join(lines)
    
    def generate_html(self, scan_result: ScanResult) -> str:
        """Generate HTML format report
        
        Args:
            scan_result: Scan results to format
            
        Returns:
            HTML string containing formatted report
        """
        severity_counts = self._get_severity_counts(scan_result.vulnerabilities)
        grouped = self.group_by_severity(scan_result.vulnerabilities)
        
        # Severity color mapping
        severity_colors = {
            "critical": "#dc3545",
            "high": "#fd7e14",
            "medium": "#ffc107",
            "low": "#17a2b8",
            "info": "#6c757d"
        }
        
        html_parts = []
        html_parts.append("<!DOCTYPE html>")
        html_parts.append("<html lang='en'>")
        html_parts.append("<head>")
        html_parts.append("    <meta charset='UTF-8'>")
        html_parts.append("    <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
        html_parts.append("    <title>Security Scan Report</title>")
        html_parts.append("    <style>")
        html_parts.append("        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }")
        html_parts.append("        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }")
        html_parts.append("        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }")
        html_parts.append("        h2 { color: #555; margin-top: 30px; }")
        html_parts.append("        h3 { color: #666; margin-top: 20px; }")
        html_parts.append("        .summary { background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }")
        html_parts.append("        .summary-item { margin: 10px 0; }")
        html_parts.append("        .severity-badge { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; margin-right: 10px; }")
        html_parts.append("        .vulnerability { border: 1px solid #ddd; padding: 20px; margin: 15px 0; border-radius: 5px; background-color: #fafafa; }")
        html_parts.append("        .vulnerability h4 { margin-top: 0; color: #333; }")
        html_parts.append("        .vuln-meta { color: #666; font-size: 0.9em; margin: 10px 0; }")
        html_parts.append("        .code-snippet { background-color: #f4f4f4; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; overflow-x: auto; font-family: 'Courier New', monospace; font-size: 0.9em; }")
        html_parts.append("        .recommendation { background-color: #e7f3ff; border-left: 4px solid #007bff; padding: 10px; margin: 10px 0; }")
        html_parts.append("    </style>")
        html_parts.append("</head>")
        html_parts.append("<body>")
        html_parts.append("    <div class='container'>")
        html_parts.append("        <h1>Security Scan Report</h1>")
        
        # Summary section
        html_parts.append("        <div class='summary'>")
        html_parts.append("            <h2>Executive Summary</h2>")
        html_parts.append(f"            <div class='summary-item'><strong>Scan Date:</strong> {escape(scan_result.timestamp)}</div>")
        html_parts.append(f"            <div class='summary-item'><strong>Target Path:</strong> {escape(scan_result.target_path)}</div>")
        html_parts.append(f"            <div class='summary-item'><strong>Files Scanned:</strong> {scan_result.files_scanned}</div>")
        html_parts.append(f"            <div class='summary-item'><strong>Scan Duration:</strong> {scan_result.scan_duration:.2f}s</div>")
        html_parts.append(f"            <div class='summary-item'><strong>Total Vulnerabilities:</strong> {len(scan_result.vulnerabilities)}</div>")
        html_parts.append("            <h3>Vulnerabilities by Severity</h3>")
        
        for severity_name, count in severity_counts.items():
            color = severity_colors[severity_name]
            html_parts.append(f"            <div class='summary-item'>")
            html_parts.append(f"                <span class='severity-badge' style='background-color: {color};'>{severity_name.upper()}</span>")
            html_parts.append(f"                {count}")
            html_parts.append(f"            </div>")
        
        html_parts.append("        </div>")
        
        # Detailed vulnerabilities
        html_parts.append("        <h2>Detailed Findings</h2>")
        
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM,
                        SeverityLevel.LOW, SeverityLevel.INFO]:
            if severity in grouped:
                color = severity_colors[severity.value]
                html_parts.append(f"        <h3><span class='severity-badge' style='background-color: {color};'>{severity.value.upper()}</span> Severity</h3>")
                
                for vuln in grouped[severity]:
                    html_parts.append("        <div class='vulnerability'>")
                    html_parts.append(f"            <h4>{escape(vuln.title)}</h4>")
                    html_parts.append(f"            <div class='vuln-meta'>")
                    html_parts.append(f"                <strong>ID:</strong> {escape(vuln.id)} | ")
                    html_parts.append(f"                <strong>File:</strong> {escape(vuln.file_path)}:{vuln.line_number} | ")
                    html_parts.append(f"                <strong>Type:</strong> {escape(vuln.vulnerability_type.value)}")
                    if vuln.cwe_id:
                        html_parts.append(f" | <strong>CWE:</strong> {escape(vuln.cwe_id)}")
                    html_parts.append(f"            </div>")
                    html_parts.append(f"            <p><strong>Description:</strong> {escape(vuln.description)}</p>")
                    html_parts.append(f"            <div class='code-snippet'>")
                    html_parts.append(f"                <strong>Code Snippet:</strong><br>")
                    html_parts.append(f"                <pre>{escape(vuln.code_snippet)}</pre>")
                    html_parts.append(f"            </div>")
                    html_parts.append(f"            <div class='recommendation'>")
                    html_parts.append(f"                <strong>Recommendation:</strong> {escape(vuln.recommendation)}")
                    html_parts.append(f"            </div>")
                    html_parts.append("        </div>")
        
        html_parts.append("    </div>")
        html_parts.append("</body>")
        html_parts.append("</html>")
        
        return "\n".join(html_parts)
