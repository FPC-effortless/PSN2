# Security Scanner

Automated security vulnerability detection for Python ML codebases.

## Project Structure

```
security_scanner/
├── __init__.py           # Package initialization
├── models.py             # Core data models (Vulnerability, ScanResult, etc.)
├── detectors/            # Vulnerability detectors
├── parsers/              # File parsers (Python, Jupyter, config)
├── reporting/            # Report generation (JSON, Markdown, HTML)
├── cache/                # Result caching for incremental scans
└── filters/              # False positive filtering

tests/
└── test_models.py        # Unit and property-based tests for data models
```

## Core Data Models

### Enums
- **SeverityLevel**: CRITICAL, HIGH, MEDIUM, LOW, INFO
- **VulnerabilityType**: 10 categories (code injection, secrets, path traversal, etc.)
- **FileType**: PYTHON, NOTEBOOK, CONFIG_JSON, CONFIG_YAML, REQUIREMENTS

### Dataclasses
- **Vulnerability**: Represents a detected security issue
- **FileInfo**: Information about files to scan
- **ScanResult**: Results from a security scan
- **ScanConfig**: Configuration for scanning
- **DetectorConfig**: Configuration for individual detectors

## Testing

Run tests with pytest:
```bash
pytest tests/test_models.py -v
```

Property-based tests use Hypothesis for comprehensive validation.
