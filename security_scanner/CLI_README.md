# Security Scanner CLI

Command-line interface for the security vulnerability scanner.

## Installation

The CLI is part of the security_scanner package. No additional installation is required.

## Usage

### Basic Usage

Scan the current directory:
```bash
python -m security_scanner.cli
```

Scan a specific path:
```bash
python -m security_scanner.cli --path /path/to/code
```

### Output Formats

Generate JSON report:
```bash
python -m security_scanner.cli --format json --output report.json
```

Generate HTML report:
```bash
python -m security_scanner.cli --format html --output report.html
```

Generate Markdown report (default):
```bash
python -m security_scanner.cli --format markdown --output report.md
```

### Filtering Results

Filter by minimum severity:
```bash
python -m security_scanner.cli --severity high
```

Exclude files or directories:
```bash
python -m security_scanner.cli --exclude "test/*" "*.pyc" "__pycache__"
```

### Configuration File

Use a configuration file for complex settings:
```bash
python -m security_scanner.cli --config security-config.json
```

Example configuration file (`security-config.json`):
```json
{
  "target_path": ".",
  "exclude_patterns": ["test/*", "*.pyc"],
  "min_severity": "medium",
  "fail_on_severity": ["critical", "high"],
  "suppression_rules": {
    "INJ001": "Known false positive in legacy code"
  }
}
```

### CI/CD Integration

Fail build on critical or high severity vulnerabilities:
```bash
python -m security_scanner.cli --fail-on critical high
```

This will exit with code 1 if any critical or high severity vulnerabilities are found.

### Performance Options

Enable incremental scanning (only scan changed files):
```bash
python -m security_scanner.cli --incremental --cache-dir .security-cache
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--path` | Target directory or file to scan | Current directory |
| `--config` | Path to configuration file | None |
| `--output` | Output file path | stdout |
| `--format` | Output format (json, markdown, html) | markdown |
| `--severity` | Minimum severity level (critical, high, medium, low, info) | info |
| `--exclude` | Glob patterns for files to exclude | None |
| `--incremental` | Scan only changed files | False |
| `--cache-dir` | Directory for caching scan results | None |
| `--fail-on` | Exit with error on specified severity levels | None |

## Exit Codes

- **0**: Success (no vulnerabilities or below fail-on threshold)
- **1**: Vulnerabilities found that meet fail-on criteria
- **2**: Error during execution

## Examples

### Example 1: Quick Scan

Scan current directory and output to console:
```bash
python -m security_scanner.cli
```

### Example 2: CI/CD Pipeline

Scan with strict criteria for CI/CD:
```bash
python -m security_scanner.cli \
  --path src/ \
  --format json \
  --output security-report.json \
  --fail-on critical high \
  --exclude "test/*" "docs/*"
```

### Example 3: Incremental Scan

Fast incremental scan with caching:
```bash
python -m security_scanner.cli \
  --incremental \
  --cache-dir .security-cache \
  --severity medium
```

### Example 4: Custom Configuration

Use configuration file with custom suppression rules:
```bash
python -m security_scanner.cli --config .security-config.json
```

## Demo

Run the demo script to see the CLI in action:
```bash
python security_scanner/demo_cli.py
```

## Integration with CI/CD

### GitHub Actions

```yaml
- name: Security Scan
  run: |
    python -m security_scanner.cli \
      --format json \
      --output security-report.json \
      --fail-on critical high
  
- name: Upload Report
  uses: actions/upload-artifact@v2
  with:
    name: security-report
    path: security-report.json
```

### GitLab CI

```yaml
security_scan:
  script:
    - python -m security_scanner.cli --format json --output security-report.json --fail-on critical high
  artifacts:
    reports:
      security: security-report.json
```

## Troubleshooting

### Configuration File Not Found

Ensure the configuration file path is correct and the file exists:
```bash
python -m security_scanner.cli --config /full/path/to/config.json
```

### No Vulnerabilities Detected

Check that:
1. The target path contains Python files
2. The severity filter is not too restrictive
3. Files are not excluded by patterns

### Exit Code 2 (Error)

Check the error message in stderr for details. Common issues:
- Invalid configuration file format
- Permission denied accessing files
- Invalid command-line arguments
