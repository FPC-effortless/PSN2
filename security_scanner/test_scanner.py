"""Tests for SecurityScanner class"""

import tempfile
from pathlib import Path

from security_scanner.models import ScanConfig, SeverityLevel
from security_scanner.scanner import SecurityScanner


def test_scanner_initialization():
    """Test that SecurityScanner initializes correctly with detectors"""
    config = ScanConfig(
        target_path=".",
        exclude_patterns=[],
        min_severity=SeverityLevel.INFO,
        incremental=False,
        cache_dir=None,
        suppression_rules={},
        fail_on_severity=[]
    )
    
    scanner = SecurityScanner(config)
    
    # Verify detectors are initialized
    assert len(scanner.detectors) == 4, f"Expected 4 detectors, got {len(scanner.detectors)}"
    
    detector_names = [d.name for d in scanner.detectors]
    assert "InjectionDetector" in detector_names
    assert "SecretsDetector" in detector_names
    assert "PathTraversalDetector" in detector_names
    assert "DeserializationDetector" in detector_names
    
    print("✓ Scanner initialization test passed")


def test_file_discovery():
    """Test that file discovery finds Python files"""
    # Create a temporary directory with test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test files
        (tmppath / "test.py").write_text("print('hello')")
        (tmppath / "config.json").write_text("{}")
        (tmppath / "notebook.ipynb").write_text("{}")
        (tmppath / "readme.txt").write_text("readme")
        
        # Create subdirectory with more files
        subdir = tmppath / "subdir"
        subdir.mkdir()
        (subdir / "module.py").write_text("def foo(): pass")
        
        # Initialize scanner
        config = ScanConfig(
            target_path=str(tmppath),
            exclude_patterns=[],
            min_severity=SeverityLevel.INFO,
            incremental=False,
            cache_dir=None,
            suppression_rules={},
            fail_on_severity=[]
        )
        scanner = SecurityScanner(config)
        
        # Discover files
        files = scanner.discover_files(str(tmppath))
        
        # Verify correct files were discovered
        file_names = [Path(f.path).name for f in files]
        assert "test.py" in file_names
        assert "config.json" in file_names
        assert "notebook.ipynb" in file_names
        assert "module.py" in file_names
        assert "readme.txt" not in file_names  # Should be excluded
        
        print(f"✓ File discovery test passed - found {len(files)} files")


def test_scan_simple_file():
    """Test scanning a simple Python file"""
    # Create a temporary Python file with a vulnerability
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        test_file = tmppath / "vulnerable.py"
        
        # Write code with eval() vulnerability
        test_file.write_text("""
# Test file with vulnerability
user_input = input("Enter code: ")
result = eval(user_input)  # Code injection vulnerability
print(result)
""")
        
        # Initialize scanner
        config = ScanConfig(
            target_path=str(test_file),
            exclude_patterns=[],
            min_severity=SeverityLevel.INFO,
            incremental=False,
            cache_dir=None,
            suppression_rules={},
            fail_on_severity=[]
        )
        scanner = SecurityScanner(config)
        
        # Run scan
        result = scanner.scan(str(test_file))
        
        # Verify results
        assert result.files_scanned == 1
        assert len(result.vulnerabilities) > 0, "Expected to find vulnerabilities"
        
        # Check that eval() was detected
        vuln_titles = [v.title for v in result.vulnerabilities]
        assert any("eval" in title.lower() for title in vuln_titles), "Expected to detect eval() vulnerability"
        
        print(f"✓ Scan test passed - found {len(result.vulnerabilities)} vulnerabilities")


def test_exclusion_patterns():
    """Test that exclusion patterns work correctly"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        
        # Create test files
        (tmppath / "include.py").write_text("print('hello')")
        
        # Create __pycache__ directory (should be excluded by default)
        pycache = tmppath / "__pycache__"
        pycache.mkdir()
        (pycache / "cached.py").write_text("print('cached')")
        
        # Initialize scanner
        config = ScanConfig(
            target_path=str(tmppath),
            exclude_patterns=["exclude"],
            min_severity=SeverityLevel.INFO,
            incremental=False,
            cache_dir=None,
            suppression_rules={},
            fail_on_severity=[]
        )
        scanner = SecurityScanner(config)
        
        # Discover files
        files = scanner.discover_files(str(tmppath))
        
        # Verify __pycache__ files are excluded
        file_paths = [f.path for f in files]
        assert not any("__pycache__" in p for p in file_paths), "__pycache__ files should be excluded"
        assert any("include.py" in p for p in file_paths), "include.py should be found"
        
        print("✓ Exclusion patterns test passed")


if __name__ == "__main__":
    test_scanner_initialization()
    test_file_discovery()
    test_scan_simple_file()
    test_exclusion_patterns()
    print("\n✅ All scanner tests passed!")
