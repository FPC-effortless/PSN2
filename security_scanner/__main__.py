"""Entry point for running security_scanner as a module

This allows the scanner to be run with:
    python -m security_scanner
"""
import sys

from security_scanner.cli import main

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nScan interrupted by user.", file=sys.stderr)
        sys.exit(130)
