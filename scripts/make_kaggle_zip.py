"""Build psn2_kaggle_v3.zip — clean package for Kaggle upload."""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.parent

EXCLUDE_DIRS = {'.git', '__pycache__', '.pytest_cache', 'docs', 'tests'}
EXCLUDE_FILES = {
    '.gitignore', 'prd.json',
    'DEPLOYMENT_SUMMARY.md', 'FINAL_STATUS.md',
    'KAGGLE_READY.md', 'TRAINING_READINESS_REVIEW.md',
}
EXCLUDE_SUFFIXES = {'.zip', '.tmp', '.pyc', '.bat', '.sh'}
EXCLUDE_PREFIXES = {'update_kaggle_dataset'}

OUT = ROOT / 'psn2_kaggle_v4.zip'

def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    parts = rel.parts
    # Skip if any parent dir is excluded
    for part in parts[:-1]:
        if part in EXCLUDE_DIRS:
            return False
    name = path.name
    if name in EXCLUDE_FILES:
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    for prefix in EXCLUDE_PREFIXES:
        if name.startswith(prefix):
            return False
    return True

files = [p for p in ROOT.rglob('*') if p.is_file() and should_include(p)]
files.sort()

if OUT.exists():
    OUT.unlink()

with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in files:
        arcname = str(f.relative_to(ROOT)).replace('\\', '/')
        zf.write(f, arcname)

size_mb = OUT.stat().st_size / 1_048_576
print(f"Created: {OUT.name}  ({size_mb:.1f} MB, {len(files)} files)")
