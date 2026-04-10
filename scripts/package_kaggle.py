"""
Package the PSN-2 repo for Kaggle upload.
Works on Windows, Linux, and macOS.

Usage:
    python scripts/package_kaggle.py

Output:
    psn2_kaggle_full.zip  (~50MB, wikitext excluded)
"""
import zipfile
import os
from pathlib import Path

OUTFILE = "psn2_kaggle_full.zip"

# Directories and files to include
INCLUDE_DIRS = ["psn2", "configs", "scripts", "tests"]
INCLUDE_FILES = ["train.py", "evaluate.py", "requirements.txt", "kaggle_train.ipynb"]

# Data subdirs to include (wikitext excluded — 550MB, upload separately)
INCLUDE_DATA = [
    "data/d5_arc_agi2",
    "data/d3_tom",
    "data/d3_tomi",
    "data/d5_gsm8k",
    "data/d6_bbh",
]

# Patterns to skip
SKIP_SUFFIXES = {".pyc"}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".git", "artifacts", "ARC-AGI"}


def should_skip(path: Path) -> bool:
    if path.suffix in SKIP_SUFFIXES:
        return True
    for part in path.parts:
        if part in SKIP_DIRS:
            return True
    return False


def add_dir(zf: zipfile.ZipFile, directory: str):
    for root, dirs, files in os.walk(directory):
        # Prune skip dirs in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            if not should_skip(fpath):
                zf.write(fpath)


root = Path(__file__).parent.parent
os.chdir(root)

print(f"Packaging PSN-2 for Kaggle → {OUTFILE}")

with zipfile.ZipFile(OUTFILE, "w", zipfile.ZIP_DEFLATED) as zf:
    for d in INCLUDE_DIRS:
        if Path(d).exists():
            add_dir(zf, d)
            print(f"  + {d}/")
        else:
            print(f"  ! {d}/ not found, skipping")

    for f in INCLUDE_FILES:
        if Path(f).exists():
            zf.write(f)
            print(f"  + {f}")
        else:
            print(f"  ! {f} not found, skipping")

    for d in INCLUDE_DATA:
        if Path(d).exists():
            add_dir(zf, d)
            print(f"  + {d}/")
        else:
            print(f"  ! {d}/ not found, skipping")

size_mb = Path(OUTFILE).stat().st_size / 1e6
print(f"\nCreated {OUTFILE} ({size_mb:.1f} MB)")
print()
print("NOTE: data/d4_wikitext/ (550MB) excluded.")
print("      Upload it as a separate Kaggle dataset if needed for D4 training.")
print()
print("Next steps:")
print("  1. kaggle.com/datasets → New Dataset → Upload psn2_kaggle_full.zip")
print("     Kaggle will ask for a name (e.g. 'psn2-kaggle') and auto-unzip it.")
print("  2. New Notebook → Add dataset → attach 'psn2-kaggle'")
print("  3. Settings → Accelerator → GPU T4 x2 (or P100)")
print("  4. Upload kaggle_train.ipynb")
print("  5. Set STAGE = 'D1' in the config cell and run all cells")
print()
print("For subsequent sessions:")
print("  - Download /kaggle/working/artifacts/latest.pt from the previous session")
print("  - Upload it as a new Kaggle dataset (e.g. 'psn2-checkpoint')")
print("  - Attach it to the next notebook session")
print("  - The notebook auto-detects and resumes from it")
