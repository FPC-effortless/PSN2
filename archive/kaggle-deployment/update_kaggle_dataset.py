#!/usr/bin/env python3
"""
Update Kaggle Dataset with Relation Prediction Fixes
Cross-platform Python script to create updated zip file
"""
import os
import shutil
import zipfile
from pathlib import Path

def main():
    print("=== PSN-2 Kaggle Dataset Updater ===")
    print()
    print("This will create psn2_kaggle_full_v2.zip with the relation prediction fixes.")
    print()

    # Create temporary directory
    temp_dir = Path("psn2_kaggle_temp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    print("Copying files...")

    # Files and directories to copy
    items_to_copy = [
        "psn2",
        "configs",
        "data",
        "scripts",
        "train.py",
        "train_sequential.py",
        "evaluate.py",
        "README.md",
        "PRD.md",
        "CHANGELOG.md",
        "RELATION_PREDICTION_FIX.md",
        "SEQUENTIAL_TRAINING_GUIDE.md",
    ]

    for item in items_to_copy:
        src = Path(item)
        if not src.exists():
            print(f"  Warning: {item} not found, skipping...")
            continue
        
        dst = temp_dir / item
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '.pytest_cache', '.git', '*.egg-info'
            ))
        else:
            shutil.copy2(src, dst)
        print(f"  ✓ {item}")

    print("\nCleaning up...")
    # Additional cleanup
    for pattern in ['__pycache__', '.pytest_cache', '*.pyc']:
        for path in temp_dir.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    print("Creating zip file...")
    zip_path = Path("psn2_kaggle_full_v2.zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in temp_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)

    # Cleanup temp directory
    shutil.rmtree(temp_dir)

    # Get file size
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print()
    print(f"✓ Created psn2_kaggle_full_v2.zip ({size_mb:.1f} MB)")
    print()
    print("Next steps:")
    print("1. Go to kaggle.com/datasets")
    print("2. Find your 'psn2-kaggle' dataset")
    print("3. Click 'New Version'")
    print("4. Upload psn2_kaggle_full_v2.zip")
    print("5. Add version notes: 'Relation prediction fixes - bond formation and masked entity context'")
    print("6. Click 'Create'")
    print()
    print("Your existing Kaggle notebooks will automatically use the new version!")

if __name__ == "__main__":
    main()
