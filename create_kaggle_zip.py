"""Create Kaggle-compatible zip with forward slashes."""
import zipfile
import os
from pathlib import Path

def create_kaggle_zip(source_dir, output_zip):
    """Create zip with Unix-style paths (forward slashes)."""
    source_path = Path(source_dir)
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in source_path.rglob('*'):
            if file_path.is_file():
                # Get relative path and convert to Unix style (forward slashes)
                arcname = file_path.relative_to(source_path.parent)
                arcname_str = str(arcname).replace('\\', '/')
                
                print(f"Adding: {arcname_str}")
                zipf.write(file_path, arcname_str)
    
    # Get file size
    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n✅ Created {output_zip} ({size_mb:.1f} MB)")
    print(f"   All paths use forward slashes (/) - Kaggle compatible!")

if __name__ == "__main__":
    create_kaggle_zip("psn2_kaggle_package", "psn2_kaggle_final.zip")
    
    print("\n" + "=" * 60)
    print("READY TO UPLOAD")
    print("=" * 60)
    print("\n1. Go to: https://www.kaggle.com/datasets")
    print("2. Click: 'New Dataset'")
    print("3. Upload: psn2_kaggle_final.zip")
    print("4. Title: 'PSN-2 Complete Training Package'")
    print("5. Tags: pytorch, neural-networks, arc-agi, deep-learning")
