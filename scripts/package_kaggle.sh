#!/usr/bin/env bash
# Packages the repo for upload to Kaggle as a dataset.
# Usage: bash scripts/package_kaggle.sh
set -euo pipefail

OUTFILE="psn2_kaggle_full.tar.gz"

tar -czf "$OUTFILE" \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='artifacts' \
    --exclude="$OUTFILE" \
    .

SIZE=$(du -sh "$OUTFILE" | cut -f1)
echo "Created $OUTFILE ($SIZE)"
echo ""
echo "Next steps:"
echo "  1. Go to kaggle.com/datasets → New Dataset"
echo "  2. Upload $OUTFILE"
echo "  3. Create a new Notebook, attach the dataset"
echo "  4. Upload kaggle_train.ipynb or paste its cells"
echo "  5. Enable GPU (T4 x2 or P100) under Settings → Accelerator"
echo "  6. Run all cells"
