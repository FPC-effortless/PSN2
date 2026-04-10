#!/usr/bin/env bash
# Packages the repo + training data for upload to Kaggle as a dataset.
# Usage: bash scripts/package_kaggle.sh
set -euo pipefail

OUTFILE="psn2_kaggle_full.zip"

echo "Packaging PSN-2 for Kaggle..."

# Remove old archive if present
rm -f "$OUTFILE"

# Create zip including data/ directory
zip -r "$OUTFILE" \
    psn2/ \
    configs/ \
    data/ \
    scripts/ \
    tests/ \
    train.py \
    evaluate.py \
    requirements.txt \
    kaggle_train.ipynb \
    -x "**/__pycache__/*" \
    -x "**/*.pyc" \
    -x "**/.pytest_cache/*" \
    -x "**/artifacts/*" \
    -x "**/.git/*" \
    -x "data/ARC-AGI-2-main/*" \
    -x "data/BIG-Bench-Hard-main/*" \
    -x "data/grade-school-math-master/*" \
    -x "data/d4_wikitext/*"   # wikitext is 550MB — too large for Kaggle dataset upload

SIZE=$(du -sh "$OUTFILE" | cut -f1)
echo ""
echo "Created $OUTFILE ($SIZE)"
echo ""
echo "Data included:"
echo "  data/d5_arc_agi2/   (ARC-AGI-2 tasks)"
echo "  data/d3_tom/        (Theory of Mind)"
echo "  data/d3_tomi/       (ToMi NLI)"
echo "  data/d5_gsm8k/      (GSM8K math)"
echo "  data/d6_bbh/        (BIG-Bench Hard)"
echo ""
echo "NOTE: data/d4_wikitext/ (550MB) excluded from zip."
echo "      Upload it as a separate Kaggle dataset and set:"
echo "      cfg['data_dir'] to point to its mount path."
echo ""
echo "Next steps:"
echo "  1. kaggle.com/datasets → New Dataset → Upload $OUTFILE"
echo "  2. New Notebook → Add dataset attachment"
echo "  3. Settings → Accelerator → GPU T4 x2 (or P100)"
echo "  4. Upload kaggle_train.ipynb or paste cells"
echo "  5. Set STAGE = 'D1' in the config cell"
echo "  6. Run all cells"
echo ""
echo "For subsequent sessions:"
echo "  - Download artifacts/latest.pt from previous session"
echo "  - Upload as a separate dataset attachment"
echo "  - The notebook auto-detects and resumes from it"
