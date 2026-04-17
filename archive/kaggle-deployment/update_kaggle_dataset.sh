#!/bin/bash
# Update Kaggle Dataset with Relation Prediction Fixes
# Run this script to create a new zip file with your updated code

set -e

echo "=== PSN-2 Kaggle Dataset Updater ==="
echo ""
echo "This will create psn2_kaggle_full_v2.zip with the relation prediction fixes."
echo ""

# Create temporary directory
TEMP_DIR="psn2_kaggle_temp"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

echo "Copying files..."

# Copy all necessary files
cp -r psn2 "$TEMP_DIR/"
cp -r configs "$TEMP_DIR/"
cp -r data "$TEMP_DIR/"
cp -r scripts "$TEMP_DIR/"
cp train.py "$TEMP_DIR/"
cp train_sequential.py "$TEMP_DIR/"
cp evaluate.py "$TEMP_DIR/"
cp README.md "$TEMP_DIR/"
cp PRD.md "$TEMP_DIR/"
cp CHANGELOG.md "$TEMP_DIR/"
cp RELATION_PREDICTION_FIX.md "$TEMP_DIR/"
cp SEQUENTIAL_TRAINING_GUIDE.md "$TEMP_DIR/"

# Remove unnecessary files
echo "Cleaning up..."
find "$TEMP_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$TEMP_DIR" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
find "$TEMP_DIR" -name ".git" -type d -exec rm -rf {} + 2>/dev/null || true

# Create zip file
echo "Creating zip file..."
cd "$TEMP_DIR"
zip -r ../psn2_kaggle_full_v2.zip . -q
cd ..

# Cleanup
rm -rf "$TEMP_DIR"

# Get file size
SIZE=$(du -h psn2_kaggle_full_v2.zip | cut -f1)
echo ""
echo "✓ Created psn2_kaggle_full_v2.zip ($SIZE)"
echo ""
echo "Next steps:"
echo "1. Go to kaggle.com/datasets"
echo "2. Find your 'psn2-kaggle' dataset"
echo "3. Click 'New Version'"
echo "4. Upload psn2_kaggle_full_v2.zip"
echo "5. Add version notes: 'Relation prediction fixes - bond formation and masked entity context'"
echo "6. Click 'Create'"
echo ""
echo "Your existing Kaggle notebooks will automatically use the new version!"
