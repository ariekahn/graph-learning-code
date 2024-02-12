#!/bin/bash
set -euo pipefail

# Get PROJECT_DIR
source ../../config.cfg

OUTPUT_DIR=$PROJECT_DIR/derived/mvpa_classifier

mkdir -p $OUTPUT_DIR/all

pdfunite $OUTPUT_DIR/sub*/images/*masks.pdf $OUTPUT_DIR/all/masks.pdf
pdfunite $OUTPUT_DIR/sub*/images/*blocks_nodes.pdf $OUTPUT_DIR/all/blocks_nodes.pdf
pdfunite $OUTPUT_DIR/sub*/images/*sample_pes.pdf $OUTPUT_DIR/all/sample_pes.pdf
pdfunite $OUTPUT_DIR/sub*/images/*voxel_selection.pdf $OUTPUT_DIR/all/voxel_selection.pdf
pdfunite $OUTPUT_DIR/sub*/images/*zscore_distribution.pdf $OUTPUT_DIR/all/zscore_distribution.pdf
pdfunite $OUTPUT_DIR/sub*/images/confusion*full.pdf $OUTPUT_DIR/all/confusion_full.pdf
