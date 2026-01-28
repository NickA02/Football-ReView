#!/bin/bash

# Football Field Analysis Pipeline Test Script
# This script runs the refactored pipeline with sample data

# Activate conda environment
conda activate fv

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Football Field Analysis Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if video file is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No video file provided${NC}"
    echo -e "${YELLOW}Usage: ./run.sh <video_file> [options]${NC}"
    echo ""
    echo "Examples:"
    echo "  ./run.sh input_video.mp4"
    echo "  ./run.sh input_video.mp4 --device cpu"
    echo "  ./run.sh input_video.mp4 --player-conf 0.6 --pose-conf 0.8"
    echo "  ./run.sh input_video.mp4 --target-fps -1  # Process all frames"
    echo "  ./run.sh input_video.mp4 --target-fps 15  # Process at 15 FPS"
    echo "  ./run.sh input_video.mp4 --enable-player-poses=false"
    echo ""
    echo "Common Options:"
    echo "  --player-conf <0.0-1.0>    Player detection confidence (default: 0.5)"
    echo "  --pose-conf <0.0-1.0>      Pose estimation confidence (default: 0.7)"
    echo "  --target-fps <number>      Target processing FPS (default: 30, use -1 for all)"
    echo "  --device <cpu|mps|auto>    Processing device (default: auto)"
    echo "  --enable-player-poses      Enable/disable player detection (default: true)"
    echo ""
    exit 1
fi

VIDEO_FILE="$1"
shift  # Remove first argument (video file)

# Check if video file exists
if [ ! -f "$VIDEO_FILE" ]; then
    echo -e "${RED}Error: Video file '$VIDEO_FILE' not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Video file found: $VIDEO_FILE${NC}"

# Define model paths
HASH_MODEL="detection/models/hash.pt"
NUMBER_MODEL="detection/models/numbers.pt"
PLAYER_MODEL="detection/models/player_detection.pt"
POSE_MODEL="detection/models/player_pose.pt"

# Check if models exist
echo -e "${BLUE}Checking model files...${NC}"

if [ ! -f "$HASH_MODEL" ]; then
    echo -e "${RED}Error: Hash model not found at $HASH_MODEL${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Hash model found${NC}"

if [ ! -f "$NUMBER_MODEL" ]; then
    echo -e "${RED}Error: Number model not found at $NUMBER_MODEL${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Number model found${NC}"

if [ ! -f "$PLAYER_MODEL" ]; then
    echo -e "${YELLOW}⚠ Player model not found (optional)${NC}"
    PLAYER_MODEL=""
fi

if [ ! -f "$POSE_MODEL" ]; then
    echo -e "${YELLOW}⚠ Pose model not found (optional)${NC}"
    POSE_MODEL=""
fi

# Check/create field template
FIELD_TEMPLATE="2D_projection/field_template.png"
if [ ! -f "$FIELD_TEMPLATE" ]; then
    echo -e "${YELLOW}⚠ Field template not found, generating...${NC}"
    if [ -f "2D_projection/make_field_template.py" ]; then
        python 2D_projection/make_field_template.py
        if [ -f "$FIELD_TEMPLATE" ]; then
            echo -e "${GREEN}✓ Field template generated${NC}"
        else
            echo -e "${RED}Error: Failed to generate field template${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: make_field_template.py not found${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ Field template found${NC}"
fi

# Create output directory
OUTPUT_DIR="output"
mkdir -p "$OUTPUT_DIR"

# Generate output filenames based on input video name
VIDEO_BASENAME=$(basename "$VIDEO_FILE" | sed 's/\.[^.]*$//')
OUTPUT_VIDEO="$OUTPUT_DIR/${VIDEO_BASENAME}_output.mp4"
OUTPUT_JSON="$OUTPUT_DIR/${VIDEO_BASENAME}_metadata.json"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Running Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Input:  $VIDEO_FILE"
echo -e "Output: $OUTPUT_VIDEO"
echo -e "JSON:   $OUTPUT_JSON"
echo ""

# Build command
CMD="python main.py \
  --video \"$VIDEO_FILE\" \
  --hash-model \"$HASH_MODEL\" \
  --number-model \"$NUMBER_MODEL\" \
  --field-template \"$FIELD_TEMPLATE\" \
  --out \"$OUTPUT_VIDEO\" \
  --write-json \"$OUTPUT_JSON\""

# Add player models if available
if [ -n "$PLAYER_MODEL" ] && [ -n "$POSE_MODEL" ]; then
    CMD="$CMD --player-model \"$PLAYER_MODEL\" --pose-model \"$POSE_MODEL\""
    echo -e "${GREEN}✓ Player detection enabled${NC}"
else
    CMD="$CMD --enable-player-poses=false"
    echo -e "${YELLOW}⚠ Player detection disabled (models not found)${NC}"
fi

# Add any additional arguments passed to the script
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
    echo -e "${BLUE}Additional options: $@${NC}"
fi

echo ""
echo -e "${YELLOW}Executing:${NC}"
echo "$CMD"
echo ""

# Run the pipeline
eval $CMD

# Check if outputs were created
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Results${NC}"
echo -e "${BLUE}========================================${NC}"

if [ -f "$OUTPUT_VIDEO" ]; then
    VIDEO_SIZE=$(du -h "$OUTPUT_VIDEO" | cut -f1)
    echo -e "${GREEN}✓ Output video created: $OUTPUT_VIDEO ($VIDEO_SIZE)${NC}"
else
    echo -e "${RED}✗ Output video not created${NC}"
fi

if [ -f "$OUTPUT_JSON" ]; then
    JSON_SIZE=$(du -h "$OUTPUT_JSON" | cut -f1)
    FRAME_COUNT=$(grep -o '"frame":' "$OUTPUT_JSON" | wc -l | tr -d ' ')
    echo -e "${GREEN}✓ Metadata JSON created: $OUTPUT_JSON ($JSON_SIZE, $FRAME_COUNT frames)${NC}"
else
    echo -e "${RED}✗ Metadata JSON not created${NC}"
fi

echo ""
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo ""
