# Pipeline Refactoring

This directory contains a refactored, modular version of the football field analysis pipeline.

## Directory Structure

```
pipeline/
├── __init__.py                 # Package initialization
├── detection/                  # Detection modules
│   ├── __init__.py
│   ├── field_detection.py     # Hash marks and yard numbers
│   └── player_detection.py    # Player and pose detection
├── geometry/                   # Geometric transformations
│   ├── __init__.py
│   └── homography.py          # Homography estimation and refinement
├── models/                     # Model loading
│   ├── __init__.py
│   └── loader.py              # YOLO model loading utilities
├── processing/                 # Detection processing
│   ├── __init__.py
│   └── fusion.py              # Detection fusion and anchor generation
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── device.py              # Device resolution (CPU/GPU/MPS)
│   └── mappings.py            # Class ID to yard mappings
└── visualization/              # Visualization modules
    ├── __init__.py
    └── overlay.py             # Drawing overlays on frames
```

## Main Entry Point

**`main.py`** - The main pipeline execution script. This file orchestrates the entire pipeline:
- Loads models
- Processes video frame by frame
- Applies detection, fusion, and homography estimation
- Generates output video and JSON metadata

## Module Descriptions

### `pipeline.detection`
Contains detection modules for field elements and players:
- **`field_detection.py`**: Detects hash marks and yard numbers using YOLO models
- **`player_detection.py`**: Two-step player detection and pose estimation

### `pipeline.geometry`
Geometric transformation utilities:
- **`homography.py`**: Homography matrix estimation, quality assessment, and temporal smoothing

### `pipeline.models`
Model management:
- **`loader.py`**: Loads YOLO models for various detection tasks

### `pipeline.processing`
Detection processing and fusion:
- **`fusion.py`**: Fuses hash mark and yard number detections to create anchor points for homography estimation

### `pipeline.utils`
Utility functions:
- **`device.py`**: Automatic device selection (CPU/CUDA/MPS)
- **`mappings.py`**: Maps class IDs to yard line values

### `pipeline.visualization`
Visualization and overlay rendering:
- **`overlay.py`**: Draws detection results on video frames

## Usage

Run the pipeline using `main.py`:

```bash
python main.py \
  --video /path/to/video.mp4 \
  --hash-model detection/models/hash.pt \
  --number-model detection/models/numbers.pt \
  --field-template 2D_projection/field_template.png \
  --player-model detection/models/players.pt \
  --pose-model detection/models/player_pose.pt \
  --out output_video.mp4 \
  --write-json output_metadata.json
```

## Benefits of Refactoring

1. **Modularity**: Each component has a clear, single responsibility
2. **Testability**: Individual modules can be tested in isolation
3. **Maintainability**: Easier to locate and modify specific functionality
4. **Reusability**: Components can be imported and used in other projects
5. **Readability**: Clear organization makes the codebase easier to understand
6. **Extensibility**: New detection methods or processing steps can be added without affecting existing code

## Migration from Old Pipeline

The original `pipeline.py` has been preserved. The new modular structure maintains the exact same functionality but with better organization. To use the new version, simply use `main.py` instead of `pipeline.py` with the same command-line arguments.
