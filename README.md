# Football Field Analysis Pipeline - Refactored

This repository contains a refactored, modular implementation of the football field analysis pipeline.

## 📁 Project Structure

```
comp992/
├── main.py                    # Main pipeline entry point
├── pipeline.py                # Original monolithic script (preserved)
├── test_imports.py            # Module import verification script
│
├── pipeline/                  # Refactored modular pipeline
│   ├── README.md             # Detailed module documentation
│   ├── detection/            # Detection modules
│   ├── geometry/             # Geometric transformations
│   ├── models/               # Model loading utilities
│   ├── processing/           # Detection fusion and processing
│   ├── utils/                # Utility functions
│   └── visualization/        # Overlay rendering
│
├── detection/                 # Model weights and training
│   ├── models/               # Trained YOLO models (.pt files)
│   ├── training/             # Training scripts and datasets
│   └── example_usage/        # Usage examples
│
├── 2D_projection/            # Field template generation
└── 3D_projection/            # 3D projection utilities
```

## 📚 Documentation Files

- **`QUICKSTART.md`** - Quick start guide with command examples
- **`REFACTORING_SUMMARY.md`** - Detailed refactoring breakdown
- **`ARCHITECTURE.md`** - System architecture and data flow diagrams
- **`pipeline/README.md`** - Module-level documentation

## 🚀 Quick Start

### 1. Verify Installation

Test that all modules can be imported correctly:

```bash
python test_imports.py
```

### 2. Run the Pipeline

Basic usage with hash and number detection:

```bash
python main.py \
  --video path/to/video.mp4 \
  --hash-model detection/models/hash.pt \
  --number-model detection/models/numbers.pt \
  --field-template 2D_projection/field_template.png \
  --out output.mp4
```

With player detection and pose estimation:

```bash
python main.py \
  --video path/to/video.mp4 \
  --hash-model detection/models/hash.pt \
  --number-model detection/models/numbers.pt \
  --field-template 2D_projection/field_template.png \
  --player-model detection/models/players.pt \
  --pose-model detection/models/player_pose.pt \
  --out output.mp4 \
  --write-json metadata.json
```

## 🔧 Key Features

### Detection
- **Hash Mark Detection**: Identifies field hash marks using keypoints or line detection
- **Yard Number Recognition**: Detects and classifies yard line numbers (10, 20, 30, 40, 50)
- **Player Detection**: Two-step player detection with pose estimation
- **Pose Estimation**: 17-keypoint COCO skeleton detection for players

### Processing
- **Detection Fusion**: Combines hash marks and numbers to create anchor correspondences
- **Temporal Caching**: Maintains high-quality detections across frames
- **Anchor Propagation**: Intelligently propagates yard values to neighboring hash lines

### Geometry
- **Homography Estimation**: RANSAC-based homography for field projection
- **Quality Assessment**: Reprojection error and corner drift validation
- **Temporal Smoothing**: EMA-based smoothing for stable transformations

### Visualization
- **Rich Overlays**: Draws all detections, poses, and yardline estimates
- **Color-Coded**: Different colors for different detection types
- **Confidence Display**: Shows detection confidence scores

## 📊 Outputs

### Video Output
- Annotated video with overlays showing:
  - Hash mark detections (red lines/points)
  - Yard number bounding boxes (green)
  - Player bounding boxes (orange)
  - Pose skeletons (cyan bones, magenta keypoints)
  - Current yardline estimate (yellow text)

### JSON Metadata
Per-frame data including:
- Yard value and confidence
- Anchor correspondences (image ↔ field)
- Homography matrix
- Quality metrics (reprojection error, corner drift)
- Player detections with keypoint coordinates

## 🏗️ Architecture

The pipeline follows a modular design with clear separation of concerns:

```
Input → Detection → Processing → Geometry → Quality Control → Visualization → Output
```

See `ARCHITECTURE.md` for detailed data flow diagrams.

## 🔄 Migration from Original Pipeline

The refactored code maintains **100% functional compatibility** with the original `pipeline.py`:

```bash
# Old way
python pipeline.py --video input.mp4 --hash-model hash.pt ...

# New way (identical functionality, better code organization)
python main.py --video input.mp4 --hash-model hash.pt ...
```

## 📦 Module Usage

Individual modules can be imported and used independently:

```python
from pipeline.detection import detect_hash_and_numbers
from pipeline.geometry import estimate_homography
from pipeline.visualization import draw_overlay

# Use modules as needed...
```

## ⚙️ Advanced Configuration

### Device Selection
```bash
--device auto   # Auto-detect (MPS > CUDA > CPU)
--device cpu    # Force CPU
--device mps    # Force Apple Silicon GPU
--device 0      # CUDA GPU 0
```

### Homography Tuning
```bash
--smooth-alpha 0.8           # Temporal smoothing (0=none, 1=max)
--max-reproj-error 12.0      # Quality threshold (pixels)
--max-corner-drift 40.0      # Stability threshold (pixels)
--hold-last                  # Reuse last good homography
```

### Yard Detection Tuning
```bash
--conf-alpha 0.85            # Confidence smoothing
--min-ema-conf 0.25          # Minimum confidence threshold
```

### Player Detection
```bash
--player-conf 0.5            # Player detection threshold
--enable-player-poses        # Enable pose estimation
```

## 🧪 Testing

Run the import test to verify all modules load correctly:

```bash
python test_imports.py
```

Expected output:
```
Testing imports...
  ✓ Importing pipeline.utils...
  ✓ Importing pipeline.models...
  ✓ Importing pipeline.detection...
  ✓ Importing pipeline.processing...
  ✓ Importing pipeline.geometry...
  ✓ Importing pipeline.visualization...

✅ All imports successful!
```

## 🤝 Contributing

With the modular structure, contributions are easier:
1. Each module has a single, clear responsibility
2. Changes are localized to specific modules
3. New features can be added as new modules
4. Testing is more straightforward

## 📝 License

[Your license information here]

## 👥 Authors

[Your author information here]

---

For more details, see:
- **Quick Start**: `QUICKSTART.md`
- **Refactoring Details**: `REFACTORING_SUMMARY.md`
- **Architecture**: `ARCHITECTURE.md`
- **Module Documentation**: `pipeline/README.md`
