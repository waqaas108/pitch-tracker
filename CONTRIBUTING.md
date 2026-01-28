# Contributing to Pitch Tracker

## Getting Started

1. Clone the repo and set up the environment (see README.md)
2. Run the synthetic demo to verify everything works:
   ```bash
   python run_stereo_prototype.py --step all --samples 5000 --epochs 20
   python run_full_pipeline.py --synthetic
   ```

## Priority Tasks

### 1. Ball Detection (HIGH PRIORITY)

The biggest gap is reliable ball detection on real video. The current YOLO model uses COCO's generic "sports ball" class.

**What's needed:**
- Collect ~1000 labeled frames of baseballs in flight
- Various angles, lighting conditions, motion blur levels
- Train YOLOv8 on this data

**Files to modify:**
- `src/detection/ball_detector.py` - Update to use custom model
- Create `data/baseball_detection/` for training data

**Resources:**
- [Roboflow](https://roboflow.com/) - Easy labeling tool
- [YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)

### 2. Real Video Testing (HIGH PRIORITY)

We need ground truth data to validate accuracy.

**What's needed:**
- Videos recorded alongside Rapsodo/TrackMan
- Compare our predictions to pro system output
- Document failure modes

**Ideal test setup:**
- iPhone mounted on tripod, fixed position
- Record pitch with iPhone + Rapsodo simultaneously
- Export Rapsodo data as ground truth

### 3. Break Estimation Improvement (MEDIUM PRIORITY)

Current break estimation has ~1-2" error. This is acceptable but not pro-grade.

**Potential improvements:**
- Add pitch-type-specific priors to optimizer
- Constrain spin parameters to realistic ranges
- Run multiple optimizer restarts, take best

**Files to modify:**
- `src/physics/trajectory_fitter.py` - Add constraints
- `src/data/pitch_types.py` - Add spin constraints per type

### 4. Seam Tracking for Spin (MEDIUM PRIORITY)

For accurate spin estimation, we need to track seam rotation.

**Approach:**
1. Extract high-res crops of ball from each frame
2. Train CNN to detect seam orientation angle
3. Compute spin axis from rotation between frames

**Files to create:**
- `src/spin/seam_detector.py`
- `src/spin/spin_estimator.py`

**Reference:** This is how Rapsodo works.

### 5. iOS App (LOWER PRIORITY)

Eventually need a native app for capture + analysis.

**Requirements:**
- Record stereo video (wide + ultrawide simultaneously)
- Export models to CoreML
- Display results overlay on video

## Code Style

- Python 3.10+
- Type hints for function signatures
- Docstrings for public functions
- Keep files under 500 lines

## Testing

Run synthetic tests before submitting:
```bash
python run_full_pipeline.py --synthetic
```

All pitch types should show velocity error < 1 mph.

## Questions?

Open an issue or reach out to the team.
