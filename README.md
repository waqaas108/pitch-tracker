# Pitch Tracker

iPhone-based baseball pitch tracking using physics-constrained ML. Think TrackMan, but with just your phone.

## Current Status

This is a **working prototype** that demonstrates the core approach. It's not production-ready but provides a solid foundation for development.

**What works well:**
- Physics simulation (ballistic + Magnus effect)
- Stereo camera depth estimation
- Physics-constrained trajectory fitting
- Velocity estimation (~0.1 mph error on synthetic data)

**What needs work:**
- Ball detection on real video (needs fine-tuned YOLO)
- Break estimation (~1-2" error, acceptable but not pro-grade)
- Spin estimation (rough approximation from trajectory)
- Real-world camera calibration

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/pitch-tracker.git
cd pitch-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (GPU recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib tqdm opencv-python ultralytics

# Train the physics model (required first time)
python run_stereo_prototype.py --step all --samples 10000 --epochs 50

# Test on synthetic data
python run_full_pipeline.py --synthetic
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     iPhone Capture                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Wide Camera │  │  Ultrawide  │  │   LiDAR     │         │
│  │   (main)    │  │  (stereo)   │  │  (depth)    │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
└─────────┼────────────────┼────────────────┼─────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────┐
│                   Ball Detection (YOLO)                     │
│         Detects baseball in each frame → bounding box       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Ball Tracking (Kalman)                    │
│         Links detections across frames → 2D trajectory      │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                   Stereo Triangulation                      │
│         Wide + Ultrawide → 3D positions + depth             │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                Physics-Constrained Optimizer                │
│    Fits velocity, spin_rate, spin_axis, release_angles     │
│    to match observed trajectory using L-BFGS-B              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      Output Metrics                         │
│  • Velocity (mph)      • Horizontal Break (inches)          │
│  • Spin Rate (rpm)     • Vertical Break (inches)            │
│  • Pitch Type          • Plate Location                     │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
pitch_tracker/
├── src/
│   ├── physics/
│   │   ├── trajectory.py        # Ballistic + Magnus simulation
│   │   ├── stereo_camera.py     # iPhone dual-camera model
│   │   └── trajectory_fitter.py # Physics-constrained optimization
│   ├── detection/
│   │   ├── ball_detector.py     # YOLO-based ball detection
│   │   ├── tracker.py           # Kalman filter tracking
│   │   └── video_pipeline.py    # Video → trajectory extraction
│   ├── calibration/
│   │   ├── camera_calibrator.py # PnP camera calibration
│   │   └── field_markers.py     # Standard field geometry
│   ├── pipeline/
│   │   └── pitch_analyzer.py    # End-to-end analysis
│   ├── data/
│   │   ├── pitch_types.py       # MLB pitch profiles
│   │   └── stereo_generator.py  # Synthetic training data
│   ├── models/
│   │   └── pitch_classifier.py  # Pitch type classification
│   └── training/
│       └── train_initializer.py # Neural network training
├── run_stereo_prototype.py      # Train physics model
├── run_full_pipeline.py         # End-to-end demo
├── calibrate_camera.py          # Camera calibration tool
└── requirements.txt
```

## Usage

### 1. Train the Physics Model (Required First)

```bash
python run_stereo_prototype.py --step all --samples 10000 --epochs 50
```

This trains the neural network that initializes the physics optimizer. Takes ~10-15 minutes on a GPU.

### 2. Calibrate Your Camera

Before analyzing real video, calibrate your camera position:

```bash
# Quick calibration (if you know camera position)
python calibrate_camera.py --position 0,2.5,-2 --look-at 0,1,9 --output calibration.json

# Interactive calibration (click on field markers)
python calibrate_camera.py --video pitch.mp4 --output calibration.json
```

Common camera positions (in meters, origin at home plate):
- Behind mound, high: `--position 0,3,-2 --look-at 0,1,9`
- Behind mound, low: `--position 0,1.8,-1.5 --look-at 0,0.8,9`
- Side view (1B side): `--position 12,2,9 --look-at 0,1.2,9`
- Behind catcher: `--position 0,1.5,20 --look-at 0,1.5,0`

### 3. Analyze Video

```bash
# Single camera video
python run_full_pipeline.py --video pitch.mp4

# Stereo video (wide + ultrawide)
python run_full_pipeline.py --wide-video wide.mp4 --ultra-video ultra.mp4

# Test on synthetic data
python run_full_pipeline.py --synthetic
```

## Current Accuracy (Synthetic Data)

| Metric | Mean Error | Notes |
|--------|------------|-------|
| Velocity | 0.13 mph | Excellent - approaching TrackMan |
| H-Break | 1.41 inches | Good - needs seam tracking for better |
| V-Break | 0.91 inches | Good - same as above |
| Pitch Type | ~70% | Acceptable - confused on similar pitches |

**Note:** Real-world accuracy will be lower due to detection noise and camera calibration errors.

## Known Limitations

1. **Ball Detection**: The default YOLO model uses COCO's "sports ball" class, which isn't optimized for baseballs in flight. Expect many missed detections, especially with motion blur.

2. **Single Camera Depth**: Without stereo, depth is estimated from motion, which is rough. Stereo video significantly improves accuracy.

3. **Camera Calibration**: The physics fitter assumes known camera geometry. Incorrect calibration → incorrect metrics.

4. **Spin Estimation**: Spin rate/axis are inferred from trajectory curvature, not directly observed. For accurate spin, you need seam tracking.

5. **Processing Speed**: Physics fitting takes ~10 seconds per pitch. Not suitable for real-time yet.

## Next Steps (Priority Order)

### High Priority

1. **Fine-tune YOLO for Baseball Detection**
   - Collect/label ~1000 frames of baseballs in flight
   - Train YOLOv8 on baseball-specific data
   - Handle motion blur with augmentation
   - Target: 90%+ detection rate

2. **Improve Break Estimation**
   - Add regularization to physics optimizer
   - Constrain spin parameters to realistic ranges per pitch type
   - Consider ensemble of multiple optimizer runs

3. **Real Video Testing**
   - Collect test videos with known ground truth (Rapsodo/TrackMan data)
   - Benchmark against pro systems
   - Identify failure modes

### Medium Priority

4. **Seam Tracking for Spin**
   - Extract high-res ball crops from video
   - Train CNN to detect seam orientation
   - Estimate spin axis from seam rotation
   - This is how Rapsodo does it

5. **iPhone App Development**
   - Export models to CoreML
   - Build Swift UI for capture + analysis
   - Implement stereo video recording

6. **Speed Optimization**
   - Profile and optimize physics fitter
   - Consider GPU-accelerated optimization
   - Target: <1 second per pitch

### Lower Priority

7. **LiDAR Integration**
   - Use iPhone LiDAR for sparse but accurate depth
   - Fuse with stereo depth estimates

8. **Pitcher Pose Estimation**
   - Extract release point from body pose
   - Constrain trajectory to start from hand position
   - This is how PitcherNet works

9. **Cloud Pipeline**
   - Build API for video upload + analysis
   - Handle multiple concurrent analyses
   - Store historical data per player

## Technical References

- [PitcherNet](https://arxiv.org/html/2405.07407v1) - Waterloo/Orioles pitch tracking from video
- [Baseball Aerodynamics](https://baseballaero.com/) - Physics of pitch movement
- [TrackMan Technology](https://baseball.physics.illinois.edu/trackman.html) - How pro systems work

## Hardware Requirements

- **GPU**: NVIDIA with 6GB+ VRAM (tested on RTX 4060)
- **CPU**: Works but slower (~5x)
- **RAM**: 8GB+
- **Storage**: ~500MB for models + data

## Contributing

This is a research prototype. Key areas needing help:

1. **Data Collection**: Real pitch videos with ground truth labels
2. **Ball Detection**: Fine-tuned YOLO model for baseballs
3. **iOS Development**: Native app with stereo capture
4. **Testing**: Validation against pro tracking systems

## License

MIT
