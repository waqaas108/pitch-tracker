# Pitch Tracker

iPhone-based baseball pitch tracking using physics-constrained ML. Think TrackMan, but with just your phone.

## What This Does

Estimates pitch metrics (velocity, spin, break) from iPhone video by:
1. Using stereo cameras (wide + ultrawide) for depth estimation
2. Fitting physics parameters to observed ball trajectory
3. Neural network initialization for fast optimization

## Quick Start

```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/pitch-tracker.git
cd pitch-tracker

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scipy matplotlib tqdm

# Run the prototype
python run_stereo_prototype.py --step all --samples 10000 --epochs 50
```

## Architecture

```
iPhone Capture
├── Wide camera (main tracking @ 60fps)
├── Ultrawide camera (stereo depth)
└── LiDAR (sparse depth, future)
         │
         ▼
    Ball Detection (YOLO) ──► 2D trajectories
         │
         ▼
    Stereo Triangulation ──► 3D positions + depth
         │
         ▼
    Physics Optimizer
    ├── Neural initializer (fast starting point)
    └── L-BFGS-B optimization (fit velocity, spin, angles)
         │
         ▼
    Output: velocity, spin_rate, spin_axis, break, location
```

## Project Structure

```
pitch_tracker/
├── src/
│   ├── physics/
│   │   ├── trajectory.py        # Ballistic + Magnus simulation
│   │   ├── stereo_camera.py     # iPhone dual-camera model
│   │   └── trajectory_fitter.py # Physics-constrained optimization
│   ├── data/
│   │   ├── pitch_types.py       # MLB pitch profiles (FF, CU, SL, etc.)
│   │   └── stereo_generator.py  # Synthetic training data
│   ├── models/
│   │   ├── trajectory_regressor.py  # (v1, deprecated)
│   │   └── pitch_classifier.py      # Pitch type classification
│   └── training/
│       └── train_initializer.py # Neural initializer training
├── run_stereo_prototype.py      # Main entry point (v2)
├── run_prototype.py             # Original prototype (v1)
└── requirements.txt
```

## Two Approaches

### V1: Direct Regression (run_prototype.py)
- LSTM maps 2D trajectory → metrics
- Fast but less accurate (~50% pitch classification)
- Good for understanding the problem

### V2: Physics-Constrained Fitting (run_stereo_prototype.py) ⭐
- Stereo cameras provide depth
- Optimizer fits physics parameters to observations
- More accurate, interpretable, and robust

## Running the Prototypes

### V2 (Recommended)
```bash
# Full pipeline: generate data, train initializer, evaluate
python run_stereo_prototype.py --step all --samples 10000 --epochs 50

# Just run demo on trained model
python run_stereo_prototype.py --step demo
```

### V1 (Original)
```bash
python run_prototype.py --step all --samples 5000 --epochs 30
```

## Hardware Requirements

- GPU: NVIDIA with 6GB+ VRAM (tested on RTX 4060)
- CPU: Works but slower
- RAM: 8GB+

## Next Steps (TODO)

- [ ] Ball detection module (YOLOv8 fine-tuned on baseball)
- [ ] Real video ingestion pipeline
- [ ] iPhone app with CoreML export
- [ ] Spin estimation from seam tracking
- [ ] LiDAR depth integration

## How It Works

The key insight from [research](https://arxiv.org/html/2405.07407v1): instead of learning a direct mapping from pixels to metrics, we:

1. **Observe** the ball in 2D from two cameras
2. **Triangulate** approximate 3D positions
3. **Optimize** physics parameters (v₀, spin, angles) that explain observations
4. **Extract** metrics from the fitted trajectory

This is more robust because:
- Physics constraints ensure plausible predictions
- Stereo depth resolves scale ambiguity
- Optimizer finds exact solution, not approximation

## References

- [PitcherNet](https://arxiv.org/html/2405.07407v1) - Waterloo/Orioles pitch tracking from video
- [PitchLab](https://apps.apple.com/pl/app/pitchlab-baseball/id6738223162) - iPhone pitch tracking app
- [Baseball Aerodynamics](https://baseballaero.com/) - Physics of pitch movement
