# HELLBAT — Acoustic Drone Detection at the Edge

Real-time acoustic drone detector. Built for the Army xTech National Security Hackathon, May 2026.

Two MEMS microphones → ESP32-S3 → Jetson Orin Nano. Bio-inspired modulation features feed a probabilistic classifier; GCC-PHAT on stereo audio gives bearing. <200ms latency, no GPS, no RF emissions.

## Hardware

- 2× INMP441 MEMS microphones (I2S)
- ESP32-S3 dev board (USB CDC)
- NVIDIA Jetson Orin Nano (CUDA)

## Pipeline

```
mics → ESP32 (stereo I2S → USB) → Jetson
       → auditory spectrogram + Gabor filterbank
       → modulation profile spectrogram (RSF tensor)
       → tSVD → cohort GMM → log-likelihood ratio
       → drone / no-drone decision
       → GCC-PHAT bearing (parallel branch)
```

## Repo layout

```
features/         streaming GPU feature pipeline (CuPy)
training/         tSVD + cohort GMM model code
scripts/
  compute_features.py    one-shot feature extraction
  train.py               fit + eval (~30s after features cached)
  infer.py               wav-file inference
  realtime.py            ESP32 → Jetson live inference
data/output/      symlink to dataset (109k clips, not in repo)
models/           tsvd.pkl + gmm.pkl
```

## Setup

Requires Python 3.12, CUDA-capable GPU, Poetry.

```bash
poetry env use python3.12
poetry install
```

## Usage

```bash
# Train (after features are cached)
poetry run python training/train.py

# Inference on a wav file
poetry run python scripts/infer.py path/to/audio.wav

# Live inference from ESP32
poetry run python scripts/realtime.py
```

## Performance

- 98.8% accuracy, 0.978 AUC on test set
- 5.3× real-time on Jetson Orin Nano
- 109,370 training clips across 10 cohorts (drone + 9 nodrone)

## Status

v1 prototype. Single-node detection with bearing. Mesh networking, multi-node fusion, and CNN classifier are follow-on work.

## Author

Jared Childers — MS Applied Mathematics, Johns Hopkins