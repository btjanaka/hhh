# Runs on BirdVox dataset.
python -m hhh.detector \
  --labels-csv-path data/bird-audio-detection/birdvox-labels.csv \
  --wav-dir data/bird-audio-detection/birdvox-wav \
  --features-pkl-path birdvox-features.pkl \
  --labels-npy-path birdvox-labels.npy \
  --model-save-path birdvox-all-detector.pth \
  --tensorboard-dir birdvox-all-tensorboard-logs \
  --dsp all \
  --batch-size 16 \
  --epochs 100 \
  
  # --use-saved-features \
  # --model-load-path birdvox-all-detector.pth \
  # --continue-training \