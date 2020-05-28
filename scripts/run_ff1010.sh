# Runs on FF1010 dataset.
python -m hhh.detector \
  --labels-csv-path data/bird-audio-detection/ff1010-labels.csv \
  --wav-dir data/bird-audio-detection/ff1010-wav \
  --features-pkl-path ff1010-features.pkl \
  --labels-npy-path ff1010-labels.npy \
  --model-save-path ff1010-all-detector.pth \
  --tensorboard-dir ff1010-all-tensorboard-logs \
  --dsp all \
  --batch-size 16 \
  --epochs 100 \

  # --use-saved-features \
  # --model-load-path ff1010-all-detector.pth \
  # --continue-training \