# Runs on Warblr dataset.
python -m hhh.detector \
  --labels-csv-path data/bird-audio-detection/warblr-labels.csv \
  --wav-dir data/bird-audio-detection/warblr-wav \
  --features-pkl-path warblr-features.pkl \
  --labels-npy-path warblr-labels.npy \
  --model-save-path warblr-detector.pth \
  --tensorboard-dir warblr-tensorboard-logs \
  --dsp all \
  --batch-size 16 \
  --epochs 120

  # --model-load-path warblr-detector.pth \
  # --use-saved-features \
  # --continue-training \
