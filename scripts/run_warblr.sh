# Runs on Warblr dataset.
python -m hhh.detector \
  --labels-csv-path data/bird-audio-detection/warblr-labels.csv \
  --wav-dir data/bird-audio-detection/warblr-wav \
  --features-pkl-path warblr-features.pkl \
  --labels-npy-path warblr-labels.npy \
  --model-save-path warblr-all-detector.pth \
  --tensorboard-dir warblr-all-tensorboard-logs \
  --dsp all \
  --batch-size 16 \
  --epochs 100 \

  # --use-saved-features \
  # --model-load-path warblr-all-detector.pth \
  # --continue-training \
