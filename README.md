# High Entropy Homies

<div style="display:block; margin: 0px auto; width:200px; text-align: center">

![dance](docs/team-logo.png)

</div>

Group 1 in Pierre Baldi's Spring 2020 offering of CS 172B: Neural Networks and
Deep Learning.

- Taneisha Arora (arorat@uci.edu)
- Thanasi Bakis (abakis@uci.edu)
- Theja Krishna (takrishn@uci.edu)
- Bryon Tjanaka (btjanaka@uci.edu)

## Project Overview

We seek to detect bird sounds from 10-second audio recordings, as described in
the
[DCASE 2018 Challenge](http://dcase.community/challenge2018/task-bird-audio-detection).

## Instructions

### Installation

Install the dependencies:

```
pip install -r requirements.txt
```

And you should be good to go.

### Training a Detector

The main interface is in the `hhh.detector` script. Run
`python -m hhh.detector -h` for full help info.

Given a dataset with labels in `LABELS.csv` and WAV audio files in `WAV_DIR`,
train a detector with:

```bash
python -m hhh.detector \
  --labels-csv-path LABELS.csv \
  --wav-dir WAV_DIR \
  --features-npy-path FEATURES.npy \
  --labels-npy-path LABELS.npy
```

`FEATURES.npy` and `LABELS.npy` will store computed features and labels of the
dataset, to avoid re-computing in the future. Next time the script is run, pass
in the `--use-saved-features` flag to use these cached features and labels.

The above command will also save the model to `detector.pth`. This filepath can
be changed with the `--model-save-path` flag. To load a model, use the
`--model-load-path` flag. Note that if you load a model, it will not be trained
further by default. To continue training, pass the `--continue-training` flag.

#### Tensorboard

Metrics are logged to the directory indicated in the `--tensorboard-dir` flag
(`tensorboard-logs/` by default). To view these metrics, run

```bash
tensorboard --logdir <DIR>
```

And go to http://localhost:6006 in your browser.

#### Hyperparemeters

You can modify several of the training hyperparameters, including training
epochs (`--epochs`) and batch size (`--batch-size`).

#### Miscellaneous Parameters

Pass the `--force-cpu` flag to force training on a CPU.
