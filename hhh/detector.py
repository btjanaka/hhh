"""Detect bird sounds.

Usage:
    python -m hhh.detector
"""
import argparse
import pathlib
import time

import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# TODO: consider using the full RGB image (though it might not matter much)
def extract_features(filename):
    """Returns the MFCC."""
    try:
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        #  mfccsscaled = np.mean(mfccs.T, axis=0)  # Average over time
    except Exception as err:
        print(f"Error encountered while parsing file {filename}: {err}")
        return None
    return mfccs


def retrieve_data(labels_csv_path: pathlib.Path, wav_dir: pathlib.Path,
                  feature_npy_path: pathlib.Path,
                  label_npy_path: pathlib.Path) -> ("features", "labels"):
    """Loads the dataset."""
    labels_csv = pd.read_csv(labels_csv_path)
    features = []
    labels = []

    for _, row in tqdm(list(labels_csv.iterrows())[:10]):
        filename = wav_dir / f"{row.itemid}.wav"
        label = row.hasbird
        feature = extract_features(filename)
        features.append(feature)
        labels.append(label)

    features = np.array(features, dtype=np.float32)
    features = features.reshape(
        (features.shape[0], 1, features.shape[1], features.shape[2]))
    labels = np.array(labels, dtype=np.float32)

    np.save(feature_npy_path, features)
    np.save(label_npy_path, labels)

    return features, labels


def make_dataloaders(features, labels,
                     batch_size: int) -> ("trainloader", "testloader"):
    """Make dataloaders from the features and labels."""
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2, random_state=42)

    trainset = TensorDataset(torch.from_numpy(features_train),
                             torch.from_numpy(labels_train))
    testset = TensorDataset(torch.from_numpy(features_test),
                            torch.from_numpy(labels_test))

    trainloader = DataLoader(trainset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)
    testloader = DataLoader(testset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)

    return trainloader, testloader


def make_detector_model(features, labels, device) -> nn.Module:
    """Returns a new instance of the detector model."""
    # Input: (1,40,431)
    return nn.Sequential(
        nn.Conv2d(in_channels=1,
                  out_channels=16,
                  kernel_size=3,
                  stride=1,
                  padding=1),  # -> (16,40,431)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # -> (16,20,215)
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=16,
                  out_channels=32,
                  kernel_size=3,
                  stride=1,
                  padding=1),  # -> (32,20,215)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # -> (32,10,107)
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=32,
                  out_channels=64,
                  kernel_size=3,
                  stride=1,
                  padding=1),  # -> (64,10,107)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # -> (64,5,53)
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=64,
                  out_channels=128,
                  kernel_size=3,
                  stride=1,
                  padding=1),  # -> (128,5,53)
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),  # -> (128,2,26)
        nn.Dropout(0.2),
        nn.AvgPool2d(kernel_size=(2, 26)),  # -> (128,1) (Global Avg Pooling)
        nn.Flatten(),
        nn.Linear(in_features=128,
                  out_features=1),  # -> z = log ( p(bird=1) / p(bird=0) )
        nn.Sigmoid(),  # -> p(bird = 1) = sigmoid(z)
    ).to(device)


def fit(detector, trainloader, epochs: int, model_path: str, device):
    """Trains the detector model."""
    bce_loss = nn.BCELoss()  # Double check this
    optimizer = optim.Adam(detector.parameters())

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1} ===")
        total_loss = 0.0

        # Iterate through batches in the (shuffled) training dataset.
        for batch_i, data in enumerate(trainloader):
            features = data[0].to(device)
            labels = data[1].to(device)

            optimizer.zero_grad()

            outputs = detector(features).view(features.shape[0])
            loss = bce_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()

        print("Loss:", total_loss)

        # Bookmark
        torch.save(detector.state_dict(), model_path)


def parse_args():
    """All script options."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Filepaths
    parser.add_argument("--labels-csv-path",
                        type=str,
                        default="data/bird-audio-detection/ff1010-labels.csv",
                        help="Location of CSV labels.")
    parser.add_argument("--wav-dir",
                        type=str,
                        default="data/bird-audio-detection/ff1010-wav/",
                        help="Directory holding WAV files")
    parser.add_argument("--features-npy-path",
                        type=str,
                        default="ff1010-features.npy",
                        help="File for generated features, saved as numpy.")
    parser.add_argument("--labels-npy-path",
                        type=str,
                        default="ff1010-labels.npy",
                        help="File for generated labels, saved as numpy.")
    parser.add_argument("--use-saved-features",
                        action="store_true",
                        help=("Pass this flag to load features and labels from "
                              "the files specified in --features-npy-path and "
                              "--labels-npy-path"))
    parser.add_argument("--model-load-path",
                        type=str,
                        default=None,
                        help=("Filepath of a model to load. "
                              "Leave as None to avoid loading."))
    parser.add_argument("--model-save-path",
                        type=str,
                        default="detector.pth",
                        help="Filepath to save the model.")

    # Computation
    parser.add_argument("--force-cpu",
                        action="store_true",
                        help=("Pass this flag to force PyTorch to use the CPU. "
                              "Otherwise, the GPU will be used if available."))

    # Algorithm hyperparameters
    parser.add_argument("--epochs",
                        type=int,
                        default=72,
                        help="Number of epochs to train the model.")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Training (and testing) batch size")

    return parser.parse_args()


def main():
    """Run everything (duh)."""
    args = parse_args()

    # Choose device.
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    print("Loading data")
    if args.use_saved_features:
        print("Using cached features and labels")
        features = np.load(args.features_npy_path)
        labels = np.load(args.labels_npy_path)
    else:
        print("Generating new features and labels from data")
        features, labels = retrieve_data(pathlib.Path(args.labels_csv_path),
                                         pathlib.Path(args.wav_dir),
                                         pathlib.Path(args.features_npy_path),
                                         pathlib.Path(args.labels_npy_path))
    trainloader, testloader = make_dataloaders(features, labels,
                                               args.batch_size)

    print("Making model")
    detector = make_detector_model(features, labels, device)

    # Load the model if so desired.
    if args.model_load_path is not None:
        print(f"Loaded model from {args.model_load_path}")
        detector.load_state_dict(torch.load(args.model_load_path))

    # Train.
    print("Start training")
    start = time.time()
    fit(detector, trainloader, args.epochs, args.model_save_path, device)
    end = time.time()
    print("End training")
    print(f"=== Training time: {end - start} s ===")

    torch.save(detector.state_dict(), args.model_save_path)
    print(f"Model saved to {args.model_save_path}")

    # TODO: evaluation


if __name__ == "__main__":
    main()
