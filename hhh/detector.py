"""Detect bird sounds.

Usage:
    See README
"""
import argparse
import pathlib
import time

import librosa
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hhh.models


def extract_features(filename):
    """Returns the MFCC."""
    try:
        audio, sample_rate = librosa.load(filename, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
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

    for _, row in tqdm(list(labels_csv.iterrows())):
        filename = wav_dir / f"{row.itemid}.wav"
        label = row.hasbird
        feature = extract_features(filename)
        features.append(feature)
        labels.append(label)

    # Pad each feature to have the same length in the time dimension.
    max_length = max(map(lambda x: x.shape[1], features))
    for idx, feature in enumerate(features):
        features[idx] = np.pad(feature,
                               ((0, 0), (0, max_length - feature.shape[1])))

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
        sklearn.model_selection.train_test_split(features, labels, test_size=0.2, random_state=42)

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
    return hhh.models.BasicModel().to(device)


def fit(detector, trainloader, epochs: int, model_path, tensorboard_writer,
        device):
    """Trains the detector model."""
    bce_loss = nn.BCELoss()  # Double check this
    optimizer = optim.Adam(detector.parameters())

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1} ===")
        total_loss = 0.0

        # Iterate through batches in the (shuffled) training dataset.
        for _, data in enumerate(trainloader):
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
        tensorboard_writer.add_scalar("epoch_loss", total_loss, epoch)

        # Bookmark
        torch.save(detector.state_dict(), model_path)


def calculate_auc(detector, dataloader, tensorboard_writer, device):
    """Calculates ROC AUC score with data from the given dataloader."""
    with torch.no_grad():  # Gradient should not be tracked during evaluation.
        all_labels, all_outputs = [], []
        for _, data in enumerate(dataloader):
            features = data[0].to(device)
            labels = data[1].to(device)

            outputs = detector(features)
            all_labels.extend(labels.tolist())
            all_outputs.extend(outputs.tolist())

        auc_score = sklearn.metrics.roc_auc_score(all_labels, all_outputs)
        print(f"AUC on {len(all_labels)} audio files: {auc_score:.4f}")
        tensorboard_writer.add_scalar("AUC", auc_score)
        return auc_score


def main():
    """Run everything (duh)."""

    def parse_args():
        """All script options."""
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        # Filepaths
        parser.add_argument(
            "--labels-csv-path",
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
        parser.add_argument(
            "--use-saved-features",
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
        parser.add_argument(
            "--continue-training",
            action="store_true",
            default=None,
            help=("If the model was loaded from a path, pass "
                  "this parameter to continue training it (by "
                  "default, no training happens on loaded models."))
        parser.add_argument("--tensorboard-dir",
                            type=str,
                            default="tensorboard-logs",
                            help="Directory for writing logs for tensorboard")

        # Computation
        parser.add_argument(
            "--force-cpu",
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

    args = parse_args()

    # Choose device.
    use_cuda = torch.cuda.is_available() and not args.force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    # Create a writer for logging output.
    tensorboard_writer = SummaryWriter(args.tensorboard_dir)

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

    # Train a new model or continue training an existing model.
    if args.model_load_path is None or args.continue_training:
        print("Start training")
        start = time.time()
        fit(detector, trainloader, args.epochs, args.model_save_path,
            tensorboard_writer, device)
        end = time.time()
        print("End training")
        print(f"=== Training time: {end - start} s ===")

        torch.save(detector.state_dict(), args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

    calculate_auc(detector, testloader, tensorboard_writer, device)
    tensorboard_writer.close()


if __name__ == "__main__":
    main()
