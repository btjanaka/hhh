"""Detect bird sounds.

Usage:
    python -m hhh.detector
"""
import os
import pathlib
import struct
import time
from datetime import datetime

import librosa
import librosa.display
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.callbacks import ModelCheckpoint
from keras.layers import (Activation, Conv2D, Convolution2D, Dense, Dropout,
                          Flatten, GlobalAveragePooling2D, MaxPooling2D)
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils, to_categorical


# TODO: consider using the full RGB image (though it might not matter much)
def extract_features(file_name):
    """Returns the MFCC as a GREYSCALE image."""
    try:
        audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccsscaled = np.mean(mfccs.T, axis=0)  # greyscaled
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None

    return mfccsscaled


# Read data in and extract features.
fulldatasetpath = pathlib.Path('data/bird-audio-detection/')
labels = pd.read_csv(fulldatasetpath / "ff1010-labels.csv")
data = []

for index, row in labels.iterrows():
    filename = fulldatasetpath / "ff1010-wav" / row.itemid
    label = row.hasbird
    feature = extract_features(filename)
    data.append([feature, label])

data = np.array(data)
features = data[:, 0]
labels = data[:, 1]

# 80-20 split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.2, random_state=42)

num_channels = 1  # (greyscale)

features_train = features_train.reshape(features_train.shape[0], data.shape[0],
                                        data.shape[1], num_channels)
features_test = features_test.reshape(features_test.shape[0], data.shape[0],
                                      data.shape[1], num_channels)

num_labels = data.shape[1]



use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)

pytorch_model = nn.Sequential(
    nn.Conv2D(in_channels=num_channels,
              out_channels=16,
              kernel_size=2,
              stride=1),
    nn.ReLU(),
    nn.MaxPool2d(pool_size=2),
    nn.Dropout(0.2),
    nn.Conv2D(in_channels=16, out_channels=32, kernel_size=2, stride=1),
    nn.MaxPool2d(pool_size=2),
    nn.Dropout(0.2),
    nn.Conv2D(in_channels=32, out_channels=64, kernel_size=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(pool_size=2),
    nn.Dropout(0.2),
    nn.Conv2D(in_channels=64, out_channels=128, kernel_size=2, stride=1),
    nn.ReLU(),
    nn.MaxPool2d(pool_size=2),
    nn.Dropout(0.2),
    nn.AvgPool2d(kernel_size=(data.shape[0] // 16, data.shape[1]//16)) ), # Global Avg Pooling -> one number per channel
    nn.Linear(in_features=128, out_features=1),
    # output = log ( p(bird=1) / p(bird=0) )
    # p(bird = 1) = sigmoid(output)
    nn.Sigmoid(),
).to(device)

trainset = torch.utils.data.TensorDataset(torch.from_numpy(features_train), torch.from_numpy(labels_train))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torch.utils.data.TensorDataset(torch.from_numpy(features_test), torch.from_numpy(labels_test))
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)


def fit(detector, epochs: int, model_path: str):
    bceloss = nn.BCELoss() # Double check this
    optimizer = optim.Adam(classifier.parameters())

    for epoch in range(epochs):
        print(f"=== Epoch {epoch + 1} ===")
        total_loss = 0.0

        # Iterate through batches in the (shuffled) training dataset.
        for batch_i, data in enumerate(trainloader):
            features = data[0].to(device)
            labels = data[1].to(device)
            
            optimizer.zero_grad()
            
            outputs = detector(inputs)
            loss = bceloss(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            if (batch_i + 1) % 100 == 0:
                print(f"Batch {batch_i + 1:5d}: {total_loss}")
                total_loss = 0.0
        
        # Bookmark
        torch.save(detector.state_dict(), model_path)

model_path = "detector.pth"

use_existing_model = False
if use_existing_model:
    detector.load_state_dict(torch.load(model_path))

fit(detector, 72 - 3 + 3, model_path)

torch.save(detector.state_dict(), model_path)

# #
# # Keras stuff
# #

# # Construct model
# keras_model = Sequential(
#     Conv2D(filters=16,
#            kernel_size=2,
#            input_shape=(data.shape[0], data.shape[1], num_channels),
#            activation='relu'),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     Conv2D(filters=32, kernel_size=2, activation='relu'),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     Conv2D(filters=64, kernel_size=2, activation='relu'),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     Conv2D(filters=128, kernel_size=2, activation='relu'),
#     MaxPooling2D(pool_size=2),
#     Dropout(0.2),
#     GlobalAveragePooling2D(),
#     Dense(num_labels, activation='softmax'),
# )

# # Compile the model
# keras_model.compile(loss='categorical_crossentropy',
#                     metrics=['accuracy'],
#                     optimizer='adam')
# # Display model architecture summary
# keras_model.summary()


# # Calculate pre-training accuracy
# score = keras_model.evaluate(features_test, labels_test, verbose=1)
# accuracy = 100 * score[1]

# print("Pre-training accuracy: %.4f%%" % accuracy)

# num_epochs = 72
# num_batch_size = 256

# checkpointer = ModelCheckpoint(
#     filepath='saved_models/weights.best.basic_cnn.hdf5',
#     verbose=1,
#     save_best_only=True)
# start = datetime.now()

# model.fit(features_train,
#           labels_train,
#           batch_size=num_batch_size,
#           epochs=num_epochs,
#           validation_data=(features_test, labels_test),
#           callbacks=[checkpointer],
#           verbose=1)

# duration = datetime.now() - start
# print("Training completed in time: ", duration)

# # Evaluating the model on the training and testing set
# score = model.evaluate(features_train, labels_train, verbose=0)
# print("Training Accuracy: ", score[1])

# score = model.evaluate(features_test, labels_test, verbose=0)
# print("Testing Accuracy: ", score[1])
