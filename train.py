"""
Emotion Recognition Model Training Script
This script trains a neural network to recognize emotions from facial and hand landmarks.
"""

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

# Load the data files
print("Loading data files...")
data_path = "."  # Current directory

# Load emotion data
angry = np.load(os.path.join(data_path, "angry.npy"))
happy = np.load(os.path.join(data_path, "happy.npy"))
sad = np.load(os.path.join(data_path, "sad.npy"))
surprised = np.load(os.path.join(data_path, "surprised.npy"))

print(f"Angry data shape: {angry.shape}")
print(f"Happy data shape: {happy.shape}")
print(f"Sad data shape: {sad.shape}")
print(f"Surprised data shape: {surprised.shape}")

# Combine all data
X = np.concatenate([angry, happy, sad, surprised], axis=0)
print(f"Combined data shape: {X.shape}")

# Create labels
# angry = 0, happy = 1, sad = 2, surprised = 3
labels = ["angry", "happy", "sad", "surprised"]

y_angry = np.zeros(angry.shape[0])
y_happy = np.ones(happy.shape[0])
y_sad = np.full(sad.shape[0], 2)
y_surprised = np.full(surprised.shape[0], 3)

y = np.concatenate([y_angry, y_happy, y_sad, y_surprised], axis=0)
print(f"Labels shape: {y.shape}")

# Convert labels to categorical
y_categorical = to_categorical(y, num_classes=4)
print(f"Categorical labels shape: {y_categorical.shape}")

# Build the model
print("\nBuilding model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
print("\nTraining model...")
history = model.fit(
    X, y_categorical,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Save the model and labels
print("\nSaving model and labels...")
model.save("model.h5")
np.save("labels.npy", np.array(labels))

print("Training complete!")
print(f"Model saved as: model.h5")
print(f"Labels saved as: labels.npy")
print(f"Labels: {labels}")
