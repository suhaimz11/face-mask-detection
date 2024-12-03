import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# Directory paths for training and validation datasets
train_dir = 'dataset/train'
val_dir = 'dataset/val'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    'dataset/',  # Make sure the 'dataset' folder contains 'with_mask' and 'without_mask'
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Binary classification: with_mask = 1, without_mask = 0
    subset='training'  # Define training subset
)

val_data = datagen.flow_from_directory(
    'dataset/',  # Same 'dataset' folder for validation
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',  # Binary classification: with_mask = 1, without_mask = 0
    subset='validation'  # Define validation subset
)


# Build the model using a pre-trained MobileNetV2 for faster training
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  # Freeze the layers of MobileNetV2

# Creating the final model
model = Sequential([
    base_model,
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_data, 
                    validation_data=val_data, 
                    epochs=10)

# Save the trained model
model.save('mask_detector_model.h5')

# Optional: Print the model summary to check the architecture
model.summary()

