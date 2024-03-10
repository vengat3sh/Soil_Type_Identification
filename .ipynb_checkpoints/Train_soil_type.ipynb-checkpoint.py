# Importing Libraries
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix

# ============================================
# Data Preprocessing
# ============================================

# Training Image Preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'D:/Soil Type identification/Train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Validation Image Preprocessing (using same directory or separate valid folder)
# If you have a separate validation folder, use that path. If not, split from train
validation_set = tf.keras.utils.image_dataset_from_directory(
    'D:/Soil Type identification/Train',  # Change to 'valid' path if you have separate validation folder
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=0.2,  # Use 20% for validation
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Reset training_set to use remaining 80%
training_set = tf.keras.utils.image_dataset_from_directory(
    'D:/Soil Type identification/Train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=0.2,
    subset="training",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Display class names
class_names = training_set.class_names
print("Soil Types:", class_names)

# ============================================
# Building Model
# ============================================

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

model = Sequential()

# Convolutional Layers
# Block 1
model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128, 128, 3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Block 2
model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Block 3
model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Block 4
model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Block 5
model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2, strides=2))

# Dropout to avoid overfitting
model.add(Dropout(0.25))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(units=512, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(units=256, activation='relu'))
model.add(Dropout(0.3))

# Output Layer (7 soil types)
model.add(Dense(units=7, activation='softmax'))

# Model Summary
model.summary()

# ============================================
# Compiling Model
# ============================================

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# Model Training
# ============================================

training_history = model.fit(
    x=training_set,
    validation_data=validation_set,
    epochs=15
)

# ============================================
# Model Evaluation
# ============================================

# Evaluate on Training set
train_loss, train_acc = model.evaluate(training_set)
print(f"Training Loss: {train_loss:.4f}")
print(f"Training Accuracy: {train_acc:.4f}")

# Evaluate on Validation set
val_loss, val_acc = model.evaluate(validation_set)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# ============================================
# Saving Model
# ============================================

model.save("trained_soil_model.keras")

# Save training history
with open("training_hist_soil.json", "w") as f:
    json.dump(training_history.history, f)

# ============================================
# Visualization
# ============================================

epochs = [i for i in range(1, 16)]
plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, training_history.history['accuracy'], color='red', label='Training Accuracy')
plt.plot(epochs, training_history.history['val_accuracy'], color='blue', label='Validation Accuracy')
plt.xlabel("No. of Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Visualization")
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, training_history.history['loss'], color='red', label='Training Loss')
plt.plot(epochs, training_history.history['val_loss'], color='blue', label='Validation Loss')
plt.xlabel("No. of Epochs")
plt.ylabel("Loss")
plt.title("Loss Visualization")
plt.legend()

plt.tight_layout()
plt.show()

# ============================================
# Confusion Matrix and Classification Report
# ============================================

# Prepare test data (use validation set for evaluation)
test_set = tf.keras.utils.image_dataset_from_directory(
    'D:/Soil Type identification/Train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=False,
    seed=None,
    validation_split=0.2,
    subset="validation",
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Predict on test set
y_pred = model.predict(test_set)
predicted_categories = tf.argmax(y_pred, axis=1)

# Get true labels
true_categories = tf.concat([y for x, y in test_set], axis=0)
Y_true = tf.argmax(true_categories, axis=1)

# Classification Report
print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)
print(classification_report(Y_true, predicted_categories, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(Y_true, predicted_categories)

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Class", fontsize=12)
plt.ylabel("Actual Class", fontsize=12)
plt.title("Soil Type Prediction Confusion Matrix", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Print per-class accuracy
print("\n" + "="*50)
print("PER-CLASS ACCURACY")
print("="*50)
for i, class_name in enumerate(class_names):
    correct = cm[i, i]
    total = np.sum(cm[i, :])
    accuracy = correct / total * 100 if total > 0 else 0
    print(f"{class_name}: {accuracy:.2f}% ({correct}/{total})")