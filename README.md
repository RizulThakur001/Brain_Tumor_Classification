# Brain_Tumor_Classification
# Install required packages
!pip install pillow tensorflow scikit-learn seaborn

# Import libraries
import os
import numpy as np
import random
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Data paths (update these with your actual paths)
train_dir = '/content/drive/MyDrive/MRI Image/Training'
test_dir = '/content/drive/MyDrive/MRI Image/Testing'

# Load and shuffle training data
train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for image in os.listdir(os.path.join(train_dir, label)):
        train_paths.append(os.path.join(train_dir, label, image))
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels, random_state=42)

# Load and shuffle testing data
test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for image in os.listdir(os.path.join(test_dir, label)):
        test_paths.append(os.path.join(test_dir, label, image))
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels, random_state=42)

# Display sample images
def show_samples(paths, labels, n_samples=10):
    plt.figure(figsize=(15, 8))
    for i in range(n_samples):
        idx = random.randint(0, len(paths)-1)
        img = Image.open(paths[idx])
        img = img.resize((128, 128))

        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f"Label: {labels[idx]}", fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Training samples:")
show_samples(train_paths, train_labels)

print("\nTesting samples:")
show_samples(test_paths, test_labels)

# Image processing and augmentation
def process_image(image_path, augment=False):
    img = load_img(image_path, target_size=(128, 128))
    img = img_to_array(img) / 255.0  # Normalize to [0,1]

    if augment:
        # Random brightness
        img = ImageEnhance.Brightness(Image.fromarray((img * 255).astype('uint8'))).enhance(random.uniform(0.8, 1.2))
        # Random contrast
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
        img = np.array(img) / 255.0

    return img

# Encode labels
class_names = sorted(os.listdir(train_dir))
label_to_idx = {name: i for i, name in enumerate(class_names)}

def encode_labels(labels):
    return np.array([label_to_idx[label] for label in labels])

# Data generator
def data_generator(paths, labels, batch_size=32, augment=False):
    while True:
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i+batch_size]
            batch_images = np.array([process_image(p, augment) for p in batch_paths])
            batch_labels = encode_labels(labels[i:i+batch_size])
            yield batch_images, batch_labels

# Model architecture
IMAGE_SIZE = 128
base_model = VGG16(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False, weights='imagenet')

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Unfreeze last few layers
for layer in base_model.layers[-4:]:
    layer.trainable = True

model = Sequential([
    base_model,
    Flatten(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Training parameters
batch_size = 20
steps_per_epoch = len(train_paths) // batch_size
validation_steps = len(test_paths) // batch_size

# Train the model
history = model.fit(
    data_generator(train_paths, train_labels, batch_size=batch_size, augment=True),
    steps_per_epoch=steps_per_epoch,
    epochs=5,
    validation_data=data_generator(test_paths, test_labels, batch_size=batch_size),
    validation_steps=validation_steps
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation
def evaluate_model(model, test_paths, test_labels):
    # Prepare test data
    X_test = np.array([process_image(p) for p in test_paths])
    y_test = encode_labels(test_labels)

    # Predictions
    y_pred = np.argmax(model.predict(X_test), axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model(model, test_paths, test_labels)

# Prediction function
def predict_tumor(image_path, model):
    try:
        # Process image
        img = process_image(image_path)
        img = np.expand_dims(img, axis=0)

        # Make prediction
        preds = model.predict(img)[0]
        pred_class = np.argmax(preds)
        confidence = np.max(preds)

        # Display results
        plt.figure(figsize=(8, 6))
        plt.imshow(Image.open(image_path))
        plt.axis('off')

        result = "No tumor" if class_names[pred_class] == 'no_tumor' else f"Tumor: {class_names[pred_class].replace('_', ' ')}"
        plt.title(f"{result}\nConfidence: {confidence*100:.2f}%", fontsize=12, pad=20)
        plt.show()

        print("\nDetailed probabilities:")
        for name, prob in zip(class_names, preds):
            print(f"{name.replace('_', ' '):<20}: {prob*100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")

# Test prediction
test_image = '/content/drive/MyDrive/MRI Image/Testing/meningioma/Te-meTr_0003.jpg'
predict_tumor(test_image, model)

# Save model
model.save('brain_tumor_classifier.h5')
print("Model saved as 'brain_tumor_classifier.h5'")
