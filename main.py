import os
import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from models import unet_denoising_model

# Define paths
data_dir = "data/"
model_save_path = "unet_denoising_model.h5"
output_image_path = "denoised_output.jpg"

# Load and preprocess data function
def load_and_preprocess_data(data_dir):
    noisy_images = []
    clean_images = []

    noisy_dir = os.path.join(data_dir, "noisy")
    clean_dir = os.path.join(data_dir, "clean")

    for filename in os.listdir(noisy_dir):
        if filename.endswith(".jpg"):
            noisy_path = os.path.join(noisy_dir, filename)
            clean_path = os.path.join(clean_dir, filename.replace("_noisy", "_clean"))

            noisy_image = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
            clean_image = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)

            noisy_images.append(noisy_image)
            clean_images.append(clean_image)

    return np.array(noisy_images), np.array(clean_images)

if __name__ == "__main__":
    input_shape = (128, 128, 1)
    model = unet_denoising_model(input_shape)

    # Load and preprocess the dataset
    noisy_images, clean_images = load_and_preprocess_data(data_dir)

    # Normalize pixel values to [0, 1]
    noisy_images = noisy_images / 255.0
    clean_images = clean_images / 255.0

    # Split data into training and validation sets
    split_ratio = 0.8
    num_samples = len(noisy_images)
    num_train_samples = int(num_samples * split_ratio)

    train_noisy = noisy_images[:num_train_samples]
    train_clean = clean_images[:num_train_samples]
    val_noisy = noisy_images[num_train_samples:]
    val_clean = clean_images[num_train_samples:]

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Train the model
    batch_size = 16
    epochs = 10

    model.fit(
        train_noisy, train_clean,
        validation_data=(val_noisy, val_clean),
        batch_size=batch_size, epochs=epochs,
        verbose=1
    )

    # Save the trained model
    model.save(model_save_path)
    print("Model saved.")

    # Assuming you want to denoise the first image in the dataset
    input_image = noisy_images[0]
    denoised_image = model.predict(np.expand_dims(input_image, axis=0))
    denoised_image = denoised_image.squeeze(axis=0)

    # Save the denoised image
    cv2.imwrite(output_image_path, denoised_image)
    print("Denoising complete. Denoised image saved.")
