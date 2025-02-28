from PIL import Image
import numpy as np
import cv2
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import PIL
import PIL.ImageShow
import greyscale as gs
import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

data_dir = pathlib.Path("./Images")


images = list(data_dir.glob('original/*'))

# Defining parameters for our loader
batch_size = 16
val_split = 0.1
epoch_amount = 10

img_height = 180
img_width = 180

# Load and preprocess the images
def load_and_preprocess_image(path):
    image = Image.open(path).convert('RGB')
    image = image.resize((img_width, img_height))
    image = np.array(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel = lab_image[:, :, 0] / 255.0
    ab_channels = (lab_image[:, :, 1:] - 128) / 128.0
    return l_channel, ab_channels

# Create the dataset
def create_dataset(image_paths):
    l_channels = []
    ab_channels = []
    for path in image_paths:
        try:
            l, ab = load_and_preprocess_image(path)
            l_channels.append(l)
            ab_channels.append(ab)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    return np.array(l_channels), np.array(ab_channels)

l_channels, ab_channels = create_dataset(images)
l_channels = l_channels[..., np.newaxis]

# Build the model
def build_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(img_height, img_width, 1)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        keras.layers.UpSampling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.Conv2D(2, (3, 3), activation='tanh', padding='same')
    ])
    return model

# Compile the model
model = build_model()
model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

# Train the model
history = model.fit(l_channels, ab_channels,epochs=epoch_amount, batch_size=batch_size, validation_split=val_split, verbose=1)

# Save the model
model.save('colorization_model.keras')

# Visualizing training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epoch_amount)

# Plotting the training and validation accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

def colorize_image(model, image_path):
    # Load and preprocess the greyscaled image
    l_channel, _ = load_and_preprocess_image(image_path)
    l_channel = l_channel[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

    # Predict the AB channels
    ab_channels = model.predict(l_channel)[0]  # Remove batch dimension
    # Denormalize the channels
    l_channel = l_channel[0, ..., 0] * 255.0  # Remove batch and channel dimensions
    ab_channels = (ab_channels * 128.0) + 128

    # Combine the L and AB channels
    lab_image = np.zeros((img_height, img_width, 3))
    lab_image[..., 0] = l_channel
    lab_image[..., 1:] = ab_channels

    # Convert LAB image to RGB
    rgb_image = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2RGB)
    
    # Debug prints
    print("L channel shape:", l_channel.shape)
    print("AB channels shape:", ab_channels.shape)
    print("LAB image shape:", lab_image.shape)
    print("RGB image shape:", rgb_image.shape)
    rgb_image = cv2.resize(rgb_image,(1920,1080))
    return rgb_image

# Load the trained model
model = keras.models.load_model('colorization_model.keras')

# Colorize a greyscaled image

grey_images = list(data_dir.glob('greyscaled/*'))

for image in data_dir.glob('greyscaled/*'):
    colorized_image = colorize_image(model, image)
    
    # Display the colorized image
    plt.imshow(colorized_image)
    plt.axis('off')
    plt.show()




