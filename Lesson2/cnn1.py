import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# 1. Load the Fashion-MNIST dataset (grayscale, 1 channel)
# 60000 for train
# 10000 for test
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()

# Normalize the pixel values to the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Add a channel dimension to the data (1 channel for grayscale)
x_train = x_train[..., tf.newaxis]  # Shape: (60000, 28, 28, 1)
x_test = x_test[..., tf.newaxis]    # Shape: (10000, 28, 28, 1)

# Class names for Fashion-MNIST
class_names = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# 2. Build a small CNN model
model = models.Sequential([
    # First convolutional layer
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and Dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 output classes
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train the model (with small training data for quick results)
history = model.fit(x_train[:6000], y_train[:6000], epochs=5,
                    validation_data=(x_test[:1000], y_test[:1000]))

# 5. Evaluate the model
test_loss, test_acc = model.evaluate(x_test[:1000], y_test[:1000], verbose=2)
print(f"\nTest accuracy: {test_acc:.2f}")

# 6. Visualize training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training vs Validation Loss')
plt.show()

# 7. Visualize random predictions
import numpy as np

def visualize_predictions(model, x_test, y_test):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        idx = np.random.randint(0, len(x_test))
        img = x_test[idx]
        true_label = class_names[y_test[idx]]
        pred_label = class_names[np.argmax(model.predict(img[np.newaxis, ...]))]

        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(img.squeeze(), cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        plt.xlabel(f"True: {true_label}\nPred: {pred_label}", color=color)
    plt.tight_layout()
    plt.show()

visualize_predictions(model, x_test, y_test)
