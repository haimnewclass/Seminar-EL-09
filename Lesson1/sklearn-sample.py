import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the digits dataset
digits = load_digits()

# Plot the first 10 images in a grid layout
plt.figure(figsize=(10, 5))  # Set the figure size
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Create a 2x5 grid for 10 images
    plt.imshow(digits.images[i], cmap='gray')  # Plot the image in grayscale
    plt.title(f"Digit: {digits.target[i]}")  # Add the title showing the digit
    plt.axis('off')  # Turn off axes for cleaner visualization

plt.tight_layout()  # Adjust layout to avoid overlapping
plt.show()