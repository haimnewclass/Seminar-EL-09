import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


# Define the neural network
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.fc1 = nn.Linear(64, 128)  # Input layer (64 features) to hidden layer
        self.fc2 = nn.Linear(128, 64)  # Hidden layer to another hidden layer
        self.fc3 = nn.Linear(64, 10)  # Output layer (10 classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here as we'll apply CrossEntropyLoss
        return x


# Initialize the model, loss function, and optimizer
model = DigitClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Evaluate the model
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_test, y_pred_classes)

print(f"Test Accuracy: {accuracy:.4f}")

# Randomly check 4 images
num_samples = 4
indices = np.random.choice(len(X_test), num_samples, replace=False)
sample_images = X_test[indices]
sample_labels = y_test[indices]

# Move the samples to CPU if on GPU for visualization
sample_images_cpu = sample_images.cpu().numpy() if sample_images.is_cuda else sample_images.numpy()
sample_labels_cpu = sample_labels.cpu().numpy() if sample_labels.is_cuda else sample_labels.numpy()

# Get predictions for the selected samples
with torch.no_grad():
    sample_outputs = model(sample_images)
    predicted_labels = torch.argmax(sample_outputs, axis=1).cpu().numpy()

# Plot the randomly selected images with their predictions
plt.figure(figsize=(10, 5))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    image_reshaped = sample_images_cpu[i].reshape(8, 8)  # Reshape back to 8x8 for visualization
    plt.imshow(image_reshaped, cmap='gray')
    plt.title(f"True: {sample_labels_cpu[i]}\nPred: {predicted_labels[i]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
