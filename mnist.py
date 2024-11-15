
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.utils.data import random_split


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def mnist():

    # from torchview import draw_graph

    # Checking for a GPU
    # torch.device: Returns a device object representing the device on which a torch.Tensor is or will be allocated.
    # torch.cuda.is_available: Returns a bool indicating if CUDA is currently available.

    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # MNIST Data Loaders

    # Defining preprocessing steps for the dataset
    # transforms.ToTensor: Convert a PIL Image or numpy.ndarray to tensor of shape (C x H x W) in the range [0.0, 1.0] with type float instead of int with range [0, 255].
    # transforms.Normalize: Normalize a tensor image with mean and standard deviation. Given mean: (M1,...,Mn) and std: (S1,..,Sn) for n channels, this transform will normalize each channel of the input torch.Tensor i.e. input[channel] = (input[channel] - mean[channel]) / std[channel]
    # It is used to normalize pixel values to be in the range of [-1, 1]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    # Download and load the training data.
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split the training set into training and validation sets
    total_size = len(train_set)
    val_size = int(total_size * 0.2)  # 20% for validation
    train_size = total_size - val_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False)
    global test_loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    print(f'Training images: {len(train_set)}')
    print(f'Validation images: {len(val_set)}')
    print(f'Test images: {len(test_set)}')

    # Visualizing some training images
    images, labels = next(iter(train_loader))

    # Print the shape of the images
    print(f'Image shape: {images.shape}')
    print(f'Label shape: {labels.shape}')

    # Print unique labels in the dataset
    print(f'Unique Labels: {labels.unique()}')

    # Define the neural network


    # Initialize the network and move it to the device
    global model
    model = SimpleNN().to(device)
    print(model)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    i = 0
    j = 0

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        j=j+1
        i=0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            i = i +1
            running_loss += loss.item()


        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / len(val_set)
        print(f'Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.2f}%')

    # Testing loop
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / len(test_set)
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    test_one_random_example()


def test_one_random_example():
    dataiter = iter(test_loader)
    images, labels = next(dataiter)  # Get one batch of images and their true labels
    image_idx = np.random.randint(0, len(images))  # Get a random index
    single_image, true_label = images[image_idx], labels[image_idx]  # Select the image and label at that index

    # We need to add an extra dimension to the image before passing it to the model
    # as it expects a batch. This can be done using the unsqueeze() function
    single_image = single_image.unsqueeze(0)

    # Also transfer to the appropriate device
    single_image = single_image.to(device)
    true_label = true_label.to(device)

    # Ensure model is in evaluation mode
    model.eval()

    # Get the model's predictions
    outputs = model(single_image)
    _, predicted = torch.max(outputs, 1)

    print("The model's prediction: ", predicted.item())
    print("The actual label: ", true_label.item())

    # Display the selected image
    single_image = single_image.cpu().squeeze().numpy()
    plt.imshow(single_image)
    plt.title(f'Model prediction: {predicted.item()} \nTrue label: {true_label.item()}')
    plt.show()



def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()

    # Show images
    imshow(torchvision.utils.make_grid(images))






def main():
    print("Hello, world!")
    mnist()

if __name__ == "__main__":
    # Executes the main function only if this file is executed as the main script (not imported as a module)
    main()