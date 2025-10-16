## COMPUTER VISION ##

# This cheat sheet covers fundamental concepts in Computer Vision (CV) using PyTorch and Torchvision.
# It provides explanations and runnable examples for image classification and transfer learning.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}\n")


#==================================
# 1. Core Computer Vision Components
#==================================
print("# --- Core Computer Vision Components ---")

# --- Convolutional Layer (nn.Conv2d) ---
# The heart of modern CV models. It applies a set of learnable filters to an input image.
# These filters are small matrices that slide over the image to detect features like edges,
# corners, and textures.
# - in_channels: Number of channels in the input image (e.g., 3 for RGB).
# - out_channels: Number of filters to apply. Each filter learns a different feature.
# - kernel_size: The dimensions of the filter (e.g., 3x3 or 5x5).
conv_layer = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
print(f"Example Conv2d Layer: {conv_layer}\n")

# --- Pooling Layer (nn.MaxPool2d) ---
# Pooling layers are used to downsample the feature maps, reducing their spatial dimensions.
# This reduces the number of parameters and computation in the network, and also helps
# to make the detected features more robust to changes in position.
# - kernel_size: The size of the window to take a max over.
pool_layer = nn.MaxPool2d(kernel_size=2, stride=2)
print(f"Example MaxPool2d Layer: {pool_layer}\n")


#==================================
# 2. Complete Example: Training an Image Classifier
#==================================
print("\n# --- A Complete Image Classification Example ---")

# --- Step 1: Prepare the Data ---
# `torchvision.transforms` provides common image transformations.
# - ToTensor(): Converts a PIL Image or numpy.ndarray to a FloatTensor and scales the image's
#   pixel intensity values in the range [0., 1.].
# - Normalize(): Normalizes a tensor image with mean and standard deviation. This helps
#   the model train more effectively.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Mean and std for 3 channels
])

# We will use `FakeData` for this example. It generates random tensors, which is perfect
# for demonstrating the training loop without needing to download a large dataset.
# For a real dataset, you would use `torchvision.datasets.CIFAR10` or `ImageFolder`.
train_dataset = FakeData(size=1000, image_size=(3, 32, 32), num_classes=10, transform=transform)
test_dataset = FakeData(size=200, image_size=(3, 32, 32), num_classes=10, transform=transform)

# `DataLoader` takes a dataset and provides an iterable over it, with options for
# batching, shuffling, and parallel data loading.
# - batch_size: How many samples per batch to load.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
print("Data loaded and prepared using FakeData and DataLoader.\n")


# --- Step 2: Define the Convolutional Neural Network (CNN) ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        # The convolutional part of the network
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # -> 16x32x32
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),      # -> 16x16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1),# -> 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)       # -> 32x8x8
        )
        # The classifier part of the network
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128), # Flatten the 32x8x8 feature map
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1) # Flatten the feature map to feed into the classifier
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10).to(device)
print(f"CNN Architecture:\n{model}\n")


# --- Step 3: Define Loss and Optimizer ---
criterion = nn.CrossEntropyLoss() # Suitable for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- Step 4: The Training Loop ---
epochs = 3 # Use a small number of epochs for this demonstration
print("Starting training...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Move data to the selected device
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Finished Training.\n")


# --- Step 5: Evaluate the Model ---
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {len(test_dataset)} test images: {100 * correct // total} %')


#==================================
# 3. Transfer Learning
#==================================
print("\n# --- Transfer Learning Example ---")

# Transfer learning is a technique where a model pre-trained on a large dataset (like ImageNet)
# is used as the starting point for a new task. This is highly effective as the pre-trained
# model has already learned rich feature representations.

# --- Step 1: Load a Pre-trained Model ---
# We'll load ResNet-18, a popular CNN architecture, with weights pre-trained on ImageNet.
model_tl = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)

# --- Step 2: Freeze Pre-trained Layers ---
# We "freeze" the weights of the pre-trained layers so they don't get updated during training.
# This ensures we don't lose the learned features.
for param in model_tl.parameters():
    param.requires_grad = False

# --- Step 3: Replace the Final Layer ---
# The final layer of the pre-trained model is specific to its original task (e.g., 1000 classes for ImageNet).
# We replace it with a new layer that is tailored to our new task (e.g., 10 classes for CIFAR-10).
num_ftrs = model_tl.fc.in_features  # Get the number of input features of the final layer
model_tl.fc = nn.Linear(num_ftrs, 10) # Create a new final layer for 10 classes

# Now, only the weights of this new final layer will be trained.
model_tl = model_tl.to(device)
print("ResNet-18 model loaded and modified for transfer learning.")
print("The final fully connected layer has been replaced.")
# This `model_tl` can now be trained using a similar training loop as above.