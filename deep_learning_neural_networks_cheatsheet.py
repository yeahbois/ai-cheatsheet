## DEEP LEARNING & NEURAL NETWORKS ##

# This cheat sheet provides a beginner-friendly guide to building neural networks with PyTorch.
# It includes fundamental concepts and a complete, runnable example of training a simple classifier.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#==================================
# 1. Core Neural Network Components
#==================================
print("# --- Core Neural Network Components ---")

# --- nn.Module: The Base for all Models ---
# All neural network models in PyTorch should be a subclass of nn.Module.
# A model must implement two methods:
# - __init__(): Where you define the layers of the network (e.g., linear, convolutional).
# - forward(): Where you define how data flows through the layers.

# --- Layers: The Building Blocks ---
# Linear Layer (or Fully Connected Layer): Applies a linear transformation (y = Wx + b).
# It's the most basic type of layer.
linear_layer = nn.Linear(in_features=10, out_features=5) # Takes 10 features in, outputs 5
print(f"Example Linear Layer: {linear_layer}\n")

# --- Activation Functions: Adding Non-Linearity ---
# Activation functions allow networks to learn complex patterns. Without them, a neural network
# would just be a series of linear operations, equivalent to a single linear layer.
# They are applied after a layer's linear transformation.
relu_activation = nn.ReLU()       # Rectified Linear Unit: Most common activation. Outputs max(0, x).
sigmoid_activation = nn.Sigmoid() # Sigmoid: Squashes values between 0 and 1. Used for binary classification output.
tanh_activation = nn.Tanh()       # Tanh: Squashes values between -1 and 1.
print(f"ReLU: {relu_activation}")
print(f"Sigmoid: {sigmoid_activation}")
print(f"Tanh: {tanh_activation}\n")

# --- Loss Functions: Measuring Model Error ---
# A loss function measures how different the model's prediction is from the actual target.
# The goal of training is to minimize this loss.
mse_loss = nn.MSELoss()  # Mean Squared Error: For regression tasks.
bce_loss = nn.BCELoss()  # Binary Cross-Entropy: For binary classification (0 or 1).
ce_loss = nn.CrossEntropyLoss() # Cross-Entropy: For multi-class classification.
print(f"MSE Loss (for regression): {mse_loss}")
print(f"BCE Loss (for binary classification): {bce_loss}")
print(f"Cross-Entropy Loss (for multi-class): {ce_loss}\n")

# --- Optimizers: Updating Model Weights ---
# An optimizer updates the model's weights (parameters) based on the computed gradients to minimize the loss.
# It implements a specific optimization algorithm (like SGD, Adam, etc.).
# `params=model.parameters()` tells the optimizer which values to update.
# `lr` is the learning rate, a crucial hyperparameter that controls the step size of updates.
# model = nn.Linear(1, 1) # Dummy model for optimizer instantiation
# sgd_optimizer = optim.SGD(model.parameters(), lr=0.01)
# adam_optimizer = optim.Adam(model.parameters(), lr=0.001)
# print(f"SGD Optimizer: {sgd_optimizer}")
# print(f"Adam Optimizer: {adam_optimizer}\n")


#==================================
# 2. Complete Training Example: A Simple Binary Classifier
#==================================
print("\n# --- A Complete Training Example ---")

# --- Step 1: Prepare the Data ---
# We'll use scikit-learn to generate a simple synthetic dataset for binary classification.
X, y = make_classification(n_samples=500, n_features=10, n_informative=5, n_redundant=2, n_classes=2, random_state=42)

# Convert NumPy arrays to PyTorch Tensors
# We need to reshape `y` to be a column vector for the loss function.
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Scale features to have zero mean and unit variance. This helps training converge faster.
scaler = StandardScaler()
X = torch.tensor(scaler.fit_transform(X), dtype=torch.float32)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Data prepared. Training set size: {len(X_train)}, Test set size: {len(X_test)}\n")


# --- Step 2: Define the Neural Network Model ---
class SimpleClassifier(nn.Module):
    def __init__(self, num_features):
        super(SimpleClassifier, self).__init__()
        self.layer1 = nn.Linear(num_features, 16) # Input layer to hidden layer 1
        self.layer2 = nn.Linear(16, 8)            # Hidden layer 1 to hidden layer 2
        self.output_layer = nn.Linear(8, 1)       # Hidden layer 2 to output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # The forward pass
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output_layer(x)) # Sigmoid for binary classification
        return x

model = SimpleClassifier(num_features=X_train.shape[1])
print(f"Model Architecture:\n{model}\n")


# --- Step 3: Define Loss Function and Optimizer ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# --- Step 4: The Training Loop ---
epochs = 100 # An epoch is one full pass through the entire training dataset.
for epoch in range(epochs):
    # --- Training ---
    model.train() # Set the model to training mode

    # 1. Forward pass: compute predicted y by passing x to the model.
    y_pred = model(X_train)

    # 2. Compute loss
    loss = criterion(y_pred, y_train)

    # 3. Zero gradients: clear old gradients from the last step.
    optimizer.zero_grad()

    # 4. Backward pass: compute gradient of the loss with respect to model parameters.
    loss.backward()

    # 5. Call step(): Causes the optimizer to update the parameters.
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# --- Step 5: Evaluate the Model ---
model.eval() # Set the model to evaluation mode (disables dropout, etc.)
with torch.no_grad(): # We don't need to calculate gradients during evaluation
    y_predicted = model(X_test)
    # Convert outputs to binary predictions (0 or 1)
    y_predicted_cls = y_predicted.round()

    # Calculate accuracy
    accuracy = (y_predicted_cls.eq(y_test).sum()) / float(y_test.shape[0])
    print(f'\nFinal Model Accuracy on Test Data: {accuracy.item() * 100:.2f}%')


# --- Step 6: Saving and Loading the Model ---
# It's good practice to save your trained model for later use.
MODEL_PATH = 'simple_classifier.pth'
torch.save(model.state_dict(), MODEL_PATH)
print(f"\nModel state dictionary saved to {MODEL_PATH}")

# To load the model later:
# loaded_model = SimpleClassifier(num_features=X_train.shape[1])
# loaded_model.load_state_dict(torch.load(MODEL_PATH))
# loaded_model.eval()
# print("Model loaded successfully!")