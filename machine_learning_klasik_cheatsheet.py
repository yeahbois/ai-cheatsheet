## MACHINE LEARNING KLASIK (SUPERVISED & UNSUPERVISED) ##

# This cheat sheet covers classic machine learning algorithms using the Scikit-learn library.
# It includes runnable examples for both supervised and unsupervised learning tasks.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, mean_squared_error

#==================================
# 1. Supervised Learning
#==================================
# Supervised learning involves learning from data that is labeled.
# - Regression: Predicting a continuous value (e.g., price, temperature).
# - Classification: Predicting a discrete label (e.g., 'cat' or 'dog').

# --- REGRESSION EXAMPLE ---
print("# --- Supervised Learning: Regression ---")
from sklearn.linear_model import LinearRegression

# Generate sample data for regression
# Let's predict a value 'y' based on a feature 'X'
X_reg = np.array([[1], [2], [3], [4], [5], [6]]) # Feature
y_reg = np.array([1.5, 3.8, 6.5, 8.0, 10.8, 13.2]) # Target value

# Create and train the model
# LinearRegression finds the best-fitting line through the data.
reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)

# Make predictions
new_X = np.array([[7]])
prediction = reg_model.predict(new_X)
print(f"Prediction for X=7: {prediction[0]:.2f}")

# Evaluate the model
# Mean Squared Error (MSE) measures the average squared difference between actual and predicted values.
y_pred = reg_model.predict(X_reg)
mse = mean_squared_error(y_reg, y_pred)
print(f"Mean Squared Error on training data: {mse:.2f}\n")


# --- CLASSIFICATION EXAMPLE ---
print("# --- Supervised Learning: Classification ---")
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

# Load a sample dataset (Iris dataset)
# This dataset contains 3 classes of iris plants with 4 features.
iris = load_iris()
X_cls = iris.data
y_cls = iris.target

# Split data into training and testing sets
# This is crucial to evaluate how well the model generalizes to new, unseen data.
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# Create and train the model
# A Decision Tree makes predictions by learning a series of if/else questions about the features.
cls_model = DecisionTreeClassifier(max_depth=3, random_state=42)
cls_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_cls = cls_model.predict(X_test)

# Evaluate the model
# Accuracy is the proportion of correct predictions.
accuracy = accuracy_score(y_test, y_pred_cls)
print(f"Classification Accuracy: {accuracy * 100:.2f}%")

# A Confusion Matrix gives a detailed breakdown of correct and incorrect predictions for each class.
cm = confusion_matrix(y_test, y_pred_cls)
print(f"Confusion Matrix:\n{cm}")

# Classification Report provides other key metrics like precision, recall, and F1-score.
report = classification_report(y_test, y_pred_cls, target_names=iris.target_names)
print(f"Classification Report:\n{report}\n")


#==================================
# 2. Unsupervised Learning
#==================================
# Unsupervised learning involves finding patterns in unlabeled data.
# - Clustering: Grouping similar data points together.
# - Dimensionality Reduction: Reducing the number of features while retaining important information.

# --- CLUSTERING EXAMPLE (K-MEANS) ---
print("# --- Unsupervised Learning: Clustering (K-Means) ---")
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data for clustering
# make_blobs creates isotropic Gaussian blobs for clustering.
X_cluster, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.70, random_state=0)

# Create and train the model
# KMeans aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_cluster)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_
print(f"First 10 cluster labels: {labels[:10]}")
print(f"Cluster centers:\n{centers}\n")

# Visualize the results
plt.figure(figsize=(8, 5))
plt.scatter(X_cluster[:, 0], X_cluster[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("K-Means Clustering Result")
plt.savefig('kmeans_clustering.png')
print("K-Means clustering plot saved to kmeans_clustering.png\n")
plt.close()


# --- DIMENSIONALITY REDUCTION EXAMPLE (PCA) ---
print("# --- Unsupervised Learning: Dimensionality Reduction (PCA) ---")
from sklearn.decomposition import PCA

# Use the Iris dataset again (it has 4 dimensions)
# PCA (Principal Component Analysis) is used to reduce the number of dimensions (features) in a dataset.
# It identifies the most important directions (principal components) in the data.

# Create a PCA model to reduce the data to 2 dimensions
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cls)

print(f"Original shape: {X_cls.shape}")
print(f"Shape after PCA: {X_pca.shape}\n")

# Visualize the PCA-transformed data
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_cls, cmap='viridis', edgecolor='k', s=70)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA of Iris Dataset")
plt.savefig('pca_reduction.png')
print("PCA plot saved to pca_reduction.png\n")
plt.close()


#==================================
# 3. Model Evaluation & Tuning
#==================================
print("# --- Model Evaluation & Tuning ---")
# Using the classification task from before

# --- Cross-Validation ---
# A technique to evaluate a model by splitting data into several folds and training/testing on different combinations.
# This gives a more robust estimate of model performance.
scores = cross_val_score(cls_model, X_cls, y_cls, cv=5) # 5-fold cross-validation
print(f"Cross-Validation Accuracy Scores: {scores}")
print(f"Average CV Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})\n")


# --- Hyperparameter Tuning (Grid Search) ---
# Most models have parameters (hyperparameters) that can be tuned.
# Grid Search systematically tries all combinations of a given set of hyperparameters.
param_grid = {'max_depth': [2, 3, 4, 5], 'min_samples_leaf': [1, 2, 3]}

# n_jobs=-1 uses all available CPU cores
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters found by Grid Search: {grid_search.best_params_}")
print(f"Best score achieved: {grid_search.best_score_:.2f}\n")

# Evaluate the best model found by Grid Search
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Accuracy of the best model on test set: {accuracy_score(y_test, y_pred_best):.2f}")