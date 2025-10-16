## NATURAL LANGUAGE PROCESSING (NLP) ##

# This cheat sheet provides a beginner-friendly introduction to NLP using PyTorch and the Hugging Face library.
# It covers fundamental concepts and provides runnable examples for text classification.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

#==================================
# 1. Core NLP Components with PyTorch
#==================================
print("# --- Core NLP Components with PyTorch ---")

# --- nn.Embedding: Representing Words as Vectors ---
# Machine learning models can't understand text directly. We need to convert words into numbers.
# An Embedding layer is a lookup table that stores a vector for each word in the vocabulary.
# When you pass an index (representing a word) to the layer, it returns the corresponding vector.
# These vectors are learned during training.

# Parameters:
# - num_embeddings: The size of the vocabulary (how many unique words).
# - embedding_dim: The size of the vector for each word.
vocab_size = 1000
embedding_dim = 50
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# Example: Get the vector for the word at index 10
word_index = torch.tensor([10], dtype=torch.long)
word_vector = embedding_layer(word_index)
print(f"Example Embedding Layer: {embedding_layer}")
print(f"Shape of vector for one word: {word_vector.shape}\n") # (1, 50)


# --- Recurrent Neural Networks (RNNs): Processing Sequences ---
# RNNs are designed to handle sequential data like text. They have a "memory" that allows
# them to retain information from previous steps in the sequence.
# - nn.LSTM (Long Short-Term Memory): A popular and powerful type of RNN that can learn
#   long-range dependencies, avoiding the vanishing gradient problem of simple RNNs.

# Parameters:
# - input_size: The size of the input features (e.g., embedding_dim).
# - hidden_size: The size of the hidden state (the "memory").
# - num_layers: Number of stacked LSTM layers.
# - batch_first=True: Makes the input/output tensors have the batch dimension first (batch, seq, feature).
hidden_size = 64
lstm_layer = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=1, batch_first=True)
print(f"Example LSTM Layer: {lstm_layer}\n")


#==================================
# 2. Complete Example: Text Classification with PyTorch
#==================================
print("\n# --- A Complete Text Classification Example ---")

# --- Step 1: Prepare the Data ---
# For this example, we'll create a small, synthetic dataset.
# In a real-world scenario, you would use a library like `torchtext` or Hugging Face `datasets`.
texts = ["this movie is great", "i really enjoyed this film", "what a waste of time", "i did not like this at all"]
labels = [1, 1, 0, 0] # 1 for positive, 0 for negative

# --- Text Preprocessing ---
# 1. Create a vocabulary: a mapping from words to integer indices.
word_to_idx = {}
for sentence in texts:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
vocab_size = len(word_to_idx)
print(f"Vocabulary: {word_to_idx}")
print(f"Vocabulary Size: {vocab_size}\n")

# 2. Tokenize and encode sentences into integer sequences.
sequences = []
for sentence in texts:
    seq = [word_to_idx[word] for word in sentence.split()]
    sequences.append(torch.tensor(seq))

# 3. Pad sequences so they all have the same length.
# Models typically require inputs to be of a uniform size.
# `pad_sequence` pads with 0s by default.
padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
X = padded_sequences
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

print(f"Padded sequences (X):\n{X}")
print(f"Labels (y):\n{y}\n")

# Create a DataLoader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=2)


# --- Step 2: Define the Text Classification Model ---
class SimpleTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SimpleTextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text shape: (batch_size, seq_length)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_length, embed_dim)

        # We only need the final hidden state of the LSTM
        _, (hidden, _) = self.lstm(embedded)
        # hidden shape: (num_layers, batch_size, hidden_dim)

        # Squeeze to remove the num_layers dimension
        hidden = hidden.squeeze(0)
        # hidden shape: (batch_size, hidden_dim)

        output = self.fc(hidden)
        output = self.sigmoid(output)
        return output

# Instantiate the model
embedding_dim = 32
hidden_dim = 16
output_dim = 1 # One output neuron for binary classification
model = SimpleTextClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
print(f"Model Architecture:\n{model}\n")


# --- Step 3: Define Loss and Optimizer ---
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# --- Step 4: The Training Loop ---
print("Starting training...")
epochs = 50
for epoch in range(epochs):
    model.train()
    for texts_batch, labels_batch in loader:
        optimizer.zero_grad()
        predictions = model(texts_batch)
        loss = criterion(predictions, labels_batch)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


# --- Step 5: Evaluate / Run Inference ---
model.eval()
with torch.no_grad():
    # Test with a known positive sentence
    test_text = "this film is great"
    test_seq = torch.tensor([word_to_idx[w] for w in test_text.split()])
    prediction = model(test_seq.unsqueeze(0)) # Add batch dimension
    print(f"\nPrediction for '{test_text}': {prediction.item():.3f} (Closer to 1 is positive)")

    # Test with a known negative sentence
    test_text = "i did not like this"
    test_seq = torch.tensor([word_to_idx[w] for w in test_text.split()])
    prediction = model(test_seq.unsqueeze(0))
    print(f"Prediction for '{test_text}': {prediction.item():.3f} (Closer to 0 is negative)")


#==================================
# 3. Introduction to Hugging Face `transformers`
#==================================
print("\n# --- Introduction to Hugging Face `transformers` ---")
# Hugging Face provides easy access to thousands of pre-trained models (like BERT, GPT)
# for a wide variety of NLP tasks. This is the standard approach for most modern NLP applications.

from transformers import pipeline

# --- Using a `pipeline` for a Zero-Shot Task ---
# The `pipeline` is the easiest way to use a pre-trained model for inference.
# "Zero-shot" means the model can classify text into labels it has never seen during its training.
print("\n# -- Zero-Shot Classification Pipeline --")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sequence_to_classify = "The new AI regulations will have a big impact."
candidate_labels = ['politics', 'business', 'technology', 'sports']
result = classifier(sequence_to_classify, candidate_labels)
print(f"Text: '{sequence_to_classify}'")
print(f"Classification Result: {result['labels'][0]} (Score: {result['scores'][0]:.2f})")

# --- Using a `pipeline` for Text Generation ---
print("\n# -- Text Generation Pipeline --")
generator = pipeline('text-generation', model='gpt2')
generated_text = generator("In a world where AI is king,", max_length=25, num_return_sequences=1)
print("Generated text:")
print(generated_text[0]['generated_text'])