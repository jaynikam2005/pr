
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import matplotlib.pyplot as plt

# --- Configuration Constants ---
# Use the top 10,000 most frequently occurring words
MAX_WORDS = 10000
# Cut reviews after this many words
MAX_LEN = 200
# Number of samples per gradient update
BATCH_SIZE = 32
# Number of epochs to train for
EPOCHS = 10
# Dimension of the dense embedding
EMBEDDING_DIM = 128
# Number of units in the LSTM layer
LSTM_UNITS = 128

# 1. Load the IMDB dataset
print(f"Loading data with max_words={MAX_WORDS}...")
# The dataset is pre-tokenized, where words are represented by integers
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=MAX_WORDS)

# 2. Preprocess Data: Pad sequences
print(f"Padding sequences to max_len={MAX_LEN}...")
# Pad sequences with zeros to ensure uniform length. 'post' means padding at the end.
x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"Training sequences shape: {x_train.shape}")
print(f"Test sequences shape: {x_test.shape}")

# 3. Define the LSTM Model for Binary Classification
print("Building LSTM model...")
model = Sequential([
    # Embedding layer: Turns integer indices into dense vectors of fixed size
    Embedding(input_dim=MAX_WORDS,
              output_dim=EMBEDDING_DIM,
              input_length=MAX_LEN),

    # LSTM layer: The core recurrent layer for capturing sequential dependencies
    LSTM(units=LSTM_UNITS, dropout=0.2, recurrent_dropout=0.2),

    # Dense output layer: Single neuron with sigmoid for binary classification (0 or 1)
    Dense(1, activation='sigmoid')
])

# Compile the model
# Loss: Binary Crossentropy for binary classification
# Optimizer: 'adam' is a good general-purpose choice
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Train the Model
print("Training the model...")
history = model.fit(x_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.1,  # Use 10% of training data for validation
                    verbose=1)

# Evaluate the model on the test data
loss, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# 5. Plot Accuracy and Loss
def plot_history(history):
    """Plots training and validation accuracy and loss."""

    # Plot Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()


    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.tight_layout()
    plt.show()

plot_history(history)

# 6. Visualize Example Predictions

# Get the word index to decode the integer sequences back to text
word_index = imdb.get_word_index()
# Map index 3 to the word "UNK" (unknown) and reserve 0, 1, 2 for padding, start, and unknown
reverse_word_index = {value + 3: key for (key, value) in word_index.items()}
reverse_word_index[0] = "<PAD>"
reverse_word_index[1] = "<START>"
reverse_word_index[2] = "<UNK>"

def decode_review(text_indices):
    """Converts a sequence of indices back to a readable string."""
    # Use list comprehension to map indices to words, handling the offset
    return ' '.join([reverse_word_index.get(i, '?') for i in text_indices])

# Select a small sample for prediction
sample_indices = [5, 10, 15] # Indices in the test set
x_sample = x_test[sample_indices]
y_true_sample = y_test[sample_indices]

# Make predictions (returns probabilities)
predictions = model.predict(x_sample).flatten()
# Convert probabilities to binary class (0 or 1)
y_pred_class = (predictions > 0.5).astype(int)

print("\n--- Example Predictions ---")
for i, idx in enumerate(sample_indices):
    review_text = decode_review(x_test[idx][:50]) # Show first 50 words
    true_sentiment = 'Positive' if y_true_sample[i] == 1 else 'Negative'
    predicted_sentiment = 'Positive' if y_pred_class[i] == 1 else 'Negative'
    confidence = predictions[i]

    print(f"\nReview #{idx} (True: {true_sentiment}):")
    print(f"  Snippet: \"{review_text.replace('<PAD>', '').strip()}...\"")
    print(f"  Prediction: {predicted_sentiment} (Confidence: {confidence:.4f})")

