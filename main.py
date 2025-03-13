import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from Adam import Adam
from SGD import SGD
from SimpleNN import SimpleNN

# Load MNIST dataset and convert to NumPy arrays
mnist = fetch_openml('mnist_784', version=1)
X = mnist['data'].to_numpy().astype(np.float32)
y = mnist['target'].to_numpy().astype(int)

# Normalize data to the range [0, 1]
X /= 255.0

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y.reshape(-1, 1))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Ensure that data is in NumPy array format
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# Training function
def train(model, optimizer, X_train, y_train, X_test, y_test, epochs, batch_size=64):
    history = {'train_loss': [], 'test_loss': [], 'test_accuracy': []}
    num_batches = X_train.shape[0] // batch_size

    for epoch in range(epochs):
        # Shuffle training data
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        epoch_loss = 0

        for i in range(num_batches):
            # Get batch data
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            # Forward pass
            output = model.forward(X_batch)
            loss = np.mean(-np.sum(y_batch * np.log(output + 1e-8), axis=1))  # Cross-entropy loss
            epoch_loss += loss

            # Backward pass
            model.backward(X_batch, y_batch)

            # Get gradients
            grads = model.get_gradients()

            # Update weights
            optimizer.step(grads)

        # Average loss per epoch
        avg_epoch_loss = epoch_loss / num_batches
        history['train_loss'].append(avg_epoch_loss)

        # Evaluate on test set
        test_output = model.forward(X_test)
        test_loss = np.mean(-np.sum(y_test * np.log(test_output + 1e-8), axis=1))
        test_preds = np.argmax(test_output, axis=1)
        test_labels = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_preds == test_labels)

        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        if epoch % 5 == 0:
            print(f'Epoch {epoch}: Train Loss: {avg_epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    return history


# Model parameters
input_size = 784  # 28x28 pixels
hidden_size = 128
output_size = 10
epochs = 30
batch_size = 64

# Training with Adam optimizer
print("Training with Adam optimizer...")
model_adam = SimpleNN(input_size, hidden_size, output_size)
parameters_adam = model_adam.get_parameters()
adam_optimizer = Adam(parameters_adam, lr=0.001)
adam_history = train(model_adam, adam_optimizer, X_train, y_train, X_test, y_test, epochs, batch_size)

# Training with SGD optimizer
print("\nTraining with SGD optimizer...")
model_sgd = SimpleNN(input_size, hidden_size, output_size)
parameters_sgd = model_sgd.get_parameters()
sgd_optimizer = SGD(parameters_sgd, lr=0.1)
sgd_history = train(model_sgd, sgd_optimizer, X_train, y_train, X_test, y_test, epochs, batch_size)

# Plotting the results
import matplotlib.pyplot as plt

# Plot training loss
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(adam_history['train_loss'], label='Adam')
plt.plot(sgd_history['train_loss'], label='SGD')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot test loss
plt.subplot(1, 3, 2)
plt.plot(adam_history['test_loss'], label='Adam')
plt.plot(sgd_history['test_loss'], label='SGD')
plt.title('Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot test accuracy
plt.subplot(1, 3, 3)
plt.plot(adam_history['test_accuracy'], label='Adam')
plt.plot(sgd_history['test_accuracy'], label='SGD')
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
