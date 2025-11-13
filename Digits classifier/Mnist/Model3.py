# %%
import numpy as np
import tensorflow as tf

# %%
# Load MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# %%
# Normalize and flatten
X = X_train.reshape(-1, 28*28) / 255.0
XT = X_test.reshape(-1, 28*28) / 255.0

# One-hot encoding
def hot_value(n):
    return np.array([1 if i==n else 0 for i in range(10)])

Y = np.array([hot_value(i) for i in y_train])
YT = np.array([hot_value(i) for i in y_test])

# %%
# Activations
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Loss
def cross_ent(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

# Accuracy
def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))

# %%
# Hyperparameters
np.random.seed(42)
input_size = 784
hidden1 = 128
hidden2 = 64
output_size = 10

lr = 0.01
epochs = 20
batch_size = 128
momentum = 0.9

# %%
# Initialize weights with He init
W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
b1 = np.zeros((1, hidden1))
W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
b2 = np.zeros((1, hidden2))
W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
b3 = np.zeros((1, output_size))

# Initialize momentum terms
vW1 = np.zeros_like(W1); vb1 = np.zeros_like(b1)
vW2 = np.zeros_like(W2); vb2 = np.zeros_like(b2)
vW3 = np.zeros_like(W3); vb3 = np.zeros_like(b3)

# %%
# Forward pass
def forward(X, W1, b1, W2, b2, W3, b3):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = relu(Z2)
    Z3 = A2 @ W3 + b3
    A3 = softmax(Z3)
    cache = (Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

# %%
# Backward pass
def backward(X, y, cache, W1, W2, W3):
    Z1, A1, Z2, A2, Z3, A3 = cache
    m = X.shape[0]

    dZ3 = A3 - y
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    dA2 = dZ3 @ W3.T
    dZ2 = dA2 * relu_derivative(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2, dW3, db3

# %%
# Training loop with mini-batches and momentum
num_batches = X.shape[0] // batch_size

for epoch in range(epochs):
    # Shuffle data
    perm = np.random.permutation(X.shape[0])
    X_shuff = X[perm]; Y_shuff = Y[perm]

    for i in range(num_batches):
        start = i*batch_size
        end = start + batch_size
        X_batch = X_shuff[start:end]
        Y_batch = Y_shuff[start:end]

        y_pred, cache = forward(X_batch, W1, b1, W2, b2, W3, b3)
        loss = cross_ent(Y_batch, y_pred)
        dW1, db1, dW2, db2, dW3, db3 = backward(X_batch, Y_batch, cache, W1, W2, W3)

        # Update weights with momentum
        vW1 = momentum * vW1 - lr * dW1; W1 += vW1
        vb1 = momentum * vb1 - lr * db1; b1 += vb1
        vW2 = momentum * vW2 - lr * dW2; W2 += vW2
        vb2 = momentum * vb2 - lr * db2; b2 += vb2
        vW3 = momentum * vW3 - lr * dW3; W3 += vW3
        vb3 = momentum * vb3 - lr * db3; b3 += vb3

    # Epoch metrics
    y_train_pred, _ = forward(X, W1, b1, W2, b2, W3, b3)
    train_acc = accuracy(Y, y_train_pred)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}")

# %%
# Test accuracy
y_test_pred, _ = forward(XT, W1, b1, W2, b2, W3, b3)
test_acc = accuracy(YT, y_test_pred)
print("Test Accuracy:", test_acc)

