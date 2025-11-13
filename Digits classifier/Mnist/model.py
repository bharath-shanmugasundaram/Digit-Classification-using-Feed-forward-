import numpy as np
import tensorflow as tf

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

def equalizer(n):
    lis = np.zeros(10, dtype=int)
    lis[n] = 1
    return lis

X = np.array([i.flatten() for i in X_train])
Y = np.array([equalizer(i) for i in y_train])

W = np.random.randn(784, 10) * 0.01
b = np.zeros((1, 10))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m

epochs = 1000
alpha = 0.00005

for epoch in range(epochs):
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    loss = cross_entropy(Y, y_pred)

    dz = y_pred - Y
    dW = (X.T @ dz) / X.shape[0]
    db = np.sum(dz, axis=0, keepdims=True) / X.shape[0]

    W -= alpha * dW
    b -= alpha * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

np.save("weights.npy", W)
np.save("bias.npy", b)
