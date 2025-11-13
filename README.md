# 🧠 MNIST Digit Classifier — Softmax Regression From Scratch (NumPy Only)

This project implements a **Multiclass Softmax Regression (Multinomial Logistic Regression)** model **from scratch using only NumPy** — no Keras, no PyTorch.  
The MNIST dataset is loaded through TensorFlow **only for the data**, and the entire training pipeline (forward pass, loss, gradients, update rules) is written manually.

---

## 📌 Project Overview

- **Task**: Classify handwritten digits (0–9) from the MNIST dataset  
- **Model**: Softmax Regression (Single-layer Neural Network)  
- **Frameworks Used**:  
  - NumPy (main implementation)  
  - TensorFlow (only for loading dataset)  
- **Training**: Gradient Descent  
- **Loss**: Cross-Entropy  
- **Activation**: Softmax  

This is a perfect beginner-friendly demonstration of training a neural network using **pure math + NumPy**.

---

## 📂 Dataset

The MNIST dataset contains:

- **60,000 training images**
- **10,000 test images**
- Each image is **28 × 28 (784 pixels)**

The dataset is loaded using:

```python
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
