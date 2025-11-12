
#pip install "numpy<2.0" --force-reinstall

import numpy as np
import matplotlib.pyplot as plt

# AND gate dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # Inputs
y = np.array([0,0,0,1])                   # Output

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0
lr = 0.1   # learning rate

# Activation function
def step(z):
    return np.where(z >= 0, 1, 0)

# Training the perceptron
for _ in range(20):  # epochs
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = step(z)
        w += lr * (y[i] - y_pred) * X[i]
        b += lr * (y[i] - y_pred)

# Print final weights
print("Weights:", w, "Bias:", b)

# Plot decision boundary
x1 = np.linspace(-0.2, 1.2, 100)
x2 = -(w[0]*x1 + b)/w[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
plt.plot(x1, x2, 'g--')
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# OR gate dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])   # Inputs
y = np.array([0,1,1,1]) # Output

# Initialize weights and bias
w = np.zeros(X.shape[1])
b = 0
lr = 0.1   # learning rate

# Activation function
def step(z):
    return np.where(z >= 0, 1, 0)

# Training the perceptron
for _ in range(20):  # epochs
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = step(z)
        w += lr * (y[i] - y_pred) * X[i]
        b += lr * (y[i] - y_pred)

# Print final weights
print("Weights:", w, "Bias:", b)

# Plot decision boundary
x1 = np.linspace(-0.2, 1.2, 100)
x2 = -(w[0]*x1 + b)/w[1]

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')
plt.plot(x1, x2, 'g--')
plt.title("Perceptron Decision Boundary (AND Gate)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

