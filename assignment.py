# ============================================================================
# 1 What is deep learning, and how is it connected to artificial intelligence?
# ============================================================================
# Deep learning is a subfield of machine learning that uses neural networks with many layers
# to learn hierarchical representations. It's connected to AI because it's one of the most
# successful techniques for achieving human-like performance in vision, NLP, robotics, etc.

# ============================================================================
# 2 What is a neural network, and what are the different types of neural networks?
# ============================================================================
# A neural network is a computational model inspired by the brain, consisting of layers of
# interconnected neurons. Types: Feedforward (FNN), Convolutional (CNN), Recurrent (RNN),
# LSTM, GAN, Autoencoders, Transformers.

# ============================================================================
# 3 What is the mathematical structure of a neural network?
# ============================================================================
# Neuron: output = activation( Σ(weight_i * input_i) + bias )
# Layer: a^(l) = activation( W^(l) * a^(l-1) + b^(l) )
# Whole network is composition of such functions.

# ============================================================================
# 4 What is an activation function, and why is it essential in neural networks?
# ============================================================================
# It introduces non-linearity. Without it, the entire network would be linear,
# unable to learn complex patterns.

# ============================================================================
# 5 Could you list some common activation functions used in neural networks?
# ============================================================================
# Sigmoid, Tanh, ReLU, Leaky ReLU, PReLU, ELU, Softmax, Linear.

# ============================================================================
# 6 What is a multilayer neural network?
# ============================================================================
# A network with input layer, one or more hidden layers, and output layer.
# Hidden layers enable hierarchical feature learning.

# ============================================================================
# 7 What is a loss function, and why is it crucial for neural network training?
# ============================================================================
# It measures the difference between prediction and true target, giving a scalar error
# signal that guides optimization.

# ============================================================================
# 8 What are some common types of loss functions?
# ============================================================================
# MSE, MAE (regression); Binary/Categorical Cross-Entropy (classification);
# Hinge Loss, KL Divergence.

# ============================================================================
# 9 How does a neural network learn?
# ============================================================================
# By iteratively adjusting weights and biases to minimize the loss using gradient descent
# and backpropagation.

# ============================================================================
# 10 What is an optimizer in neural networks, and why is it necessary?
# ============================================================================
# An optimizer defines the parameter update rule based on gradients. It's necessary to
# efficiently minimize the loss and converge.

# ============================================================================
# 11 Could you briefly describe some common optimizers?
# ============================================================================
# SGD, Momentum, Adagrad, RMSprop, Adam.

# ============================================================================
# 12 Can you explain forward and backward propagation in a neural network?
# ============================================================================
# Forward: input → layers → prediction. Backward: compute gradients of loss w.r.t.
# parameters using chain rule from output to input.

# ============================================================================
# 13 What is weight initialization, and how does it impact training?
# ============================================================================
# Initial values of weights before training. Bad init → vanishing/exploding gradients
# or slow convergence. Good init (Xavier, He) stabilizes training.

# ============================================================================
# 14 What is the vanishing gradient problem in deep learning?
# ============================================================================
# Gradients become extremely small as they are backpropagated through many layers,
# preventing early layers from learning. Common with sigmoid/tanh.

# ============================================================================
# 15 What is the exploding gradient problem?
# ============================================================================
# Gradients grow exponentially large, causing unstable updates and divergence.
# Common in RNNs or poorly initialized deep nets.

# ============================================================================
# 16 How do you create a simple perceptron for basic binary classification?
# ============================================================================
import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = 1 if linear >= 0 else 0
                update = self.lr * (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return np.where(linear >= 0, 1, 0)

# Example usage (uncomment to test)
# X = np.array([[0,0], [0,1], [1,0], [1,1]])
# y = np.array([0,0,0,1])  # AND gate
# p = Perceptron(lr=0.1, epochs=10)
# p.fit(X, y)
# print(p.predict(X))

# ============================================================================
# 17 How can you build a neural network with one hidden layer using Keras?
# ============================================================================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

input_dim = 10  # example
model = Sequential([
    Dense(64, activation='relu', input_shape=(input_dim,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# To actually train: model.fit(X_train, y_train, epochs=10)

# ============================================================================
# 18 How do you initialize weights using the Xavier (Glorot) initialization method in Keras?
# ============================================================================
from tensorflow.keras.initializers import GlorotUniform

layer = Dense(64, activation='relu', kernel_initializer=GlorotUniform())
# Or after model creation:
for layer in model.layers:
    if hasattr(layer, 'kernel_initializer'):
        layer.kernel_initializer = GlorotUniform()

# ============================================================================
# 19 How can you apply different activation functions in a neural network in Keras?
# ============================================================================
from tensorflow.keras.layers import Activation

# Method 1: inside Dense
model.add(Dense(64, activation='tanh'))
# Method 2: separate Activation layer
model.add(Dense(64))
model.add(Activation('relu'))
# Method 3: for output layer
model.add(Dense(10, activation='softmax'))

# ============================================================================
# 20 How do you add dropout to a neural network model to prevent overfitting?
# ============================================================================
from tensorflow.keras.layers import Dropout

model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.5),   # drops 50% of neurons randomly
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# ============================================================================
# 21 How do you manually implement forward propagation in a simple neural network?
# ============================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_prop(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return A2, (Z1, A1, Z2)

# ============================================================================
# 22 How do you add batch normalization to a neural network model in Keras?
# ============================================================================
from tensorflow.keras.layers import BatchNormalization

model = Sequential([
    Dense(64, input_shape=(input_dim,)),
    BatchNormalization(),
    Activation('relu'),
    Dense(32),
    BatchNormalization(),
    Activation('relu'),
    Dense(1, activation='sigmoid')
])

# ============================================================================
# 23 How can you visualize the training process with accuracy and loss curves?
# ============================================================================
import matplotlib.pyplot as plt

# Assuming 'history' from model.fit(...)
# history = model.fit(X_train, y_train, epochs=50, validation_split=0.2)

def plot_training_history(history):
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.title('Loss curves')
    
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy curves')
    plt.show()

# ============================================================================
# 24 How can you use gradient clipping in Keras to control the gradient size and prevent exploding gradients?
# ============================================================================
from tensorflow.keras.optimizers import Adam

optimizer_clipvalue = Adam(clipvalue=1.0)   # clips each gradient element to [-1,1]
optimizer_clipnorm = Adam(clipnorm=1.0)     # clips global L2 norm to 1.0
model.compile(optimizer=optimizer_clipnorm, loss='mse')

# ============================================================================
# 25 How can you create a custom loss function in Keras?
# ============================================================================
import tensorflow as tf

def custom_huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small = tf.abs(error) <= delta
    small_loss = 0.5 * tf.square(error)
    large_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small, small_loss, large_loss)

model.compile(optimizer='adam', loss=custom_huber_loss)

# ============================================================================
# 26 How can you visualize the structure of a neural network model in Keras?
# ============================================================================
from tensorflow.keras.utils import plot_model

# Text summary
model.summary()

# Save as image
plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

print("All answers are embedded as comments and code in this file.")