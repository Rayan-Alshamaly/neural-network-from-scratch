import math
from neural import *
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1+math.exp(-x))

def derivative_sigmoid(sigmoid_y):
    return sigmoid_y*(1-sigmoid_y)

digits = load_digits()
X = digits.data / 16.0  # Normalisation [0, 1]
y = digits.target

Y = np.zeros((len(y), 10))
for i, target in enumerate(y):
    Y[i, target] = 1.0
    
X_train, X_test, y_train, y_test, Y_train, Y_test = train_test_split(X, y, Y, test_size=0.25, random_state=42)

neural_network = Network(64,[48,10],0.2,sigmoid,derivative_sigmoid)

n_train = 50

for k in range(n_train):
    sum_error = 0.0
    for inputs, expected in zip(X_train, Y_train):
        outputs = neural_network.train(inputs, expected)
        sum_error += np.sum((np.array(expected) - np.array(outputs)) ** 2)
    
    if (k + 1) % 10 == 0 or k == 0:
        print(f"Époque {k+1}/{n_train} - Erreur quadratique totale : {sum_error:.4f}")

correct_guess = 0
for inputs, target in zip(X_test, y_test):
    prediction = neural_network.evaluate(inputs)
    if prediction == target:
        correct_guess += 1

accuracy = (correct_guess / len(y_test)) * 100
print(f"\nPrécision sur le jeu de test : {accuracy:.2f}%")
