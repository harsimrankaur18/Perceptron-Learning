# Python version--3.6.0
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Plot Mistakes
def plotMistakes(mistakes_arr):
    plt.plot(mistakes_arr)
    plt.xlabel("Number of Passes")
    plt.ylabel("Number of Mistakes")
    plt.title("Number of Mistakes in Passes by Perceptron Algorithm")
    plt.show()


# Perceptron Algorithm
def perceptron(X, y, b, w, max_passes, mistakes_arr):
    for i in range(1, max_passes + 1):
        mistakes = 0
        for j in range(len(X)):
            if y[j] * (np.dot(X[j], w) + b) <= 0:
                # Mistake
                w = w + (y[j] * X[j])
                b = b + y[j]
                mistakes = mistakes + 1
        mistakes_arr[i - 1] = mistakes
    # print("Mistakes", mistakes_arr)
    return mistakes_arr


def main():
    # input
    X = pd.read_csv("spambase_X.csv", header=None).values.T
    # labels
    y = pd.read_csv("spambase_y.csv", header=None)
    y = np.ravel(y)

    # Weight & Bias
    b = 0
    w = np.zeros((len(X[0]),), dtype=np.int)
    max_passes = 500
    mistakes_arr = np.zeros((max_passes,), dtype=np.int)

    mistakes_arr = perceptron(X, y, b, w, max_passes, mistakes_arr)

    plotMistakes(mistakes_arr)


if __name__ == '__main__':
    main()
