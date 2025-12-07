import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-10, 10, 0.1)

def show_plot(x, y_func, y_derivative, title="Activation Function"):
    # setting the axes at the centre
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # plot the function
    ax.plot(x, y_func, color="#d35400", linewidth=3, label="Function")
    ax.plot(x, y_derivative, color="#1abd15", linewidth=3, label="Derivative")
    
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False)
    plt.show()

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# PReLU (Parametric ReLU)
def prelu(x, alpha=0.25):
    return np.where(x > 0, x, alpha * x)
    
def prelu_derivative(x, alpha=0.25):
    return np.where(x > 0, 1, alpha)


# ELU (Exponential Linear Unit)
def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))


# Softplus
def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return 1 / (1 + np.exp(-x))


# Softsign
def softsign(x):
    return x / (1 + np.abs(x))

def softsign_derivative(x):
    return 1 / (1 + np.abs(x))**2


if __name__ == "__main__":
    y_sigmoid = sigmoid(x)
    y_sigmoid_derivative = sigmoid_derivative(x)
    show_plot(x, y_sigmoid, y_sigmoid_derivative, "Sigmoid")

    y_prelu = prelu(x)
    y_prelu_derivative = prelu_derivative(x)
    show_plot(x, y_prelu, y_prelu_derivative, "PReLU")

    y_elu = elu(x)
    y_elu_derivative = elu_derivative(x)
    show_plot(x, y_elu, y_elu_derivative, "ELU")

    y_softplus = softplus(x)
    y_softplus_derivative = softplus_derivative(x)
    show_plot(x, y_softplus, y_softplus_derivative, "Softplus")

    y_softsign = softsign(x)
    y_softsign_derivative = softsign_derivative(x)
    show_plot(x, y_softsign, y_softsign_derivative, "Softsign")
