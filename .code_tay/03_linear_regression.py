import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


class MyLinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss = []

    def loss_function(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def train(self, x, y):
        self.w = np.random.rand(x.shape[1], 1)
        self.b = 0

        for epoch in range( self.epochs):
            y_hat = np.dot(x, self.w) + self.b
            self.loss.append(self.loss_function(y_hat, y))
            dw = (1 / x.shape[0]) * np.dot(x.T, (y_hat - y))
            db = (1 / x.shape[0]) * np.sum(y_hat - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {self.loss[-1]}")


def create_data():
    # Tạo data mẫu
    x , y = make_regression(n_samples=200, n_features=3, noise=15, random_state=42)

    # Reshape vì make_regression trả về y có shape (200,), nhưng chúng ta cần nó có shape (200, 1) để phù hợp với các thuật toán học máy.
    y = y.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

        
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_data()

    model = MyLinearRegression()
    model.train(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = model.loss_function(y_pred, y_test)
    print(f"Mean Squared Error: {mse}")
    