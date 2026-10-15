import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler  # chuẩn hoá Standard

class MyLogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 360, eps = 1e-9):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.eps = eps
        self.loss = []

    def loss_function(self, y_hat, y):
        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        return -np.mean(y * np.log(y_hat) + (1-y) * np.log(1- y_hat))

    def sigmoid(self, z):
        return 1 / (1+ np.exp(-z))

    def train(self, x, y):
        self.w = np.random.rand(x.shape[1], 1)
        self.b = 0

        for epoch in range (self.epochs):
            y_hat = self.sigmoid( x @ self.w + self.b )
            self.loss.append(self.loss_function(y_hat, y))

            dw = (1 / x.shape[0]) * np.dot(x.T, (y_hat - y))
            db = (1 / x.shape[0]) * np.sum(y_hat - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 10 == 0:
                print(f"Epoch {epoch}  |   Loss: {self.loss[-1]}")


    def predict(self, x):
        return self.sigmoid( x @ self.w + self.b )


def create_data():
    df = pd.read_csv("../.data/iris/iris_2class_big.csv")
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1]
    y = y.map({"Iris-setosa": 0, "Iris-versicolor": 1})

    print(x.shape)
    print(y.shape)

    print("\nCác nhãn hiện tại của y")
    print(np.unique(y))

    y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_data()

    mylinear = MyLogisticRegression()
    mylinear.train(x_train, y_train)
    y_pred = mylinear.predict(x_test)
    testLoss = mylinear.loss_function(y_pred, y_test)
    y_pred = (y_pred >= 0.5)

    acc = np.mean(y_pred == y_test)
    print(f"Độ chính xác: {acc * 100:.2f}%")
    print(f"BCE trên tập test là: {testLoss}")