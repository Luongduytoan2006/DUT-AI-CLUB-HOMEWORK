from __unknown__ import df
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # chuẩn hoá Standard

encode = {
    name: num for num, name in enumerate(list_class)
}

list_class = []

class MySoftmaxRegression:
    def __init__(self, lr = 0.01, esp = 1e-9, epochs = 1000, numClass = 2):
        self.lr = lr
        self.epochs = epochs
        self.numClass = numClass
        self.W = None
        self.b = None
        self.loss = []

    def softmax(self, z):
        z = z - np.max(z, axis = 1, keepdims = True)
        e = np.exp(z)
        return e / np.sum(e, axis=1, keepdims = True)

    def forward(self, X):
        return self.softmax(X@self.W + self.b)

    def loss_func(self, y, y_hat):
        return np.mean(-np.sum( y*np.log(y_hat) , axis = 1))

    def accuracy(self, y, y_hat):
        y_pred = np.argmax(y_hat, axis=1)
        y_true = np.argmax(y, axis=1)
        return np.mean(y_pred == y_true)

    def train(self, X, y):
        self.W = np.random.rand(X.shape[1], self.numClass)
        self.b = 0

        # chuẩn hoá y thành dãy chấm
        y = np.eye(self.numClass)[y]

        for epoch in range(self.epochs):
            y_hat = self.forward(X)
            loss_epoch = self.loss_func(y, y_hat)
            self.loss.append(loss_epoch)

            error = y_hat - y
            dw = (1 / X.shape[0]) * np.dot(X.T, error)
            db = (1 / X.shape[0]) * np.sum(error, axis=0, keepdims=True)

            self.W -= self.lr * dw
            self.b -= self.lr * db

            temp_acc = self.accuracy(y_hat, y)

            if epoch % 10 == 0:
                print(f"Epoch: {epoch}   | Acc: {temp_acc}  | Loss: {self.loss[-1]}")



def create_data():
    df = pd.read_csv("../../.data/iris/iris_3class_big.csv")
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1]
    
    list_class = np.unique(y)

    encode = {
        name: num for num, name in enumerate(list_class)
    }

    y = y.map(encode)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=42)


    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = create_data()

    myModel = MySoftmaxRegression(epochs = 100000, numClass=len(encode))
    myModel.train(x_train, y_train)

    y_hat = myModel.forward(x_test)
    y_test = np.eye(len(encode))[y_test]

    print(f"Tập test Loss = {myModel.loss_func(y_test, y_hat)} ")
    print(f"Tập test Accuracy: {myModel.accuracy(y_test, y_hat)}")
        
