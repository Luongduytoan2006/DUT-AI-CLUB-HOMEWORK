import math

def binary_crossentropy(y, y_hat):
    return -y*math.log(y_hat) - (1-y)*math.log(1-y_hat)

prediction = [0.3, 0.8, 0.7, 0.4, 0.6, 0.8, 0.9, 0.2] 
label = [0, 0, 0, 0, 1, 1, 1, 1]

loss = [binary_crossentropy(y, y_hat) for y, y_hat in zip(label, prediction)]
print(loss)


print(40*"=")


prediction1 = [0.7, 0.8, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.8, 0.7]
label1 = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]

loss1 = [binary_crossentropy(y, y_hat) for y, y_hat in zip(label1, prediction1)]
print(loss1)