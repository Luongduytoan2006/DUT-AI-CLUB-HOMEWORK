import math

def softmax(data: list) -> list:
    max_value = max(data)
    data = [x-max_value for x in data]
    
    exp_data = [math.exp(x) for x in data]
    sum_exp_data = sum(exp_data)
    ans = [x/sum_exp_data for x in exp_data]
    
    return ans

data1 = [2.0, 1.0, 0.1]
print(softmax(data1))  

data2 = [1000, 2000, 1500, 800]
print(softmax(data2))