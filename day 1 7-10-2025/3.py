import math

def binary_step_function(x):
    if x < 0:
        return 0
    else:
        return 1
    
def sigmoid_function(x):
    return 1 / (1 + math.exp(-x))

def elu_function(x):
    alpha = 0.01
    if x < 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x

try:
    x_input = input("Enter a number: ")
    x = float(x_input)
except ValueError:
    print("Error: Input must be a number.")
    exit()

while True:
    print(20*"=")
    print("Choose an activation function:")
    print("1. Binary Step Function")
    print("2. Sigmoid Function")
    print("3. ELU Function")
    print("0. Exit")
    choice = input("Enter your choice (0/1/2/3): ")
    if choice == "0":
        break
    elif choice in ["1", "2", "3"]:
        if choice == "1":
            result = binary_step_function(x)
            print(f"Binary Step Function result: {result}")
        elif choice == "2":
            result = sigmoid_function(x)
            print(f"Sigmoid Function result: {result}")
        elif choice == "3":
            result = elu_function(x)
            print(f"ELU Function result: {result}")
    else:
        print("Invalid choice. Please try again.")