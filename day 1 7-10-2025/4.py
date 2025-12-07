import random
import math

def calculate_loss(loss_name, predict, target):
    if loss_name == 'MAE':
        return abs(target - predict)
    elif loss_name == 'MSE':
        return (target - predict) ** 2
    elif loss_name == 'RMSE':
        return math.sqrt((target - predict) ** 2)
    return 0

def run_regression_simulation():
    num = input("Enter the number of samples: ")

    if not num.isnumeric():
        print("Error: Number of samples must be an integer number")
        return

    num = int(num)
    loss_name = input("Enter the loss function (MAE, MSE, RMSE): ")

    if loss_name not in ['MAE', 'MSE', 'RMSE']:
        print(f"Error: Loss function '{loss_name}' is not valid.")
        return

    for i in range(num):
        predict_value = random.uniform(0, 10)
        target_value = random.uniform(0, 10)

        loss = calculate_loss(loss_name, predict_value, target_value)

        print(f"Sample-{i} >> loss_name: {loss_name}, predict: {predict_value:.2f}, target: {target_value:.2f}, loss: {loss:.2f}")


if __name__ == "__main__":
    run_regression_simulation()