from b1 import show_plot
import numpy as np


def f(x):
    return 3* (x**4) - 4* (x**2) - 6*x - 3

def df(x):
    return 12* (x**3) - 8*x - 6

def gradient_descent(eta, x_cur, max_epoch):
    epoch = 0
    while epoch < max_epoch:
        x_prev = x_cur
        grad = df(x_prev)
        x_cur = x_prev - eta * grad
        epoch += 1
        
        # Kiểm tra hội tụ
        if abs(x_cur - x_prev) < 1e-6:
            break
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, eta: {eta}, x: {x_cur:.6f}, f(x): {f(x_cur):.6f}")

    return [epoch, eta, x_cur, f(x_cur)]

if __name__ == "__main__":
    x = 2
    eta = 0.01  # Learning rate nhỏ hơn để ổn định
    max_epoch = 1000
    ans = gradient_descent(eta, x, max_epoch)
    print("="*40)
    print("Result after gradient descent:")
    print(f"Epoch: {ans[0]}")
    print(f"eta: {ans[1]}")
    print(f"x: {ans[2]:.6f}")
    print(f"f(x): {ans[3]:.6f}")

    # Vẽ đồ thị hàm số và đạo hàm
    x = np.arange(ans[2]-0.5, ans[2]+0.5, 0.01)
    y_func = f(x)
    y_derivative = df(x)

    show_plot(x, y_func, y_derivative, " y = 3x^4 - 4x^2 - 6x - 3 and its Derivative")


