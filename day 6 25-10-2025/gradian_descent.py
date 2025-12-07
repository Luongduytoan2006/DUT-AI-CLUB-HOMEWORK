# func: x^2 -2x + 1 , tìm x dể f(x) min

def f(x):
    return x**2 - 2*x + 1

def df(x):
    return 2*x - 2

def gradian_descent(eta, x_cur, max_epoch):
    x_prev = 0.1
    epoch = 0
    while abs(x_cur - x_prev) > 1e-6 and epoch < max_epoch:
        x_prev = x_cur
        x_cur = x_prev - eta * df(x_prev)
        if df(x_cur) * df(x_prev) <= 0:
            eta = eta / 2
        epoch += 1

        if(epoch % 10 == 0):
            print(f"Epoch: {epoch}, eta: {eta}, x: {x_cur}, f(x): {f(x_cur)}")

    return [epoch, eta, x_cur, f(x_cur)]

if __name__ == "__main__":
    x = 12345678901234567890
    eta = 143
    max_epoch = 1000
    ans = gradian_descent(eta, x ,max_epoch)
    print("="*40)
    print("Result after gradient descent:")
    print(f"Epoch: {ans[0]}")
    print(f"eta: {ans[1]}")
    print(f"x: {ans[2]}")
    print(f"f(x): {ans[3]}")
