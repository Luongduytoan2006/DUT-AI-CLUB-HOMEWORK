# 🧠 Linear Regression (file `03_linear_regression.py`)

> Dự đoán **số liên tục** (giá nhà, nhiệt độ...).

---

## Các bước

1. **Chuẩn bị data**: `make_regression()` → `y.reshape(-1,1)` → `train_test_split`.
2. **Khởi tạo tham số**: `w = np.random.rand(n_features, 1)`, `b = 0`.
3. **Vòng lặp train (epochs)**:
   - Dự đoán: `y_hat = x @ w + b`
   - Loss: **MSE** = `np.mean((y_hat - y)**2)`
   - Gradient: `dw = (1/n) * x.T @ (y_hat - y)`, `db = (1/n) * np.sum(y_hat - y)`
   - Cập nhật: `w -= lr*dw`, `b -= lr*db`
4. **Dự đoán & đánh giá**: `predict()` rồi tính MSE trên test.

## Bản chất
Tìm đường thẳng khớp data bằng cách hạ dần loss theo gradient.

## Shape chính
- `x (n, feat)`, `w (feat, 1)`, `y (n, 1)`, `y_hat (n, 1)`.
