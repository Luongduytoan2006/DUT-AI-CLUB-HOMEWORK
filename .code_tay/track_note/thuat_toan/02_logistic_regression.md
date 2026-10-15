# 🧠 Logistic Regression (file `04_logistic_regression.py`)

> Phân loại **2 lớp** (spam / không, setosa / versicolor...).

---

## Khác Linear ở đâu
- Thêm **sigmoid** bọc ngoài: `y_hat = sigmoid(x@w + b)` → xác suất (0,1).
- Loss đổi MSE → **BCE** (Binary Cross-Entropy).
- Cần `np.clip` tránh `log(0)`.

## Các bước
1. **Data**: đọc CSV iris → tách X/y → **map nhãn chữ → 0/1** → `reshape(-1,1)` → split.
2. **Khởi tạo**: `w = np.random.rand(n_features, 1)`, `b = 0`.
3. **Train (epochs)**:
   - `y_hat = sigmoid(x @ w + b)`
   - loss = BCE = `-mean(y*log(ŷ) + (1-y)*log(1-ŷ))` (clip trước)
   - gradient **giống linear**: `dw = (1/n) x.T @ (y_hat - y)`, `db = (1/n) sum(...)`
   - cập nhật w, b
4. **Dự đoán → phân lớp**: `y_pred = (y_hat >= 0.5)` (ngưỡng 0.5).
5. **Đánh giá**: accuracy = `np.mean(y_pred == y_test)`.

## Shape chính
- `w (feat, 1)`, `y (n, 1)`, `y_hat (n, 1)`.

## ⚠️ Điểm cần sửa/ôn (từ bài này)
- Class tên `MyLogisticRegression` nhưng biến ở main là `mylinear` → đặt tên nhất quán.
- Đã `import StandardScaler` nhưng **chưa dùng** → nên chuẩn hóa X.
- `import LinearRegression` thừa.
