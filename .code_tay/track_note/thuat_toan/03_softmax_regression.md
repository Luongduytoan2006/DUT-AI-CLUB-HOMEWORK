# 🧠 Softmax Regression (file `05_softmax_regression.ipynb`)

> Phân loại **nhiều lớp** (≥3). Iris 3 lớp. Data: `.data/iris/iris_3class_big.csv` (1500 dòng).

---

## Khác Logistic ở đâu
- **W có C cột** (C = số lớp) thay vì 1 → z ra **C số** thay vì 1.
- **sigmoid → softmax** (C số → C xác suất cộng=1).
- **BCE → Cross-Entropy**; nhãn y phải **one-hot** `(n,C)`.
- Phân lớp: **argmax** thay vì ngưỡng 0.5.
- Gradient **vẫn giống** logistic: `X.T @ (y_hat - y_onehot)`.

---

## Các bước (trình tự chuẩn)

1. **Đọc data**: `df = pd.read_csv(...)`, `x = df.iloc[:,:-1].values`, `y = df.iloc[:,-1]`.
2. **Encode nhãn**: `list_class = np.unique(y)` → `encode = {name:num for num,name in enumerate(list_class)}` → `y = y.map(encode)`.
   - ⚠️ Softmax **KHÔNG** `reshape(-1,1)` (giữ y 1D cho one-hot).
3. **Split**: `train_test_split(x, y, test_size=0.2, random_state=42)`.
4. **(Nên có) Scale**: `StandardScaler` fit train, transform test.
5. **Khởi tạo**: `W = np.random.rand(n_features, numClass)`, `b = np.zeros((1, numClass))`.
6. **One-hot y**: `y = np.eye(numClass)[y]` → `(n, C)`.  ⚠️ y phải int, 1D.
7. **Train (epochs)**:
   - `z = X @ W + b`
   - `y_hat = softmax(z)` — nhớ **trừ max, axis=1, keepdims**
   - `loss = np.mean(-np.sum(y*np.log(y_hat), axis=1))` (clip trước log)
   - `error = y_hat - y`
   - `dw = (1/n) * X.T @ error`  → `(feat, C)`
   - `db = (1/n) * np.sum(error, axis=0, keepdims=True)` → `(1, C)`
   - `W -= lr*dw`, `b -= lr*db`
8. **Predict**: `y_hat = forward(X)` → `y_pred = np.argmax(y_hat, axis=1)`.
9. **Accuracy**: đưa cả 2 về vị trí rồi so:
   - `y_pred = np.argmax(y_hat, axis=1)`, `y_true = np.argmax(y_onehot, axis=1)`
   - `np.mean(y_pred == y_true)`.

## Bản chất
Mỗi lớp 1 cột trọng số → chấm điểm mỗi lớp → softmax thành xác suất → chọn lớp cao nhất.
Train: kéo xác suất lớp đúng lên cao bằng cross-entropy + gradient.

---

## 🐞 CÁC BUG ĐÃ GẶP Ở BÀI NÀY (nhớ đời!)

1. **softmax để `axis=0`** → SAI. Phải `axis=1` (chuẩn hóa mỗi mẫu).
   - Test: `forward(X).sum(axis=1)` phải toàn `1.0`.
2. **`y.reshape(-1,1)` rồi `np.eye(C)[y]`** → ra `(n,1,C)` thừa chiều → broadcast nổ.
   - Fix: giữ y **1D** trước one-hot.
3. **accuracy dùng `np.max`** → SAI. Phải `np.argmax` (cần vị trí, không phải giá trị). Và `axis=1`.
4. **`db = np.sum(error)`** ra 1 số → nên `axis=0, keepdims=True` cho ra `(1,C)`.
5. `numClass` mặc định để 2 → nhớ truyền `numClass=len(encode)` khi tạo model.

## 🎯 Mốc kiểm tra
- Loss epoch đầu ≈ `-log(1/3) ≈ 1.10`.
- Loss giảm dần. Accuracy iris ~ 90-97% (baseline sklearn = 97%).

## Shape chính (n=1200, feat=4, C=3)
`X(1200,4)` `W(4,3)` `b(1,3)` `z/y_hat(1200,3)` `y_onehot(1200,3)` `error(1200,3)` `dw(4,3)` `db(1,3)`
