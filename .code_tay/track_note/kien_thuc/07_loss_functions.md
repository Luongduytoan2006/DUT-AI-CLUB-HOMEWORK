# 📚 Loss Functions (hàm mất mát)

> Loss = đo model sai BAO NHIÊU (1 con số). Để theo dõi, KHÔNG trực tiếp update W.
> Cái update W là **gradient** (xem `08_gradient_descent`).

---

## 1. MSE — Mean Squared Error (Linear Regression)

```python
np.mean((y_hat - y) ** 2)
```
- Dùng cho bài **dự đoán số** (regression).
- Ý nghĩa: trung bình bình phương sai lệch.

---

## 2. BCE — Binary Cross-Entropy (Logistic, 2 lớp)

```python
y_hat = np.clip(y_hat, eps, 1 - eps)     # chống log(0)
-np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
```
- Dùng cho **2 lớp**.

---

## 3. Cross-Entropy (Softmax, nhiều lớp)

```python
y_hat = np.clip(y_hat, eps, 1 - eps)
np.mean(-np.sum(y * np.log(y_hat), axis=1))    # y là ONE-HOT (n,C)
```
- `axis=1`: cộng qua các lớp TRONG 1 mẫu → ra loss mỗi mẫu `(n,)`, rồi `mean` → 1 số.
- Vì y one-hot (chỉ 1 chỗ =1) → thực chất = `-log(ŷ tại lớp đúng)`.

### Cách "chấm" hoạt động
```
ŷ tại lớp đúng cao (0.99) -> -log = 0.01  (phạt nhẹ, đoán đúng & tự tin)
ŷ tại lớp đúng thấp (0.10) -> -log = 2.30  (phạt nặng, đoán sai)
```
→ model train để đẩy ŷ tại lớp đúng lên cao.

### 🎯 Mốc kiểm tra vàng
- Loss epoch đầu (W random, C lớp) ≈ `-log(1/C)`.
- Iris 3 lớp: `-log(1/3) ≈ 1.10`. Ra tầm đó = forward + loss ĐÚNG.
- Loss phải **giảm dần** qua epoch. Tăng = lộn dấu gradient hoặc lr quá to.

---

## ⚠️ Luôn nhớ
- **clip trước log** — nếu không, `log(0) = nan` làm hỏng cả train.
- Loss `axis=1`, nhưng db (gradient bias) `axis=0` — đừng lộn.

---

## ⚠️ Còn cần hiểu sâu
- Vì sao đạo hàm cross-entropy + softmax rút gọn đúng bằng `y_hat - y_onehot`? (chưa hiểu bản chất, mới học kết quả)
