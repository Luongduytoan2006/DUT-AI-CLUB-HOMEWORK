# 📚 argmax vs max & các hàm numpy hay dùng

---

## 1. max vs argmax — GIÁ TRỊ vs VỊ TRÍ (hay lộn!)

```python
probs = np.array([0.1, 0.2, 0.7])
np.max(probs)     # -> 0.7   (GIÁ TRỊ lớn nhất)
np.argmax(probs)  # -> 2     (VỊ TRÍ của giá trị lớn nhất)
```

- **max** = lấy con số lớn nhất.
- **argmax** = lấy **chỉ số/vị trí** của số lớn nhất. (`arg` = argument = vị trí)

**Khi phân loại: LUÔN dùng `argmax`** — vì cần biết "lớp nào thắng" (vị trí), không cần giá trị %.

```python
np.argmax(y_hat, axis=1)   # mỗi dòng -> vị trí lớp có % cao nhất -> nhãn dự đoán
```

---

## 2. Mẹo: argmax của one-hot = nhãn số gốc

```python
one_hot = [0, 0, 1]
np.argmax(one_hot)  # -> 2   (số 1 nằm ở vị trí 2)
```
→ Nếu `y` đang là one-hot `(n,C)`, muốn về nhãn số `(n,)` thì: `np.argmax(y, axis=1)`.
Dùng khi tính accuracy mà y đã bị one-hot.

---

## 3. So sánh & đếm đúng (tính accuracy)

```python
y_pred == y_true          # -> mảng True/False
np.mean(y_pred == y_true) # True=1, False=0 -> trung bình = tỉ lệ đúng = accuracy
```
**Điều kiện:** `y_pred` và `y_true` phải **cùng shape `(n,)`**. Lệch shape → broadcast sai → ra `(n,n)`.

---

## 4. clip — chặn giá trị

```python
np.clip(y_hat, eps, 1 - eps)   # ép mọi giá trị vào [eps, 1-eps]
```
Dùng trước `np.log` để tránh `log(0) = -inf → nan`. (eps = 1e-9)

---

## 5. exp / log

- `np.exp(z)` = e^z. Trong softmax. **Bẫy tràn số:** z lớn → e^z = inf. Chống bằng trừ max (xem note softmax).
- `np.log(x)` = ln(x). Trong cross-entropy. **Bẫy:** log(0) = -inf → cần clip.
