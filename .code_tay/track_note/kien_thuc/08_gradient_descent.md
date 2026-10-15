# 📚 Gradient Descent & cập nhật trọng số

> Đây mới là cái THỰC SỰ update W (không phải loss). Loss chỉ để xem.

---

## Công thức gốc

```
W := W - lr * dW      (dW = ∂Loss/∂W = độ dốc của loss theo W)
b := b - lr * db
```
- `lr` = learning rate (bước đi).
- Đi **ngược** hướng dốc để loss giảm (xuống núi).

---

## Gradient rút gọn (Linear / Logistic / Softmax — GIỐNG NHAU!)

```python
error = y_hat - y            # Linear/Logistic: (n,1) ; Softmax: (n,C) one-hot
dw = (1/n) * np.dot(X.T, error)          # shape = shape W
db = (1/n) * np.sum(error, axis=0, keepdims=True)   # softmax: (1,C)
W -= lr * dw
b -= lr * db
```

**Cả 3 thuật toán dùng CÙNG công thức `X.T @ (y_hat - y)`.** Chỉ khác:
- Linear: không có activation.
- Logistic: y_hat qua sigmoid, y là (n,1).
- Softmax: y_hat qua softmax, y là one-hot (n,C), db cần `axis=0, keepdims`.

---

## error là gì & khác loss thế nào

| | Loss | Error = `y_hat - y` |
|---|---|---|
| Là | 1 con số | ma trận `(n,C)` |
| Trả lời | sai BAO NHIÊU | sai theo HƯỚNG NÀO, chỗ nào |
| Dùng | người xem | máy chỉnh W (vào dw, db) |

- error dương ở 1 ô = lớp đó đoán **dư** → W bị kéo xuống.
- error âm = đoán **thiếu** → W kéo lên.
- Về toán: **error chính là đạo hàm của loss theo z** → nên nó (không phải loss) mới update W.

---

## Vì sao có `X.T` (ép shape)

`dw` phải cùng shape W. Ghép: `X.T (feat×n) @ error (n×C) = dw (feat×C)` ✅.
Để `X @ error` thẳng thì n không khớp → nhân không được. Nên phải transpose.

---

## Bảng dịch toán → numpy

| Toán | numpy |
|---|---|
| Xᵀ | `X.T` |
| A·B (nhân ma trận) | `np.dot(A,B)` hoặc `A @ B` |
| Σ theo chiều | `np.sum(..., axis=?, keepdims=True)` |
| 1/n | `1/X.shape[0]` |
| W := W − lr·dW | `W -= lr * dw` |

---

## ⚠️ Bẫy
- **Dấu**: phải `y_hat - y` và `W -= lr*dw`. Đảo → loss tăng.
- **Thứ tự nhân**: `X.T @ error`, KHÔNG `error @ X.T`.
- **db softmax**: `axis=0, keepdims=True` → ra `(1,C)`. `np.sum(error)` trơn ra 1 số là sai (bias không học riêng từng lớp).
- Test: `dw.shape == W.shape` và `db.shape == b.shape`.
