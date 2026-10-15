# 📚 Activation Functions (hàm kích hoạt)

> Hàm bọc ngoài `z = X@W+b` để biến điểm thô thành xác suất.

---

## 1. Sigmoid (cho 2 lớp — Logistic)

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
```
- Biến 1 số thực → 1 xác suất trong khoảng **(0, 1)**.
- Ra **1 số** cho mỗi mẫu. Phân lớp: so ngưỡng `0.5`.
- Dùng khi: **2 lớp** (nhị phân).

---

## 2. Softmax (cho nhiều lớp — Softmax Regression)

```python
def softmax(z):                      # z shape (n, C)
    z = z - np.max(z, axis=1, keepdims=True)   # trừ max MỖI DÒNG -> chống tràn e^
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)  # chia tổng MỖI DÒNG
```

Công thức: `ŷ_i = e^(z_i) / Σ e^(z_j)`

- Biến **C số** → **C xác suất cộng lại = 1**.
- Ra **nhiều số** (mỗi lớp 1 xác suất). Phân lớp: `argmax`.
- Dùng khi: **nhiều lớp** (≥ 3).

### 🔥 Mẹo chống tràn (QUAN TRỌNG)
- `e^z` với z lớn → **tràn số (inf)**. VD `e^1000 = inf`.
- Fix: **trừ tất cả z cho `max(z)` của mỗi dòng** trước khi `exp`.
- Kết quả % **KHÔNG đổi** (tử & mẫu cùng nhân `e^(-max)` → triệt tiêu), nhưng số lớn nhất giờ là `e^0=1` → an toàn.
- `axis=1` vì chuẩn hóa theo mỗi mẫu (mỗi dòng).

### Vì sao softmax cộng lại = 1
Mỗi phần tử = "phần của mình / tổng tất cả" → gộp lại đương nhiên = 1 → là xác suất hợp lệ.

---

## Sigmoid vs Softmax

| | Sigmoid | Softmax |
|---|---|---|
| Số lớp | 2 | ≥ 3 |
| Vào → ra | 1 số → 1 xác suất | C số → C xác suất (cộng=1) |
| Chọn lớp | ngưỡng 0.5 | argmax |
| W có | 1 cột | C cột |

> Logistic thực ra = softmax 2 lớp rút gọn. Cùng 1 họ.
