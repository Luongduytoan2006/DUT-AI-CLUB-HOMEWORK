# 🔴 LƯU Ý QUAN TRỌNG NHẤT — Type & Shape dữ liệu

> Đây là chỗ tôi vấp NHIỀU NHẤT. Đa số bug ML không báo lỗi đỏ — nó **chạy được nhưng ra số sai**.
> Nguyên nhân gần như luôn là: **sai shape** hoặc **sai kiểu dữ liệu**. Đọc file này trước mỗi lần code.

---

## ⭐ Nguyên tắc vàng: IN SHAPE RA SAU MỖI BƯỚC

```python
print(x.shape, y.shape)     # làm hoài, ở mọi bước
print(y.dtype)              # kiểm tra kiểu
```
Bug thầm lặng (chạy được mà sai) chỉ lộ ra khi nhìn shape. **Nghi ngờ gì → print shape trước.**

---

## 1. Bẫy `reshape(-1, 1)` — Logistic CẦN, Softmax KHÔNG

| | Logistic | Softmax |
|---|---|---|
| y nên là | `(n, 1)` → cần `reshape(-1,1)` | `(n,)` **1D** → KHÔNG reshape |
| Lý do | y_hat ra `(n,1)`, y phải khớp | one-hot cần y 1D để index |

**Bug kinh điển:** `y` đang `(n,1)` mà đem `np.eye(C)[y]` → ra `(n, 1, C)` (thừa 1 chiều!)
→ `error = y_hat - y` thành `(n,C) - (n,1,C)` → **broadcast nổ thành `(n,n,C)`** → sai hết mà không báo lỗi.

**Cách tránh:** trước khi `np.eye`, đảm bảo `y` là 1D. Test: `print(y_onehot.shape)` phải `(n, C)`, KHÔNG phải `(n,1,C)`.

---

## 2. `.values` / `.to_numpy()` — pandas → numpy

- `df["cot"]` là **Series (pandas)** → có `.map()`, `.head()`.
- `df["cot"].values` là **mảng numpy** → có `.reshape()`, dùng cho tính toán.
- **Bẫy:** `.map()` KHÔNG chạy trên numpy array. Nếu lỡ `.values` rồi thì phải bọc `pd.Series(arr).map(...)`.

Thứ tự đúng: **map (khi còn là Series) → rồi mới `.values`**.

---

## 3. Kiểu dữ liệu (dtype) — phải là SỐ mới train được

- Sau `.map(encode)`: kiểm tra `y.dtype` phải là `int64`, KHÔNG phải `object` (còn chữ).
- `np.eye(C)[y]`: `y` phải là **int**. Nếu là float → lỗi index. Ép: `y.astype(int)`.
- `dtype=object` = pandas đang giữ chữ/hỗn hợp → chưa encode xong.

---

## 4. Bảng shape chuẩn cho 1 bài phân lon loại (iris: n=1200, feat=4, C=3)

| Biến | Shape | Ghi chú |
|---|---|---|
| `X` | `(n, 4)` | mỗi dòng 1 mẫu, mỗi cột 1 feature |
| `y` (số) | `(n,)` | 1D cho softmax |
| `W` | `(4, 3)` | (n_features × n_class) |
| `b` | `(1, 3)` | mỗi lớp 1 bias |
| `z = X@W+b` | `(n, 3)` | điểm thô |
| `y_hat` (softmax) | `(n, 3)` | xác suất, mỗi dòng cộng=1 |
| `y_onehot` | `(n, 3)` | eye(3)[y] |
| `error = y_hat - y_onehot` | `(n, 3)` | phải cùng shape mới trừ được |
| `dw = X.T @ error` | `(4, 3)` | **phải == shape W** |
| `db` | `(1, 3)` | sum(error, axis=0, keepdims) |

**Quy tắc tự kiểm:** `dw.shape` phải luôn `== W.shape`. Lệch là sai thứ tự nhân.

---

## 5. Checklist debug khi "chạy được mà sai"

1. `print` shape mọi biến — có cái nào thừa/thiếu chiều không?
2. `y_hat.sum(axis=1)` có toàn `1.0` không? (softmax đúng chưa)
3. `dw.shape == W.shape`? (gradient đúng chiều chưa)
4. Loss epoch đầu ~ `-log(1/C)`? (iris 3 lớp ≈ 1.10) → forward đúng chưa
5. Loss có **giảm dần** không? Tăng = lộn dấu hoặc lr quá to.
6. `dtype` có phải số không? Còn `object` = chưa encode.

---

_Ghi sau khi làm xong Softmax (05) — đây là bài dính nhiều bug shape nhất._
