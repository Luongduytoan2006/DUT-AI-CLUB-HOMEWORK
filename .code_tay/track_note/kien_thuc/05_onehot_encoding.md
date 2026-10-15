# 📚 Encoding nhãn & One-hot

> 2 thứ đều "biến thành số" nhưng KHÁC TẦNG — hay lẫn.

---

## Phân biệt 3 khái niệm (đừng lẫn!)

| Tên | Làm gì | Ví dụ | Khi nào |
|---|---|---|---|
| **Encode nhãn** | chữ → số | `"setosa" → 0` | BẮT BUỘC, ngay sau đọc data |
| **One-hot** | số → vector | `2 → [0,0,1]` | lúc train (tính loss/gradient) |
| **Decode** | số → chữ | `2 → "setosa"` | KHÔNG bắt buộc, chỉ để người đọc |

---

## 1. Encode nhãn (chữ → số)

**Cách tay (hiểu bản chất):**
```python
list_class = np.unique(y)           # np.unique tự SORT sẵn -> thứ tự cố định
encode = { name: num for num, name in enumerate(list_class) }
y = y.map(encode)                   # y phải là Series (pandas), không phải numpy
```
- `np.unique` đã sort → không cần `sorted()` nữa.
- `enumerate` cho cặp `(số, tên)`.
- `.map()` phải gán lại (không sửa tại chỗ).

**Cách thư viện (dùng nhiều thực tế):**
- `sklearn LabelEncoder`: `.fit()`, `.transform()`, `.inverse_transform()`, xem bảng ở `.classes_`.
- `pd.factorize(y)` → trả về `(mảng số, list nhãn)`. Thêm `sort=True` để cố định.

---

## 2. One-hot (số → vector)

**Mẹo `np.eye` (không cần for):**
```python
y_onehot = np.eye(C)[y]    # C = SỐ LỚP;  y phải là int, 1D
```
Vì sao chạy được: `np.eye(C)` là ma trận đơn vị — **dòng thứ k = one-hot của lớp k**.
Đưa mảng `y` vào index → rút đúng các dòng → ra bảng one-hot `(n, C)`.

```
np.eye(3) =
 [[1,0,0],   <- lớp 0
  [0,1,0],   <- lớp 1
  [0,0,1]]   <- lớp 2
```

**C là SỐ LỚP** (không phải số dòng data, không phải số feature). C = độ dài 1 vector one-hot.
Số dòng đến từ `y` lúc index.

**Cách for (chậm, để hiểu):**
```python
oh = np.zeros((n, C))
for i in range(n):
    oh[i, y[i]] = 1
```

### ⚠️ Bẫy chí mạng
- `y` phải **1D `(n,)`**. Nếu `(n,1)` → `eye[y]` ra `(n,1,C)` thừa chiều → hỏng.
- `y` phải **int**. Float → lỗi index.
- Test: `y_onehot.sum(axis=1)` phải toàn `1.0`, shape `(n,C)`.

---

## 3. Decode (số → chữ) — KHÔNG bắt buộc

- Train/loss/accuracy: **KHÔNG cần decode** (số so số là xong).
- Chỉ cần khi: in kết quả cho người đọc, predict data thật, xem confusion matrix.
- Cách: đảo dict `decode = {v:k for k,v in encode.items()}` rồi `decode[2]`.
