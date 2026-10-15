# 📚 axis & keepdims — CHỖ HAY LỘN NHẤT

> Sai `axis` là bug thầm lặng số 1 trong bài softmax. Đọc kỹ.

---

## 1. axis nhìn thế nào cho khỏi lộn

Với ma trận `(n dòng, C cột)` — mỗi **dòng = 1 mẫu**, mỗi **cột = 1 lớp**:

```
             cột0  cột1  cột2      <- axis=1 chạy NGANG qua đây
      dòng0 [  .    .    .  ]
      dòng1 [  .    .    .  ]
      dòng2 [  .    .    .  ]
        ↑
   axis=0 chạy DỌC xuống đây
```

- **`axis=0`** = gộp theo **DỌC** (xuống các dòng) → **gộp qua các MẪU**, kết quả có shape `(C,)`
- **`axis=1`** = gộp theo **NGANG** (qua các cột) → **gộp qua các LỚP trong 1 mẫu**, kết quả `(n,)`

**Mẹo nhớ:** axis nào thì chiều đó **biến mất** (bị gộp lại).
- `sum(axis=0)` → mất chiều dòng → còn `(C,)`
- `sum(axis=1)` → mất chiều cột → còn `(n,)`

---

## 2. Dùng axis nào ở đâu (BẢNG CHỐT cho softmax)

| Việc | axis | Vì sao |
|---|---|---|
| **softmax** (max & sum) | **`axis=1`** | chuẩn hóa MỖI DÒNG (mỗi mẫu 1 phân phối) |
| **loss** cross-entropy (sum) | **`axis=1`** | cộng qua các lớp TRONG 1 mẫu |
| **argmax** khi predict | **`axis=1`** | mỗi mẫu chọn 1 lớp thắng |
| **db** (sum error) | **`axis=0`** | gộp qua các MẪU, giữ chiều lớp → ra `(1,C)` |

→ Nhớ: hầu hết là **axis=1** (làm theo mẫu). Chỉ **db dùng axis=0** (gộp mẫu để cập nhật bias mỗi lớp).

**Bug tôi từng mắc:** softmax để `axis=0` (sai) nhưng loss để `axis=1` (đúng) → không nhất quán → xác suất vô nghĩa.
Test: `forward(X).sum(axis=1)` phải toàn `1.0`.

---

## 3. keepdims=True — để làm gì

- Không có `keepdims`: `sum(axis=1)` của `(n,C)` → ra `(n,)` (mất chiều)
- Có `keepdims=True`: → ra `(n,1)` (giữ chiều, thành cột)

**Vì sao cần:** để phép chia **broadcast đúng**. `(n,C) / (n,1)` → chia từng dòng đúng. Còn `(n,C) / (n,)` → lỗi hoặc broadcast sai.

→ Trong softmax và db: **luôn để `keepdims=True`**.

---

## ⚠️ Test nhanh
```python
z = np.random.rand(4, 3)
print(np.sum(z, axis=0).shape)   # (3,)  -> gộp mẫu
print(np.sum(z, axis=1).shape)   # (4,)  -> gộp lớp
print(np.sum(z, axis=1, keepdims=True).shape)  # (4,1)
```
