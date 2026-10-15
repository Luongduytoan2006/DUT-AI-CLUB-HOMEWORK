# 📚 NumPy — Cú pháp cơ bản

> Syntax numpy đã tự gõ tay và hiểu. Thêm mỗi khi học được cái mới.

---

## 1. Tạo ma trận (file `01_numpy.py`)

- Tạo thủ công: `np.array([[1,2,3],[4,5,6],[7,8,9]])`
- Tạo tuần tự: `np.arange(1, 10).reshape(3, 3)` → 1..9 xếp thành 3x3
- Toàn số 1: `np.ones((5, 1))`
- Toàn số 0: `np.zeros((n, C))` — hay dùng để dựng bảng trống rồi điền (vd one-hot thủ công)
- **Ma trận đơn vị**: `np.eye(C)` → ma trận vuông C×C, đường chéo = 1 (xem file one-hot)

## 2. Random (file `02_...`)

- Số thực [0,1): `np.random.rand(5, 2)` → shape (5,2)
- Số nguyên: `np.random.randint(0, 10, (5, 2))`
- Dùng khởi tạo W: `np.random.rand(n_features, n_class)`

## 3. Slice / Index

- Một hàng: `matrix[1]` → hàng thứ 2 (index từ 0)
- Một cột: `matrix[:, 1]` → cột thứ 2
- Vùng con: `matrix[0:2, 1:3]`
- **Boolean mask**: `matrix[matrix % 2 == 0]` → lọc số chẵn
- **Fancy indexing**: đưa 1 mảng làm chỉ số → rút nhiều dòng cùng lúc.
  - `np.eye(3)[[0,2,1]]` → rút dòng 0,2,1 → chính là one-hot (xem `05_onehot`)

## 4. Ghép / biến hình

- Ghép cột: `np.concatenate((bias, x), axis=1)` (axis=1 = ghép ngang, thêm cột)
- **Reshape**: `y.reshape(-1, 1)` → `(200,)` thành `(200,1)`. `-1` = "tự tính chiều còn lại".
- Chuyển vị: `x.T` → đổi (a,b) thành (b,a). Dùng nhiều trong gradient `x.T @ error`.

## 5. Thống kê

- `.mean()`, `.sum()` — có thể thêm `axis` (xem file `03_axis`)
- `np.max` vs `np.argmax` — **giá trị** vs **vị trí** (xem file `04_argmax_max`)

---

## ⚠️ Còn yếu / cần ôn
- Phân biệt `np.concatenate` vs `np.hstack`/`np.vstack`/`np.stack`.
- Broadcasting khi cộng `b` vô hướng/`(1,C)` vào ma trận `(n,C)`.
