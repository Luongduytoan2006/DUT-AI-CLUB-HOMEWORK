# 📚 Ma trận trọng số W & Xử lý data

---

## 1. W quyết định SỐ ĐẦU RA (rất quan trọng!)

**Số CỘT của W = số đầu ra.** Vì quy tắc nhân ma trận:
```
X (n × feat) @ W (feat × SỐ_CỘT) = z (n × SỐ_CỘT)
```

| | Số cột W | z ra | Dùng |
|---|---|---|---|
| Linear | 1 | 1 số | dự đoán số |
| Logistic | 1 | 1 số → sigmoid | 2 lớp |
| Softmax | **C (số lớp)** | C số → softmax | nhiều lớp |

→ Muốn ra mấy lớp thì cho W bấy nhiêu cột. **Mỗi lớp có 1 cột trọng số riêng** — cột k chấm điểm "hợp với lớp k cỡ nào".

**Khởi tạo:**
```python
W = np.random.rand(X.shape[1], numClass)   # (n_features, n_class)
b = np.zeros((1, numClass))                # mỗi lớp 1 bias, KHÔNG phải (n,1)
```
- b đi theo **lớp** (C cái), không theo mẫu. `b` shape `(1, C)` để broadcast xuống n dòng.

---

## 2. Xử lý data — quy trình chuẩn

```python
df = pd.read_csv(...)
x = df.iloc[:, :-1].values      # mọi cột trừ cuối -> features (numpy)
y = df.iloc[:, -1]              # cột cuối -> nhãn (còn là Series!)
y = y.map(encode)              # chữ -> số (khi CÒN là Series)
# softmax: KHÔNG reshape(-1,1). logistic: có reshape.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

- `.iloc[:, :-1]` = tất cả cột trừ cột cuối. `.iloc[:, -1]` = cột cuối.
- Encode lúc `y` còn là **Series** (mới có `.map`).
- `random_state=42` = cố định kết quả chia (tái lập được).

---

## 3. StandardScaler — chuẩn hóa feature

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)   # fit + transform trên TRAIN
x_test  = scaler.transform(x_test)        # CHỈ transform trên test (không fit lại!)
```
- Đưa feature về mean 0, std 1 → giúp gradient descent hội tụ nhanh, ổn định.
- **Quy tắc vàng:** `fit` CHỈ trên train. Test chỉ `transform`. Fit cả test = rò rỉ dữ liệu (data leakage).
- ⚠️ Đã import ở bài 04 nhưng chưa dùng — bài softmax nên áp vào (nếu loss khó giảm thường do chưa scale).

---

## ⚠️ Còn yếu / cần ôn
- `np.concatenate` vs `hstack/vstack`.
- Khi nào gộp bias vào X vs để `b` riêng.
