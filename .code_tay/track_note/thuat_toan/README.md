# 🧠 FOLDER THUẬT TOÁN — Index

> Quy trình / các bước tự tay dựng lại từng thuật toán. KHÔNG chép code — chỉ "khung xương".
> Mỗi thuật toán làm xong = 1 file `XX_ten.md`.

## Danh sách file

| File | Thuật toán | Loại |
|---|---|---|
| `01_linear_regression.md` | Linear Regression | Dự đoán số |
| `02_logistic_regression.md` | Logistic Regression | Phân loại 2 lớp |
| `03_softmax_regression.md` | Softmax Regression | Phân loại nhiều lớp |
| `99_checklist_sap_toi.md` | Danh sách chưa làm + ý tưởng ôn | — |

## Mạch tiến hóa (mỗi bài kế thừa bài trước)
```
Linear:   z = X@W+b                    -> MSE
Logistic: sigmoid(z)  [W 1 cột]        -> BCE      -> ngưỡng 0.5
Softmax:  softmax(z)  [W C cột]        -> CrossEnt -> argmax
                                          (y one-hot)
→ Gradient CẢ 3 giống nhau: X.T @ (y_hat - y)
```

## Mỗi file gồm
1. Khác bài trước ở đâu
2. Các bước (trình tự)
3. Bản chất
4. Bug đã gặp
5. Shape chính + mốc kiểm tra
