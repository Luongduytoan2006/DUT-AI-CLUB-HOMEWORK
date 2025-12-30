# Đề Bài: Validation Strategy And Hyperparameter Optimizer

---

## Bài tập 1: Minh họa "Lời nguyền số chiều" (Curse of Dimensionality) và Random Search

### Mục tiêu

Chứng minh lý thuyết "Grid Search gặp vấn đề bùng nổ tổ hợp, trong khi Random Search hiệu quả hơn với cùng ngân sách".

### Đề bài

Giả sử bạn cần tune Hyperparameter cho mô hình `RandomForestClassifier` trên bộ dữ liệu **Digits** (nhận dạng số viết tay).

**Phần A (Lý thuyết & Tính toán)**: Định nghĩa một không gian Grid Search lớn (6 tham số). Hãy viết code in ra tổng số lượng tổ hợp cần thử nghiệm nếu chạy Grid Search. (Không chạy fit, chỉ tính toán số lượng).

**Phần B (Thực hành)**: Sử dụng `RandomizedSearchCV`. Thiết lập `n_iter=50` (chỉ chạy 50 lần thử).

### Yêu cầu

So sánh số lượng tổ hợp lý thuyết của Grid Search so với 50 lần chạy thực tế của Random Search để thấy sự chênh lệch về tài nguyên.

---

## Bài tập 2: TPE (Tree-structured Parzen Estimator) - Sức mạnh của "Memory"

### Mục tiêu

Thực hành phương pháp TPE (thuật toán mặc định của Optuna) để thấy cách nó "học" từ quá khứ (sử dụng $l(x)$ và $g(x)$ như lý thuyết đã nêu) thay vì dò ngẫu nhiên.

### Đề bài

- Sử dụng thư viện **Optuna**.
- Sử dụng bộ dữ liệu **Wine** (phân loại rượu).
- Tối ưu hóa mô hình **LightGBM** (hoặc XGBoost/RandomForest).

### Điểm nhấn

Trong hàm `objective`, hãy thêm `trial.report` và **Pruning** (tùy chọn) để mô phỏng quy trình chuyên nghiệp.

Visualize lịch sử tối ưu hóa để thấy thuật toán hội tụ về vùng tham số tốt như thế nào.

---

## Bài tập 3: So sánh trực tiếp Random Search (Vô nhớ) vs. TPE (Có nhớ)

### Mục tiêu

Chứng minh nhận định "TPE chọn $x$ để tối đa hóa tỷ số $l(x)/g(x)$, tức là chọn giá trị có khả năng xuất hiện cao trong nhóm tốt".

### Đề bài

1. Định nghĩa một hàm mục tiêu toán học phức tạp (ví dụ hàm **Rastrigin** hoặc một hàm tự tạo có nhiều cực trị địa phương) để mô phỏng hàm loss của mô hình Deep Learning.
2. Chạy `RandomSampler` (mô phỏng Random Search) 50 lần.
3. Chạy `TPESampler` (Bayesian Optimization) 50 lần.
4. So sánh giá trị tối ưu tìm được của cả hai. Bạn sẽ thấy TPE thường tìm được giá trị tốt hơn hoặc hội tụ nhanh hơn.

---

## Tổng kết

Ba bài tập này giúp bạn:

- **Bài tập 1**: Hiểu được vấn đề "Curse of Dimensionality" và ưu điểm của Random Search so với Grid Search.
- **Bài tập 2**: Nắm được cách hoạt động của TPE và khả năng "học từ quá khứ" của nó.
- **Bài tập 3**: So sánh trực tiếp hiệu quả giữa Random Search (vô nhớ) và TPE (có nhớ) trong tối ưu hóa hyperparameter.
