# Bài tập 2: Tiền xử lý dữ liệu - Thêm cột Bias
# Lần này, hãy gõ tiếp vào file hiện tại của bạn các bước sau:
# Tạo một ma trận $X$ bất kỳ có kích thước 5x2 (đại diện cho 5 mẫu dữ liệu, mỗi mẫu có 2 đặc trưng/features). 
# Bạn có thể tự gõ số hoặc dùng hàm random của Numpy đều được.
# Tạo một mảng chứa toàn bộ là số 1, có kích thước 5x1.Nối (concatenate/stack) mảng toàn số 1 đó vào bên trái của ma trận $X$.
# In ma trận kết quả ra màn hình (Lúc này nó phải là một ma trận kích thước 5x3, với cột ngoài cùng bên trái toàn số 1).


import numpy as np

x = np.random.rand(5,2)
print(x)

bias = np.ones((5, 1))
print(bias)

x_with_bias = np.concatenate((bias, x), axis=1)
print(x_with_bias)


x = np.random.randint(0, 10, (5, 2))
print(x)