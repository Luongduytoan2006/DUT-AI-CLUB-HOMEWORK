# Bài tập 1: Khởi động tay với Ma trận (NumPy)
# Bạn hãy mở file Python hoặc Jupyter Notebook lên và tự tay gõ đoạn code giải quyết yêu cầu sau (nếu quên hàm, hãy thử search Google tài liệu của Numpy):
# Import thư viện numpy.
# Khởi tạo một ma trận (2D array) có kích thước 3x3, chứa các con số tuần tự từ 1 đến 9.
# Lấy ra (slice) toàn bộ hàng thứ 2 của ma trận đó và in ra màn hình.
# Tính và in ra giá trị trung bình (mean) của tất cả các phần tử trong ma trận.
# Tính tổng của tất cả các phần tử trong ma trận và in ra kết quả.
# Tính tổng các số chẵn trong ma trận và in ra kết quả.



import numpy as np

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.arange(1, 10).reshape(3, 3)

print(matrix)
print(matrix2)

print("Lấy ra toàn bộ ma trận hàng thứ 2")
print(matrix[1])

print("Tính và in ra giá trị trung bình (mean) của tất cả các phần tử trong ma trận")
print(matrix.mean())


print("Tính tổng của tất cả các phần tử trong ma trận và in ra kết quả")
print(matrix.sum())

print("Tính tổng các số chẵn trong ma trận và in ra kết quả")
even_sum = matrix[matrix % 2 == 0]
print(even_sum)
print(even_sum.sum())