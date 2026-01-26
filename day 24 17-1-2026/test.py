import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print("Đây là phép nhân thông thường:")
print(a*b)

print("Đây là phép nhân (tích vô hướng) ma trận (dot product):")
print(np.dot(a, b))

a_col = a.reshape(3,1)   # shape (3,1)
b_row = b.reshape(1,3)   # shape (1,3)
print("Đây là phép ma trân nhân ma trận (multi matrix)")
print(a_col)
print(b_row)
print(a_col @ b_row)
