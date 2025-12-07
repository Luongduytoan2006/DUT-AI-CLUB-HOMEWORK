import numpy as np

arr = np.array([ [1,2,3],
                 [4,5,6],
                 [7,8,9]])


print(arr[::2,::2])
print("\n")
print(arr[-3:2,0:3])

print(40*"=")

check = arr > 5
print(arr[check])