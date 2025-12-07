import math

def nni_2d(data, k):    
    height = len(data)
    width = len(data[0])
    new_data = [[0]*(k*width) for _ in range(k*height)]
    
    for i in range(k*height):
        for j in range(k*width):
            y = math.floor(i/k)
            x = math.floor(j/k)
            new_data[i][j] = data[y][x]
            
    return new_data


# Test case 1
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
k = 2
ans = [[1, 1, 2, 2, 3, 3], 
       [1, 1, 2, 2, 3, 3],
       [4, 4, 5, 5, 6, 6],
       [4, 4, 5, 5, 6, 6], 
       [7, 7, 8, 8, 9, 9], 
       [7, 7, 8, 8, 9, 9]]

result = nni_2d(data, k)
if result == ans:
    print("✅ Test case 1 passed")
else:
    print("❌ Test case 1 failed")


# Test case 2
data = [[1, 2],
        [4, 5],
        [7, 8]]
k = 2
ans = [[1, 1, 2, 2], 
       [1, 1, 2, 2], 
       [4, 4, 5, 5], 
       [4, 4, 5, 5], 
       [7, 7, 8, 8],
       [7, 7, 8, 8]]

result = nni_2d(data, k)
if result == ans:
    print("✅ Test case 2 passed")
else:
    print("❌ Test case 2 failed")


# Test case 3 (multiple choice từ đề bài)
data = [[1, 2, 3, 4]]
k = 2
result = nni_2d(data, k)
ans = [[1, 1, 2, 2, 3, 3, 4, 4],
       [1, 1, 2, 2, 3, 3, 4, 4]]
if result == ans:
    print("✅ Test case 3 passed")
else:
    print("❌ Test case 3 failed")
