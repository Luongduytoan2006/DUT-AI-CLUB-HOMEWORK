def h_nni_2d(data, k):    
    result = []
    for row in data:
        new_row = []
        for x in row:
            for _ in range(k):
                new_row.append(x)
        result.append(new_row)
    return result


# Test case 1
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
k = 2
ans = [[1, 1, 2, 2, 3, 3], 
       [4, 4, 5, 5, 6, 6], 
       [7, 7, 8, 8, 9, 9]]

result = h_nni_2d(data, k)
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
       [4, 4, 5, 5], 
       [7, 7, 8, 8]]

result = h_nni_2d(data, k)
if result == ans:
    print("✅ Test case 2 passed")
else:
    print("❌ Test case 2 failed")


# Test case 3 
data = [[3],
        [4]]
k = 3
result = h_nni_2d(data, k)
ans = [[3, 3, 3],
       [4, 4, 4]]
if result == ans:
    print("✅ Test case 3 passed")
else:
    print("❌ Test case 3 failed")

