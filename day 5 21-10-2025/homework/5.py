def v_nni_2d(data, k):    
    result = []
    for row in data:
        for _ in range(k):
            temp = row
            result.append(temp)
    return result


# Test case 1
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
k = 2
ans = [[1, 2, 3],
       [1, 2, 3],
       [4, 5, 6],
       [4, 5, 6],
       [7, 8, 9],
       [7, 8, 9]]

result = v_nni_2d(data, k)
if result == ans:
    print("✅ Test case 1 passed")
else:
    print("❌ Test case 1 failed")


# Test case 2
data = [[1, 2],
        [4, 5],
        [7, 8]]
k = 2
ans = [[1, 2], 
       [1, 2], 
       [4, 5], 
       [4, 5], 
       [7, 8], 
       [7, 8]]

result = v_nni_2d(data, k)
if result == ans:
    print("✅ Test case 2 passed")
else:
    print("❌ Test case 2 failed")


# Test case 3
data = [[3, 7]]
k = 2
result = v_nni_2d(data, k)
ans = [[3, 7],
       [3, 7]]
if result == ans:
    print("✅ Test case 3 passed")
else:
    print("❌ Test case 3 failed")
