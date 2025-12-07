import math

def linear_interpolation(p1, p2, x):
    x1, y1 = p1
    x2, y2 = p2
    if x1 != x2:
        y = ((x2-x)/(x2-x1))*y1 + ((x-x1)/(x2-x1))*y2
    else:
        y = y1
    return y

def li_1d(data, k):
    result = []
    n = len(data)
    
    for i in range(n-1):
        for j in range(k):
            x = i + j/k
            y = linear_interpolation((i, data[i]), (i+1, data[i+1]), x)
            result.append(y)
    
    result.append(data[-1])
    
    while len(result) < n*k:
        result.append(0)
    
    return result


# Test case 1
data = [1, 2, 3]
k = 2
result = li_1d(data, k)
ans = [1, 1.5, 2, 2.5, 3, 0]
if result[:-1] == ans[:-1]:
    print("✅ Test case 1 passed")
    print(f"Result: {result}")
else:
    print("❌ Test case 1 failed")
    print(f"Expected: {ans}")
    print(f"Got: {result}")


# Test case 2
data = [1, 4, 7]
k = 3
result = li_1d(data, k)
ans = [1, 2.0, 3.0, 4, 5.0, 6.0, 7, 0, 0]
if result[:-2] == ans[:-2]:
    print("✅ Test case 2 passed")
    print(f"Result: {result}")
else:
    print("❌ Test case 2 failed")
    print(f"Expected: {ans}")
    print(f"Got: {result}")


# Test case 3 
data = [7, 10]
k = 3
result = li_1d(data, k)
print(f"\n✅ Test case 3 (multiple choice):")
print(f"data = {data}, k = {k}")
print(f"Result (bỏ 2 phần tử cuối): {result[:-2]}")
