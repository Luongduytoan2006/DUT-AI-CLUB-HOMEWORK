def nni_1d(data, k):
    result = []
    for x in data:
        for _ in range(k):
            result.append(x)
    return result

def nni_1d_v2(data, k):
    return [x for x in data for _ in range(k)]


# Test case 1
data = [1, 2, 3]
k = 2
ans = [1, 1, 2, 2, 3, 3]
result = nni_1d(data, k)
if result == ans:
    print("✅ Test case 1 passed")
else:
    print("❌ Test case 1 failed")

# Test case 2
data = [1, 2, 3]
k = 3
ans = [1, 1, 1, 2, 2, 2, 3, 3, 3]
result = nni_1d(data, k)
if result == ans:
    print("✅ Test case 2 passed")
else:
    print("❌ Test case 2 failed")

data = [5, 7]
k = 3
ans = [5, 5, 5, 7, 7, 7]
result = nni_1d(data, k)
if result == ans:
    print("✅ Test case 3 passed")
else:
    print("❌ Test case 3 failed")

