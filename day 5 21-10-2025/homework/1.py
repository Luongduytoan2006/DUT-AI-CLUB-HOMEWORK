def compute_sum(data):
    result = []
    for row in data:
        row_sum = sum(row)
        result.append(row_sum)

    return result

# test case
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]
ans  = [6, 15, 24]
if compute_sum(data) == ans:
    print("Test case 1 passed")
else:
    print("Test case 1 failed")

data = [[1, 2],
        [3, 4],
        [5, 6]]
ans  = [3, 7, 11]
if compute_sum(data) == ans:
    print("Test case 2 passed")
else:
    print("Test case 2 failed")
