def compose(data):
    res = []
    for i in range(len(data[0])):
        col = [row[i] for row in data]
        res.append(col)
    return res


# test case
data = [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]] 
ans  = [[1, 4, 7],
        [2, 5, 8],
        [3, 6, 9]]
if compose(data) == ans:
    print("Test case 1 passed")
else:
    print("Test case 1 failed")

data = [[1, 2, 3],
        [4, 5, 6]]
ans  = [[1, 4],
        [2, 5],
        [3, 6]]
if compose(data) == ans:
    print("Test case 2 passed")
else:
    print("Test case 2 failed")