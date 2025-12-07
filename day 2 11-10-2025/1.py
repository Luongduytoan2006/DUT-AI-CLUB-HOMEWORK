from collections import deque

n , k = map(int, input("Enter n and k: ").split())
a = list(map(int, input("Enter the elements of a: ").split()))

if n < k:
    print("n must be greater than or equal to k")
    exit()

temp = []
for i in range(k):
    temp.append(a[i])

temp = deque(temp)

print(max(temp), end=' ')

for i in range(k+1, n):
    temp.popleft()
    temp.append(a[i])
    print(max(temp), end=' ')

