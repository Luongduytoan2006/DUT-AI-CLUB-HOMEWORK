n = int(input("Enter n: "))
a = list(map(int, input().split()))

m = int(input("Enter m: "))
b = list(map(int, input().split()))

a_set = set(a)
b_set = set(b)

giao_nhau = a_set.intersection(b_set)

ans = [ a[i] for i in range(len(a)) if a[i] in giao_nhau ]

print(f"a giao nhau b: {ans}")

