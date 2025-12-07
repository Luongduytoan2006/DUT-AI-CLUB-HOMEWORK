def pascal_triangle(n) -> list[list[int]]:
    triangle = []
    for i in range(n):
        row = [1] * (i + 1)
        for j in range(1, i):
            row[j] = triangle[i - 1][j - 1] + triangle[i - 1][j]
        triangle.append(row)
    return triangle

def fibonaci(n) -> list[int]:
    if n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        ans = [0, 1]
        for i in range(2, n):
            ans.append(ans[i - 1] + ans[i - 2])
        return ans

while True:
    print(20 * "=")
    print("Choose an option:")
    print("1. Pascal's Triangle")
    print("2. Fibonacci sequence")
    print("0. Exit")
    choice = input("Enter your choice (0/1/2): ")
    if choice == "0":
        break
    elif choice in ["1", "2"]:
        try:
            n = input("Enter n: ")
            n = int(n)
        except ValueError:
            print("Error: n must be an integer.")
            continue

        if n < 1:
            print("Error: n must be a positive integer.")
            continue

        if choice == "1":
            ans = pascal_triangle(n)
            for row in ans:
                print(row)
        elif choice == "2":
            ans = fibonaci(n)
            print(ans)
    else:
        print("Invalid choice. Please try again.")