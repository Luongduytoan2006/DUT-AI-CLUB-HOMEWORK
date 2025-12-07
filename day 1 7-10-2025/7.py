def reverse_number(n: int) -> int:
    ans = 0
    while n > 0:
        temp = n % 10
        ans = ans * 10 + temp
        n //= 10
    return ans

n = int(input("Enter a positive integer: "))
while n < 0:
    print("Error: n must be a positive integer.")
    n = int(input("Enter a positive integer: "))

ans = reverse_number(n)
print(f"Reversed number: {ans}")