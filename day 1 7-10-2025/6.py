import math

pi = math.pi

def sinx(x,n) -> float:
    ans = x
    tuso = x
    heso = 1
    mauso = 1
    for i in range(1,n):
        tuso *= x * x
        heso *= -1
        mauso *= (2*i) * (2*i+1)
        temp = heso * (tuso / mauso)
        ans += temp
    return ans

def cosx(x,n) -> float:
    ans = 1
    tuso = 1
    heso = 1
    mauso = 1
    for i in range(1,n):
        tuso *= x * x
        heso *= -1
        mauso *= (2*i-1) * (2*i)
        temp = heso * (tuso / mauso)
        ans += temp
    return ans

def sinhx(x,n) -> float:
    ans = x
    tuso = x
    mauso = 1
    for i in range(1,n):
        tuso *= x * x
        mauso *= (2*i) * (2*i+1)
        temp = tuso / mauso
        ans += temp
    return ans

def coshx(x,n) -> float:
    ans = 1
    tuso = 1
    mauso = 1
    for i in range(1,n):
        tuso *= x * x
        mauso *= (2*i-1) * (2*i)
        temp = tuso / mauso
        ans += temp
    return ans

x = float(input("Enter x (degrees): "))
n = int(input("Enter n: "))

while( n < 0 ):
    print("Error: n must not be a negative integer.")
    n = int(input("Enter n: "))

x = x * pi / 180

print(f"sin({x}) = {sinx(x,n):.6f}")
print(f"cos({x}) = {cosx(x,n):.6f}")
print(f"sinh({x}) = {sinhx(x,n):.6f}")
print(f"cosh({x}) = {coshx(x,n):.6f}")
