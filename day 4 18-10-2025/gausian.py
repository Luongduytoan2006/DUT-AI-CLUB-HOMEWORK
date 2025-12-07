import math
import matplotlib.pyplot as plt

def gaussian(x, mean, variance):
    mau = math.sqrt(2*math.pi*variance)
    tu = math.exp( -(1/(2*variance)) * (x-mean)**2 )
    return tu/mau

x = [i for i in range(0, 21)]
y = [gaussian(i, 10, 3) for i in x]

plt.plot(x, y)
plt.title("Gaussian Distribution")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid()
plt.show()