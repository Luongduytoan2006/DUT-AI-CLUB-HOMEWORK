import math
import cv2
import matplotlib.pyplot as plt

def computeXDerivative(image):
    h, w = len(image), len(image[0])
    gx = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            left  = image[i][j-1] if j-1 >= 0 else image[i][j]
            right = image[i][j+1] if j+1 < w else image[i][j]
            val = int(right) - int(left)
            gx[i][j] = max(0, min(255, val + 128))
    return gx

def computeYDerivative(image):
    h, w = len(image), len(image[0])
    gy = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            up   = image[i-1][j] if i-1 >= 0 else image[i][j]
            down = image[i+1][j] if i+1 < h else image[i][j]
            val = int(down) - int(up)
            gy[i][j] = max(0, min(255, val + 128))
    return gy

def computeMagnitudeXY(image):
    gx = computeXDerivative(image)
    gy = computeYDerivative(image)
    h, w = len(image), len(image[0])
    mag = [[0]*w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            dx = gx[i][j] - 128
            dy = gy[i][j] - 128
            val = int(round(math.sqrt(dx*dx + dy*dy)))
            mag[i][j] = max(0, min(255, val))
    return mag

if __name__ == "__main__":
    cat_img = cv2.imread("cat.jpeg", 0)
    cat_img = cv2.resize(cat_img, (400, 400), interpolation=cv2.INTER_AREA)
    cat_img_list = cat_img.tolist()
    x_derivative = computeXDerivative(cat_img_list)
    y_derivative = computeYDerivative(cat_img_list)
    gradient_magnitude = computeMagnitudeXY(cat_img_list)

    # Hiển thị
    fig = plt.figure(figsize=(10, 7))

    fig.add_subplot(2, 2, 1)
    plt.imshow(cat_img, cmap="gray")
    plt.axis('off')
    plt.title("Input image")

    fig.add_subplot(2, 2, 2)
    plt.imshow(x_derivative, cmap="gray")
    plt.axis('off')
    plt.title("Gradient in X-direction")

    fig.add_subplot(2, 2, 3)
    plt.imshow(y_derivative, cmap="gray")
    plt.axis('off')
    plt.title("Gradient in X-direction")

    fig.add_subplot(2, 2, 4)
    plt.imshow(gradient_magnitude, cmap="gray")
    plt.axis('off')
    plt.title("Gradient Magnitude")

    plt.tight_layout()
    plt.show()
