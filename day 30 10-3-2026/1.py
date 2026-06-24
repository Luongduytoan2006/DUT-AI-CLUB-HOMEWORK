import cv2
import numpy as np
from scipy.signal import wiener

class ImageRestoration:
    @staticmethod
    def sobel(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return cv2.convertScaleAbs(sobel_combined)

    @staticmethod
    def laplacian(image):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

    @staticmethod
    def median_filter(image, ksize=5):
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def wiener_filter(image, kernel_size=5):
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for i in range(3):
                result[:, :, i] = wiener(image[:, :, i], (kernel_size, kernel_size))
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            filtered = wiener(image, (kernel_size, kernel_size))
            return np.clip(filtered, 0, 255).astype(np.uint8)
