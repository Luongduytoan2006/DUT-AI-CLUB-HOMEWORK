import cv2
import matplotlib.pyplot as plt

# Load bộ lọc đã được huấn luyện sẵn của OpenCV [cite: 96, 200]
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đọc ảnh
img = cv2.imread('../cho_quang.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Haar hoạt động trên ảnh xám [cite: 49, 362]

# detectMultiScale: scaleFactor=1.3 (giảm kích thước ảnh 30% mỗi lần quét), minNeighbors=5 [cite: 53, 304]
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

print(f"Tìm thấy {len(faces)} khuôn mặt!")

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

# Chuyển BGR sang RGB để hiển thị đúng màu trên Jupyter
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
