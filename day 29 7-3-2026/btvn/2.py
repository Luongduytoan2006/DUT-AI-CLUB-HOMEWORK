import cv2
import pickle
import numpy as np
import os

# Khởi tạo camera và bộ phân loại khuôn mặt
camera = cv2.VideoCapture(0)
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_data = []  # Mảng chứa dữ liệu pixel khuôn mặt
i = 0

name = input("Nhập tên người dùng: ").strip()

print("Bắt đầu quét khuôn mặt... Nhấn 'ESC' để dừng hoặc đợi máy lấy đủ 10 mẫu.")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Lỗi Camera!")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = facecascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

    # Sắp xếp để lấy khuôn mặt có diện tích lớn nhất
    face_coordinates = sorted(face_coordinates, key=lambda x: x[2] * x[3], reverse=True)

    for (a, b, w, h) in face_coordinates:
        # Cắt vùng khuôn mặt và resize về 50x50
        fc = frame[b:b+h, a:a+w, :]
        r = cv2.resize(fc, (50, 50))

        if len(face_data) < 10:
            face_data.append(r)
            print(f"Đã thu thập mẫu thứ: {len(face_data)}")

        # Vẽ khung khuôn mặt
        cv2.rectangle(frame, (a, b), (a+w, b+h), (0, 255, 0), 2)

    cv2.imshow('Collecting Face Data', frame)

    # Dừng khi đủ 10 mẫu hoặc nhấn ESC
    if cv2.waitKey(1) == 27 or len(face_data) >= 10:
        break

camera.release()
cv2.destroyAllWindows()

# Chuyển mảng về dạng numpy array và reshape thành vector phẳng
face_data = np.asarray(face_data)
face_data = face_data.reshape(len(face_data), -1)

# Tạo thư mục dataset nếu chưa tồn tại
os.makedirs('dataset', exist_ok=True)

# Xử lý lưu tên (Names)
names_path = 'dataset/names.pkl'
if not os.path.exists(names_path):
    names = [name] * len(face_data)
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)
else:
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * len(face_data)
    with open(names_path, 'wb') as f:
        pickle.dump(names, f)

# Xử lý lưu dữ liệu khuôn mặt (Faces)
faces_path = 'dataset/faces.pkl'
if not os.path.exists(faces_path):
    with open(faces_path, 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open(faces_path, 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, face_data, axis=0)
    with open(faces_path, 'wb') as f:
        pickle.dump(faces, f)

print(f"Đã lưu {len(face_data)} mẫu khuôn mặt cho '{name}' vào thư mục dataset/")
