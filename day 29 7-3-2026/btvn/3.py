import cv2 # [cite: 193]
import pickle # [cite: 196]
import numpy as np # [cite: 195]
from sklearn.neighbors import KNeighborsClassifier # [cite: 267]

# Tải dữ liệu từ file lưu trữ ở Bài tập 2 [cite: 191, 202]
with open('dataset/faces.pkl', 'rb') as w:
    faces = pickle.load(w) # [cite: 205]

with open('dataset/names.pkl', 'rb') as file: # [cite: 206]
    labels = pickle.load(file) # [cite: 209]

print('Kích thước ma trận khuôn mặt:', faces.shape) # [cite: 213]

# Khởi tạo mô hình KNN với 5 láng giềng gần nhất
knn = KNeighborsClassifier(n_neighbors=5)

# Máy bắt đầu "học" dữ liệu đã có
knn.fit(faces, labels)

camera = cv2.VideoCapture(0) # [cite: 210, 212]
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # [cite: 198, 201]

while True: # [cite: 217]
    ret, frame = camera.read() # [cite: 220]
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # [cite: 226]
        # Phát hiện vị trí khuôn mặt [cite: 227]
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 5)

        for (a, b, w, h) in face_coordinates: # [cite: 230]
            # Cắt vùng khuôn mặt từ khung hình camera [cite: 232]
            fc = frame[b:b+h, a:a+w, :]
            
            # Chuẩn hóa ảnh về kích thước 50x50 và duỗi phẳng thành vector [cite: 236]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1, -1)
            
            # Dự đoán danh tính dựa trên khoảng cách L2 [cite: 235, 793]
            text = knn.predict(r)

            # Hiển thị tên và khung hình chữ nhật [cite: 238, 241]
            cv2.putText(frame, text[0], (a, b-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.rectangle(frame, (a, b), (a+w, b+h), (0, 0, 255), 2)

        cv2.imshow('livetime face recognition', frame) # [cite: 244]
        
        # Nhấn phím ESC (mã 27) để thoát [cite: 246, 248]
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error") # [cite: 252]
        break

camera.release() # [cite: 257]
cv2.destroyAllWindows() # [cite: 256]