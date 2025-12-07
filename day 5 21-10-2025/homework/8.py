import cv2
import numpy as np
import math
import os

def resize_nni(source, source_h, source_w, target_h, target_w):    
    new_data = [[0]*target_w for _ in range(target_h)]
    
    w_scale_factor = source_w/target_w 
    h_scale_factor = source_h/target_h
    
    for i in range(target_h):
        for j in range(target_w):
            src_y = math.floor(i * h_scale_factor)
            src_x = math.floor(j * w_scale_factor)
            new_data[i][j] = source[src_y][src_x]
            
    return new_data

if os.path.exists('nature.jpg'):
    print("✅ Đang xử lý ảnh nature.jpg...")
    
    image = cv2.imread('nature.jpg', 0).tolist()
    
    height = len(image)
    width = len(image[0])
    
    print(f"Kích thước ảnh gốc: {height}x{width}")
    print(f"Kích thước ảnh mới: {height*3}x{width*3}")
    
    new_image = resize_nni(image, height, width, height*3, width*3)
    cv2.imwrite('nature_resized_3x.jpg', np.array(new_image))
    
    print("✅ Đã lưu ảnh mới: nature_resized_3x.jpg")
else:
    print("❌ Không tìm thấy file nature.jpg")
