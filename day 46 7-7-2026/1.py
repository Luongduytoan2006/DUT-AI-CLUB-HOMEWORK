"""Bai 1 - Code mo hinh R-CNN don gian.

Yeu cau:
- Su dung selective search tu thu vien (cv2).
- Dung ResNet18 lam Feature Extractor de trich xuat dac trung.
- Dung lop Linear de phan loai cac region proposals.
- Phong cach code de hieu, co ban, phu hop de hoc tap.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

# ----------------------------- SELECTIVE SEARCH -----------------------------

class RegionProposal_SelectiveSearch:
    """Su dung Selective Search cua OpenCV de tim cac vung co kha nang chua object."""
    def __init__(self, max_proposals=2000):
        self.max_proposals = max_proposals
        
    def get_proposals(self, image):
        # image la numpy array (H, W, 3) doc bang OpenCV
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(image)
        
        # Dung che do 'fast' de toc do chay nhanh hon, hoac 'quality' de chinh xac hon
        ss.switchToSelectiveSearchFast()
        
        # rects tra ve co dang (x, y, w, h)
        rects = ss.process()
        
        # Gioi han so luong proposals (vung de xuat)
        if len(rects) > self.max_proposals:
            rects = rects[:self.max_proposals]
            
        # Chuyen tu toa do (x, y, w, h) sang (x1, y1, x2, y2) cho de bieu dien
        boxes = []
        for (x, y, w, h) in rects:
            boxes.append([x, y, x + w, y + h])
            
        return boxes

# ----------------------------- FEATURE EXTRACTOR -----------------------------

class ResNet18FeatureExtractor(nn.Module):
    """Su dung ResNet18 de rut trich dac trung, bo di lop cuoi cung (FC)."""
    def __init__(self):
        super().__init__()
        # Load mo hinh ResNet18 da duoc train truoc tren ImageNet
        resnet = models.resnet18(pretrained=True)
        
        # Lay tat ca cac lop tru lop Fully Connected cuoi cung
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x):
        # x co kich thuoc: (batch_size, 3, 224, 224)
        x = self.features(x)
        
        # x hien tai co kich thuoc: (batch_size, 512, 1, 1)
        # Ta can flatten de chuyen no thanh vector 1D cho moi anh trong batch
        x = x.view(x.size(0), -1) # Kich thuoc sau flatten: (batch_size, 512)
        return x

# ----------------------------- RCNN MODEL -----------------------------

class RCNN(nn.Module):
    """Mo hinh R-CNN ket hop Selective Search, Feature Extractor va Classifier."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Rut trich dac trung
        self.feature_extractor = ResNet18FeatureExtractor()
        
        # 2. Phan loai (Classification) bang Linear layer
        # ResNet18 tra ve vector dac trung co kich thuoc 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        # 3. Bo bien doi anh (Transform) cho dau vao cua ResNet18
        # Anh dau vao mang ResNet can phai resize ve 224x224
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def process_regions(self, image, boxes):
        """Cat cac vung tu anh goc va resize de dua vao mang ResNet18."""
        regions = []
        for (x1, y1, x2, y2) in boxes:
            # Cat anh tu array (H, W, C) -> chu y y truoc x trong numpy index
            crop_img = image[y1:y2, x1:x2]
            
            # Kiem tra neu vung cat qua nho hoac bi loi
            if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
                continue
                
            # Resize, dua ve Tensor va them vao mang
            region_tensor = self.transform(crop_img)
            regions.append(region_tensor)
            
        # Ghép cac tensor don le thanh 1 batch
        if len(regions) > 0:
            return torch.stack(regions) # Tra ve (N, 3, 224, 224)
        else:
            return torch.empty(0)

    def forward(self, region_batch):
        """
        Dau vao la batch cac region da duoc crop va resize san.
        (N, 3, 224, 224)
        """
        # Rut trich dac trung
        features = self.feature_extractor(region_batch) # Kich thuoc: (N, 512)
        
        # Phan loai cac vung do
        preds = self.classifier(features) # Kich thuoc: (N, num_classes)
        return preds


# ----------------------------- MAIN DE KIEM TRA -----------------------------

def main():
    print("Kiem tra kien truc R-CNN tu code...")
    
    # Tao anh gia (numpy array 500x500x3) de gia lap anh tu camera hay du lieu
    image_fake = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    
    # 1. Chay Selective Search
    print("- Chay Selective Search...")
    ss = RegionProposal_SelectiveSearch(max_proposals=5) # Lay 5 region cho nhanh
    boxes = ss.get_proposals(image_fake)
    print(f"  So luong region proposals: {len(boxes)}")
    
    # 2. Khoi tao R-CNN
    print("- Khoi tao mo hinh R-CNN...")
    # Bo dataset Garbage Classification co nhieu loai, gia su ta co 6 class rac + 1 background = 7
    num_classes = 7 
    model = RCNN(num_classes=num_classes)
    
    # 3. Tien xu ly vung (Crop + Resize)
    print("- Cat va resize cac vung anh de lay feature...")
    region_batch = model.process_regions(image_fake, boxes)
    print(f"  Kich thuoc batch anh vao model: {region_batch.shape}")
    
    # 4. Chuyen tiep qua mo hinh
    if len(region_batch) > 0:
        print("- Chuyen tiep (Forward) qua Feature Extractor va Classifier...")
        out = model(region_batch)
        print(f"  Kich thuoc dau ra du doan: {out.shape} -> (So vung, So Classes)")
        print("Thanh cong! R-CNN xay dung hop ly va hoat dong tot.")
    else:
        print("Loi: Khong crop duoc vung nao hop le.")

if __name__ == "__main__":
    main()
