"""Bai 2 - Code mo hinh Fast R-CNN kem theo lop RoI Pooling tu build.

Yeu cau:
- Tu code lop RoI Pooling theo dung ly thuyet.
- Tu code cls_head, bbox_head theo dung ly thuyet.
- So sanh qua trinh inference va training va tu rut ra nhan xet.

-----------------------------------------------------------------------------------------
SO SANH VA NHAN XET (TRAINING vs INFERENCE TRONG FAST R-CNN):

1. Qua trinh Training (Huan luyen):
   - Dau vao gom anh va Ground Truth (bounding boxes thuc te tu nhan vung).
   - Su dung Selective Search (hoac RPN trong Faster RCNN sau nay) de sinh ra Region Proposals (RoIs).
   - Buoc loc mau (Sampling): Tinh IoU cua RoIs voi Ground Truth de chon ra cac mau positive (IoU >= 0.5) 
     va negative/background (IoU tu 0.1 den < 0.5) de dua vao batch (thuong la 25% positive, 75% negative).
   - Tinh Loss (Mat mat): Gom 2 phan: Classification Loss (Cross Entropy cho cls_head) 
     va Regression Loss (Smooth L1 cho bbox_head de tinh chinh toa do box).
   - Lan truyen nguoc (Backprop): Cap nhat trong so cua toan bo mang (Feature Extractor, cls_head, bbox_head).

2. Qua trinh Inference (Kiem thu / Du doan tren anh thuc te):
   - Dau vao chi co anh goc, KHONG co ground truth.
   - Su dung Selective Search de sinh RoIs. 
   - Dua TAY CA cac proposals (hoac luong lon box) qua RoI Pooling roi vao cls_head, bbox_head ma 
     KHONG co buoc chon mau positive/negative.
   - Ket qua tra ve se chua rat nhieu bounding box du doan trong len nhau.
   - Bat buoc phai ap dung Non-Maximum Suppression (NMS) hoac Soft-NMS de loai bo cac box trung lap, 
     chi giu lai box co confidence score (do tin cay) cao nhat cho moi object.

3. Nhan xet ca nhan:
   - Fast R-CNN cai tien toc do dot pha so voi R-CNN vi anh chi di qua Feature Extractor MOT LAN DUY NHAT 
     de tao ra Feature Map. Cac vung sau do moi duoc "cat" tren Feature Map qua RoI Pooling, giup tiet kiem 
     thoi gian tinh toan chong cheo cho hang ngan boxes.
   - Qua trinh training phuc tap va doi hoi phai co cac thuat toan lay mau (sampling) va loss da muc tieu, 
     trong khi inference thi nhe nhanh hon nhung phu thuoc lon vao thuat toan loc nhieu la NMS de ra ket qua dep.
-----------------------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------- ROI POOLING LAYER -----------------------------

class RoIPooling(nn.Module):
    """Lop RoI Pooling tu code theo dung ly thuyet de mang feature map ve kich thuoc co dinh."""
    def __init__(self, output_size, spatial_scale=1.0):
        super().__init__()
        # output_size la kich thuoc dau ra, vd: int (7) -> thanh (7, 7) hoac truyen thang tuple
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size
            
        # spatial_scale la ty le kich thuoc giua feature map va anh goc
        # Vi du anh dau vao 224x224, feature map tu ResNet la 14x14 -> scale = 14/224 = 1/16
        self.spatial_scale = spatial_scale

    def forward(self, feature_map, rois):
        """
        - feature_map: Tensor dang (1, C, H, W) - chi xet batch_size = 1 cho don gian.
        - rois: Tensor cac regions (N, 5), voi 5 cot la (batch_index, x1, y1, x2, y2).
        """
        num_rois = rois.size(0)
        channels = feature_map.size(1)
        out_H, out_W = self.output_size
        
        # Khoi tao tensor chua ket qua dau ra (N, C, 7, 7)
        output = torch.zeros(num_rois, channels, out_H, out_W).to(feature_map.device)
        
        for i in range(num_rois):
            roi = rois[i]
            batch_idx = int(roi[0].item())
            
            # Map toa do box tu anh goc xuong toa do tren feature map bang cach nhan ty le
            x1 = int(round(roi[1].item() * self.spatial_scale))
            y1 = int(round(roi[2].item() * self.spatial_scale))
            x2 = int(round(roi[3].item() * self.spatial_scale))
            y2 = int(round(roi[4].item() * self.spatial_scale))
            
            # Tranh truong hop toa do vuot qua bien hoac kich thuoc <= 0
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = max(x1 + 1, x2)
            y2 = max(y1 + 1, y2)
            
            # Cat vung feature map (Crop)
            roi_feature = feature_map[batch_idx, :, y1:y2, x1:x2]
            
            # Dung Adaptive Max Pool de ep vung moi cat ve cung mot kich thuoc (7x7) bat ke box to nho ra sao
            # Chuyen mang tu (C, H_roi, W_roi) -> (1, C, H_roi, W_roi) de Pool xu ly dung
            roi_feature = roi_feature.unsqueeze(0)
            pooled_feature = nn.functional.adaptive_max_pool2d(roi_feature, self.output_size)
            
            # Luu vao ket qua sau khi squeeze bo chieu batch tam thoi di
            output[i] = pooled_feature.squeeze(0)
            
        return output

# ----------------------------- FAST RCNN MODEL -----------------------------

class FastRCNN(nn.Module):
    """Kien truc mo hinh Fast R-CNN."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # 1. Feature Extractor
        # Dung cac lop tich chap cua ResNet18, bo AvgPool va Fully Connected di
        resnet = models.resnet18(pretrained=True)
        
        # Lay tat ca cac lop tru 2 lop cuoi cung
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        
        # Tu ResNet18 thi anh dau vao qua nhieu lop conv se nho di 32 lan
        self.spatial_scale = 1.0 / 32.0 
        
        # 2. RoI Pooling layer 
        # Muc tieu la keo cac roi co kich thuoc khac nhau tren feature map ve thanh (7x7) de dua vao lop Linear
        self.roi_pool = RoIPooling(output_size=(7, 7), spatial_scale=self.spatial_scale)
        
        # 3. Fully Connected chung sau RoI Pool
        # ResNet18 co so kenh dau ra la 512, mat na pool ra la 7x7 -> kich thuoc flatten = 512 * 7 * 7
        in_features = 512 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # 4. Classification Head (Phan loai object)
        # Dau ra la diem (logits) cho tung lop trong num_classes
        self.cls_head = nn.Linear(1024, num_classes)
        
        # 5. Bounding Box Regression Head (Tinh chinh box)
        # Dau ra: 4 gia tri (dx, dy, dw, dh) cho MOI class -> kich thuoc = num_classes * 4
        self.bbox_head = nn.Linear(1024, num_classes * 4)

    def forward(self, image, rois):
        """
        - image: Tensor anh (1, 3, H, W)
        - rois: Tensor cac box region (N, 5) -> 5 gia tri do la (batch_index, x1, y1, x2, y2)
        """
        # Buoc 1: Chay Toan Bo anh qua Feature Extractor MỘT LẦN DUY NHẤT
        feature_map = self.feature_extractor(image)
        
        # Buoc 2: Chay qua RoI Pooling de lay feature cho tung roi
        # Dau ra tu day co dang chuan: (N, C, 7, 7)
        pooled_features = self.roi_pool(feature_map, rois)
        
        # Buoc 3: Flatten (trai phang) de dua vao lop Fully Connected chung
        flattened = pooled_features.view(pooled_features.size(0), -1) # Kich thuoc: (N, 512*7*7)
        fc_out = self.fc(flattened) # Kich thuoc: (N, 1024)
        
        # Buoc 4: Chay song song qua 2 head
        cls_scores = self.cls_head(fc_out) # (N, num_classes)
        bbox_preds = self.bbox_head(fc_out) # (N, num_classes * 4)
        
        return cls_scores, bbox_preds

# ----------------------------- MAIN DE KIEM TRA -----------------------------

def main():
    print("Kiem tra kien truc Fast R-CNN va RoI Pooling tu code...")
    
    # Khoi tao mo hinh
    num_classes = 7 # 6 class rac + 1 background de test
    model = FastRCNN(num_classes=num_classes)
    
    # Tao anh gia (Batch size = 1, 3 channels, 224x224)
    image_fake = torch.randn(1, 3, 224, 224) 
    
    # Tao 3 cai RoIs gia lap tu Selective Search 
    # Cot 1 la batch_index (do chi co 1 anh ne index = 0)
    rois_fake = torch.tensor([
        [0, 10.0, 20.0, 100.0, 150.0],
        [0, 50.0, 50.0, 200.0, 200.0],
        [0, 5.0, 5.0, 50.0, 50.0]
    ])
    
    print("- Chuyen tiep (Forward) qua Fast R-CNN...")
    cls_scores, bbox_preds = model(image_fake, rois_fake)
    
    print(f"  Kich thuoc Image dau vao : {image_fake.shape}")
    print(f"  Kich thuoc RoIs dau vao  : {rois_fake.shape} (3 vung)")
    print(f"  Kich thuoc du doan Class : {cls_scores.shape} (3 vung x 7 classes)")
    print(f"  Kich thuoc du doan BBox  : {bbox_preds.shape} (3 vung x 28 toa do)")
    print("Thanh cong! Fast R-CNN kem RoI Pooling chay hop ly.")

if __name__ == "__main__":
    main()
