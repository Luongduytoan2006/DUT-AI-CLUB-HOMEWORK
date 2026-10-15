# -*- coding: utf-8 -*-
"""YOLOv1 inference tren Jetson bang OpenCV DNN + numpy (KHONG can torch)."""
import cv2
import numpy as np

# --- Cau hinh (khop voi luc train) ---
CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
S, B, C = 7, 2, 6
IMG_SIZE = 448
CONF_THRESH = 0.04      # model demo (10 epoch) conf thap -> nguong nhe
NMS_IOU = 0.45

# Mau cho tung lop (BGR)
COLORS = [(0, 255, 0), (255, 128, 0), (0, 128, 255),
          (255, 0, 255), (0, 255, 255), (128, 0, 255)]


class Detector:
    def __init__(self, onnx_path):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        # Thu dung CUDA backend (neu OpenCV build co CUDA), khong thi CPU
        try:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except Exception:
            pass

    def infer(self, frame_bgr):
        """frame BGR bat ky -> list (x1,y1,x2,y2,score,cls) theo pixel goc."""
        h0, w0 = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame_bgr, scalefactor=1 / 255.0,
            size=(IMG_SIZE, IMG_SIZE), swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()          # (1, S, S, B*5+C) hoac phang
        pred = np.array(out).reshape(S, S, B * 5 + C)
        return self._decode(pred, w0, h0)

    def _decode(self, pred, w0, h0):
        boxes, scores, classes = [], [], []
        logits = pred[:, :, B * 5:]             # (S,S,C) logit tho
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        cls_prob = e / e.sum(axis=-1, keepdims=True)   # softmax -> xac suat
        cls_idx = cls_prob.argmax(axis=-1)      # (S,S)
        cls_score = cls_prob.max(axis=-1)       # (S,S)
        cell = IMG_SIZE / S
        sx, sy = w0 / IMG_SIZE, h0 / IMG_SIZE   # scale ve anh goc

        for gj in range(S):
            for gi in range(S):
                for b in range(B):
                    o = b * 5
                    conf = float(pred[gj, gi, o + 4])
                    sc = conf * float(cls_score[gj, gi])
                    if sc < CONF_THRESH:
                        continue
                    x, y, w, h = pred[gj, gi, o:o + 4]
                    cx = (gi + x) * cell
                    cy = (gj + y) * cell
                    bw, bh = w * IMG_SIZE, h * IMG_SIZE
                    x1 = (cx - bw / 2) * sx; y1 = (cy - bh / 2) * sy
                    x2 = (cx + bw / 2) * sx; y2 = (cy + bh / 2) * sy
                    boxes.append([x1, y1, x2 - x1, y2 - y1])  # x,y,w,h cho NMSBoxes
                    scores.append(sc)
                    classes.append(int(cls_idx[gj, gi]))

        results = []
        if not boxes:
            return results
        idxs = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESH, NMS_IOU)
        if len(idxs) == 0:
            return results
        for i in np.array(idxs).flatten():
            x, y, w, h = boxes[i]
            results.append((int(x), int(y), int(x + w), int(y + h),
                            scores[i], classes[i]))
        return results


def draw(frame, results):
    """Ve bounding box + nhan len frame (sua truc tiep)."""
    for x1, y1, x2, y2, score, cls in results:
        color = COLORS[cls % len(COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "%s %.2f" % (CLASS_NAMES[cls], score)
        yt = max(y1 - 6, 12)
        cv2.putText(frame, label, (x1, yt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame






