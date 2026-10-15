# -*- coding: utf-8 -*-
"""Tien ich xu ly output YOLOv8, viet tuong thich Python 3.6/OpenCV 4.1."""
import cv2
import numpy as np


PERSON_CLASS_ID = 0


def letterbox(image, size=640):
    """Resize giu ti le va pad anh ve hinh vuong."""
    height, width = image.shape[:2]
    ratio = min(float(size) / height, float(size) / width)
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(image, (new_width, new_height))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - new_width) // 2
    pad_y = (size - new_height) // 2
    canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    return canvas, ratio, pad_x, pad_y


def make_blob(image, size=640):
    canvas, ratio, pad_x, pad_y = letterbox(image, size)
    blob = cv2.dnn.blobFromImage(
        canvas, 1.0 / 255.0, (size, size), swapRB=True, crop=False)
    return blob, ratio, pad_x, pad_y


def decode_people(output, ratio, pad_x, pad_y, width, height,
                  conf_threshold=0.35, nms_threshold=0.45):
    """Giai ma output YOLOv8 (1,84,8400), chi giu class person=0."""
    pred = np.asarray(output)
    pred = np.squeeze(pred, axis=0)
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    person_scores = pred[:, 4 + PERSON_CLASS_ID]
    keep = person_scores >= conf_threshold
    if not np.any(keep):
        return []

    xywh = pred[keep, :4]
    scores = person_scores[keep]
    cx, cy, box_w, box_h = xywh.T
    x = (cx - box_w / 2.0 - pad_x) / ratio
    y = (cy - box_h / 2.0 - pad_y) / ratio
    box_w = box_w / ratio
    box_h = box_h / ratio
    nms_boxes = np.stack([x, y, box_w, box_h], axis=1)

    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), scores.tolist(), conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    results = []
    for index in np.asarray(indices).reshape(-1):
        bx, by, bw, bh = nms_boxes[index]
        x1 = max(0, min(width - 1, int(bx)))
        y1 = max(0, min(height - 1, int(by)))
        x2 = max(0, min(width - 1, int(bx + bw)))
        y2 = max(0, min(height - 1, int(by + bh)))
        results.append((x1, y1, x2, y2, float(scores[index])))
    return results


def draw_people(image, detections):
    for x1, y1, x2, y2, score in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, "person %.2f" % score, (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return image

