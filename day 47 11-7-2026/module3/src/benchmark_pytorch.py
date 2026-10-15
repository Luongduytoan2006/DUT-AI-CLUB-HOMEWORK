# -*- coding: utf-8 -*-
"""Chay YOLOv8n PT hoac PTH state_dict bang PyTorch/Ultralytics CPU."""
import argparse
import json
import os
import platform
import sys
import time

import cv2
import torch
from ultralytics import YOLO

from common import draw_people


def load_model(model_format, model_path):
    if model_format == "pt":
        return YOLO(model_path)
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if "model" in payload:
        wrapper = YOLO("yolov8n.yaml")
        wrapper.model = payload["model"].float().eval()
        return wrapper
    model = YOLO("yolov8n.yaml")
    model.model.load_state_dict(payload["state_dict"], strict=True)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=["pt", "pth"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", default="data/people.avi")
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=True)
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    result = {
        "format": args.format.upper(),
        "runtime": "PyTorch/Ultralytics CPU",
        "status": "failed",
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_bytes": os.path.getsize(args.model),
    }
    started = time.time()
    cap = None
    writer = None
    try:
        model = load_model(args.format, args.model)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("Khong mo duoc video")
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 10.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(
            args.output, cv2.VideoWriter_fourcc(*"MJPG"), fps_in,
            (width, height))
        frames = 0
        pipeline_seconds = 0.0
        person_detections = 0
        frames_with_people = 0
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            prediction = model.predict(
                source=frame, imgsz=640, conf=args.conf,
                device="cpu", verbose=False)[0]
            detections = []
            if prediction.boxes is not None:
                xyxy = prediction.boxes.xyxy.cpu().numpy()
                scores = prediction.boxes.conf.cpu().numpy()
                classes = prediction.boxes.cls.cpu().numpy()
                for box, score, class_id in zip(xyxy, scores, classes):
                    if int(class_id) != 0:
                        continue
                    detections.append((int(box[0]), int(box[1]),
                                       int(box[2]), int(box[3]), float(score)))
            pipeline_seconds += time.time() - t0
            person_detections += len(detections)
            frames_with_people += int(bool(detections))
            writer.write(draw_people(frame, detections))
            frames += 1
        result.update({
            "status": "success",
            "frames": frames,
            "frames_with_people": frames_with_people,
            "person_detections": person_detections,
            "pipeline_seconds": round(pipeline_seconds, 3),
            "pipeline_fps": round(frames / pipeline_seconds, 3),
            "wall_seconds": round(time.time() - started, 3),
            "output_video": args.output,
        })
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
    result["wall_seconds"] = round(time.time() - started, 3)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
