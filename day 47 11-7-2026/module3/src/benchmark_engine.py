# -*- coding: utf-8 -*-
"""Baseline native Jetson: chay TensorRT engine tren cung video."""
import argparse
import json
import os
import platform
import sys
import time

import cv2

from common import draw_people
from tensorrt_detector import TensorRTDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.engine")
    parser.add_argument("--video", default="data/people.avi")
    parser.add_argument("--output", default="results/engine_people.avi")
    parser.add_argument("--json", default="results/engine.json")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    result = {
        "format": "TensorRT engine",
        "runtime": "TensorRT + PyCUDA",
        "status": "failed",
        "model": args.model,
        "python": sys.version.split()[0],
        "os": platform.platform(),
    }
    cap = None
    writer = None
    detector = None
    started = time.time()
    try:
        detector = TensorRTDetector(args.model)
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
        infer_seconds = 0.0
        person_detections = 0
        frames_with_people = 0
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            detections = detector.infer_people(frame, args.conf)
            infer_seconds += time.time() - t0
            person_detections += len(detections)
            frames_with_people += int(bool(detections))
            writer.write(draw_people(frame, detections))
            frames += 1
        result.update({
            "status": "success",
            "frames": frames,
            "frames_with_people": frames_with_people,
            "person_detections": person_detections,
            "inference_seconds": round(infer_seconds, 3),
            "inference_fps": round(frames / infer_seconds, 3) if infer_seconds else 0,
            "pipeline_seconds": round(infer_seconds, 3),
            "pipeline_fps": round(frames / infer_seconds, 3) if infer_seconds else 0,
            "output_video": args.output,
        })
    except Exception as exc:
        result["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        if detector is not None:
            detector.close()
    result["wall_seconds"] = round(time.time() - started, 3)
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
