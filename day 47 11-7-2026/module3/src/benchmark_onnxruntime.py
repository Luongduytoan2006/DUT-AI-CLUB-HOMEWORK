# -*- coding: utf-8 -*-
"""Chay YOLOv8n ONNX that bang ONNX Runtime CPU va xuat video."""
import argparse
import json
import os
import platform
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort

from common import decode_people, draw_people, make_blob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolov8n.onnx")
    parser.add_argument("--video", default="data/people.avi")
    parser.add_argument("--output", default="results/onnx_people.avi")
    parser.add_argument("--json", default="results/onnx.json")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    result = {
        "format": "ONNX",
        "runtime": "ONNX Runtime CPU",
        "status": "failed",
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
    }
    started = time.time()
    cap = None
    writer = None
    try:
        session = ort.InferenceSession(
            args.model, providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
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
        output = None
        while frames < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.time()
            blob, ratio, pad_x, pad_y = make_blob(frame)
            output = session.run(None, {input_name: blob})[0]
            detections = decode_people(
                output, ratio, pad_x, pad_y, width, height, args.conf)
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
            "output_shape": list(np.asarray(output).shape),
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

