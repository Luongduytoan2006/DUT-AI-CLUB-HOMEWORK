# -*- coding: utf-8 -*-
"""YOLOv8 TensorRT runner toi gian, tuong thich JetPack 4.6/TensorRT 8.2."""
import gc

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

from common import decode_people, make_blob


class TensorRTDetector(object):
    def __init__(self, engine_path):
        cuda.init()
        self.cuda_context = cuda.Device(0).make_context()
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as handle, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        for binding in self.engine:
            shape = tuple(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            device = cuda.mem_alloc(host.nbytes)
            record = {"host": host, "device": device, "shape": shape}
            self.bindings.append(int(device))
            if self.engine.binding_is_input(binding):
                self.inputs.append(record)
            else:
                self.outputs.append(record)
        self.cuda_context.pop()

    def infer_people(self, frame, conf_threshold=0.35):
        height, width = frame.shape[:2]
        blob, ratio, pad_x, pad_y = make_blob(frame)
        self.cuda_context.push()
        try:
            np.copyto(self.inputs[0]["host"], blob.ravel())
            cuda.memcpy_htod_async(
                self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(
                self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
            self.stream.synchronize()
            output = self.outputs[0]["host"].reshape(self.outputs[0]["shape"])
        finally:
            self.cuda_context.pop()
        return decode_people(
            output, ratio, pad_x, pad_y, width, height, conf_threshold)

    def close(self):
        """Giai phong object CUDA/TRT khi context van con active."""
        try:
            self.cuda_context.push()
            for record in self.inputs + self.outputs:
                record["device"].free()
            self.bindings = []
            self.inputs = []
            self.outputs = []
            self.stream = None
            self.context = None
            self.engine = None
            gc.collect()
            self.cuda_context.pop()
            self.cuda_context.detach()
        except Exception:
            pass
