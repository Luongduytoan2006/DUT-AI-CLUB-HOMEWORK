# Ket qua tren Jetson Nano

| Dinh dang | Runtime can dung | Trang thai | FPS pipeline | Ghi chu / loi |
|---|---|---:|---:|---|
| ONNX | ONNX Runtime CPU | success | 2.213 | - |
| PT | PyTorch/Ultralytics CPU | success | 1.509 | - |
| PTH | PyTorch/Ultralytics CPU | success | 1.481 | - |
| TensorRT engine (tham chieu) | TensorRT + PyCUDA | success | 8.226 | - |

## Ket luan dung

- `.onnx` chay bang ONNX Runtime CPU trong Python 3.10, khong can PyTorch.
- `.pt` va `.pth` deu la quy uoc ten checkpoint PyTorch; duoi file khong tu tao kha nang tuong thich.
- `.pt` la checkpoint Ultralytics; `.pth` benchmark thanh cong luu full `nn.Module`. File state_dict-only rieng cho thay code kien truc phai khop chinh xac moi nap duoc.
- Kha nang chay phu thuoc CPU architecture, Python, CUDA/JetPack va runtime da cai; khong phai Windows hay Linux tu quyet dinh bang duoi file.
- TensorRT `.engine` la binary da toi uu cho dung TensorRT/GPU Jetson; day la baseline native cua NVIDIA trong bai thu.
