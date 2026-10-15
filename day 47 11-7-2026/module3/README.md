# Module 3 - So sanh ONNX, PT va PTH tren Jetson Nano

Project nay nhan dien **nguoi** trong video OpenCV `vtest.avi`, sau do ghi lai
ket qua thuc te cua ba cach luu/chay model YOLOv8n.

## Cau truc

```text
module3/
|-- data/people.avi
|-- models/yolov8n.onnx
|-- models/yolov8n.pt
|-- models/yolov8n_state_dict.pth
|-- models/yolov8n_full_model.pth
|-- models/yolov8n.engine
|-- src/
|   |-- benchmark_onnx.py
|   |-- benchmark_engine.py
|   |-- check_pytorch_format.py
|   |-- common.py
|   `-- make_report.py
|-- results/
|   |-- comparison.md
|   `-- onnx_people.avi
|-- pyproject.toml
`-- run_all.sh
```

## Moi truong

Jetson Nano nay dung JetPack 4.6, Python 3.6.9 va cac goi he thong aarch64.
Tao venv bang uv nhung cho phep thay OpenCV/NumPy cua he thong:

```bash
uv venv --python /usr/bin/python3 --system-site-packages
```

Chay toan bo thi nghiem, mac dinh 30 frame cho moi dinh dang:

```bash
cd /home/team2/team2/module3
bash run_all.sh 30
```

- `.venv`: Python 3.6 system-site-packages cho TensorRT/PyCUDA cua JetPack.
- `.venv-modern`: Python 3.10 aarch64 voi PyTorch CPU, ONNX Runtime va Ultralytics.

Danh sach goi truc tiep nam trong `requirements-modern.txt`; script
`setup_modern.sh` dung de tao lai venv khi da co Python aarch64 trong `.python/`.

Ket qua doc tai `results/comparison.md`. Video ONNX da ve bounding box nam tai
`results/onnx_people.avi`.

## Dieu can hieu

`.pt` va `.pth` khong dai dien cho hai he dieu hanh. Theo quy uoc PyTorch, ca hai
deu co the la file duoc tao boi `torch.save`. Noi dung moi quan trong:

- `yolov8n.pt`: checkpoint Ultralytics gom model va metadata.
- `yolov8n_state_dict.pth`: state_dict thu nghiem, cho thay kien truc code phai
  khop chinh xac voi weights.
- `yolov8n_full_model.pth`: full `nn.Module` PyTorch dung trong benchmark thanh
  cong, vi no giu dung cau truc cua checkpoint YOLOv8 doi dau.
- `yolov8n.onnx`: graph tinh trung gian, runtime khac PyTorch co the doc.
- `yolov8n.engine`: binary TensorRT da compile rieng cho GPU/runtime NVIDIA; duoc
  them lam baseline de thay dang nao thuc su phu hop Jetson.

Video nguon: OpenCV repository, `samples/data/vtest.avi`.
