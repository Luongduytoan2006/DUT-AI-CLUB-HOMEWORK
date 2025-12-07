# 🐍 Hướng Dẫn Cài Đặt và Quản Lý Python - Chi Tiết Toàn Diện

## 📑 Mục Lục
1. [Cài Đặt Python](#1-cài-đặt-python)
2. [Quản Lý Python với pyenv (Windows)](#2-quản-lý-python-với-pyenv-windows)
3. [Virtual Environment - Môi Trường Ảo](#3-virtual-environment---môi-trường-ảo)
4. [Jupyter Notebook & Kernel](#4-jupyter-notebook--kernel)
5. [VS Code Configuration](#5-vs-code-configuration)
6. [Dấu Hiệu Nhận Biết](#6-dấu-hiệu-nhận-biết)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Cài Đặt Python

### 🔹 Cách 1: Cài Trực Tiếp (Đơn Giản)

**Bước 1: Download Python**
```
https://www.python.org/downloads/
```
- Tải Python 3.10+ hoặc 3.11
- ✅ **QUAN TRỌNG:** Tích ✅ "Add Python to PATH"

**Bước 2: Kiểm Tra**
```powershell
python --version
# Hoặc
py --version
```

**Kết quả mong đợi:**
```
Python 3.10.11
```

---

### 🔹 Cách 2: Cài Qua Microsoft Store (Windows)

1. Mở **Microsoft Store**
2. Tìm "Python 3.11" hoặc "Python 3.10"
3. Click **Install**

**Ưu điểm:**
- ✅ Tự động thêm vào PATH
- ✅ Quản lý updates dễ dàng
- ✅ Không conflict với Python khác

---

## 2. Quản Lý Python với pyenv (Windows)

### 📦 Cài Đặt pyenv-win

**Bước 1: Cài qua PowerShell (Admin)**
```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

**Hoặc qua Git:**
```powershell
git clone https://github.com/pyenv-win/pyenv-win.git "$HOME\.pyenv"
```

**Bước 2: Thêm vào PATH**
```
Biến môi trường System:
PYENV = C:\Users\<YourUsername>\.pyenv\pyenv-win
PYENV_ROOT = C:\Users\<YourUsername>\.pyenv\pyenv-win
PYENV_HOME = C:\Users\<YourUsername>\.pyenv\pyenv-win

Path:
%PYENV%\bin
%PYENV%\shims
```

**Bước 3: Khởi động lại Terminal**

---

### 🎯 Các Lệnh pyenv Quan Trọng

```powershell
# Xem phiên bản Python có thể cài
pyenv install --list

# Cài Python 3.10.11
pyenv install 3.10.11

# Xem các Python đã cài
pyenv versions

# Set Python global (toàn hệ thống)
pyenv global 3.10.11

# Set Python local (chỉ folder hiện tại)
pyenv local 3.10.11

# Xem Python đang dùng
pyenv version
```

---

## 3. Virtual Environment - Môi Trường Ảo

### ❓ Virtual Environment Là Gì?

**Virtual Environment (venv)** là môi trường Python **riêng biệt** cho mỗi project:
- ✅ Mỗi project có thư viện riêng
- ✅ Không conflict giữa các project
- ✅ Dễ quản lý dependencies

---

### 🔹 Tạo Virtual Environment

#### **Cách 1: Dùng `venv` (Built-in)**

```powershell
# Di chuyển đến thư mục project
cd "D:\Duy Toan\Python\DUT AI Club\Homework"

# Tạo virtual environment tên .venv
python -m venv .venv

# Hoặc tên khác
python -m venv myenv
```

---

#### **Cách 2: Dùng `virtualenv`**

```powershell
# Cài virtualenv
pip install virtualenv

# Tạo venv
virtualenv .venv
```

---

### 🔹 Kích Hoạt Virtual Environment

#### **Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1

# Nếu lỗi Execution Policy:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### **Windows CMD:**
```cmd
.venv\Scripts\activate.bat
```

#### **Git Bash:**
```bash
source .venv/Scripts/activate
```

---

### 🔹 Dấu Hiệu Đã Kích Hoạt

```powershell
(.venv) PS D:\Duy Toan\Python\DUT AI Club\Homework>
^^^^^
Tên venv xuất hiện
```

---

### 🔹 Tắt Virtual Environment

```powershell
deactivate
```

---

### 🔹 Cài Thư Viện Trong Venv

```powershell
# Kích hoạt venv trước
.\.venv\Scripts\Activate.ps1

# Cài thư viện
pip install numpy pandas matplotlib opencv-python

# Hoặc từ file requirements
pip install -r requirements.txt
```

---

### 🔹 Global vs Local Environment

| Loại | Mô Tả | Khi Nào Dùng |
|------|-------|--------------|
| **Global** | Python cài trên máy | Dùng chung toàn hệ thống |
| **Virtual Env** | Python riêng cho project | **Khuyên dùng** cho mỗi project |

**Ví dụ:**
```
Global Python 3.10.11
├── pip install requests  → Cài vào global
│
Project A/
├── .venv/  → Python riêng cho Project A
│   └── pip install django
│
Project B/
├── .venv/  → Python riêng cho Project B
    └── pip install flask
```

---

## 4. Jupyter Notebook & Kernel

### 📘 Kernel Là Gì?

**Kernel** = Môi trường Python mà Jupyter Notebook sử dụng để chạy code.

- Mỗi notebook chọn 1 kernel
- Kernel có thể là global Python hoặc venv

---

### 🔹 Cài Jupyter

#### **Cài Global:**
```powershell
pip install jupyter notebook jupyterlab
```

#### **Cài Trong Venv:**
```powershell
# Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# Cài jupyter
pip install jupyter ipykernel
```

---

### 🔹 Thêm Venv Làm Jupyter Kernel

```powershell
# Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# Đăng ký kernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"
```

**Giải thích:**
- `--name=.venv`: Tên kernel (internal)
- `--display-name "Python (.venv)"`: Tên hiển thị trong Jupyter

---

### 🔹 Xem Danh Sách Kernel

```powershell
jupyter kernelspec list
```

**Kết quả:**
```
Available kernels:
  .venv       C:\Users\YourName\AppData\Roaming\jupyter\kernels\.venv
  python3     C:\Users\YourName\AppData\Roaming\jupyter\kernels\python3
```

---

### 🔹 Xóa Kernel

```powershell
jupyter kernelspec uninstall .venv
```

---

### 🔹 Chọn Kernel Trong VS Code

**Bước 1:** Mở file `.ipynb`

**Bước 2:** Click **"Select Kernel"** ở góc trên bên phải

**Bước 3:** Chọn kernel:
- `Python 3.10.11` → Global Python
- `Python (.venv)` → Virtual environment

**Bước 4:** Chạy cell để test

---

### 🔹 Chọn Kernel Trong Jupyter Notebook

1. Mở notebook
2. **Kernel** → **Change Kernel**
3. Chọn kernel mong muốn

---

## 5. VS Code Configuration

### 🔧 Cài Extension

**Bắt buộc:**
- ✅ **Python** (Microsoft)
- ✅ **Jupyter** (Microsoft)
- ✅ **Pylance** (Microsoft)

**Khuyên thêm:**
- ✅ **Python Debugger**
- ✅ **autoDocstring**

---

### 🔧 Chọn Python Interpreter Trong VS Code

**Cách 1: Command Palette**
```
Ctrl + Shift + P
→ Python: Select Interpreter
→ Chọn .venv hoặc global
```

**Cách 2: Status Bar**
- Click Python version ở góc dưới bên phải
- Chọn interpreter

---

### 🔧 Settings.json

**File:** `.vscode/settings.json`

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe",
  "jupyter.kernels.filter": [
    {
      "path": "${workspaceFolder}/.venv/Scripts/python.exe",
      "type": "pythonEnvironment"
    }
  ],
  "python.terminal.activateEnvironment": true
}
```

---

## 6. Dấu Hiệu Nhận Biết

### ✅ Dấu Hiệu Đang Dùng Venv

#### **1. Terminal:**
```powershell
(.venv) PS D:\Project>
^^^^^
```

#### **2. VS Code Status Bar:**
```
🐍 Python 3.10.11 ('.venv': venv)
```

#### **3. Jupyter Notebook:**
```
Kernel: Python (.venv)
```

---

### ✅ Dấu Hiệu Đang Dùng Global

#### **1. Terminal:**
```powershell
PS D:\Project>
(Không có tên venv)
```

#### **2. VS Code Status Bar:**
```
🐍 Python 3.10.11 (global)
```

#### **3. Kiểm Tra:**
```powershell
python -c "import sys; print(sys.executable)"
```

**Kết quả:**
```
# Global:
C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe

# Venv:
D:\Project\.venv\Scripts\python.exe
```

---

## 7. Troubleshooting

### ❌ Lỗi: "python không được nhận dạng"

**Nguyên nhân:** Python chưa được thêm vào PATH

**Giải pháp:**
```powershell
# Kiểm tra PATH
echo $env:PATH

# Thêm Python vào PATH (PowerShell Admin)
$env:Path += ";C:\Users\YourName\AppData\Local\Programs\Python\Python310"
$env:Path += ";C:\Users\YourName\AppData\Local\Programs\Python\Python310\Scripts"

# Lưu vĩnh viễn
[System.Environment]::SetEnvironmentVariable("Path", $env:Path, [System.EnvironmentVariableTarget]::User)
```

---

### ❌ Lỗi: "Activate.ps1 không chạy được"

**Nguyên nhân:** PowerShell Execution Policy bị chặn

**Giải pháp:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

### ❌ Jupyter Không Thấy Kernel .venv

**Giải pháp:**
```powershell
# Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# Cài ipykernel
pip install ipykernel

# Đăng ký kernel
python -m ipykernel install --user --name=.venv --display-name "Python (.venv)"

# Khởi động lại VS Code
```

---

### ❌ Import Lỗi Trong Jupyter

**Nguyên nhân:** Kernel đang dùng Python global, không có thư viện

**Giải pháp:**
1. Chọn đúng kernel `.venv`
2. Hoặc cài thư viện trong notebook:
```python
!pip install opencv-python matplotlib
```

---

## 8. Workflow Chuẩn Cho Mỗi Project

### 📋 Checklist Setup Project Mới

```powershell
# 1. Tạo thư mục project
mkdir MyProject
cd MyProject

# 2. Tạo virtual environment
python -m venv .venv

# 3. Kích hoạt venv
.\.venv\Scripts\Activate.ps1

# 4. Cài thư viện cần thiết
pip install numpy pandas matplotlib jupyter ipykernel

# 5. Đăng ký Jupyter kernel
python -m ipykernel install --user --name=myproject --display-name "Python (MyProject)"

# 6. Tạo requirements.txt
pip freeze > requirements.txt

# 7. Mở VS Code
code .

# 8. Chọn Python Interpreter
# Ctrl+Shift+P → Python: Select Interpreter → .venv

# 9. Mở/Tạo notebook
# Chọn kernel "Python (MyProject)"
```

---

## 9. File Cấu Trúc Project Chuẩn

```
MyProject/
├── .venv/                  # Virtual environment (KHÔNG commit)
├── .vscode/
│   └── settings.json       # VS Code config
├── data/                   # Dữ liệu
├── notebooks/              # Jupyter notebooks
│   └── analysis.ipynb
├── src/                    # Source code
│   ├── __init__.py
│   └── main.py
├── tests/                  # Unit tests
├── .gitignore             # Ignore .venv, __pycache__
├── requirements.txt       # Danh sách thư viện
└── README.md              # Hướng dẫn project
```

---

## 10. Lệnh Tham Khảo Nhanh

### 📌 Python & pip

```powershell
# Kiểm tra version
python --version
pip --version

# Cài thư viện
pip install <package>

# Gỡ thư viện
pip uninstall <package>

# Xem thư viện đã cài
pip list

# Xuất danh sách thư viện
pip freeze > requirements.txt

# Cài từ requirements.txt
pip install -r requirements.txt
```

---

### 📌 Virtual Environment

```powershell
# Tạo venv
python -m venv .venv

# Kích hoạt
.\.venv\Scripts\Activate.ps1    # PowerShell
.venv\Scripts\activate.bat      # CMD
source .venv/Scripts/activate   # Git Bash

# Tắt
deactivate

# Xóa venv
rmdir /s .venv                  # CMD
rm -r .venv                     # PowerShell/Bash
```

---

### 📌 Jupyter

```powershell
# Khởi động Jupyter Notebook
jupyter notebook

# Khởi động JupyterLab
jupyter lab

# Xem kernel
jupyter kernelspec list

# Thêm kernel
python -m ipykernel install --user --name=mykernel

# Xóa kernel
jupyter kernelspec uninstall mykernel
```

---

### 📌 pyenv

```powershell
# Xem Python có thể cài
pyenv install --list

# Cài Python
pyenv install 3.10.11

# Xem Python đã cài
pyenv versions

# Set global
pyenv global 3.10.11

# Set local
pyenv local 3.10.11

# Xem Python đang dùng
pyenv version
```

---

## 11. Tips & Best Practices

### ✨ Mẹo Hay

1. **Luôn dùng Virtual Environment** cho mỗi project
2. **Commit `requirements.txt`**, KHÔNG commit `.venv/`
3. **Đặt tên venv rõ ràng:** `.venv`, `venv`, hoặc `env`
4. **Update pip thường xuyên:**
   ```powershell
   python -m pip install --upgrade pip
   ```
5. **Backup requirements trước khi update:**
   ```powershell
   pip freeze > requirements_backup.txt
   ```

---

### 🎯 Gitignore Template

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
.venv/
venv/
env/
ENV/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# IDE
.vscode/
.idea/

# Data
*.csv
*.pkl
*.npy
data/
```

---

## 📚 Tài Liệu Tham Khảo

- **Python Official:** https://docs.python.org/3/
- **pyenv-win:** https://github.com/pyenv-win/pyenv-win
- **Virtual Environments:** https://docs.python.org/3/tutorial/venv.html
- **Jupyter:** https://jupyter.org/documentation
- **VS Code Python:** https://code.visualstudio.com/docs/python/python-tutorial

---

## 💡 Tóm Tắt Nhanh

| Task | Command |
|------|---------|
| Tạo venv | `python -m venv .venv` |
| Kích hoạt venv | `.\.venv\Scripts\Activate.ps1` |
| Cài thư viện | `pip install <package>` |
| Thêm Jupyter kernel | `python -m ipykernel install --user --name=.venv` |
| Chọn kernel VS Code | `Ctrl+Shift+P` → Select Kernel |
| Kiểm tra Python đang dùng | `python -c "import sys; print(sys.executable)"` |

---

**🎉 Chúc bạn setup thành công!**
