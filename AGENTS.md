# AGENTS.md — Quy tắc làm việc trong repo DUT_AI

Đây là repo học AI/ML của DuyToan (sinh viên DUT). Đọc kỹ quy tắc dưới đây trước khi làm bất cứ việc gì.

## 🔴 QUY TẮC TỐI QUAN TRỌNG: Folder `.code_tay/` — CODE TAY TUYỆT ĐỐI

- **KHÔNG BAO GIỜ tự viết / sửa / gợi ý code ML** trong `.code_tay/` cho user.
- Đây là nơi user **tự gõ tay từng dòng để hiểu bản chất** (xem `.code_tay/rule.txt`).
- Nếu user hỏi về code trong đây, chỉ được:
  - Giải thích **lý thuyết / công thức / toán** (để user hiểu rồi tự gõ). Giải thích bản chất gốc kèm theo cách giải thích Feynman cho user hiểu những cái trừu tượng.
  - Chỉ ra lỗi/bug và **gợi ý hướng suy nghĩ**, KHÔNG viết code sửa hộ.
  - Gợi ý tài liệu để tra cứu.
- **Được phép** viết/cập nhật file **note** trong `.code_tay/track_note/` (đây là ghi chú, không phải code bài tập):
  - `note_kien_thuc.md` — kiến thức đã học (syntax, tính chất, nhân ma trận...).
  - `note_thuat_toan.md` — các bước làm từng thuật toán (linear, logistic, CNN...).

## ✅ Các folder khác — ĐƯỢC dùng AI bình thường

- `day X <ngày>/` (các buổi học), `MobileNet/`, notebook thử nghiệm, script tiện ích... → hỗ trợ code, sửa, giải thích như bình thường.
- Được viết code, debug, refactor, tạo notebook giúp user ở những nơi này.

## 📁 Cấu trúc repo (tóm tắt)

- `day X <ngày>/` — bài học theo buổi (mỗi buổi thường có cặp `.py` + `.ipynb`).
- `.code_tay/` — **khu code tay** (xem quy tắc đỏ ở trên).
  - `01_numpy.py`, `02..`, `03_linear_regression.py`, `04_logistic_regression.py` — bài tự gõ.
  - `track_note/` — ghi chú tiến độ học.
  - `rule.txt` — tuyên ngôn "code tay, không copy AI".
- `.data/` — dataset (iris...).
- `MobileNet/` — thử nghiệm model.

## 🛠️ Môi trường

- Python 3.12.3, quản lý bằng `uv` (venv). Torch bản GPU (cu128), RTX 4060.
- Xem `PYTHON_SETUP_GUIDE.md` và `requirements.txt` để biết chi tiết.

## 🗣️ Ngôn ngữ

- User giao tiếp bằng **tiếng Việt** → trả lời bằng tiếng Việt.
