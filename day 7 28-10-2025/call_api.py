from openai import OpenAI
import os

input_path = os.path.join(os.path.dirname(__file__), "input.txt")
output_path = os.path.join(os.path.dirname(__file__), "output.txt")
with open(input_path, "r", encoding="utf-8") as f:
    question = f.read().strip()

client = OpenAI(
    api_key=os.getenv("GOOGLE_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia thuật toán,Z luyện ICPC nhiều năm."},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content

    # Ghi kết quả vào file output.txt
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Câu hỏi: {question}\n\n")
        f.write(f"Trả lời:\n{answer}")

    print("✅ Đã ghi kết quả vào output.txt")
    
except Exception as e:
    print(f"❌ Lỗi: {e}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Câu hỏi: {question}\n\n")
        f.write(f"Lỗi: {str(e)}")