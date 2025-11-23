import os
import google.generativeai as genai
from getpass import getpass

# Nhập API Key để kiểm tra
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    api_key = getpass("Nhập Google Gemini API Key của bạn: ")
    os.environ["GOOGLE_API_KEY"] = api_key

genai.configure(api_key=api_key)

print("\nĐang kết nối tới Google để lấy danh sách model...\n")
try:
    print(f"{'Tên Model':<30} | {'Hỗ trợ generateContent?':<25}")
    print("-" * 60)
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"{m.name:<30} | {'Có':<25}")
    print("\n✅ Hãy chọn một trong các tên ở cột bên trái (bỏ chữ 'models/' đi) để điền vào file app.py")
    print("Ví dụ: nếu thấy 'models/gemini-1.5-flash', hãy dùng 'gemini-1.5-flash'")
except Exception as e:
    print(f"❌ Lỗi: {e}")