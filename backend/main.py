from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import io

# Cài thêm: pip install pypdf python-docx
from pypdf import PdfReader
import docx

# Import Class đã đóng gói từ file ai_model.py (như mình hướng dẫn ở bước trước)
from ai_model import AI_Summarizer

app = FastAPI(title="Hệ thống Tóm tắt Đa văn bản")

# ==========================================
# 1. LOAD MODEL (CHỈ CHẠY 1 LẦN KHI KHỞI ĐỘNG)
# ==========================================
print("Đang tải AI Model... Vui lòng đợi...")
engine = AI_Summarizer(
    model_path="my_model/best_extractive_sentence_model.bin",
    tokenizer_path="my_model/tokenizer",
    use_mock=False
)
print("Tải Model thành công! Server đã sẵn sàng.")

# ==========================================
# 2. CẤU HÌNH CORS CHO REACT
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 3. API XỬ LÝ CHÍNH
# ==========================================
@app.post("/api/summarize")
async def handle_summarization(
        texts: List[str] = Form(default=[]),
        files: List[UploadFile] = File(default=[])
):
    combined_text = ""

    # 1. Gom text nhập tay
    for t in texts:
        if t.strip():
            combined_text += t + "\n"

    # 2. Xử lý an toàn file PDF và DOCX từ bộ nhớ (BytesIO)
    for idx, file in enumerate(files):
        # Đọc file nhị phân
        content = await file.read()

        try:
            if file.filename.endswith('.pdf'):
                pdf_reader = PdfReader(io.BytesIO(content))
                for page in pdf_reader.pages:
                    combined_text += page.extract_text() + " "

            elif file.filename.endswith('.docx'):
                doc = docx.Document(io.BytesIO(content))
                for para in doc.paragraphs:
                    combined_text += para.text + " "

            elif file.filename.endswith('.txt'):
                combined_text += content.decode('utf-8') + " "

        except Exception as e:
            # Báo lỗi về Frontend nếu file bị hỏng
            raise HTTPException(status_code=400, detail=f"Không thể đọc file {file.filename}. Lỗi: {str(e)}")

    if not combined_text.strip():
        raise HTTPException(status_code=400, detail="Bạn chưa nhập nội dung hoặc file trống!")

    try:
        # 3. Chạy qua AI Engine đã load sẵn trên RAM
        result = engine.process(combined_text)
        return {"summary": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi AI Engine: {str(e)}")