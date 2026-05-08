import React, { useState } from 'react';

function App() {
  // Các state (trạng thái) để lưu trữ dữ liệu trên màn hình
  const [textInput, setTextInput] = useState('');
  const [files, setFiles] = useState([]);
  const [summary, setSummary] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Hàm này chạy khi người dùng bấm nút "Tóm tắt"
  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setSummary(''); // Xóa kết quả cũ trên màn hình

    // FormData giống như một cái thùng carton để đóng gói text và file gửi đi
    const formData = new FormData();

    if (textInput.trim() !== '') {
       formData.append('texts', textInput);
    }

    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      // ĐÃ SỬA THÀNH 127.0.0.1 CHO CHẮC CÚ
      const response = await fetch('http://127.0.0.1:8000/api/summarize', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setSummary(data.summary); // Hiển thị kết quả lấy được từ Backend

    } catch (error) {
      console.error("Chi tiết lỗi:", error);
      setSummary("Lỗi kết nối. Hãy chắc chắn bạn đã bật Backend (uvicorn main:app).");
    }

    setIsLoading(false);
  };

  return (
    <div style={{ maxWidth: '800px', margin: '50px auto', fontFamily: 'Arial' }}>
      <h1>Hệ Thống Tóm Tắt Đa Văn Bản</h1>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
        <div>
          <label><strong>1. Nhập văn bản trực tiếp:</strong></label>
          <textarea
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            rows="6"
            style={{ width: '100%', marginTop: '8px' }}
            placeholder="Dán văn bản cần tóm tắt vào đây..."
          />
        </div>

        <div>
          <label><strong>2. Hoặc tải lên nhiều file (TXT, PDF, Word):</strong></label><br/>
          <input
            type="file"
            multiple
            // ĐÃ MỞ KHÓA CHO PDF VÀ DOCX
            accept=".txt,.pdf,.docx"
            onChange={(e) => setFiles(e.target.files)}
            style={{ marginTop: '8px' }}
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          style={{ padding: '10px', fontSize: '16px', backgroundColor: '#007BFF', color: 'white', border: 'none', cursor: 'pointer' }}
        >
          {isLoading ? 'Đang tóm tắt...' : 'Tóm tắt ngay'}
        </button>
      </form>

      {summary && (
        <div style={{ marginTop: '30px', padding: '20px', border: '1px solid #ccc', borderRadius: '5px', backgroundColor: '#f9f9f9' }}>
          <h3>Kết quả tóm tắt:</h3>
          <p style={{ whiteSpace: 'pre-wrap', lineHeight: '1.6' }}>{summary}</p>
        </div>
      )}
    </div>
  );
}

export default App;