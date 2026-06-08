# Trích xuất vùng tòa nhà từ ảnh chụp từ trên cao (Building Footprint Extraction)

Dự án này xây dựng một quy trình (pipeline) phân đoạn dấu chân công trình từ ảnh hàng không. Việc tự động hóa nhận diện vùng công trình nhằm mục đích hỗ trợ cập nhật bản đồ đô thị, ước lượng mật độ xây dựng và theo dõi biến động cho Hệ thống thông tin địa lý (GIS). 

Bài toán được mô hình hóa dưới dạng phân đoạn nhị phân (binary segmentation) ở mức điểm ảnh, trong đó đầu vào là các patch ảnh RGB và đầu ra là mặt nạ (mask) phân định vùng công trình và nền.

## 📊 Tập dữ liệu (Dataset)
* **Nguồn dữ liệu:** Sử dụng bộ dữ liệu chuẩn **Inria Aerial Image Labeling Dataset** bao gồm ảnh RGB và mặt nạ nhị phân ground truth.
* **Tiền xử lý:** Ảnh được chia theo tỷ lệ 70% huấn luyện (Train), 15% kiểm chứng (Validation) và 15% kiểm thử (Test) ở mức ảnh gốc để tránh rò rỉ dữ liệu.
* **Kích thước đầu vào:** Dữ liệu được cắt thành các patch có kích thước 512x512 pixel.
* **Tăng cường ảnh:** Áp dụng kỹ thuật CLAHE trên kênh sáng LAB để làm rõ biên và các vùng có độ tương phản thấp.

## 🧠 Các phương pháp tiếp cận
Dự án triển khai và so sánh 4 phương pháp từ cổ điển đến học sâu trên cùng một điều kiện thực nghiệm:
1.  **K-Means:** Phương pháp học không giám sát, phân cụm điểm ảnh dựa trên đặc trưng màu RGB và HSV (K=4).
2.  **Otsu Thresholding:** Phương pháp phân ngưỡng tự động nhằm tối đa hóa phương sai giữa 2 lớp trên kênh độ bão hòa (Saturation).
3.  **Linear SVM:** Mô hình học có giám sát, phân loại từng điểm ảnh dựa trên các vector đặc trưng thủ công bao gồm màu sắc, độ sáng, gradient Sobel và thống kê cục bộ.
4.  **U-Net:** Mạng nơ-ron tích chập (CNN) với kiến trúc Encoder-Decoder kết hợp Skip Connections để học ngữ cảnh không gian và hình dạng công trình.

*Tất cả các phương pháp đều đi qua bước hậu xử lý hình thái học (Morphological Closing/Opening) và lọc thành phần liên thông để loại bỏ nhiễu.*

## 🏆 Kết quả thực nghiệm
Đánh giá định lượng trên 300 patch kiểm thử (Test set) cho thấy mô hình **U-Net** đạt hiệu suất vượt trội nhất trong các mô hình khảo sát:

| Phương pháp | IoU | Dice | Precision | Recall | F1-Score | Area Error (%) | Thời gian (s/patch) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K-Means** | 0.230 | 0.347 | 0.347 | 0.469 | 0.347 | 20.093 | 0.0828 |
| **Otsu** | 0.223 | 0.333 | 0.233 | 0.802 | 0.333 | 64.691 | 0.0075 |
| **SVM** | 0.295 | 0.416 | 0.315 | 0.774 | 0.416 | 34.436 | 0.0565 |
| **U-Net** | **0.544** | **0.665** | **0.698** | 0.673 | **0.665** | **5.258** | 0.0396 |

*Bảng dữ liệu trích xuất từ Báo cáo*

**Nhận xét:**
* U-Net cải thiện rõ rệt mức độ chồng lấp không gian và giảm thiểu đáng kể sai số diện tích dự đoán (Area Error chỉ ở mức 5.258).
* Các phương pháp truyền thống (K-Means, Otsu) rất nhanh nhưng gặp khó khăn lớn trong việc phân biệt mái nhà với đường hoặc sân bê tông do chỉ dựa vào đặc trưng màu sắc.
* SVM tốt hơn nhóm baseline nhưng vẫn đưa ra nhiều dự đoán thừa (False Positives) do giới hạn của đặc trưng cục bộ và ranh giới tuyến tính.

## 📂 Cấu trúc thư mục (Repository Structure)
```text
computer_vision/
│
├── src/
│   ├── data/                 # Thư mục chứa tập dữ liệu (ảnh RGB và mask)
│   ├── model/                # Thư mục lưu trữ các checkpoint mô hình (ví dụ: U-Net weights)
│   └── final-cv.ipynb        # Jupyter Notebook chứa toàn bộ mã nguồn: Tiền xử lý, Huấn luyện, Đánh giá
│
└── README.md                 # Tài liệu mô tả dự án (File này)