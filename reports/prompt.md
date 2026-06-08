Bạn là một hệ thống đa tác nhân học thuật, có nhiệm vụ hoàn thiện một báo cáo nghiên cứu bằng tiếng Việt dựa trên các tài liệu đầu vào sau:

1. File code chính: building_footprint_final.py
2. Notebook thí nghiệm: final_building_footprint.ipynb
3. Khung báo cáo Markdown đã chuẩn hóa: khung_bao_cao_building_footprint_da_chinh_sua.md

Mục tiêu cuối cùng:
- Viết hoàn chỉnh báo cáo học thuật dài khoảng 20–25 trang.
- Chủ đề báo cáo: phân đoạn dấu chân công trình từ ảnh hàng không bằng các phương pháp thị giác máy tính truyền thống, học máy và học sâu.
- Định dạng đầu ra bắt buộc:
  1. File PDF hoàn chỉnh.
  2. File LaTeX nguồn biên dịch bằng XeLaTeX.
- Báo cáo viết bằng tiếng Việt học thuật, rõ ràng, mạch lạc, không viết theo kiểu gạch đầu dòng sơ sài.
- Không chèn mã nguồn/code vào nội dung báo cáo.
- Có thể nhắc đến thuật toán, quy trình, thư viện, mô hình, tham số, nhưng chỉ mô tả bằng văn bản học thuật, bảng hoặc sơ đồ, tuyệt đối không đưa block code Python vào báo cáo.
- Việt hóa tối đa các thuật ngữ có thể. Với thuật ngữ chuyên ngành khó dịch hoặc dễ gây hiểu nhầm, dùng dạng: “thuật ngữ tiếng Việt (English term)”.
  Ví dụ:
  - phân đoạn ngữ nghĩa (semantic segmentation)
  - dấu chân công trình (building footprint)
  - mặt nạ nhị phân (binary mask)
  - học sâu (deep learning)
  - mạng U-Net (U-Net network)
  - hệ số Dice (Dice coefficient)
  - giao trên hợp (Intersection over Union, IoU)

Vai trò các tác nhân:

Agent 1 – Tác nhân phân tích mã nguồn:
- Đọc kỹ file Python và notebook.
- Trích xuất logic thí nghiệm, không sao chép code.
- Xác định:
  - bài toán nghiên cứu;
  - dữ liệu đầu vào;
  - quy trình tiền xử lý;
  - các phương pháp sử dụng: K-Means, Otsu, SVM, U-Net;
  - hậu xử lý hình thái học;
  - các chỉ số đánh giá;
  - các file kết quả đầu ra;
  - các giới hạn của phương pháp.
- Chuyển các chi tiết triển khai thành mô tả học thuật bằng lời văn, không đưa mã nguồn.

Agent 2 – Tác nhân viết học thuật:
- Viết báo cáo theo cấu trúc trong file Markdown khung.
- Giữ bố cục hợp lý cho báo cáo 20–25 trang.
- Chương 2 “Cơ sở lý thuyết” phải thuần lý thuyết:
  - chỉ giải thích khái niệm, nguyên lý, công thức, ưu điểm, hạn chế chung;
  - không viết “trong đề tài này sử dụng...”;
  - không mô tả cấu hình code;
  - không nói “hàm trong code thực hiện...”.
- Chương 3 “Phương pháp nghiên cứu” mới trình bày cách đề tài áp dụng lý thuyết:
  - dữ liệu;
  - chia tập;
  - cắt ảnh thành patch;
  - tiền xử lý CLAHE;
  - phân đoạn bằng K-Means;
  - phân đoạn bằng Otsu;
  - phân đoạn bằng SVM;
  - phân đoạn bằng U-Net;
  - hậu xử lý;
  - đánh giá;
  - xuất kết quả.
- Chương 4 trình bày thực nghiệm và kết quả:
  - nếu có sẵn kết quả trong file CSV hoặc notebook, hãy dùng kết quả đó;
  - nếu không có kết quả thực nghiệm cụ thể, tạo bảng mẫu có ghi rõ “cần điền sau khi chạy thực nghiệm”, không bịa số liệu.
- Chương 5 nêu kết luận, hạn chế và hướng phát triển.

Agent 3 – Tác nhân kiểm duyệt nội dung:
- Kiểm tra để bảo đảm:
  - không có đoạn code Python trong báo cáo;
  - Chương 2 không lẫn nội dung triển khai cụ thể của đề tài;
  - thuật ngữ được Việt hóa nhất quán;
  - văn phong học thuật, không quá dài dòng;
  - độ dài phù hợp 20–25 trang;
  - các chương liên kết logic với nhau;
  - không bịa đặt kết quả thực nghiệm nếu dữ liệu kết quả chưa có.
- Nếu phát hiện phần nào quá dài, hãy rút gọn.
- Nếu phát hiện phần nào quá sơ sài, hãy bổ sung vừa đủ.

Agent 4 – Tác nhân định dạng LaTeX/XeLaTeX:
- Chuyển báo cáo sang LaTeX hoàn chỉnh, biên dịch được bằng XeLaTeX.
- Dùng cấu trúc báo cáo học thuật tiếng Việt.
- Thiết lập:
  - khổ giấy A4;
  - lề trái 3.0 cm, lề phải 2.0 cm, lề trên 2.5 cm, lề dưới 2.5 cm;
  - font hỗ trợ tiếng Việt, ưu tiên Times New Roman nếu có;
  - giãn dòng khoảng 1.3 hoặc 1.5;
  - đánh số chương, mục, tiểu mục rõ ràng;
  - có mục lục tự động;
  - có danh mục bảng nếu có bảng;
  - có danh mục hình nếu có hình;
  - bảng biểu căn chỉnh đẹp, không tràn trang;
  - công thức toán học trình bày chuẩn.
- Không dùng pdfLaTeX. Bắt buộc dùng XeLaTeX.
- Xuất ra:
  1. main.tex
  2. report.pdf
  3. thư mục figures nếu có hình
  4. thư mục tables nếu có bảng phụ
  5. bibliography.bib nếu có tài liệu tham khảo

Yêu cầu chi tiết về nội dung báo cáo:

# Trang bìa
copy ở báo cáo giữa kỳ

Yêu cầu văn phong:
- Tiếng Việt học thuật.
- Không dùng văn nói.
- Không viết quá dài ở phần cơ sở lý thuyết.
- Mỗi đoạn nên có luận điểm rõ ràng.
- Tránh lặp ý giữa Chương 2 và Chương 3.
- Tránh dùng quá nhiều tiếng Anh nếu có thể Việt hóa.
- Khi cần dùng tiếng Anh, đặt trong ngoặc đơn sau thuật ngữ tiếng Việt.

Yêu cầu định dạng:
- Dùng XeLaTeX.
- Báo cáo phải biên dịch thành PDF không lỗi.
- Hạn chế lỗi tiếng Việt, lỗi font, lỗi bảng tràn trang.
- Các heading cần đánh số rõ ràng đến ít nhất cấp 3; chỉ dùng cấp 4 khi thật cần.
- Không để các mục chỉ có tiêu đề mà không có nội dung.
- Không để placeholder quá nhiều. Chỉ dùng placeholder cho thông tin cá nhân, cấu hình máy hoặc kết quả chưa có.

Quy trình thực hiện:
1. Đọc toàn bộ file Markdown khung.
2. Đọc và hiểu file code/notebook.
3. Lập dàn ý cuối cùng cho báo cáo 20–30 trang.
4. Viết nội dung đầy đủ bằng tiếng Việt.
5. Kiểm tra lại Chương 2 để bảo đảm thuần lý thuyết.
6. Kiểm tra lại Chương 3 để bảo đảm mô tả đúng cách triển khai của đề tài.
7. Tạo file LaTeX dùng XeLaTeX.
8. Biên dịch thành PDF.
9. Trả về các file:
   - report.pdf
   - main.tex
   - bibliography.bib nếu có
   - figures/ nếu có
   - tables/ nếu có
10. Kèm một ghi chú ngắn cho biết phần nào cần người dùng điền thêm, ví dụ: tên trường, tên sinh viên, mã sinh viên, giảng viên hướng dẫn, kết quả thực nghiệm nếu chưa có.

Lưu ý bắt buộc:
- Không chèn code Python vào báo cáo.
- Không bịa số liệu thực nghiệm.
- Không để Chương 2 lẫn nội dung triển khai cụ thể của đề tài.
- Đầu ra cuối cùng phải là PDF biên dịch bằng XeLaTeX.