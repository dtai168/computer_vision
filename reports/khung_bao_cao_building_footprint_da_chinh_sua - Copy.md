# Khung báo cáo: Phân đoạn dấu chân công trình từ ảnh hàng không

> **Định hướng dung lượng:** 20–25 trang  
> **Mức độ chi tiết:** Hàn lâm, đủ đến cấp H4  
> **Căn cứ xây dựng:** mã nguồn `building_footprint_final.py` và notebook `final_building_footprint.ipynb`  
> **Trọng tâm:** pipeline thực nghiệm, so sánh K-Means, Otsu, SVM và U-Net cho bài toán phân đoạn dấu chân công trình.

---

# Gợi ý phân bổ số trang

| Phần | Nội dung | Số trang gợi ý |
|---|---|---:|
| Tóm tắt | Bối cảnh, mục tiêu, phương pháp, kết quả chính | 0.5–1 |
| Chương 1 | Giới thiệu đề tài | 2–3 |
| Chương 2 | Cơ sở lý thuyết | 4–5 |
| Chương 3 | Phương pháp nghiên cứu | 6–7 |
| Chương 4 | Thực nghiệm và kết quả | 6–8 |
| Chương 5 | Kết luận và hướng phát triển | 2–3 |
| Tài liệu tham khảo, phụ lục | Bảng cấu hình, hình ảnh, kết quả bổ sung | 1–2 |

---

# Tóm tắt

## Bối cảnh nghiên cứu

### Bài toán trích xuất dấu chân công trình

#### Ý nghĩa thực tiễn
Trình bày ngắn gọn vai trò của việc tự động trích xuất dấu chân công trình trong quy hoạch đô thị, cập nhật bản đồ, quản lý hạ tầng và phân tích mật độ xây dựng.

### Đặc thù của ảnh hàng không

#### Khó khăn chính
Nêu các thách thức như mái nhà đa dạng màu sắc, bóng đổ, cây xanh che khuất, đường sá hoặc sân bê tông có màu tương tự công trình.

## Mục tiêu nghiên cứu

### Mục tiêu tổng quát

#### Xây dựng pipeline phân đoạn công trình
Đề tài xây dựng quy trình từ nạp dữ liệu, tiền xử lý, phân đoạn, hậu xử lý đến đánh giá kết quả.

### Mục tiêu cụ thể

#### So sánh nhiều nhóm phương pháp
So sánh các phương pháp truyền thống, học máy cổ điển và học sâu gồm K-Means, Otsu, SVM và U-Net.

#### Đánh giá bằng nhiều chỉ số
Sử dụng các chỉ số IoU, Dice, Precision, Recall, F1, sai số số lượng công trình và sai số diện tích.

## Kết quả kỳ vọng

### Xác định phương pháp hiệu quả nhất

#### Tiêu chí chính
Mô hình được đánh giá chủ yếu dựa trên Dice và IoU, đồng thời xét thêm Precision, Recall, thời gian xử lý và lỗi định tính.

---

# Chương 1. Giới thiệu

## 1.1. Bối cảnh đề tài

### 1.1.1. Ảnh hàng không và dữ liệu đô thị

#### Đặc điểm dữ liệu
Ảnh hàng không cung cấp góc nhìn từ trên xuống, có khả năng thể hiện rõ hình dạng mái nhà, đường sá, cây xanh và các đối tượng đô thị khác.

#### Giá trị trong phân tích không gian
Dữ liệu này có thể hỗ trợ cập nhật bản đồ, theo dõi biến động xây dựng và phục vụ các hệ thống GIS.

### 1.1.2. Bài toán phân đoạn dấu chân công trình

#### Định nghĩa bài toán
Phân đoạn dấu chân công trình là bài toán xác định vùng pixel thuộc công trình trên ảnh hàng không.

#### Dạng bài toán trong nghiên cứu
Trong báo cáo này, bài toán được xem là phân đoạn nhị phân gồm hai lớp: công trình và nền.

## 1.2. Lý do chọn đề tài

### 1.2.1. Lý do thực tiễn

#### Nhu cầu tự động hóa
Việc số hóa công trình thủ công tốn nhiều thời gian và khó mở rộng trên khu vực rộng lớn.

#### Ứng dụng thực tế
Kết quả phân đoạn có thể hỗ trợ quy hoạch đô thị, đánh giá mật độ xây dựng và cập nhật bản đồ nền.

### 1.2.2. Lý do khoa học

#### So sánh các hướng tiếp cận
Đề tài có ý nghĩa vì đặt các phương pháp truyền thống, học máy và học sâu trong cùng một pipeline đánh giá.

#### Đánh giá đa chiều
Ngoài chỉ số pixel-level, báo cáo còn xét lỗi số lượng công trình và lỗi diện tích, giúp đánh giá gần hơn với nhu cầu ứng dụng.

## 1.3. Mục tiêu nghiên cứu

### 1.3.1. Mục tiêu tổng quát

#### Thiết kế quy trình thực nghiệm hoàn chỉnh
Xây dựng hệ thống có khả năng đọc ảnh hàng không, dự đoán mặt nạ công trình và đánh giá chất lượng dự đoán.

### 1.3.2. Mục tiêu cụ thể

#### Tiền xử lý dữ liệu ảnh
Áp dụng CLAHE để cải thiện tương phản trước khi đưa ảnh vào các thuật toán phân đoạn.

#### Xây dựng các mô hình so sánh
Triển khai K-Means, Otsu, SVM và U-Net trên cùng tập dữ liệu.

#### Đánh giá kết quả
So sánh các mô hình bằng chỉ số định lượng và kết quả trực quan.

## 1.4. Phạm vi nghiên cứu

### 1.4.1. Phạm vi dữ liệu

#### Ảnh và mặt nạ nhị phân
Dữ liệu gồm ảnh RGB và mặt nạ ground truth tương ứng.

#### Làm việc trên patch ảnh
Ảnh được chia thành các patch kích thước 512 × 512 pixel.

### 1.4.2. Phạm vi phương pháp

#### Các phương pháp được khảo sát
Báo cáo tập trung vào bốn hướng: K-Means, Otsu, Linear SVM và U-Net.

#### Giới hạn nghiên cứu
Nghiên cứu dừng ở mặt nạ raster, chưa thực hiện chuyển đổi kết quả sang polygon GIS.

## 1.5. Cấu trúc báo cáo

### 1.5.1. Chương 1

#### Nội dung
Giới thiệu bối cảnh, lý do chọn đề tài, mục tiêu và phạm vi nghiên cứu.

### 1.5.2. Chương 2

#### Nội dung
Trình bày cơ sở lý thuyết về phân đoạn ảnh, tiền xử lý, các thuật toán và chỉ số đánh giá.

### 1.5.3. Chương 3

#### Nội dung
Mô tả phương pháp nghiên cứu và pipeline thực nghiệm.

### 1.5.4. Chương 4

#### Nội dung
Trình bày kết quả thực nghiệm, so sánh mô hình và phân tích lỗi.

### 1.5.5. Chương 5

#### Nội dung
Tổng kết đóng góp, hạn chế và hướng phát triển.

---

# Chương 2. Cơ sở lý thuyết
Hãy viết thật cô đọng phần K-Means, Otsu và SVM. Không cần đi sâu vào chứng minh toán học của SVM hay các công thức hình thái học cơ bản.

## 2.1. Phân đoạn ảnh trong viễn thám

### 2.1.1. Khái niệm phân đoạn ảnh

#### Phân đoạn ảnh
Phân đoạn ảnh là quá trình chia ảnh thành các vùng hoặc nhóm pixel có cùng đặc điểm, nhằm làm nổi bật các đối tượng quan tâm trong ảnh.

#### Phân đoạn ngữ nghĩa
Phân đoạn ngữ nghĩa gán nhãn lớp cho từng pixel. Khác với phân loại ảnh chỉ đưa ra một nhãn cho toàn ảnh, phân đoạn ngữ nghĩa tạo ra bản đồ nhãn có cùng kích thước không gian với ảnh đầu vào.

### 2.1.2. Phân đoạn ảnh viễn thám

#### Đặc điểm dữ liệu viễn thám
Ảnh viễn thám và ảnh hàng không thường có góc nhìn từ trên xuống, chứa nhiều đối tượng đô thị như công trình, đường giao thông, cây xanh, mặt nước và đất trống.

#### Khó khăn trong phân đoạn
Các đối tượng trong ảnh có thể bị ảnh hưởng bởi bóng đổ, thay đổi ánh sáng, vật liệu bề mặt, độ phân giải không gian và sự che khuất giữa các lớp đối tượng.

### 2.1.3. Phân đoạn nhị phân

#### Khái niệm
Phân đoạn nhị phân là trường hợp đặc biệt của phân đoạn ảnh, trong đó mỗi pixel được phân vào một trong hai lớp: đối tượng quan tâm hoặc nền.

#### Ý nghĩa
Cách biểu diễn này phù hợp với các bài toán cần tách riêng một loại đối tượng cụ thể khỏi phần còn lại của ảnh.

## 2.2. Bài toán trích xuất dấu chân công trình

### 2.2.1. Khái niệm dấu chân công trình

#### Định nghĩa
Dấu chân công trình là vùng chiếm dụng mặt bằng của một công trình khi quan sát từ trên cao, thường được biểu diễn dưới dạng mask raster hoặc polygon vector.

#### Biểu diễn trong ảnh
Trong ảnh hàng không, dấu chân công trình thường tương ứng với vùng mái hoặc vùng bao phủ của tòa nhà, tùy theo cách xây dựng nhãn dữ liệu.

### 2.2.2. Đặc điểm hình học và phổ ảnh

#### Đặc điểm hình học
Công trình thường có biên tương đối rõ, hình dạng có xu hướng góc cạnh, nhưng vẫn có sự đa dạng lớn về kích thước và cấu trúc.

#### Đặc điểm phổ ảnh
Màu sắc và độ sáng của công trình phụ thuộc vào vật liệu mái, điều kiện chiếu sáng và chất lượng cảm biến ảnh.

### 2.2.3. Các nguồn gây sai số

#### Sai số do nhầm lẫn đối tượng
Đường bê tông, bãi đỗ xe, sân thượng hoặc khu vực sáng màu có thể có đặc trưng gần giống công trình.

#### Sai số do che khuất
Cây xanh, bóng đổ hoặc vật thể cao có thể làm mất một phần thông tin về biên và bề mặt công trình.

## 2.3. Không gian màu và tăng cường ảnh

### 2.3.1. Không gian màu RGB

#### Khái niệm
RGB biểu diễn màu sắc thông qua ba kênh đỏ, lục và lam. Đây là không gian màu phổ biến trong lưu trữ và hiển thị ảnh số.

#### Đặc điểm
RGB trực quan và dễ xử lý, nhưng thông tin màu và độ sáng thường bị trộn lẫn, khiến một số thao tác tăng cường ảnh khó kiểm soát.

### 2.3.2. Không gian màu HSV

#### Khái niệm
HSV biểu diễn màu theo sắc độ, độ bão hòa và độ sáng.

#### Ý nghĩa xử lý ảnh
Việc tách độ bão hòa và độ sáng khỏi sắc độ giúp thuận tiện hơn trong một số bài toán phân ngưỡng, phân cụm hoặc phân tích màu sắc.

### 2.3.3. Không gian màu LAB

#### Khái niệm
LAB tách thông tin độ sáng khỏi hai thành phần màu đối lập.

#### Ý nghĩa xử lý ảnh
Do kênh sáng được tách riêng, LAB thường được sử dụng trong các kỹ thuật tăng cường tương phản mà không làm biến đổi quá mạnh thông tin màu.

### 2.3.4. CLAHE

#### Nguyên lý
CLAHE là kỹ thuật cân bằng histogram thích nghi có giới hạn tương phản. Ảnh được chia thành các vùng nhỏ, sau đó histogram cục bộ được điều chỉnh để cải thiện độ tương phản.

#### Ưu điểm
CLAHE giúp làm nổi bật chi tiết cục bộ và hạn chế hiện tượng khuếch đại nhiễu quá mức so với cân bằng histogram toàn cục.

## 2.4. Phân ngưỡng ảnh và phương pháp Otsu

### 2.4.1. Phân ngưỡng ảnh

#### Khái niệm
Phân ngưỡng là kỹ thuật chuyển ảnh xám hoặc một kênh đặc trưng thành ảnh nhị phân dựa trên một giá trị ngưỡng.

#### Mục tiêu
Mục tiêu của phân ngưỡng là tách các pixel thuộc đối tượng quan tâm khỏi nền khi hai nhóm có phân bố giá trị khác biệt.

### 2.4.2. Phương pháp Otsu

#### Nguyên lý
Otsu tự động tìm ngưỡng sao cho độ phân tách giữa hai nhóm pixel là lớn nhất, thường được mô tả thông qua việc tối đa hóa phương sai giữa lớp.

#### Điều kiện hiệu quả
Phương pháp hoạt động tốt khi histogram có xu hướng tách thành hai nhóm tương đối rõ ràng.

### 2.4.3. Ưu điểm và hạn chế

#### Ưu điểm
Otsu đơn giản, không cần dữ liệu huấn luyện và có chi phí tính toán thấp.

#### Hạn chế
Phương pháp không khai thác thông tin hình dạng hay ngữ cảnh không gian, do đó dễ bị ảnh hưởng bởi ánh sáng, bóng đổ và nhiễu nền.

## 2.5. Phân cụm K-Means

### 2.5.1. Khái niệm phân cụm

#### Học không giám sát
Phân cụm là nhóm các điểm dữ liệu dựa trên mức độ tương đồng mà không cần nhãn lớp có sẵn.

#### Không gian đặc trưng
Mỗi điểm dữ liệu được biểu diễn bằng một vector đặc trưng; khoảng cách giữa các vector phản ánh mức độ giống nhau.

### 2.5.2. Nguyên lý K-Means

#### Tâm cụm
K-Means tìm K tâm cụm sao cho tổng khoảng cách từ các điểm đến tâm cụm tương ứng là nhỏ nhất.

#### Quy trình lặp
Thuật toán luân phiên giữa bước gán điểm dữ liệu vào cụm gần nhất và bước cập nhật lại tâm cụm.

### 2.5.3. Ưu điểm và hạn chế

#### Ưu điểm
K-Means dễ triển khai, trực quan và có thể dùng làm baseline cho các bài toán phân đoạn dựa trên đặc trưng màu hoặc cường độ.

#### Hạn chế
Kết quả phụ thuộc vào số cụm, cách khởi tạo, thang đo đặc trưng và giả định rằng các cụm có thể tách biệt tốt trong không gian đặc trưng.

## 2.6. Support Vector Machine

### 2.6.1. Khái niệm SVM

#### Học có giám sát
Support Vector Machine là mô hình học có giám sát dùng cho phân loại hoặc hồi quy, trong đó mô hình học ranh giới phân tách giữa các lớp từ dữ liệu đã gán nhãn.

#### Siêu phẳng phân tách
Trong bài toán phân loại tuyến tính, SVM tìm siêu phẳng sao cho khoảng cách lề giữa các mẫu gần ranh giới và siêu phẳng là lớn nhất.

### 2.6.2. Biên mềm và tham số C

#### Biên mềm
Trong dữ liệu thực tế, hai lớp thường không tách biệt hoàn hảo. SVM biên mềm cho phép một số điểm bị phân loại sai để tăng khả năng tổng quát hóa.

#### Tham số C
Tham số C điều khiển mức phạt đối với lỗi phân loại. C lớn làm mô hình cố gắng giảm lỗi huấn luyện, trong khi C nhỏ cho phép biên mềm hơn.

### 2.6.3. SVM trong phân loại pixel

#### Biểu diễn pixel
Trong phân đoạn ảnh, mỗi pixel có thể được xem như một mẫu dữ liệu với vector đặc trưng riêng.

#### Hạn chế về ngữ cảnh
SVM phân loại dựa trên đặc trưng được cung cấp, nên khả năng khai thác hình dạng và quan hệ không gian thường hạn chế hơn các mô hình học sâu tích chập.

## 2.7. Học sâu cho phân đoạn ảnh

### 2.7.1. Mạng nơ-ron tích chập

#### Convolution
Lớp tích chập học các bộ lọc để phát hiện đặc trưng cục bộ như cạnh, góc, kết cấu và mẫu hình phức tạp hơn ở các tầng sâu.

#### Tính phân cấp đặc trưng
Các tầng đầu thường học đặc trưng đơn giản, trong khi các tầng sau học đặc trưng có tính ngữ nghĩa cao hơn.

### 2.7.2. Kiến trúc encoder-decoder

#### Encoder
Encoder giảm dần kích thước không gian để học đặc trưng ngữ cảnh và biểu diễn trừu tượng hơn.

#### Decoder
Decoder khôi phục kích thước không gian để tạo bản đồ phân đoạn tương ứng với ảnh đầu vào.

### 2.7.3. U-Net

#### Đặc điểm chính
U-Net là kiến trúc encoder-decoder có các kết nối tắt giữa hai nhánh, giúp kết hợp thông tin ngữ nghĩa cấp cao với chi tiết không gian cấp thấp.

#### Ý nghĩa của skip connection
Skip connection hỗ trợ phục hồi biên đối tượng và giảm mất mát thông tin vị trí trong quá trình downsampling.

### 2.7.4. Hàm mất mát trong phân đoạn

#### Binary cross-entropy
Binary cross-entropy thường được dùng cho phân đoạn nhị phân để đo sai khác giữa xác suất dự đoán và nhãn thực.

#### Dice loss
Dice loss tập trung vào mức độ chồng lấp giữa vùng dự đoán và vùng thật, hữu ích khi dữ liệu mất cân bằng giữa đối tượng và nền.

## 2.8. Xử lý hình thái học

### 2.8.1. Khái niệm

#### Morphology
Xử lý hình thái học là nhóm thao tác xử lý ảnh nhị phân hoặc ảnh xám dựa trên hình dạng và cấu trúc của đối tượng.

#### Phần tử cấu trúc
Các phép toán morphology sử dụng một phần tử cấu trúc để quét qua ảnh và biến đổi hình dạng vùng ảnh.

### 2.8.2. Closing và opening

#### Closing
Closing thường giúp lấp lỗ nhỏ và nối các vùng gần nhau.

#### Opening
Opening thường giúp loại bỏ nhiễu nhỏ hoặc các vùng rời rạc không mong muốn.

### 2.8.3. Thành phần liên thông

#### Khái niệm
Thành phần liên thông là tập các pixel thuộc cùng một vùng và có quan hệ láng giềng với nhau.

#### Ý nghĩa
Phân tích thành phần liên thông cho phép đếm đối tượng, lọc vùng nhỏ và đánh giá cấu trúc của mặt nạ nhị phân.

## 2.9. Chỉ số đánh giá phân đoạn

### 2.9.1. IoU

#### Ý nghĩa
IoU đo tỷ lệ giữa phần giao và phần hợp của vùng dự đoán với vùng ground truth. Chỉ số càng cao thể hiện mức độ chồng lấp càng tốt.

### 2.9.2. Dice coefficient

#### Ý nghĩa
Dice coefficient cũng đo mức độ chồng lấp giữa hai vùng, thường được sử dụng rộng rãi trong các bài toán phân đoạn khi đối tượng chiếm tỷ lệ nhỏ.

### 2.9.3. Precision và Recall

#### Precision
Precision cho biết trong số các pixel được dự đoán là đối tượng, có bao nhiêu pixel thực sự đúng.

#### Recall
Recall cho biết trong số các pixel đối tượng thực tế, có bao nhiêu pixel được mô hình phát hiện.

### 2.9.4. F1-score

#### Ý nghĩa
F1-score là trung bình điều hòa giữa Precision và Recall, phản ánh sự cân bằng giữa dự đoán đúng và khả năng phát hiện đầy đủ.

### 2.9.5. Sai số số lượng và sai số diện tích

#### Sai số số lượng
Sai số số lượng phản ánh chênh lệch giữa số đối tượng dự đoán và số đối tượng thực tế.

#### Sai số diện tích
Sai số diện tích phản ánh mức độ dự đoán thừa hoặc thiếu về tổng diện tích đối tượng.

---

# Chương 3. Phương pháp nghiên cứu

## 3.1. Tổng quan phương pháp đề xuất

### 3.1.1. Mục tiêu của quy trình thực nghiệm

#### Đầu vào
Đầu vào của quy trình là ảnh hàng không RGB và mặt nạ ground truth tương ứng cho vùng công trình.

#### Đầu ra
Đầu ra là mặt nạ dự đoán công trình, bảng chỉ số đánh giá và các hình ảnh minh họa kết quả.

### 3.1.2. Pipeline tổng thể

#### Các bước chính
Quy trình gồm: xác định dữ liệu, chia tập ảnh, tạo patch, tiền xử lý ảnh, áp dụng mô hình phân đoạn, hậu xử lý mặt nạ, đánh giá định lượng và xuất kết quả.

#### Ý nghĩa thiết kế
Việc đặt tất cả phương pháp trong cùng một pipeline giúp so sánh công bằng giữa các thuật toán truyền thống, học máy cổ điển và học sâu.

### 3.1.3. Các nhánh thực nghiệm

#### Nhánh phương pháp truyền thống
Gồm K-Means và Otsu, đóng vai trò baseline không học sâu.

#### Nhánh học máy cổ điển
Sử dụng Linear SVM trên đặc trưng pixel thủ công.

#### Nhánh học sâu
Sử dụng U-Net cho bài toán phân đoạn nhị phân công trình/nền.

## 3.2. Chuẩn bị dữ liệu

### 3.2.1. Cấu trúc dữ liệu đầu vào

#### Thư mục ảnh
Ảnh đầu vào được đặt trong thư mục `train/images`.

#### Thư mục nhãn
Mặt nạ ground truth được đặt trong thư mục `train/gt`.

### 3.2.2. Xác định đường dẫn dữ liệu

#### Cơ chế tìm dữ liệu
Chương trình hỗ trợ truyền trực tiếp đường dẫn dữ liệu, đọc từ biến môi trường hoặc tìm trong các thư mục ứng viên được cấu hình sẵn.

#### Mục đích
Cơ chế này giúp cùng một mã nguồn có thể chạy trên nhiều môi trường khác nhau như máy cá nhân hoặc Kaggle.

### 3.2.3. Ghép cặp ảnh và mặt nạ

#### Quy tắc ghép cặp
Mỗi ảnh được ghép với mặt nạ có cùng tên file hoặc cùng phần tên chính nhưng khác phần mở rộng.

#### Kiểm tra hợp lệ
Những cặp ảnh-mask không đọc được hoặc không cùng kích thước sẽ bị xem là không hợp lệ.

## 3.3. Chia tập dữ liệu và tạo manifest

### 3.3.1. Tỷ lệ chia dữ liệu

#### Train set
Tập huấn luyện chiếm 70% số ảnh.

#### Validation set
Tập validation chiếm 15% số ảnh.

#### Test set
Tập kiểm tra chiếm 15% số ảnh.

### 3.3.2. Tính tái lập

#### Random seed
Quá trình xáo trộn và lấy mẫu dùng random seed cố định để tăng khả năng tái lập kết quả.

#### Lưu manifest
Danh sách ảnh và patch sau khi chia được lưu lại, giúp kiểm tra và tái sử dụng cấu hình thực nghiệm.

### 3.3.3. Tránh rò rỉ dữ liệu

#### Chia theo ảnh gốc
Ảnh gốc được chia trước khi tạo patch, giúp tránh tình trạng patch từ cùng một ảnh xuất hiện đồng thời trong train và test.

#### Kiểm tra disjoint
Quy trình có bước kiểm tra để bảo đảm một ảnh không xuất hiện ở nhiều tập khác nhau.

## 3.4. Tạo patch ảnh

### 3.4.1. Kích thước patch

#### Giá trị sử dụng
Mỗi ảnh được chia thành các patch kích thước 512 × 512 pixel.

#### Lý do sử dụng patch
Cách tiếp cận theo patch giúp giảm yêu cầu bộ nhớ và tạo ra nhiều mẫu huấn luyện/đánh giá hơn từ ảnh gốc kích thước lớn.

### 3.4.2. Bước trượt

#### Giá trị sử dụng
Stride được đặt bằng 512, do đó các patch không chồng lấn.

#### Ảnh hưởng
Patch không chồng lấn giúp giảm thời gian xử lý, nhưng có thể bỏ qua một số lợi ích của dự đoán chồng lấn ở vùng biên patch.

### 3.4.3. Giới hạn số lượng patch

#### Mục đích
Giới hạn số patch giúp kiểm soát chi phí tính toán khi huấn luyện và đánh giá.

#### Giá trị sử dụng
Số patch tối đa lần lượt là 1200 cho train, 300 cho validation và 300 cho test.

## 3.5. Tiền xử lý ảnh trong quy trình thực nghiệm

### 3.5.1. Chuyển không gian màu

#### RGB sang LAB
Ảnh RGB được chuyển sang không gian LAB để tách kênh độ sáng khỏi thông tin màu.

#### Lý do
Việc xử lý riêng kênh sáng giúp cải thiện tương phản trong khi hạn chế làm sai lệch mạnh màu sắc tổng thể.

### 3.5.2. Áp dụng CLAHE

#### Thông số chính
CLAHE sử dụng clip limit và kích thước tile grid được khai báo trong cấu hình thí nghiệm.

#### Vai trò trong pipeline
CLAHE được dùng như bước tăng cường ảnh trước khi đưa ảnh vào các phương pháp phân đoạn.

### 3.5.3. Ảnh sau tiền xử lý

#### Chuyển về RGB
Sau khi tăng cường kênh sáng, ảnh được chuyển lại RGB để sử dụng thống nhất cho các nhánh mô hình.

#### Tính nhất quán
Việc dùng cùng một bước tiền xử lý cho các phương pháp giúp giảm sai lệch khi so sánh kết quả.

## 3.6. Phương pháp K-Means trong thực nghiệm

### 3.6.1. Đặc trưng đầu vào

#### RGB và HSV
Mỗi pixel được biểu diễn bằng đặc trưng kết hợp từ RGB và HSV đã chuẩn hóa.

#### Mục đích
Kết hợp hai không gian màu giúp mô hình phân cụm khai thác cả thông tin màu gốc và các thành phần sắc độ/độ bão hòa/độ sáng.

### 3.6.2. Thiết lập phân cụm

#### Số cụm
Số cụm K được đặt là 4.

#### Tiêu chí lặp
Thuật toán sử dụng tiêu chí dừng dựa trên số vòng lặp tối đa và sai số hội tụ.

### 3.6.3. Tạo mặt nạ dự đoán

#### Chọn cụm công trình
Sau khi phân cụm, một cụm được chọn làm cụm đại diện cho vùng công trình dựa trên đặc trưng của tâm cụm.

#### Nhị phân hóa
Các pixel thuộc cụm được chọn được gán là công trình, các pixel còn lại được gán là nền.

## 3.7. Phương pháp Otsu trong thực nghiệm

### 3.7.1. Kênh xử lý

#### HSV saturation
Ảnh sau tiền xử lý được chuyển sang HSV và lấy kênh saturation làm đầu vào cho phân ngưỡng.

#### Tăng cường cục bộ
Kênh saturation được tăng cường bằng CLAHE trước khi áp dụng Otsu.

### 3.7.2. Phân ngưỡng nhị phân

#### Otsu thresholding
Ngưỡng được xác định tự động từ histogram của kênh xử lý.

#### Dạng ngưỡng
Quy trình sử dụng dạng nhị phân đảo để tạo mặt nạ công trình/nền.

### 3.7.3. Đặc điểm vai trò

#### Baseline đơn giản
Otsu được dùng làm phương pháp baseline nhanh, giúp tạo mốc so sánh cho các phương pháp phức tạp hơn.

## 3.8. Phương pháp SVM trong thực nghiệm

### 3.8.1. Trích xuất đặc trưng pixel

#### Đặc trưng màu
Mỗi pixel được mô tả bằng các thành phần từ RGB, LAB và HSV.

#### Đặc trưng biên
Gradient Sobel được dùng để bổ sung thông tin về biên và thay đổi cường độ cục bộ.

#### Đặc trưng thống kê cục bộ
Trung bình cục bộ và độ lệch chuẩn cục bộ được tính để mô tả bối cảnh lân cận quanh pixel.

### 3.8.2. Lấy mẫu dữ liệu huấn luyện

#### Lấy mẫu theo patch
Từ mỗi patch huấn luyện, một số lượng pixel nhất định được lấy mẫu để xây dựng ma trận đặc trưng.

#### Giảm mất cân bằng lớp
Quy trình ưu tiên lấy cả pixel công trình và pixel nền khi có đủ dữ liệu, nhằm hạn chế tình trạng mô hình thiên lệch về lớp nền.

### 3.8.3. Huấn luyện Linear SVM

#### Chuẩn hóa đặc trưng
Trước khi huấn luyện, các đặc trưng được chuẩn hóa bằng StandardScaler.

#### Mô hình phân loại
Mô hình sử dụng LinearSVC với class weight cân bằng để xử lý mất cân bằng lớp.

### 3.8.4. Tuning tham số C

#### Grid search
Các giá trị C được thử gồm 0.1, 0.5 và 1.0.

#### Tiêu chí lựa chọn
Mô hình có Dice trung bình cao nhất trên validation set được chọn làm mô hình SVM cuối cùng.

### 3.8.5. Dự đoán trên patch test

#### Phân loại từng pixel
Đặc trưng của toàn bộ pixel trong patch test được trích xuất và đưa vào SVM để sinh nhãn pixel.

#### Tạo mask
Nhãn pixel được chuyển thành mặt nạ nhị phân rồi đưa qua bước hậu xử lý.

## 3.9. Phương pháp U-Net trong thực nghiệm

### 3.9.1. Chuẩn bị dữ liệu cho U-Net

#### Dataset theo patch
Mỗi mẫu dữ liệu gồm một patch ảnh và mặt nạ ground truth tương ứng.

#### Chuẩn hóa giá trị pixel
Giá trị ảnh được chuẩn hóa về khoảng [0, 1] trước khi đưa vào mạng.

### 3.9.2. Tăng cường dữ liệu

#### Biến đổi hình học
Quy trình sử dụng lật ngang, lật dọc và xoay theo bội số của 90 độ.

#### Biến đổi cường độ
Độ sáng và độ tương phản được thay đổi nhẹ nhằm tăng khả năng tổng quát hóa.

### 3.9.3. Kiến trúc U-Net sử dụng trong đề tài

#### Khối DoubleConv
Mỗi khối gồm hai lớp convolution kết hợp batch normalization và hàm kích hoạt ReLU.

#### Encoder
Encoder gồm các khối tích chập và max pooling để giảm kích thước không gian, đồng thời tăng số kênh đặc trưng.

#### Bridge
Bridge nằm giữa encoder và decoder, học đặc trưng ngữ cảnh ở mức trừu tượng cao nhất.

#### Decoder
Decoder sử dụng transposed convolution để tăng kích thước đặc trưng và kết hợp skip connection từ encoder.

#### Lớp đầu ra
Lớp convolution 1 × 1 tạo bản đồ logit một kênh cho phân đoạn nhị phân.

### 3.9.4. Huấn luyện U-Net

#### Hàm mất mát
Hàm mất mát kết hợp BCEWithLogitsLoss và Dice loss.

#### Bộ tối ưu
Mô hình được tối ưu bằng AdamW với learning rate và weight decay được cấu hình trước.

#### Dừng sớm
Quá trình huấn luyện sử dụng cơ chế early stopping dựa trên validation loss.

### 3.9.5. Chọn ngưỡng dự đoán

#### Threshold grid
Các ngưỡng 0.3, 0.4, 0.5, 0.6 và 0.7 được đánh giá trên validation set.

#### Tiêu chí chọn
Ngưỡng có Dice trung bình cao nhất được chọn để nhị phân hóa xác suất dự đoán trên tập test.

## 3.10. Hậu xử lý mặt nạ

### 3.10.1. Morphological closing

#### Thiết lập
Closing sử dụng kernel hình chữ nhật với kích thước được khai báo trong cấu hình.

#### Mục đích
Bước này giúp lấp lỗ nhỏ và nối các vùng dự đoán bị đứt đoạn.

### 3.10.2. Morphological opening

#### Thiết lập
Opening sử dụng kernel hình chữ nhật với kích thước nhỏ hơn kernel của closing.

#### Mục đích
Bước này giúp loại bỏ các vùng nhiễu nhỏ trong mặt nạ dự đoán.

### 3.10.3. Lọc thành phần liên thông

#### Ngưỡng diện tích
Các vùng liên thông có diện tích nhỏ hơn ngưỡng tối thiểu sẽ bị loại bỏ.

#### Mục đích
Việc lọc vùng nhỏ giúp giảm false positive, đặc biệt với các nhiễu rời rạc không giống công trình.

## 3.11. Đánh giá kết quả

### 3.11.1. Đánh giá từng patch

#### Metrics tính toán
Mỗi patch được đánh giá bằng IoU, Dice, Precision, Recall, F1, số vùng dự đoán, số vùng ground truth, count error và area absolute error.

#### Đo thời gian
Thời gian dự đoán được ghi lại cho từng patch để phục vụ so sánh chi phí tính toán.

### 3.11.2. Tổng hợp theo phương pháp

#### Trung bình
Kết quả của từng phương pháp được tổng hợp bằng giá trị trung bình trên tập test.

#### Độ lệch chuẩn
Độ lệch chuẩn được sử dụng để thể hiện mức độ ổn định của phương pháp trên các patch khác nhau.

### 3.11.3. Xuất kết quả thực nghiệm

#### File định lượng
Kết quả chi tiết và tổng hợp được lưu dưới dạng CSV.

#### File trực quan
Biểu đồ so sánh và ảnh overlay định tính được lưu để đưa vào phần kết quả.

#### File báo cáo
Một số bảng LaTeX và tóm tắt kết quả được sinh tự động để hỗ trợ viết báo cáo.

---

# Chương 4. Thực nghiệm và kết quả

## 4.1. Môi trường thực nghiệm

### 4.1.1. Phần cứng

#### CPU
Ghi rõ tên CPU sử dụng trong quá trình thực nghiệm.

#### GPU
Ghi rõ GPU nếu mô hình U-Net được huấn luyện bằng CUDA.

#### RAM
Ghi dung lượng RAM của máy chạy thí nghiệm.

### 4.1.2. Phần mềm

#### Ngôn ngữ lập trình
Python.

#### Thư viện chính
OpenCV, NumPy, Pandas, Scikit-learn, Matplotlib, PyTorch và Joblib.

## 4.2. Cấu hình thực nghiệm

### 4.2.1. Cấu hình dữ liệu

#### Patch
Patch size 512, stride 512.

#### Tỷ lệ chia dữ liệu
Train/validation/test tương ứng 70%/15%/15%.

### 4.2.2. Cấu hình phương pháp truyền thống

#### K-Means
Số cụm K = 4.

#### Otsu
Áp dụng trên kênh saturation sau tăng cường tương phản.

### 4.2.3. Cấu hình SVM

#### C grid
Các giá trị C gồm 0.1, 0.5 và 1.0.

#### Số pixel huấn luyện
Giới hạn tối đa 200.000 pixel.

### 4.2.4. Cấu hình U-Net

#### Huấn luyện
Epoch tối đa 20, batch size 4, learning rate 1e-3.

#### Dừng sớm
Patience bằng 5.

## 4.3. Kết quả định lượng

### 4.3.1. Bảng kết quả tổng hợp

#### Nội dung cần trình bày
Bảng gồm Method, IoU, Dice, Precision, Recall, F1, Count Error, Area Error và Seconds/Patch.

#### Nguồn dữ liệu
Lấy từ file `outputs/metrics/final_summary.csv` sau khi chạy thí nghiệm.

### 4.3.2. So sánh IoU và Dice

#### Ý nghĩa
IoU và Dice phản ánh mức độ chồng lấp giữa mask dự đoán và ground truth.

#### Cách phân tích
Phương pháp có IoU/Dice cao hơn được xem là phân đoạn chính xác hơn về mặt không gian.

### 4.3.3. So sánh Precision và Recall

#### Precision cao
Cho thấy mô hình ít dự đoán nhầm nền thành công trình.

#### Recall cao
Cho thấy mô hình ít bỏ sót vùng công trình thật.

#### Nhận xét cân bằng
Cần phân tích sự đánh đổi giữa Precision và Recall thay vì chỉ xét một chỉ số riêng lẻ.

### 4.3.4. Sai số số lượng và diện tích

#### Count error
Phản ánh khả năng mô hình tách/gộp các vùng công trình.

#### Area error
Phản ánh mức độ dự đoán thừa hoặc thiếu tổng diện tích công trình.

### 4.3.5. Thời gian xử lý

#### Seconds per patch
Dùng để so sánh chi phí tính toán giữa các phương pháp.

#### Nhận xét
Otsu thường có chi phí thấp; U-Net có thể nhanh khi dùng GPU nhưng tốn thời gian huấn luyện.

## 4.4. Kết quả định tính

### 4.4.1. Cách trình bày hình ảnh

#### Các cột nên có
Ảnh gốc, ground truth, kết quả K-Means, Otsu, SVM và U-Net.

#### Overlay lỗi
Có thể dùng màu xanh cho true positive, đỏ cho false positive và xanh dương cho false negative.

### 4.4.2. Nhận xét K-Means và Otsu

#### Điểm mạnh
Hai phương pháp đơn giản, dễ triển khai và có thể dùng làm baseline.

#### Điểm yếu
Dễ bị ảnh hưởng bởi ánh sáng, màu mái nhà, đường bê tông và bóng đổ.

### 4.4.3. Nhận xét SVM

#### Điểm mạnh
Khai thác được nhiều đặc trưng hơn so với phương pháp thresholding/phân cụm đơn giản.

#### Điểm yếu
Phân loại từng pixel nên chưa mô hình hóa tốt hình dạng tổng thể của công trình.

### 4.4.4. Nhận xét U-Net

#### Điểm mạnh
Có khả năng học đặc trưng không gian và hình dạng công trình tốt hơn.

#### Điểm yếu
Cần nhiều dữ liệu, thời gian huấn luyện và tài nguyên tính toán.

## 4.5. Phân tích lỗi

### 4.5.1. False Positive

#### Nguyên nhân
Các vùng như đường, sân bê tông hoặc mái sáng có thể bị nhận nhầm thành công trình.

### 4.5.2. False Negative

#### Nguyên nhân
Công trình nhỏ, công trình bị che khuất hoặc nằm trong vùng bóng đổ có thể bị bỏ sót.

### 4.5.3. Lỗi biên

#### Nguyên nhân
Biên công trình có thể bị co, giãn hoặc răng cưa do threshold và hậu xử lý morphology.

### 4.5.4. Ảnh hưởng của hậu xử lý

#### Tác động tích cực
Giảm nhiễu nhỏ và làm mặt nạ liền mạch hơn.

#### Tác động tiêu cực
Có thể loại bỏ công trình nhỏ hoặc làm biến dạng biên công trình.

## 4.6. Thảo luận

### 4.6.1. So sánh tổng quan các phương pháp

#### Phương pháp truyền thống
Phù hợp làm baseline, nhưng hạn chế trong bối cảnh ảnh phức tạp.

#### SVM
Cải thiện nhờ đặc trưng thủ công, nhưng vẫn thiếu khả năng học ngữ cảnh không gian mạnh.

#### U-Net
Là hướng tiếp cận phù hợp hơn cho bài toán phân đoạn công trình nếu có đủ dữ liệu và tài nguyên.

### 4.6.2. Ý nghĩa của kết quả

#### Với nghiên cứu
Kết quả cho thấy sự khác biệt giữa đặc trưng thủ công và đặc trưng học tự động.

#### Với ứng dụng
Pipeline có thể làm nền tảng cho các hệ thống tự động cập nhật bản đồ công trình.

---

# Chương 5. Kết luận và hướng phát triển

## 5.1. Kết luận

### 5.1.1. Kết quả đạt được

#### Pipeline hoàn chỉnh
Nghiên cứu đã xây dựng pipeline phân đoạn công trình gồm chuẩn bị dữ liệu, tiền xử lý, mô hình hóa, hậu xử lý và đánh giá.

#### So sánh nhiều phương pháp
Báo cáo đã so sánh K-Means, Otsu, SVM và U-Net trên cùng quy trình thực nghiệm.

### 5.1.2. Nhận xét chính

#### Phương pháp truyền thống
K-Means và Otsu đơn giản nhưng dễ bị ảnh hưởng bởi điều kiện ảnh.

#### SVM
SVM cải thiện khả năng phân loại nhờ sử dụng nhiều đặc trưng pixel.

#### U-Net
U-Net có tiềm năng tốt hơn nhờ học đặc trưng không gian và ngữ cảnh.

## 5.2. Hạn chế

### 5.2.1. Hạn chế về dữ liệu

#### Dữ liệu đầu vào
Nghiên cứu chỉ sử dụng ảnh RGB và mask nhị phân, chưa kết hợp dữ liệu độ cao hoặc đa phổ.

### 5.2.2. Hạn chế về mô hình

#### Kiến trúc U-Net
U-Net trong đề tài là phiên bản nhỏ, chưa dùng các backbone mạnh hơn.

### 5.2.3. Hạn chế về đầu ra

#### Chưa vector hóa
Kết quả mới ở dạng mask raster, chưa chuyển thành polygon phục vụ trực tiếp cho GIS.

## 5.3. Hướng phát triển

### 5.3.1. Cải tiến mô hình

#### Mô hình học sâu nâng cao
Có thể thử U-Net++, DeepLabV3+, Attention U-Net hoặc SegFormer.

### 5.3.2. Cải tiến dữ liệu

#### Tăng cường dữ liệu
Bổ sung các kỹ thuật augmentation như random crop, color jitter, noise và shadow simulation.

### 5.3.3. Cải tiến hậu xử lý

#### Vector hóa kết quả
Chuyển mask sang polygon và làm mượt biên để phục vụ ứng dụng GIS.

### 5.3.4. Cải tiến đánh giá

#### Đánh giá theo nhóm đối tượng
Có thể đánh giá riêng theo kích thước công trình, khu vực đô thị hoặc mức độ che khuất.

---

# Tài liệu tham khảo gợi ý

## Nhóm tài liệu về phân đoạn ảnh

### Semantic segmentation

#### Nội dung nên tham khảo
Các tài liệu nền tảng về segmentation, segmentation metrics và image processing.

## Nhóm tài liệu về U-Net

### Bài báo U-Net gốc

#### Nội dung nên tham khảo
Kiến trúc encoder-decoder, skip connection và ứng dụng trong segmentation.

## Nhóm tài liệu về building footprint extraction

### Ảnh viễn thám và GIS

#### Nội dung nên tham khảo
Các nghiên cứu trích xuất công trình từ ảnh vệ tinh hoặc ảnh hàng không độ phân giải cao.

## Nhóm tài liệu về bộ dữ liệu

### Inria Aerial Image Labeling Dataset

#### Nội dung nên tham khảo
Mô tả dữ liệu, ảnh đầu vào, ground truth và bối cảnh sử dụng.

---

# Phụ lục A. Cấu hình thí nghiệm

## A.1. Cấu hình dữ liệu

### A.1.1. Patch

#### Giá trị
Patch size = 512, patch stride = 512.

### A.1.2. Chia dữ liệu

#### Giá trị
Train = 70%, validation = 15%, test = 15%.

## A.2. Cấu hình tiền xử lý

### A.2.1. CLAHE

#### Giá trị
Clip limit = 2.0, tile grid = 8 × 8.

## A.3. Cấu hình hậu xử lý

### A.3.1. Morphology

#### Giá trị
Closing kernel = 9 × 9, opening kernel = 5 × 5.

### A.3.2. Lọc vùng nhỏ

#### Giá trị
Minimum building area = 200 pixel.

## A.4. Cấu hình SVM

### A.4.1. Grid search

#### Giá trị
C = 0.1, 0.5, 1.0.

### A.4.2. Số pixel huấn luyện

#### Giá trị
Tối đa 200.000 pixel.

## A.5. Cấu hình U-Net

### A.5.1. Huấn luyện

#### Giá trị
Epoch = 20, batch size = 4, learning rate = 1e-3, weight decay = 1e-4.

### A.5.2. Chọn ngưỡng

#### Giá trị
Threshold grid = 0.3, 0.4, 0.5, 0.6, 0.7.


# Gợi ý bảng và hình nên đưa vào báo cáo

## Danh mục bảng

### Bảng 1. Cấu hình thực nghiệm

#### Nội dung
Patch size, split ratio, số patch tối đa, thông số CLAHE, morphology, SVM và U-Net.

### Bảng 2. So sánh định lượng các phương pháp

#### Nội dung
Method, IoU, Dice, Precision, Recall, F1, Count Error, Area Error, Seconds/Patch.

### Bảng 3. Kết quả tuning SVM

#### Nội dung
C, validation Dice và thời gian huấn luyện.

### Bảng 4. Kết quả tuning threshold U-Net

#### Nội dung
Threshold và validation Dice.

## Danh mục hình

### Hình 1. Pipeline tổng quan

#### Nội dung
Input image → preprocessing → segmentation model → post-processing → evaluation.

### Hình 2. Ví dụ ảnh và ground truth

#### Nội dung
Ảnh hàng không và mask công trình tương ứng.

### Hình 3. Biểu đồ so sánh metric

#### Nội dung
Lấy từ file `method_comparison.png`.

### Hình 4. Kết quả định tính

#### Nội dung
So sánh ảnh gốc, ground truth, K-Means, Otsu, SVM và U-Net.

### Hình 5. Phân tích lỗi TP/FP/FN

#### Nội dung
Overlay màu thể hiện vùng đúng, vùng dự đoán sai và vùng bỏ sót.

---

