# Kịch bản thuyết trình cuối kỳ Computer Vision - Nhóm 16

- Đề tài: **Trích xuất vùng tòa nhà từ ảnh chụp từ trên cao**
- Slide gốc: `NguyenHuySon_16_presentation.pdf` gồm 21 slide
- Report gốc: `NguyenHuySon_16_report.pdf` gồm 23 trang
- Mục tiêu khi trình bày: nói rõ bài toán, pipeline, lý do chọn từng phương pháp, kết quả thực nghiệm và ý nghĩa ứng dụng GIS.

## 1. Tư duy trình bày chung

Thông điệp chính cần giữ xuyên suốt:

- Bài toán **building footprint extraction** không chỉ là tách màu mái nhà, mà là bài toán phân đoạn nhị phân trong ảnh hàng không có nhiều nhiễu: bóng đổ, cây che, đường bê tông, sân, bãi xe và mái nhà đa dạng vật liệu.
- Nhóm so sánh 4 cấp độ phương pháp trên cùng pipeline: `K-Means` và `Otsu` là baseline cổ điển, `Linear SVM` là supervised pixel-level model, còn `U-Net` là deep learning semantic segmentation.
- U-Net là phương pháp tốt nhất trong lần thực nghiệm này: `IoU = 0.544`, `Dice = 0.665`, `Area Error = 5.258`.
- Các baseline vẫn quan trọng vì giúp chứng minh vì sao bài toán này cần mô hình học ngữ cảnh không gian, không chỉ dùng ngưỡng hoặc màu sắc.
- Kết quả cần được đọc bằng cả chỉ số pixel-level và chỉ số gần với ứng dụng bản đồ như `Count Error`, `Area Error`, cùng ảnh overlay lỗi.

Thời lượng gợi ý:

- Nếu thầy/cô cho khoảng 12-15 phút: nói theo đầy đủ script dưới đây.
- Nếu chỉ có khoảng 8-10 phút: rút gọn phần lý thuyết Otsu/K-Means/SVM, tập trung vào pipeline, U-Net, bảng kết quả và phân tích lỗi.
- Nếu có demo GUI: đặt demo sau slide 17 hoặc sau slide 18, thời lượng 1-2 phút.

Phân vai gợi ý nếu 2 thành viên cùng trình bày:

- Nguyễn Huy Sơn: mở đầu, bài toán, dữ liệu, pipeline, CLAHE, Otsu, U-Net, kết quả chính, kết luận.
- Dương Văn Tài: K-Means, morphology, SVM, biểu đồ, kết quả trực quan, hạn chế và hướng phát triển.

Nếu bạn trình bày một mình, cứ dùng đại từ **"nhóm em"** để giữ đúng tinh thần báo cáo nhóm.

## 2. Kịch bản theo từng slide

### Slide 1 - Trang tiêu đề

Thời lượng: 20-30 giây

Lời thoại:

> Kính chào thầy và các bạn. Nhóm em là nhóm 16, gồm Nguyễn Huy Sơn và Dương Văn Tài. Hôm nay nhóm em trình bày đề tài **Trích xuất vùng tòa nhà từ ảnh chụp từ trên cao**, thuộc học phần Thị giác máy tính.
>
> Mục tiêu chính của đề tài là xây dựng một pipeline có thể nhận ảnh hàng không RGB và sinh ra mặt nạ vùng công trình, từ đó phục vụ các bài toán như cập nhật bản đồ đô thị, ước lượng mật độ xây dựng hoặc phân tích sử dụng đất.

Chuyển ý:

> Em xin đi vào cấu trúc bài trình bày.

### Slide 2 - Nội dung trình bày

Thời lượng: 20 giây

Lời thoại:

> Bài trình bày gồm 5 phần. Đầu tiên là bối cảnh và bài toán. Sau đó nhóm em trình bày dữ liệu và pipeline thực nghiệm. Phần thứ ba là 4 phương pháp được so sánh. Phần thứ tư là kết quả định lượng và định tính. Cuối cùng là kết luận, hạn chế và hướng phát triển.

Chuyển ý:

> Trước hết, nhóm em bắt đầu từ lý do vì sao bài toán này có ý nghĩa.

### Slide 3 - Bối cảnh và bài toán

Thời lượng: 50-60 giây

Lời thoại:

> Trong GIS, ảnh hàng không độ phân giải cao là nguồn dữ liệu rất quan trọng. Nếu trích xuất được **building footprint**, tức vùng chiếm dụng của công trình khi nhìn từ trên cao, ta có thể hỗ trợ cập nhật bản đồ, theo dõi thay đổi đô thị hoặc ước lượng mật độ xây dựng.
>
> Trong đề tài này, nhóm em mô hình hóa bài toán thành **binary segmentation**. Input là một patch ảnh RGB. Output là một mask cùng kích thước, trong đó mỗi pixel được gán là **công trình** hoặc **nền**.
>
> Khó khăn chính là ảnh đô thị rất phức tạp. Mái nhà có nhiều màu và hình dạng khác nhau; bóng đổ làm thay đổi độ sáng; cây có thể che mái; và nhiều vùng nền như đường bê tông, sân hoặc bãi đỗ xe có màu rất giống công trình. Vì vậy, nếu chỉ dựa vào màu hoặc một ngưỡng đơn giản thì dễ sinh lỗi.

Điểm cần nhấn:

- Đừng nói bài toán là "nhận diện tòa nhà" chung chung; hãy nói chính xác là `binary semantic segmentation`.
- Nhấn mạnh false positive từ đường/sân/bê tông và false negative ở mái nhỏ/bị che.

Chuyển ý:

> Từ bối cảnh đó, nhóm em đặt ra mục tiêu và phạm vi cụ thể như sau.

### Slide 4 - Mục tiêu và phạm vi nghiên cứu

Thời lượng: 45 giây

Lời thoại:

> Mục tiêu tổng quát là xây dựng một pipeline hoàn chỉnh từ dữ liệu ảnh hàng không đến mask công trình và các chỉ số đánh giá.
>
> Cụ thể, nhóm em so sánh 4 phương pháp: `K-Means`, `Otsu`, `Linear SVM` và `U-Net`. Bốn phương pháp này đại diện cho các mức độ khác nhau: không giám sát, thresholding, học máy có giám sát ở mức pixel và học sâu cho semantic segmentation.
>
> Về phạm vi, nhóm em dùng ảnh RGB và ground-truth mask từ Inria Aerial Image Labeling Dataset. Đầu ra hiện tại là mask raster, tức dạng lưới pixel. Nhóm em chưa vector hóa mask thành polygon GIS, nên đây là hướng phát triển sau.

Chuyển ý:

> Tiếp theo là dữ liệu và cách nhóm em chia tập để tránh rò rỉ dữ liệu.

### Slide 5 - Bộ dữ liệu và chia tập

Thời lượng: 60 giây

Lời thoại:

> Dữ liệu sử dụng là **Inria Aerial Image Labeling Dataset**, gồm ảnh hàng không ở 5 khu vực: Austin, Chicago, Kitsap, Tyrol-w và Vienna.
>
> Một điểm quan trọng trong thiết kế thực nghiệm là nhóm em chia dữ liệu ở mức **ảnh gốc**, không chia ngẫu nhiên từng patch ngay từ đầu. Lý do là các patch cắt từ cùng một ảnh gốc thường rất giống nhau về khu vực, màu sắc và bố cục. Nếu patch từ cùng ảnh xuất hiện cả ở train và test thì kết quả có thể bị lạc quan do data leakage.
>
> Trong lần chạy này, nhóm em có 180 ảnh gốc, chia theo tỷ lệ 70%, 15%, 15%, tương ứng 126 ảnh train, 27 ảnh validation và 27 ảnh test. Sau đó ảnh được cắt thành patch 512 x 512 pixel, giới hạn tối đa 1200 patch train, 300 patch validation và 300 patch test để cân bằng giữa độ phủ dữ liệu và tài nguyên tính toán.

Nếu bị hỏi vì sao không dùng toàn bộ patch:

> Vì Inria có dung lượng lớn, khoảng hàng chục GB. Nếu dùng toàn bộ patch thì thời gian train U-Net và tuning SVM tăng mạnh. Nhóm em chọn giới hạn patch để thực nghiệm khả thi nhưng vẫn có test set đủ lớn là 300 patch.

Chuyển ý:

> Sau khi có dữ liệu, pipeline tổng quát của nhóm em gồm các bước sau.

### Slide 6 - Pipeline thực nghiệm tổng quát

Thời lượng: 60-70 giây

Lời thoại:

> Pipeline bắt đầu bằng việc đọc ảnh RGB và mask ground truth. Sau đó dữ liệu được chia theo ảnh gốc và cắt thành patch 512 x 512.
>
> Mỗi patch đi qua bước tiền xử lý là **CLAHE trên kênh sáng LAB**. Sau đó cùng một patch được đưa vào 4 nhánh mô hình: K-Means, Otsu, SVM và U-Net.
>
> Output của mỗi nhánh là mask nhị phân. Các mask này đều đi qua hậu xử lý chung gồm morphology và connected components để giảm nhiễu nhỏ. Cuối cùng, nhóm em đánh giá bằng IoU, Dice, Precision, Recall, F1, Count Error, Area Error và thời gian suy luận.
>
> Điểm quan trọng ở đây là 4 phương pháp được đặt trong cùng một pipeline, cùng test set và cùng hậu xử lý. Nhờ vậy, kết quả so sánh tập trung vào khác biệt giữa phương pháp phân đoạn, không bị nhiễu bởi cách chia dữ liệu hay cách tính metric.

Chuyển ý:

> Em sẽ nói rõ hơn về tiền xử lý và hậu xử lý trước khi đi vào từng mô hình.

### Slide 7 - Tiền xử lý và hậu xử lý

Thời lượng: 60 giây

Lời thoại:

> Ở bước tiền xử lý, nhóm em chuyển ảnh từ RGB sang LAB. Trong không gian LAB, kênh L biểu diễn độ sáng, còn hai kênh còn lại biểu diễn thông tin màu. Nhóm em áp dụng CLAHE trên kênh L với clip limit 2.0 và tile grid 8 x 8, sau đó chuyển ảnh về RGB.
>
> Mục đích của CLAHE là tăng tương phản cục bộ, giúp các vùng mái hoặc biên công trình dễ quan sát hơn, đặc biệt trong vùng bị bóng hoặc độ tương phản thấp. Tuy nhiên CLAHE không tự giải quyết bài toán phân đoạn; nó chỉ chuẩn hóa input cho các nhánh mô hình.
>
> Với hậu xử lý, nhóm em dùng closing 9 x 9 để nối vùng đứt và lấp lỗ nhỏ, opening 5 x 5 để giảm nhiễu nhỏ, sau đó lọc connected component nhỏ hơn 200 pixel. Bước này giúp mask sạch hơn, đồng thời cho phép tính số vùng công trình.

Điểm cần nhấn:

- Morphology có hai mặt: giảm nhiễu nhưng có thể làm mất nhà nhỏ hoặc gộp các nhà gần nhau.

Chuyển ý:

> Sau đây là phương pháp đầu tiên: K-Means.

### Slide 8 - Phương pháp 1: K-Means

Thời lượng: 50-60 giây

Lời thoại:

> K-Means là baseline không giám sát. Với mỗi pixel, nhóm em dùng đặc trưng RGB kết hợp HSV, chuẩn hóa về miền số thực, rồi phân cụm với K bằng 4.
>
> Sau khi phân cụm, cần quyết định cụm nào là công trình. Trong code, nhóm em chọn cụm có tâm sáng cao nhất trên kênh Value của HSV, vì nhiều mái nhà trong ảnh hàng không thường là vùng sáng hoặc có độ phản xạ cao.
>
> Ưu điểm của K-Means là đơn giản, không cần nhãn để huấn luyện và dễ dùng làm mốc so sánh. Nhưng hạn chế rất rõ: nếu đường, sân bê tông hoặc bãi đỗ xe cũng sáng thì chúng dễ bị chọn nhầm thành công trình. Vì vậy, K-Means thường sinh false positive ở các vùng nền sáng.

Chuyển ý:

> Phương pháp baseline thứ hai là Otsu, đại diện cho nhóm thresholding.

### Slide 9 - Phương pháp 2: Otsu

Thời lượng: 50-60 giây

Lời thoại:

> Otsu là phương pháp chọn ngưỡng tự động. Ý tưởng là tìm một ngưỡng sao cho phương sai giữa hai lớp pixel là lớn nhất. Nói đơn giản, Otsu cố chia histogram thành hai nhóm tách biệt nhất có thể.
>
> Trong code của nhóm em, Otsu được áp dụng trên kênh Saturation của HSV sau khi dùng CLAHE nhẹ với clip limit 1.5. Sau đó dùng threshold dạng binary inverse để tạo mask.
>
> Ưu điểm lớn nhất là rất nhanh và không cần huấn luyện. Nhưng hạn chế là Otsu chỉ dùng một ngưỡng toàn cục. Với ảnh đô thị, histogram không tách thành hai nhóm rõ ràng, vì nền và công trình có thể có màu, độ bão hòa và độ sáng tương tự nhau. Do đó Otsu dễ dự đoán quá rộng.

Chuyển ý:

> Hai baseline vừa rồi chưa dùng nhãn huấn luyện. Vì vậy nhóm em thêm Linear SVM để xem học có giám sát ở mức pixel cải thiện được bao nhiêu.

### Slide 10 - Phương pháp 3: Linear SVM

Thời lượng: 70-80 giây

Lời thoại:

> Với Linear SVM, mỗi pixel được xem như một mẫu phân loại nhị phân: công trình hoặc nền. Điểm khác so với K-Means và Otsu là SVM được học từ ground truth mask.
>
> Nhóm em trích xuất đặc trưng thủ công cho từng pixel, gồm RGB, LAB, HSV, grayscale, Sobel gradient magnitude, local mean và local standard deviation. Tổng cộng là 13 đặc trưng cho mỗi pixel.
>
> Mô hình dùng `StandardScaler` để chuẩn hóa đặc trưng, sau đó dùng `LinearSVC` với `class_weight="balanced"` để giảm ảnh hưởng mất cân bằng giữa pixel nền và pixel công trình. Mỗi patch train lấy tối đa 1200 pixel, tổng số pixel train giới hạn 200,000. Tham số C được thử trong tập 0.1, 0.5 và 1.0 trên validation set.
>
> SVM là bước trung gian tốt: nó tận dụng được nhãn, nên thường tốt hơn baseline cổ điển. Nhưng vì vẫn phân loại từng pixel bằng đặc trưng cục bộ và ranh giới tuyến tính, SVM chưa hiểu được hình dạng hoặc ngữ cảnh rộng của cả tòa nhà.

Nếu cần nói ngắn hơn:

> Tóm lại, SVM cho thấy lợi ích của supervised learning, nhưng vẫn bị giới hạn bởi handcrafted features và thiếu spatial context.

Chuyển ý:

> Để học được ngữ cảnh không gian tốt hơn, nhóm em dùng phương pháp thứ tư là U-Net.

### Slide 11 - Phương pháp 4: U-Net

Thời lượng: 80-90 giây

Lời thoại:

> U-Net là kiến trúc encoder-decoder rất phổ biến cho semantic segmentation. Encoder giảm dần kích thước ảnh để học đặc trưng ngữ cảnh, decoder khôi phục lại mask ở độ phân giải ban đầu, còn skip connection truyền thông tin chi tiết từ encoder sang decoder để giữ biên đối tượng tốt hơn.
>
> Trong thực nghiệm, nhóm em dùng một phiên bản U-Net nhỏ gồm 3 mức encoder, một bridge và 3 mức decoder. Mỗi block chính gồm convolution 3 x 3, batch normalization và ReLU. Lớp cuối là convolution 1 x 1 để tạo logit một kênh cho bài toán nhị phân.
>
> Mô hình được train 20 epochs, batch size 4, optimizer AdamW, learning rate 10^-3 và weight decay 10^-4. Loss là tổng của `BCEWithLogitsLoss` và Dice loss. BCE giúp học phân loại pixel, còn Dice loss trực tiếp tối ưu mức chồng lấp giữa mask dự đoán và ground truth.
>
> Nhóm em cũng dùng augmentation như flip ngang, flip dọc, xoay theo bội số 90 độ, thay đổi nhẹ brightness và contrast. Sau khi train, nhóm em chọn ngưỡng nhị phân hóa trên validation set, và ngưỡng tốt nhất là 0.4.

Điểm cần nhấn:

- U-Net không chỉ nhìn từng pixel độc lập; nó học pattern vùng và hình dạng.
- Skip connection quan trọng cho biên công trình.

Chuyển ý:

> Để so sánh các phương pháp, nhóm em dùng các chỉ số sau.

### Slide 12 - Chỉ số đánh giá

Thời lượng: 60 giây

Lời thoại:

> Với bài toán segmentation, hai chỉ số quan trọng nhất là IoU và Dice. IoU đo phần giao chia cho phần hợp giữa mask dự đoán và ground truth. Dice cũng đo độ chồng lấp nhưng thường nhạy hơn khi vùng đối tượng nhỏ.
>
> Nhóm em cũng dùng Precision và Recall. Precision thấp nghĩa là mô hình dự đoán thừa nhiều nền thành công trình, tức nhiều false positive. Recall thấp nghĩa là mô hình bỏ sót công trình thật, tức nhiều false negative.
>
> Ngoài pixel-level metrics, nhóm em thêm Count Error và Area Error. Count Error đo sai số số vùng công trình, còn Area Error đo sai số tỷ lệ diện tích công trình trong patch. Hai chỉ số này gần với ứng dụng GIS hơn, vì trong bản đồ đô thị ta không chỉ cần mask đẹp mà còn cần số lượng và diện tích tương đối hợp lý.
>
> Cuối cùng là Seconds/Patch để đánh giá tốc độ suy luận trung bình.

Chuyển ý:

> Sau đây là bảng kết quả chính trên 300 patch test.

### Slide 13 - Kết quả định lượng tổng hợp

Thời lượng: 90-110 giây

Lời thoại:

> Bảng này là kết quả trung bình trên 300 patch test. Có ba điểm chính cần chú ý.
>
> Thứ nhất, U-Net đạt kết quả tốt nhất ở các chỉ số chồng lấp: IoU đạt 0.544 và Dice đạt 0.665. So với SVM, Dice tăng từ 0.416 lên 0.665, tức cải thiện khá rõ. Điều này cho thấy U-Net học được ngữ cảnh và hình dạng công trình tốt hơn so với các phương pháp dựa vào màu hoặc đặc trưng pixel cục bộ.
>
> Thứ hai, Otsu có Recall cao nhất, 0.802, nhưng Precision chỉ 0.233. Nghĩa là Otsu phát hiện được nhiều pixel công trình thật, nhưng đồng thời dự đoán nhầm rất nhiều nền thành công trình. Vì vậy, nếu chỉ nhìn Recall thì Otsu có vẻ tốt, nhưng nhìn Precision và Area Error thì thấy mask bị mở rộng quá mức.
>
> Thứ ba, U-Net có Area Error thấp nhất, 5.258, trong khi Otsu lên tới 64.691 và SVM là 34.436. Với ứng dụng bản đồ, đây là điểm rất quan trọng vì sai số diện tích thấp nghĩa là tỷ lệ vùng xây dựng dự đoán gần ground truth hơn.
>
> Về thời gian, Otsu nhanh nhất với 0.0075 giây mỗi patch. U-Net đạt 0.0396 giây mỗi patch trong điều kiện suy luận sau huấn luyện, nhanh hơn SVM và K-Means trong lần đo này. Tuy nhiên cần lưu ý thời gian của U-Net chưa tính chi phí train.

Các con số cần thuộc:

- K-Means: IoU 0.230, Dice 0.347, Area Error 20.093.
- Otsu: IoU 0.223, Dice 0.333, Recall 0.802, Area Error 64.691.
- SVM: IoU 0.295, Dice 0.416, Recall 0.774, Area Error 34.436.
- U-Net: IoU 0.544, Dice 0.665, Precision 0.698, Area Error 5.258.

Chuyển ý:

> Trước khi kết luận U-Net tốt nhất, nhóm em cũng kiểm tra phần tuning của SVM và U-Net.

### Slide 14 - Kết quả tuning SVM

Thời lượng: 45 giây

Lời thoại:

> Với SVM, nhóm em thử ba giá trị C: 0.1, 0.5 và 1.0. Validation Dice lần lượt khoảng 0.2849, 0.2851 và 0.2851. C bằng 0.5 là tốt nhất nhưng chênh lệch rất nhỏ.
>
> Điều này cho thấy trong cấu hình hiện tại, giới hạn chính của SVM không nằm ở việc chọn C, mà nằm ở bản chất mô hình: đặc trưng vẫn là handcrafted features ở mức pixel và ranh giới phân tách là tuyến tính. Vì vậy, dù có tuning C trong khoảng này, SVM vẫn khó bắt được hình dạng và ngữ cảnh phức tạp của công trình.

Chuyển ý:

> Với U-Net, tham số cần tuning sau huấn luyện là ngưỡng chuyển xác suất thành mask nhị phân.

### Slide 15 - Kết quả tuning ngưỡng U-Net

Thời lượng: 50-60 giây

Lời thoại:

> U-Net sinh ra logit hoặc xác suất cho từng pixel. Để tạo mask nhị phân, ta cần chọn một threshold. Ngưỡng mặc định thường là 0.5, nhưng chưa chắc tối ưu cho Dice.
>
> Nhóm em thử các ngưỡng 0.3, 0.4, 0.5, 0.6 và 0.7 trên validation set. Kết quả tốt nhất là threshold 0.4, với Validation Dice 0.6085.
>
> Việc ngưỡng tốt nhất thấp hơn 0.5 cho thấy nếu dùng 0.5, mô hình có thể hơi bảo thủ và bỏ sót một phần vùng công trình. Hạ threshold xuống 0.4 giúp tăng phát hiện vùng công trình, đồng thời vẫn cân bằng được Precision và Recall theo Dice.

Chuyển ý:

> Bên cạnh bảng số, nhóm em cũng trực quan hóa kết quả để dễ so sánh hơn.

### Slide 16 - So sánh nhanh các phương pháp

Thời lượng: 50-60 giây

Lời thoại:

> Biểu đồ này trực quan hóa các chỉ số IoU, Dice, Precision, Recall và F1 của 4 phương pháp.
>
> Có thể thấy U-Net dẫn đầu ở IoU, Dice, Precision và F1. Điều này phù hợp với bảng số liệu ở slide trước. Otsu nổi bật ở Recall nhưng Precision thấp, nghĩa là phương pháp này bắt được nhiều vùng công trình thật nhưng dự đoán thừa rất nhiều. SVM cũng có Recall cao nhưng Precision chưa tốt, cho thấy supervised pixel-level learning có cải thiện nhưng vẫn còn nhiều false positive.
>
> Kết luận từ biểu đồ là: nếu chỉ cần baseline nhanh, Otsu rất rẻ; nhưng nếu cần mask có chất lượng và cân bằng hơn, U-Net là lựa chọn tốt nhất trong nhóm phương pháp đã khảo sát.

Chuyển ý:

> Để hiểu lỗi cụ thể hơn, nhóm em xem kết quả overlay định tính.

### Slide 17 - Kết quả trực quan định tính

Thời lượng: 90 giây

Lời thoại:

> Ở hình overlay, nhóm em dùng ba màu để đọc lỗi. Xanh lá là true positive, tức vùng công trình dự đoán đúng. Đỏ là false positive, tức nền bị dự đoán nhầm thành công trình. Xanh dương là false negative, tức công trình thật bị bỏ sót.
>
> Khi nhìn vào các baseline như K-Means, Otsu và SVM, ta thường thấy nhiều vùng đỏ ở đường, sân hoặc bãi bê tông. Đây là các đối tượng có màu hoặc độ sáng giống mái nhà. Với Otsu, lỗi đỏ thường lớn vì một ngưỡng toàn cục dễ mở rộng vùng dự đoán quá mức.
>
> U-Net thường giữ được nhiều vùng xanh lá hơn và giảm vùng đỏ, vì mô hình học được hình dạng và ngữ cảnh xung quanh công trình. Tuy nhiên U-Net vẫn chưa hoàn hảo. Một số nhà nhỏ, mái bị cây che hoặc vùng nằm trong bóng đổ vẫn có thể thành false negative, tức màu xanh dương.
>
> Như vậy, phần hình ảnh giúp xác nhận lại bảng số: U-Net tốt hơn không chỉ ở metric mà cả ở chất lượng mask trực quan, nhưng vẫn còn lỗi ở các trường hợp khó.

Nếu có demo GUI:

> Nếu có thời gian demo, ở đoạn này nhóm em có thể mở giao diện Streamlit, upload một ảnh aerial, chọn U-Net và hiển thị mask/overlay. GUI không retrain model, chỉ load lại artifact đã train trong thư mục outputs/models và threshold đã chọn trong outputs/metrics.

Chuyển ý:

> Từ các kết quả này, nhóm em rút ra ý nghĩa thực tiễn như sau.

### Slide 18 - Ý nghĩa thực tiễn của kết quả

Thời lượng: 60 giây

Lời thoại:

> Kết quả của U-Net có ý nghĩa thực tiễn hơn vì mask có độ chồng lấp tốt và sai số diện tích thấp. Với bài toán bản đồ, điều này quan trọng vì ta không chỉ quan tâm pixel đúng sai, mà còn quan tâm tổng diện tích xây dựng có gần thực tế không.
>
> Baseline như K-Means và Otsu vẫn hữu ích. Chúng giúp tạo mốc so sánh nhanh, dễ giải thích và cho thấy rõ khó khăn của bài toán. Nếu baseline đã sai nhiều ở đường và bãi bê tông, ta có cơ sở để giải thích vì sao cần mô hình học có giám sát và học sâu.
>
> Vì vậy, đóng góp chính của đề tài không chỉ là đạt số liệu cao nhất bằng U-Net, mà là xây dựng được một pipeline so sánh nhiều phương pháp trong cùng điều kiện và phân tích được lỗi theo hướng ứng dụng GIS.

Chuyển ý:

> Từ đó nhóm em đi đến kết luận chính.

### Slide 19 - Kết luận

Thời lượng: 50-60 giây

Lời thoại:

> Nhóm em đã xây dựng một pipeline đầy đủ cho bài toán building footprint extraction, từ đọc dữ liệu, chia tập, cắt patch, tiền xử lý, chạy mô hình, hậu xử lý, đánh giá và trực quan hóa.
>
> Bốn phương pháp được so sánh gồm K-Means, Otsu, Linear SVM và U-Net. Kết quả trên 300 patch test cho thấy U-Net là phương pháp tốt nhất trong nhóm khảo sát, đạt IoU 0.544, Dice 0.665 và Area Error 5.258.
>
> SVM cải thiện so với K-Means và Otsu nhờ dùng nhãn và đặc trưng thủ công, nhưng vẫn dự đoán thừa nhiều nền do thiếu ngữ cảnh không gian rộng. Các baseline truyền thống tuy chất lượng thấp hơn nhưng có vai trò quan trọng trong việc tạo mốc so sánh và giải thích bài toán.

Chuyển ý:

> Tuy nhiên, đề tài vẫn còn một số hạn chế và hướng phát triển.

### Slide 20 - Hạn chế và hướng phát triển

Thời lượng: 70-80 giây

Lời thoại:

> Hạn chế đầu tiên là dữ liệu hiện mới dùng ảnh RGB và mask nhị phân. Nhóm em chưa khai thác thêm thông tin độ cao, ảnh đa phổ hoặc metadata địa lý. Nếu có DSM, LiDAR hoặc ảnh multi-spectral, mô hình có thể phân biệt công trình với đường và sân tốt hơn.
>
> Hạn chế thứ hai là U-Net trong đề tài là phiên bản nhỏ, chưa dùng backbone tiền huấn luyện, attention hoặc các kiến trúc segmentation hiện đại. Vì vậy, kết quả còn dư địa cải thiện.
>
> Hạn chế thứ ba là output hiện vẫn là mask raster. Để dùng trực tiếp trong GIS, cần thêm bước vector hóa sang polygon, làm mượt biên và đánh giá theo đối tượng.
>
> Hướng phát triển là thử U-Net++, Attention U-Net, DeepLabV3+ hoặc SegFormer; bổ sung augmentation phù hợp với ảnh hàng không như mô phỏng bóng đổ, biến đổi ánh sáng; đánh giá theo từng thành phố hoặc kích thước công trình; và tích hợp bước vector hóa GIS.

Chuyển ý:

> Phần trình bày của nhóm em đến đây là kết thúc.

### Slide 21 - Cảm ơn và Q&A

Thời lượng: 20 giây

Lời thoại:

> Nhóm em xin cảm ơn thầy và các bạn đã lắng nghe. Nhóm em sẵn sàng nhận câu hỏi và góp ý.

Nếu cần một câu chốt mạnh trước Q&A:

> Tóm lại, qua thực nghiệm này nhóm em thấy rằng với ảnh hàng không đô thị, phương pháp học sâu như U-Net có lợi thế rõ vì học được ngữ cảnh và hình dạng, trong khi các baseline giúp giải thích tốt các nguồn lỗi của bài toán.

## 3. Bản rút gọn 8-10 phút

Nếu thời gian bị giới hạn, nói theo khung này:

1. Slide 1-2: giới thiệu đề tài và outline trong 40 giây.
2. Slide 3-4: nói bài toán là binary segmentation cho building footprint, nhấn mạnh khó khăn trong ảnh đô thị.
3. Slide 5-7: nói dataset, split theo ảnh gốc, patch 512 x 512, CLAHE và morphology.
4. Slide 8-11: mỗi phương pháp nói 2 câu: input/features, ưu điểm, hạn chế.
5. Slide 12-13: tập trung vào IoU, Dice, Precision/Recall, Area Error; nhấn mạnh U-Net tốt nhất.
6. Slide 14-15: nói nhanh tuning SVM ít ảnh hưởng, threshold U-Net tốt nhất là 0.4.
7. Slide 16-18: giải thích U-Net dẫn đầu, Otsu/SVM recall cao nhưng precision thấp, overlay màu.
8. Slide 19-21: kết luận, hạn chế, hướng phát triển, cảm ơn.

## 4. Các câu hỏi phản biện có thể gặp

### Câu 1: Vì sao chia dữ liệu theo ảnh gốc thay vì chia patch ngẫu nhiên?

Trả lời:

> Vì các patch cắt từ cùng một ảnh gốc có thể rất giống nhau về khu vực, ánh sáng, bố cục và vật liệu. Nếu patch từ cùng ảnh xuất hiện ở cả train và test, mô hình có thể học đặc trưng khu vực thay vì tổng quát hóa thật. Chia theo ảnh gốc giúp giảm data leakage và làm đánh giá đáng tin hơn.

### Câu 2: Vì sao dùng CLAHE trên kênh L của LAB?

Trả lời:

> LAB tách độ sáng khỏi thông tin màu. Khi dùng CLAHE trên kênh L, nhóm em tăng tương phản cục bộ mà hạn chế làm biến dạng màu. Điều này hữu ích trong ảnh hàng không vì mái nhà có thể nằm trong vùng bóng hoặc tương phản thấp.

### Câu 3: Vì sao K-Means chọn cụm có Value cao nhất?

Trả lời:

> Đây là heuristic đơn giản cho baseline. Nhiều vùng mái hoặc bề mặt công trình có độ sáng cao trong ảnh hàng không, nên cụm có Value cao được chọn làm ứng viên công trình. Tuy nhiên heuristic này cũng là nguồn lỗi vì đường, sân bê tông và bãi đỗ xe sáng cũng có thể bị chọn nhầm.

### Câu 4: Vì sao Otsu có Recall cao nhưng Precision thấp?

Trả lời:

> Otsu dùng một ngưỡng toàn cục nên có xu hướng lấy rộng vùng dự đoán. Khi lấy rộng, nó bắt được nhiều pixel công trình thật nên Recall cao, nhưng đồng thời kéo theo nhiều nền bị dự đoán nhầm thành công trình nên Precision thấp.

### Câu 5: SVM dùng bao nhiêu đặc trưng cho mỗi pixel?

Trả lời:

> SVM dùng 13 đặc trưng: RGB có 3, LAB có 3, HSV có 3, grayscale có 1, Sobel gradient magnitude có 1, local mean có 1 và local standard deviation có 1. Tổng cộng là 13 đặc trưng cho mỗi pixel.

### Câu 6: Vì sao dùng Linear SVM mà không dùng RBF SVM?

Trả lời:

> Dữ liệu pixel rất lớn, tối đa 200,000 pixel train trong cấu hình hiện tại. Linear SVM nhanh và phù hợp làm supervised baseline. RBF SVM có thể mô hình hóa ranh giới phi tuyến tốt hơn nhưng chi phí train và inference cao hơn nhiều khi số mẫu lớn. Trong phạm vi đề tài, nhóm em ưu tiên Linear SVM để so sánh với U-Net.

### Câu 7: Vì sao U-Net tốt hơn SVM?

Trả lời:

> SVM phân loại từng pixel dựa trên đặc trưng cục bộ, nên khó hiểu được hình dạng tổng thể của mái nhà hoặc ngữ cảnh xung quanh. U-Net dùng convolution, encoder-decoder và skip connection, nên học được cả texture, hình dạng và quan hệ không gian. Vì vậy U-Net giảm false positive và cho Area Error thấp hơn.

### Câu 8: Vì sao threshold tốt nhất của U-Net là 0.4 chứ không phải 0.5?

Trả lời:

> 0.5 chỉ là ngưỡng mặc định. Xác suất đầu ra của model không nhất thiết được calibration hoàn hảo. Trên validation set, threshold 0.4 cho Dice cao nhất, nghĩa là nó cân bằng Precision và Recall tốt hơn trong lần chạy này. Nếu dùng 0.5, model có thể bỏ sót thêm một số vùng công trình.

### Câu 9: Vì sao U-Net có tốc độ suy luận tốt nhưng vẫn nói tốn tài nguyên?

Trả lời:

> Bảng thời gian chỉ đo inference sau khi đã train. U-Net có thể suy luận nhanh trên GPU, nhưng chi phí huấn luyện ban đầu cao hơn nhiều so với Otsu hoặc K-Means. Vì vậy khi so sánh tài nguyên, cần tách chi phí train và chi phí inference.

### Câu 10: Count Error và Area Error có ý nghĩa gì?

Trả lời:

> IoU và Dice đo chồng lấp pixel, nhưng với GIS ta còn quan tâm số vùng công trình và tổng diện tích xây dựng. Count Error phản ánh sai số số connected components, còn Area Error phản ánh sai số tỷ lệ diện tích công trình trong patch. Một mask có Dice khá nhưng gộp nhiều nhà hoặc dự đoán thừa diện tích vẫn có thể không phù hợp cho bản đồ.

### Câu 11: Hậu xử lý morphology có thể gây lỗi không?

Trả lời:

> Có. Closing và opening giúp giảm nhiễu, nhưng cũng có thể gộp các nhà gần nhau hoặc loại bỏ nhà nhỏ nếu component nhỏ hơn ngưỡng 200 pixel. Vì vậy morphology là bước cần tuning theo mục tiêu ứng dụng.

### Câu 12: Nếu làm tiếp, cải thiện quan trọng nhất là gì?

Trả lời:

> Theo nhóm em, có ba hướng quan trọng: dùng kiến trúc mạnh hơn như DeepLabV3+ hoặc SegFormer, bổ sung augmentation sát với ảnh hàng không như bóng đổ và biến đổi ánh sáng, và thêm bước vector hóa mask sang polygon GIS để kết quả dùng được trực tiếp trong bản đồ.

## 5. Checklist trước khi thuyết trình

- Thuộc 4 số chính của U-Net: `IoU 0.544`, `Dice 0.665`, `Precision 0.698`, `Area Error 5.258`.
- Nhớ giải thích Otsu: `Recall 0.802` cao nhưng `Precision 0.233` thấp vì dự đoán thừa.
- Nhớ SVM dùng 13 đặc trưng pixel, không nói nhầm thành 11.
- Nhớ split: 180 ảnh gốc -> 126 train, 27 validation, 27 test; patch 512 x 512; test 300 patch.
- Khi nói thời gian U-Net, nhấn mạnh đó là inference time, chưa tính training time.
- Khi nói output, nhấn mạnh hiện là raster mask, chưa phải polygon GIS.
- Khi nhìn overlay: xanh lá là TP, đỏ là FP, xanh dương là FN.
- Nếu bị hỏi "đóng góp là gì", trả lời: pipeline so sánh công bằng 4 phương pháp, có metrics định lượng, overlay định tính, tuning SVM/U-Net và phân tích lỗi theo hướng ứng dụng GIS.
