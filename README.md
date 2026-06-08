# Project 16 Final - Building Footprint Extraction

Workspace này là bản mở rộng cuối kỳ cho bài toán `Building Footprint Extraction from Aerial Images`.

## Mục tiêu

So sánh 4 hướng phân đoạn vùng tòa nhà trên cùng một split dữ liệu:

- `K-Means`: baseline classical từ giữa kỳ.
- `Otsu`: baseline thresholding từ giữa kỳ.
- `Linear SVM`: supervised pixel-level baseline với handcrafted features.
- `U-Net`: deep learning semantic segmentation model chạy trên Kaggle GPU.

## Cấu trúc

- `final_building_footprint_kaggle_single.ipynb`: notebook Kaggle khuyến nghị, tự chứa toàn bộ code và chỉ cần upload riêng file `.ipynb`.
- `final_building_footprint.ipynb`: notebook cũ, dùng cơ chế tự tạo module từ source nhúng.
- `src/building_footprint_final.py`: toàn bộ pipeline dùng lại được cho notebook hoặc CLI.
- `reports/main.tex`: source báo cáo cuối kỳ, đọc bảng metrics thật từ `outputs/reports`.
- `reports/report.pdf`: bản PDF báo cáo đã render.
- `file_nop/`: bộ file nộp gồm notebook, báo cáo, slide PDF, readme và kịch bản thuyết trình.
- `outputs/`: nơi notebook/script ghi metrics, figures, model và report snippets.

## Dataset

Dùng `Inria Aerial Image Labeling Dataset`, dạng thư mục:

```text
AerialImageDataset/
  train/
    images/
    gt/
```

Notebook tự tìm dataset ở các path Kaggle phổ biến. Nếu cần chỉ định tay, sửa:

```python
cfg = ExperimentConfig(dataset_root="/path/to/AerialImageDataset")
```

## Chạy trên Kaggle

1. Tạo Kaggle Notebook và bật GPU.
2. Attach dataset Inria Aerial Image Labeling.
3. Upload/import `final_building_footprint_kaggle_single.ipynb`.
4. Chạy từ trên xuống bằng `Run All`.

Notebook `final_building_footprint_kaggle_single.ipynb` không cần upload thêm `.py`; phần implementation đã nằm trong các code cell thường của notebook.

Output chính sau khi chạy:

```text
outputs/metrics/final_summary.csv
outputs/metrics/per_patch_metrics.csv
outputs/metrics/svm_tuning.csv
outputs/metrics/unet_training_history.csv
outputs/figures/method_comparison.png
outputs/figures/qualitative_grid_*.png
outputs/models/linear_svm_building_footprint.joblib
outputs/models/unet_best.pth
outputs/reports/final_metrics_table.tex
```

## Chạy nhanh bằng CLI

Local smoke test không cần dataset:

```powershell
py -3 src\building_footprint_final.py --self-test
```

Debug nhanh khi có dataset:

```powershell
py -3 src\building_footprint_final.py --dataset-root D:\datasets\AerialImageDataset --quick --skip-unet
```

Full run trên môi trường có GPU:

```powershell
py -3 src\building_footprint_final.py --dataset-root D:\datasets\AerialImageDataset
```

## Chạy GUI app

GUI dùng `Streamlit` để demo inference trên ảnh mới. Ứng dụng không retrain model; nó load lại artifact đã có trong `outputs/models` và threshold đã chọn trong `outputs/metrics/unet_selected_threshold.json`.

Cài dependency chính và dependency GUI:

```powershell
py -3 -m pip install -r requirements.txt
py -3 -m pip install -r requirements-gui.txt
```

Chạy app:

```powershell
py -3 -m streamlit run app\streamlit_app.py
```

Trong app có thể upload ảnh aerial `.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, chọn `K-Means`, `Otsu`, `SVM`, hoặc `U-Net`, rồi xem mask dự đoán, overlay và các thống kê cơ bản. Nếu upload thêm ground-truth mask, app sẽ hiển thị IoU, Dice/F1, Precision, Recall, count error, area error và error overlay theo quy ước green=TP, red=FP, blue=FN.

Đường chạy notebook/CLI phía trên vẫn là phần full training/evaluation của đề tài. GUI chỉ là ứng dụng trọn gói để trình diễn mô hình đã train.

## Ghi chú báo cáo

Không nhập tay số liệu vào báo cáo. Sau khi chạy notebook, dùng số từ:

- `outputs/metrics/final_summary.csv`
- `outputs/reports/final_metrics_table.tex`
- `outputs/reports/final_metrics_summary.tex`

Báo cáo cần tập trung vào ứng dụng geospatial: segmentation overlay, building count, area ratio, lỗi do shadow, roof color, small roofs, road/parking confusion và false merge.
