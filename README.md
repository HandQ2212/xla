# Phát hiện và phân đoạn vết nứt mặt đường bằng xử lý ảnh cổ điển

Đồ án Xử lý ảnh – Lựa chọn 2 (dự án thực tế). Không dùng học sâu hay mô hình huấn luyện sẵn.

**Đầu vào:** ảnh gốc chụp mặt đường (RGB, kích thước bất kỳ).
**Đầu ra:** mặt nạ nhị phân vết nứt cùng kích thước với ảnh gốc, và ảnh phủ mặt nạ đỏ.

## Pipeline
| Bước | Nội dung | Chương |
|---|---|---|
| Dữ liệu | CFD (118 ảnh) + CRACK500 (249 ảnh gốc), GT vẽ tay mức pixel; tune/test = 60/58 và 49/200 | – |
| Tiền xử lý | Cạnh dài → 480 px, ảnh xám, lọc trung vị 3×3, bù chiếu sáng R = 128·I/B (B = trung vị 41–61) | C2, C3, C4 |
| Xử lý chính | Làm trơn lược đồ (W = 3) + ngưỡng tam giác (Zack) trên ảnh R | C5 |
| Hậu xử lý | Phép đóng 5×5 → lọc vùng theo diện tích (≥ 30/60 px) → lọc theo độ dãn dài (≥ 3, tính từ mômen) | C7, C8 |
| Đánh giá | Dice, IoU (kèm Precision, Recall) ở độ phân giải gốc; đối chứng Otsu và Canny; nghiên cứu cắt bỏ | C8 |

## Kết quả trên tập test (Dice / IoU trung bình theo ảnh)
| Phương pháp | CFD (58 ảnh) | CRACK500 (200 ảnh) |
|---|---|---|
| Otsu (đối chứng) | 0.058 / 0.030 | 0.076 / 0.040 |
| Canny (đối chứng) | 0.053 / 0.029 | 0.047 / 0.024 |
| **Pipeline đề xuất** | **0.575 / 0.414** | **0.549 / 0.400** |

## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter matplotlib pandas
```

## Dữ liệu
Đặt thư mục `pavement crack datasets/` (gồm `CFD/` và `CRACK500/` dạng zip) **cạnh** thư mục `DoAn/`:
```
XLA/
├── pavement crack datasets/   (CFD/cfd_image.zip, CFD/cfd_gt.zip, CRACK500/valdata.zip, CRACK500/testdata.zip)
└── DoAn/
    ├── code/crack_segmentation.ipynb
    ├── samples/               5 ảnh test CFD + 5 ảnh test CRACK500, kèm GT (*_gt.png)
    ├── data/                  (sinh tự động khi chạy)
    └── results/               (sinh tự động khi chạy)
```
Nguồn tải: CFD – https://github.com/cuilimeng/CrackForest-dataset ; CRACK500 – https://github.com/fyangneil/pavement-crack-detection

## Chạy
```bash
cd code
jupyter nbconvert --to notebook --execute --inplace crack_segmentation.ipynb   # hoặc mở bằng Jupyter rồi Run All
```
* Lần chạy đầu sẽ giải nén dữ liệu vào `data/`. Toàn bộ notebook mất khoảng 6 phút trên laptop (phần tìm tham số khoảng 4.5 phút).
* Đặt `RUN_TUNING = False` ở cell đầu để bỏ qua bước tìm tham số và dùng lại `results/best_params.json`.

## Đầu ra mong đợi
* `results/best_params.json`: tham số tốt nhất cho từng tập và từng phương pháp.
* `results/tune_log_*.csv`: toàn bộ các cấu hình đã thử khi tìm tham số.
* `results/test_summary_{cfd,crack500}.csv`: bảng Dice/IoU trên tập test.
* `results/test_per_image_*.csv`: độ đo của từng ảnh.
* Trong notebook: 5 ảnh trước và sau cho mỗi bước biến đổi không gian ảnh, lược đồ và ngưỡng, ảnh phủ mặt nạ đỏ, bản đồ lỗi TP/FP/FN.
# xla
