# Phát hiện và phân đoạn vết nứt mặt đường bằng xử lý ảnh cổ điển

Đồ án Xử lý ảnh – Lựa chọn 2 (dự án thực tế). Không dùng học sâu hay mô hình huấn luyện sẵn.

**Đầu vào:** ảnh gốc RGB chụp mặt đường (Pavement Crack Segmentation Dataset: 1702×866 hoặc 1080×760).
**Đầu ra:** mặt nạ nhị phân vết nứt cùng kích thước ảnh gốc (ảnh không nứt cho mặt nạ rỗng), và ảnh gốc phủ mặt nạ đỏ.

Pipeline được xây trước trên bộ CFD (`code/crack_segmentation_cfd.ipynb`, `PIPELINE_CFD.md`), rồi chỉnh lại cho bộ dữ liệu hiện tại (`code/crack_segmentation.ipynb`, `PIPELINE.md`).

## Pipeline
| Bước | Việc làm | Chương | CFD → **Chọn cho bộ mới** |
|---|---|---|---|
| 1. Dữ liệu | Chia sẵn train 1480 / val 498 / test 231; nhãn = xám > 127 | 9 | tune 300 ảnh train phân tầng; chọn trên val |
| **2.0 Thu nhỏ (mới)** | Cạnh dài 640 px, INTER_AREA | 2 | – → **640** |
| 2.1 Ảnh xám | Y = 0.299R + 0.587G + 0.114B | 4 | – |
| 2.2 Khử nhiễu | Lọc trung vị | 3 | 3×3 → **5×5** |
| 2.3 Bù chiếu sáng | B = trung vị cửa sổ lớn; R = 128·I/B | 3 | 31 → **71** |
| 3.1 Làm trơn lược đồ | Trung bình trượt W | 5 | 3 → **3** |
| 3.2 Vùng tối D | Ngưỡng tam giác, D = (R < θ) | 5 | tự động |
| 3.3 Vùng gần biên E | Sobel \|Gx\| + \|Gy\| → Otsu → giãn ellipse | 5, 7 | 7×7 → **3×3** |
| 3.4 Mặt nạ thô | M = D ∩ E | – | – |
| 4.1 Phép đóng | Ellipse | 7 | 7×7 → **7×7** |
| 4.2 Lọc diện tích | Bỏ vùng nhỏ | 7 | 60 → **30** px |
| 4.3 Lọc độ dãn | L/S của hộp chữ nhật xoay | 7 | 2 → **3** |
| **4.4 Quyết định mức ảnh (mới)** | Tổng diện tích < t_min% ảnh thì mặt nạ rỗng | 7 | – → **0.3%** |
| 5. Đánh giá | Dice, IoU, Precision, Recall ở kích thước gốc; so với Otsu, Canny, pipeline CFD | 5 | – |

Lưới 4320 cấu hình trên 300 ảnh train; 10 cấu hình tốt nhất được xác nhận trên 498 ảnh val. Chi tiết từng bước kèm mô phỏng bằng text: xem `PIPELINE.md`.

## Kết quả trên tập test (231 ảnh, 45 ảnh không nứt)
Ảnh không nứt: dự đoán rỗng thì Dice = IoU = 1, ngược lại 0. "Dice ảnh nứt" chỉ tính trên 186 ảnh có nứt. "Gộp" là cộng TP/FP/FN của mọi ảnh rồi mới tính.

| Phương pháp | Dice | IoU | Dice ảnh nứt | Dice gộp | Ảnh lành đúng |
|---|---|---|---|---|---|
| Otsu (đối chứng) | 0.024 | 0.012 | 0.030 | 0.027 | 0/45 |
| Canny (đối chứng) | 0.018 | 0.009 | 0.022 | 0.019 | 0/45 |
| Pipeline CFD (tham số CFD) | 0.271 | 0.197 | **0.337** | 0.349 | 0/45 |
| **Pipeline mới** | **0.339** | **0.274** | 0.303 | **0.426** | **22/45** |

Mức tăng chủ yếu đến từ ảnh không nứt (bước 4.4) và từ việc báo nhầm ít hơn. Đổi lại, 45/186 ảnh có nứt nhỏ bị trả rỗng, nên Dice trên ảnh có nứt giảm nhẹ.

## Cài đặt
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt jupyter matplotlib pandas scikit-image
```

## Dữ liệu
```
DoAn/
├── data/Pavement_Crack_Segmentation_Dataset/{train,val,test}/{images/*.jpg, masks/*.png}
├── code/crack_segmentation.ipynb        (bộ hiện tại)
├── code/crack_segmentation_cfd.ipynb    (bản CFD; cần "pavement crack datasets/CFD" cạnh DoAn/)
├── samples_pcs/   10 ảnh test: <id>.jpg, <id>_gt.png, <id>_pred.jpg (kết quả phủ đỏ)
└── results/pcs/   (sinh tự động khi chạy)
```

## Chạy
```bash
cd code
jupyter nbconvert --to notebook --execute --inplace crack_segmentation.ipynb   # hoặc mở bằng Jupyter rồi Run All
```
Notebook chạy khoảng 2 phút trên laptop 10 nhân (song song theo ảnh bằng `multiprocessing`, kiểu fork). Riêng phần tìm tham số mất khoảng 2 phút.

## Đầu ra (`results/pcs/`)
* `split.json`: seed, danh sách ảnh tune / val / test.
* `tune_log.csv`: 4320 cấu hình với Dice/IoU trên tune.
* `val_log.csv`: 10 cấu hình tốt nhất, cùng P0 và tham số CFD, trên val.
* `best_params.json`: tham số đã chọn.
* `test_summary.csv`, `test_by_group.csv`, `test_per_image.csv`: kết quả test (tổng hợp, theo nhóm A/B/C, từng ảnh).
* Trong notebook: 5 ảnh trước và sau cho mỗi bước biến đổi, lược đồ và ngưỡng, ảnh phủ mặt nạ đỏ, bản đồ lỗi TP/FP/FN.
# xla
