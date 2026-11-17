# Phát hiện và phân đoạn vết nứt mặt đường bằng xử lý ảnh cổ điển

> **Notebook chọn tham số và đánh giá:** [`code/crack_segmentation.ipynb`](code/crack_segmentation.ipynb)
> **Thuật toán:** [`code/pipeline.py`](code/pipeline.py) · **App demo:** [`code/app.py`](code/app.py)

Chương trình nhận một ảnh RGB chụp mặt đường và trả về:

- **binary mask** cùng kích thước ảnh gốc, 1 là pixel vết nứt; ảnh không có vết nứt cho mask rỗng;
- **ảnh gốc phủ đỏ** vùng nghi nứt để kiểm tra trực quan.

Toàn bộ pipeline dùng kỹ thuật xử lý ảnh cổ điển (OpenCV, NumPy, scikit-image). Không dùng học sâu hay mô hình huấn luyện sẵn; tham số được chọn bằng grid search trên dữ liệu train/val.

## Pipeline

Tham số đã chọn là `P` trong [`code/pipeline.py`](code/pipeline.py).

| Bước | Việc làm | Tham số `P` |
|---|---|---|
| 2.0 Resize | Đưa cạnh dài về kích thước cố định, `INTER_AREA` | 640 px |
| 2.1 Grayscale | Y = 0.299R + 0.587G + 0.114B | – |
| 2.2 Median filter | Giảm nhiễu hạt sỏi | 5×5 |
| 2.3 Ratio image | R = 128·I / (B + 1), B là median kernel lớn của ảnh | B: 71×71 |
| 3.1 Histogram smoothing | Trung bình trượt trên histogram của R | W = 3 |
| 3.2 Vùng tối D | Triangle threshold θ, D = (R < θ) | θ tự động |
| 3.3 Vùng gần edge E | Sobel \|Gx\| + \|Gy\| → Otsu → dilation ellipse | 3×3 |
| 3.4 Mask thô | M = D ∩ E | – |
| 4.1 Closing | Nối các đoạn nứt bị đứt, ellipse | 7×7 |
| 4.2 Area filter | Bỏ connected component nhỏ | < 30 px |
| 4.3 Elongation filter | Bỏ vùng có tỉ số cạnh dài/ngắn của rotated bounding box nhỏ | L/S < 3 |
| 4.4 Image-level decision | Tổng diện tích mask nhỏ hơn ngưỡng thì trả mask rỗng | < 0.3% ảnh |
| Đầu ra | Phóng mask về kích thước gốc (nearest neighbor) | – |

## Chọn tham số

| Tập | Số ảnh | Ảnh không nứt | Dùng để |
|---|---|---|---|
| tune (lấy từ train) | 300 (A 30, B 90, C 180; seed 42) | – | Chạy toàn bộ grid search |
| val | 498 | 115 | Chọn 1 trong 10 cấu hình tốt nhất trên tune |
| test | 231 | 45 | Đánh giá cấu hình đã chọn, một lần |

Grid gồm 9 tham số, tổng cộng **4320 cấu hình**:

| Tham số | Giá trị thử | Chọn |
|---|---|---|
| Resize (cạnh dài) | 480, 640 | 640 |
| Median filter | 3, 5 | 5 |
| Background kernel | 31, 51, 71 | 71 |
| Histogram smoothing W | 3, 5 | 3 |
| Edge dilation | 3, 5, 7 | 3 |
| Closing | 5, 7 | 7 |
| Min area (px) | 30, 60, 120 | 30 |
| Min elongation L/S | 2, 3 | 3 |
| Image-level threshold (%) | 0, 0.1, 0.2, 0.3, 0.4 | 0.3 |

Mười cấu hình có Dice cao nhất trên tune được chạy lại trên toàn bộ val ở kích thước gốc. Cấu hình có Dice val cao nhất (**0.364**) được chọn làm `P`; notebook kiểm tra lại bằng `assert` rằng cấu hình này trùng với `P` trong code.

## Kết quả trên tập test (231 ảnh, 45 ảnh không nứt)

Đánh giá ở kích thước gốc. Với ảnh không nứt: mask dự đoán rỗng thì Dice = IoU = 1, ngược lại bằng 0. "Dice ảnh nứt" chỉ tính trên 186 ảnh có nứt. "Dice gộp" cộng TP, FP, FN của mọi ảnh rồi mới tính.

| Phương pháp | Dice | IoU | Dice ảnh nứt | Dice gộp | Ảnh không nứt nhận đúng |
|---|---|---|---|---|---|
| Otsu (baseline) | 0.024 | 0.012 | 0.030 | 0.027 | 0/45 |
| Canny (baseline) | 0.018 | 0.009 | 0.022 | 0.019 | 0/45 |
| **Pipeline, tham số `P`** | **0.339** | **0.274** | **0.303** | **0.426** | **22/45** |

Bước 4.4 giúp nhận đúng 22/45 ảnh không nứt, đổi lại 45/186 ảnh có vết nứt nhỏ bị trả mask rỗng. Vết trám, khe nối, vạch sơn và bóng đổ vẫn có thể bị nhận nhầm là vết nứt.

## Dữ liệu

Pavement Crack Segmentation Dataset (2.209 cặp ảnh và mask, đã chia sẵn train/val/test):
https://zenodo.org/records/20040434 (DOI: 10.5281/zenodo.20040434)

Giải nén vào `data/` theo cấu trúc:

```
data/Pavement_Crack_Segmentation_Dataset/
├── train/{images/*.jpg, masks/*.png}
├── val/{images/*.jpg, masks/*.png}
└── test/{images/*.jpg, masks/*.png}
```

Mask được đọc ở chế độ grayscale và lấy ngưỡng > 127 để thành nhãn nhị phân. Thư mục `samples_pcs/` kèm repo có 10 ảnh test (`<id>.jpg`, `<id>_gt.png`, `<id>_pred.jpg`) để chạy thử mà không cần tải toàn bộ dữ liệu.

## Cài đặt

Python 3.10 trở lên.

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

Trên Windows dùng `.venv\Scripts\Activate.ps1` thay cho lệnh trên.

```bash
python -m pip install -r requirements.txt jupyter matplotlib pandas
```

## Chạy

**App demo (Streamlit)**, dùng ngay 10 ảnh mẫu, không cần tải dữ liệu. Chạy từ thư mục gốc repo:

```bash
python -m streamlit run code/app.py
```

App mở ở http://localhost:8501: kéo thả ảnh hoặc chọn ảnh mẫu, xem mask, so sánh trước/sau, xem từng bước của pipeline, so với Otsu và Canny, bật/tắt từng bước. Ảnh mẫu có ground truth nên app hiện thêm Dice, IoU, Precision, Recall.

**Notebook chọn tham số và đánh giá** [`code/crack_segmentation.ipynb`](code/crack_segmentation.ipynb) (cần bộ dữ liệu đầy đủ):

```bash
cd code && jupyter nbconvert --to notebook --execute --inplace crack_segmentation.ipynb
```

Notebook chạy song song theo ảnh bằng `multiprocessing` (kiểu fork; trên Windows sẽ tự chạy tuần tự và lâu hơn). Trên laptop 10 nhân, grid search mất khoảng 2 phút và cả notebook khoảng 3 phút.

**Kiểm thử:**

```bash
python -m unittest discover -s tests
```

**Benchmark** trên một tập của bộ dữ liệu (mặc định `val`):

```bash
python code/benchmark.py --data-root data/Pavement_Crack_Segmentation_Dataset --split val --output results/benchmark-val
```

Không truyền `--data-root` thì benchmark chỉ chạy trên các ảnh mẫu kèm repo.

## Đầu ra

Notebook ghi vào `results/pcs/`:

| File | Nội dung |
|---|---|
| `split.json` | Seed và danh sách ảnh tune / val / test |
| `tune_log.csv` | 4320 cấu hình với Dice, IoU trên tune |
| `val_log.csv` | 10 cấu hình tốt nhất chạy lại trên val |
| `best_params.json` | Tham số đã chọn (`P`) |
| `test_summary.csv`, `test_by_group.csv`, `test_per_image.csv` | Kết quả test: tổng hợp, theo nhóm ảnh A/B/C, từng ảnh |
| `samples_pcs/` | 10 ảnh test mẫu kèm GT và kết quả phủ đỏ |

Benchmark ghi `per_image.csv`, `summary.csv` và `metadata.json` vào thư mục `--output`.

## Cấu trúc mã nguồn

| File | Vai trò |
|---|---|
| `code/pipeline.py` | Toàn bộ thuật toán, tham số `P`, độ đo và hai baseline Otsu, Canny |
| [`code/crack_segmentation.ipynb`](code/crack_segmentation.ipynb) | Notebook chọn tham số: dữ liệu, minh họa từng bước, grid search, chọn trên val, đánh giá test |
| `code/app.py`, `code/pipeline_view.py` | App demo Streamlit |
| `code/fast_median.py` | Median filter chia mảnh cho kernel lớn, cho kết quả trùng `cv2.medianBlur` |
| `code/benchmark.py` | Chạy pipeline trên một tập và ghi kết quả từng ảnh |
| `tests/` | Kiểm thử pipeline, app và benchmark |
