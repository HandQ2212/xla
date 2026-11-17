# SLIDE – ghi chú chuẩn bị slide thuyết trình

Đề tài: **Phát hiện và phân đoạn vết nứt mặt đường bằng xử lý ảnh cổ điển** (Lựa chọn 2 – dự án thực tế, không dùng học sâu).
Ghi chú này lập ngày 16/11/2026, trên nhánh `hainn/slide` (HEAD `587735f`).

---

## 1. File chính – đọc theo thứ tự này

| # | File | Vai trò | Dùng cho slide |
|---|---|---|---|
| 1 | `code/crack_segmentation.ipynb` | **Notebook chính** (bộ Pavement Crack Segmentation – PCS). Chạy toàn bộ: dữ liệu → từng bước pipeline (hình trước/sau) → kiểm chứng K1–K10 → chọn tham số → đánh giá test | Hình minh họa từng bước, bảng kết quả |
| 2 | `code/pipeline.py` | **Toàn bộ thuật toán** (199 dòng): tham số `P`, `P0`; `raw_mask` (bước 2–3), `post` (bước 4), `pipeline`, `scores`, `baseline_otsu`, `baseline_canny`. Notebook, app và benchmark đều import file này | Giải thích phương pháp |
| 3 | `PIPELINE.md` | Mô tả từng bước: tham số, lý do chọn, số đo, bảng kết quả test và bảng bỏ từng bước | **Nguồn số liệu chính cho slide** |
| 4 | `README.md` | Tóm tắt đề tài, cài đặt, lệnh chạy, kết quả test, thử nghiệm trên nhánh `hieugm` | Slide kết quả, slide demo |
| 5 | `code/app.py` + `code/pipeline_view.py` | App demo Streamlit: tải ảnh, xem từng bước, so với Otsu/Canny | Slide demo / video demo |
| 6 | `code/crack_segmentation_cfd.ipynb` + `PIPELINE_CFD.md` | Bản gốc trên bộ CFD – pipeline xuất phát, là một dòng đối chứng trong bảng kết quả | Slide "từ CFD sang bộ mới" |
| 7 | `samples_pcs/` (10 ảnh test PCS) và `samples/` (10 ảnh CFD) | Mỗi ảnh có `<id>.jpg`, `<id>_gt.png`, `<id>_pred.jpg` | Ảnh minh họa |
| 8 | `requirements.txt` | opencv-python, numpy, scikit-image, streamlit, requests | – |

### File phụ (chỉ nhắc ở slide "thử nghiệm mở rộng")

| File | Vai trò |
|---|---|
| `code/benchmark.py`, `tests/` (10 file) | Benchmark tái lập và kiểm thử: `python -m unittest discover -s tests` |
| `docs/pipeline-improvement.md`, `docs/pipeline-experiments.md` | Các hướng cải thiện đã thử và **bị loại** (tương phản cục bộ, đa độ phân giải, khôi phục mặt nạ, tái tạo vùng, trung vị 3×3) |
| `docs/external-stress-report.md` | Kiểm tra trên 100 ảnh ngoài (PaveSeg): pipeline báo nhầm 27/36 ảnh âm |
| `docs/performance-report.md` | Tối ưu tốc độ, đầu ra không đổi: 1,083× (mặc định), 1,177× (OpenCV một luồng) |
| `code/experiments.py`, `*_study.py`, `local_*.py`, `multiscale.py`… | Mã nghiên cứu, **không** nằm trong pipeline dùng thực tế (xem `code/RESEARCH.md`) |
| `docs/benchmarks/` (93 file), `plans/` (30 file) | Bằng chứng số liệu và kế hoạch của các thử nghiệm trên |

---

## 2. Code "train" chính ở đâu

Đề tài dùng xử lý ảnh cổ điển nên **không có bước huấn luyện mô hình**. Phần tương đương với "train" là **chọn tham số bằng tìm lưới**, nằm trong `code/crack_segmentation.ipynb`:

| Mục trong notebook | Hàm | Việc làm |
|---|---|---|
| "Chọn tham số: thử toàn bộ lưới trên tune…" | `tune_worker` | Thử **4320 cấu hình** trên 300 ảnh train (tập tune) |
| (ô kế tiếp) | – | Vẽ Dice tune theo từng giá trị tham số |
| (ô kế tiếp) | `val_worker` | Chạy 10 cấu hình tốt nhất trên 498 ảnh val, chọn `P`; có `assert` kiểm tra `P` trùng `pipeline.py` |
| "Bước 5 – Đánh giá trên tập test" | `test_worker` | Chấm 231 ảnh test, gồm đối chứng và bỏ từng bước |

* Lưới thử: `TUNE_SPACE` ở ô đầu notebook. Tham số đã chọn: `P` trong `code/pipeline.py` dòng 11.
* Kết quả ghi ra `results/pcs/`: `split.json`, `tune_log.csv`, `val_log.csv`, `best_params.json`, `test_summary.csv`, `test_by_group.csv`, `test_per_image.csv`. Thư mục `results/` và `data/` bị `.gitignore`, chỉ có trên máy.
* Bản CFD có phần chọn tham số tương ứng trong `code/crack_segmentation_cfd.ipynb`.

**Lưu ý:** ô đầu notebook ghi rõ các đầu ra đang lưu là kết quả lịch sử từ nhánh `mailinh`, chưa chạy lại toàn bộ sau khi chuyển hàm sang `pipeline.py`. Nên chạy lại notebook một lần trước khi nộp.

---

## 3. Thay đổi so với commit ngay trước merge

Merge `a65c170` = PR #2, nhánh `hieugm` → `main` (16/11/2026).
Commit trước merge: `2592bf7` "Update crack segmentation pipelines and samples" (14/11). So sánh `2592bf7..a65c170`: **173 file, +54.270 / −461 dòng**.

Trước merge, repo chỉ có: `README.md`, `PIPELINE.md`, `PIPELINE_CFD.md`, `requirements.txt`, hai notebook và ảnh mẫu.

| Nhóm | Thay đổi |
|---|---|
| **Notebook chính** `code/crack_segmentation.ipynb` (+150 / −458) | 28 hàm xử lý (`to_gray`, `denoise`, `illumination_ratio`, `triangle_threshold`, `edge_region`, `closing`, `elongation_filter`, `image_decision`, `pipeline`, `scores`, hai đối chứng…) được **chuyển sang `code/pipeline.py`**; notebook chỉ còn `from pipeline import …`. Mã trong notebook giảm từ 610 xuống 487 dòng, vẫn 51 ô. Giữ lại 17 hàm điều phối (`tune_worker`, `val_worker`, `test_worker`, `compare`, `summarize`…) |
| **Thêm mới trong `code/`** (32 file) | `pipeline.py`, `app.py`, `pipeline_view.py`, `fast_median.py`, `benchmark.py`, cùng các module thử nghiệm và kiểm tra ngoài miền |
| **Thêm `tests/`** (10 file) | Kiểm thử pipeline, app, benchmark, các module thử nghiệm |
| **Thêm `docs/`** (4 báo cáo + 93 file bằng chứng) và `plans/` (30 file) | Báo cáo thử nghiệm, kiểm tra ảnh ngoài, hiệu năng |
| `README.md` (+44 / −3) | Thêm lệnh chạy app Streamlit, mục "Thử nghiệm cải thiện trên nhánh `hieugm`", lệnh test và benchmark |
| `PIPELINE.md` (+1) | Một dòng ghi chú về bản `hieugm` |
| `requirements.txt` (+2) | Thêm `streamlit`, `requests` |
| **Không đổi** | `code/crack_segmentation_cfd.ipynb`, `PIPELINE_CFD.md`, ảnh mẫu |

Sáu commit được merge: `d7d4f45` (MaiLinh – thêm Streamlit), `b927189`, `d155f98`, `2d55aed`, `b2f7f3b`, `5630760` (HieuGM – thử nghiệm, kiểm tra ngoài miền, tối ưu tốc độ).

**Điểm quan trọng cho slide:** thuật toán và tham số `P` **không đổi** sau merge, nên mọi con số kết quả (Dice test 0.339…) vẫn giữ nguyên. Cải thiện duy nhất được giữ lại là tốc độ, với đầu ra trùng từng pixel. Các hướng tăng độ chính xác đều bị loại sau kiểm chứng.

Sau merge, commit `587735f` chỉ thêm `slide/SLIDE.md` (rỗng) và `.DS_Store`.

---

## 4. Yêu cầu của đề (`DoAn_XuLyAnh.pdf`) liên quan tới slide

* **10–15 slide**, trình bày 10 phút + 5 phút hỏi đáp; nộp **.pptx và .pdf**.
* Lựa chọn 2 phải có pipeline 5 bước: **Dữ liệu** (nguồn, số ảnh, cách gán nhãn, ≥ 50 ảnh đánh giá) → **Tiền xử lý** → **Xử lý chính** → **Hậu xử lý** → **Đánh giá** (độ đo trên tập kiểm tra, so với đối chứng). Mỗi bước ghi rõ tham số và lý do chọn.
* Dùng ít nhất 1 chương của học phần (chương 1–8) và ghi rõ.
* Gói nộp: `LopXX_NhomYY_TenDeTai.zip` gồm slide, `code/`, `requirements.txt`, 5–10 ảnh mẫu trong `samples/`, README, báo cáo PDF 4–10 trang (có phân công và khai báo sử dụng AI), video demo ≤ 3 phút (khuyến khích).

### Việc còn thiếu so với đề

- [ ] `PIPELINE.md` và notebook ghi bước Dữ liệu thuộc "Chương 9", trong khi đề chỉ nêu chương 1–8. Cần đối chiếu lại số chương với giáo trình.
- [ ] `PIPELINE.md` mục 3.2 ghi "Otsu gán 56.9% ảnh là nứt; tam giác gán 1.5%". Chạy lại ngày 16/11/2026 với code hiện tại ra **43.9% và 1.7%** (ảnh B_00447, đúng ảnh trong hình lược đồ); notebook cũng in 43.9% và 1.7%. Slide đã dùng số chạy lại.
- [ ] Notebook đang lưu đầu ra không đồng bộ: ô grid search dừng ở `KeyboardInterrupt`; bảng K9 ghi "không lọc diện tích 0.304" và K5 ghi "W = 5: 0.318", trong khi phần tóm tắt ngay bên dưới ghi 0.307 và 0.319. Chạy lại trên máy này ra 0.307 và 0.3185. Cần chạy lại toàn bộ notebook một lần trước khi nộp.
- [ ] README chưa có **đường dẫn tải dữ liệu** PCS (đề yêu cầu).
- [ ] Chưa có **báo cáo PDF**, bảng phân công và khai báo sử dụng AI trong repo.
- [ ] Chưa có file slide `.pptx` và `.pdf` trong thư mục `slide/` (tải từ deck ở mục 5).
- [ ] Chưa có video demo (khuyến khích, tối đa 3 phút).
- [ ] Hạn nộp ghi trong đề là 11/11/2026.

---

## 5. Slide đã dựng (15 slide, khoảng 10 phút)

Deck: https://claude.ai/artifact/Amm2T3XbnVmrvJ5q3dYVyt (riêng tư; muốn cả nhóm xem thì chia sẻ từ menu Share của trang).
Mỗi slide có ghi chú người nói kèm thời lượng gợi ý. Để nộp bài, tải deck về dạng **.pptx và .pdf** từ trang deck rồi đặt vào thư mục `slide/`.

| # | Slide | Nội dung chính | Nguồn |
|---|---|---|---|
| 1 | Bìa | Tên đề tài, giảng viên, nhóm 06, thành viên | – |
| 2 | Bài toán | Đầu vào/đầu ra; 3 khó khăn: vết nứt ≈ 0.9% pixel, chiếu sáng không đều, 19% ảnh không nứt | notebook ô đầu |
| 3 | Dữ liệu | PCS: train 1480 / val 498 / test 231; 3 nhóm A, B, C; nhãn = xám > 127; tune 300 ảnh | `PIPELINE.md` Bước 1 |
| 4 | Pipeline tổng quan | Sơ đồ 5 bước; một ảnh mẫu đi qua toàn bộ pipeline | `PIPELINE.md` mục 1 |
| 5 | Tiền xử lý | Resize 640, grayscale, median filter 5×5, ratio image R = 128·I/B; hình trước/sau illumination correction | Bước 2 |
| 6 | Xử lý chính | Triangle threshold → D; Sobel + Otsu + dilation → E; M = D ∩ E; histogram Otsu so với triangle | Bước 3 |
| 7 | Hậu xử lý | Closing 7×7, area filter < 30 px, elongation filter L/S < 3, image-level decision 0.3% | Bước 4 |
| 8 | Kiểm chứng lựa chọn | Biểu đồ thanh: thay một lựa chọn, đo Dice tune (K1–K10) | notebook "Tóm tắt kiểm chứng" |
| 9 | Chọn tham số | Grid search 4320 cấu hình → top 10 trên val → `P` (Dice val 0.364); bảng giá trị thử và giá trị đã chọn | notebook "Chọn tham số" |
| 10 | Kết quả test | Baseline Otsu 0.024 · Canny 0.018 · **pipeline của nhóm 0.339** | `PIPELINE.md` Bước 5 |
| 11 | So sánh trực quan | Năm ảnh test: gốc, ground truth, Otsu, Canny, pipeline, error map | notebook, mục trực quan hóa |
| 12 | Ablation | Biểu đồ bỏ từng bước; đánh đổi của bước 4.4 (22/45 và 45/186) | `PIPELINE.md` Bước 5 |
| 13 | Phân tích lỗi | False negative: nứt mờ ở xa; false positive: vạch sơn, khe nối, nắp cống; Dice theo nhóm B 0.554, C 0.231, A 0.126 | notebook "Thảo luận" |
| 14 | Giới hạn và thử nghiệm mở rộng | Out-of-domain PaveSeg: false positive ở 27/36 ảnh; các hướng bị loại; tăng tốc 1.08–1.18×; app Streamlit | `docs/*.md` |
| 15 | Kết luận | Ba con số chính, kỹ thuật đã dùng, hướng cải thiện | – |

Slide chỉ báo cáo pipeline hiện tại (tham số `P`): không đưa bản CFD cũ và tham số khởi đầu `P0` vào bảng hay lời nói. Các so sánh đó vẫn có trong `PIPELINE.md` nếu thầy hỏi.

Quy ước trên slide: giữ thuật ngữ tiếng Anh (mask, ground truth, threshold, histogram, median filter, closing, baseline, false positive, grid search, overfitting…) và **không nhắc số chương**. Tên năm bước lớn vẫn theo đúng chữ của đề: Dữ liệu, Tiền xử lý, Xử lý chính, Hậu xử lý, Đánh giá.

Hình trong slide lấy từ đầu ra đã lưu sẵn của `code/crack_segmentation.ipynb` và từ `samples_pcs/B_00647*.jpg`.

### Số liệu chốt để nói

* Test 231 ảnh (45 ảnh không nứt): Dice **0.339**, IoU **0.274**, Dice gộp **0.426**, ảnh lành trả rỗng đúng **22/45**.
* Tune → val → test: 0.32 → 0.36 → 0.34 (không khớp quá mức).
* Lựa chọn quan trọng nhất: ngưỡng tam giác thay Otsu (Dice tune 0.324 so với 0.082).
* Đánh đổi của 4.4: 22/45 ảnh lành đúng, nhưng 45/186 ảnh có nứt nhỏ bị trả rỗng; Dice ảnh nứt 0.303 so với 0.329 khi bỏ 4.4.
* Giới hạn: ra ngoài miền dữ liệu, pipeline báo nhầm 27/36 ảnh âm (75%).
