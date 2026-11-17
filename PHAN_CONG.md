# Phân công công việc – Nhóm 06, Lớp D23CTPM01

Đề tài: Phát hiện và phân đoạn vết nứt mặt đường bằng xử lý ảnh cổ điển.

Phần **mã nguồn** ghi theo lịch sử git của repo. Các phần **slide, thuyết trình, báo cáo, README, video, đóng gói** là việc còn lại, được chia để tổng khối lượng của bốn người tương đương nhau.

## Tổng quan

| Thành viên | Mã nguồn và thử nghiệm | Slide (soạn và trình bày) | Báo cáo | Việc khác |
|---|---|---|---|---|
| **Nguyễn Nam Hải** | Xây pipeline trên CFD rồi chuyển sang bộ PCS; notebook chính; grid search; `PIPELINE.md` | 4–7: Pipeline, Tiền xử lý, Xử lý chính, Hậu xử lý | Mục 2, 3: Phương pháp | Quản lý repo, review và merge PR, nộp bài qua email |
| **Giáp Minh Hiếu** | Tách `code/pipeline.py` dùng chung; 10 file test; benchmark; thử nghiệm cải thiện; kiểm tra ngoài miền (PaveSeg); tối ưu tốc độ | 8–10: Kiểm chứng, Chọn tham số, Kết quả | Mục 4, 5 và 7.3 | Trả lời câu hỏi về thực nghiệm khi vấn đáp |
| **Văn Thị Mai Linh** | App demo Streamlit (`code/app.py`, `code/pipeline_view.py`) | 11–14: So sánh trực quan, Ablation, Phân tích lỗi, Giới hạn | Mục 6 và 7.1–7.2 | README (cài đặt, lệnh chạy, đầu ra); video demo ≤ 3 phút |
| **Nguyễn Văn Minh Lực** | Chuẩn bị dữ liệu cho bản nộp: link tải dataset, kiểm tra nhãn và số ảnh; chạy lại notebook lần cuối và sửa các số lệch trong `PIPELINE.md` | 1–3 và 15: Bìa, Bài toán, Dữ liệu, Kết luận | Lời giới thiệu, Mục 1, 7.4 và 8 | Ghép và định dạng báo cáo PDF; xuất slide .pptx và .pdf; đóng gói file zip |

Mỗi người có 3–4 slide, khoảng 2.5 phút trình bày. Thứ tự nói: Lực (1–3) → Hải (4–7) → Hiếu (8–10) → Linh (11–14) → Lực (15, kết luận và mời hỏi đáp).

## Báo cáo (theo file `Bao_cao_XLA_Nhom_06.pdf`)

| Mục | Nội dung | Người viết |
|---|---|---|
| Mục lục, Lời giới thiệu, Phân công | Phần đầu báo cáo | Lực |
| 1. Bài toán và dữ liệu | Mục tiêu, phạm vi, bộ PCS, nhãn | Lực |
| 2. Phương pháp: tiền xử lý | Resize, grayscale, median, bù sáng | Hải |
| 3. Phương pháp: phân đoạn và hậu xử lý | Vùng tối, vùng gần biên, hình thái, quyết định mức ảnh | Hải |
| 4. Chọn tham số và thiết kế đánh giá | Grid search, độ đo, quy ước ảnh âm | Hiếu |
| 5. Kết quả định lượng | So sánh test, ablation, theo nhóm ảnh | Hiếu |
| 6. Kết quả trực quan và phân tích lỗi | Ba ca minh họa, bản đồ lỗi | Linh |
| 7.1–7.2 Thảo luận, giới hạn | Đánh đổi FP/FN, kiểm tra ngoài PCS | Linh |
| 7.3 Hiệu năng và tái lập | Benchmark, kiểm tra tương đương | Hiếu |
| 7.4 Kết luận; 8. Khai báo AI; ghép PDF | | Lực |

## Mốc thời gian gợi ý

1. Mỗi người hoàn thiện slide và ghi chú người nói của phần mình.
2. Gửi bản nháp mục báo cáo cho Lực.
3. Tập thuyết trình chung một lần, bấm giờ 10 phút.
4. Hải rà soát lần cuối và nộp `LopXX_NhomYY_TenDeTai.zip`.
