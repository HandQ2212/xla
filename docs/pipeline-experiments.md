# Khảo sát cải thiện pipeline bằng xử lý ảnh cổ điển

Nhánh `hieugm` phát triển từ `mailinh`, vốn đã bao gồm `develop`. Mục tiêu là thử cải thiện phân đoạn và sửa sai lệch giữa module, notebook, app. Không dùng học sâu, mô hình huấn luyện sẵn hay kỹ thuật ngoài học phần.

## Hai ứng viên đã bị loại khỏi pipeline sử dụng

`P` giữ nguyên các tham số và thuật toán đã đánh giá trong [PIPELINE.md](../PIPELINE.md). Hai cấu hình bên dưới đã **không đạt tiêu chí kiểm chứng trên nguồn ngoài**, nên bị loại khỏi module production và app. Mã chỉ còn trong `code/experiments.py` để tái hiện thí nghiệm; không phải cấu hình được khuyên dùng. Quyết định này thay thế ý định ban đầu cho chọn ứng viên thử nghiệm trong app.

| Cấu hình | Thay đổi so với `P` | Cơ sở học phần | Đánh đổi |
|---|---|---|---|
| `P_RECON` | `edge_mode="reconstruct"` | C7: tái tạo nhị phân bằng giãn có ràng buộc / liên thông | Khôi phục phần tối nối liền với biên, nhưng có thể kéo thêm nền tối |
| `P_THIN` | `median=3`, giữ nguyên tham số khác | C3: lọc trung vị | Ít xóa vết nứt mảnh hơn, nhưng giữ lại nhiều nhiễu hạt hơn |

Pipeline gốc lấy mặt nạ thô `M = D ∩ E`, trong đó `D` là vùng tối và `E` là vùng gần biên. Một phần bên trong vết nứt có thể thuộc `D` nhưng nằm ngoài `E`, dẫn đến mất pixel hoặc đứt đoạn.

Với `P_RECON`, đặt hạt giống `X₀ = D ∩ E`, rồi lặp `Xₖ = (Xₖ₋₁ ⊕ B) ∩ D` đến ổn định; `B` là lân cận 3×3 đầy đủ. Mã thực hiện tương đương bằng cách gán nhãn liên thông 8-láng giềng của `D` một lần, giữ các vùng có chạm hạt giống. Vùng tối tách rời hạt giống bị loại. Các bước đóng, lọc diện tích, lọc độ dãn và quyết định mức ảnh vẫn giữ nguyên.

Kiến thức này có trong tài liệu môn học tại thư mục XLA: `Chương 7-notebook.txt`, mục trích thành phần liên thông với công thức `Xₖ = (Xₖ₋₁ ⊕ B) ∩ A`; lọc trung vị nằm trong chương 3. Đây là phép xử lý ảnh xác định, không có bước học tham số từ dữ liệu.

## Kết quả khảo sát trên ảnh kèm repo

Máy hiện có 10 cặp ảnh–GT trong `samples/` và 10 cặp trong `samples_pcs/`, chưa có bộ train/val đầy đủ. Mọi điểm dưới đây tính ở kích thước ảnh gốc, Dice trung bình theo ảnh. **Cả hai thư mục đều chạy tham số PCS `P`**, vì vậy cột CFD không phải kết quả của pipeline CFD đã chọn tham số riêng trong `PIPELINE_CFD.md`.

| Cấu hình | CFD mẫu, 10 ảnh | PCS mẫu, 10 ảnh |
|---|---:|---:|
| `P` | 0.581 | 0.324 |
| `P_RECON` | 0.560 | 0.326 |
| `P_THIN` | 0.569 | 0.335 |

Hai ứng viên tăng nhẹ Dice trên PCS mẫu nhưng giảm trên CFD mẫu. Những số liệu này chỉ là khảo sát ban đầu, không đủ để giữ ứng viên trong sản phẩm. Kiểm tra sau đó trên 100 ảnh PaveSeg thật cho thấy cả hai làm tăng báo nhầm và không vượt tiêu chí đã đặt trước; xem [báo cáo kiểm chứng bên ngoài](external-stress-report.md). `P` vẫn là baseline, cũng còn nhiều thất bại trên nguồn ngoài.

Kết quả đã lưu: [bảng tổng hợp](benchmarks/samples/summary.csv), [từng ảnh](benchmarks/samples/per_image.csv), [tham số, phiên bản và mã băm dữ liệu](benchmarks/samples/metadata.json). Lệnh tái tạo: `python code/benchmark.py --output results/benchmark`. Các thời gian chạy phụ thuộc máy; điểm số được tính xác định từ ảnh và cấu hình.

Các hướng khảo sát khác cũng không đưa vào app:

| Hướng thử | CFD mẫu | PCS mẫu | Nhận xét |
|---|---:|---:|---|
| Giảm đóng ellipse 7×7 xuống 3×3 | 0.518 | 0.299 | Dễ làm vùng nứt không còn đủ liên tục để qua bộ lọc |
| Tái tạo vùng + đóng 3×3 | 0.538 | 0.321 | Chưa tốt hơn `P` ở cả hai bộ mẫu |
| Độ dãn tính từ mômen | 0.541 | 0.326 | Lợi ích PCS nhỏ, giảm rõ trên CFD |
| Giữ vùng dài hoặc mảnh theo chu vi/diện tích | 0.556 | 0.329 | Giữ được thêm nhánh nhưng cũng giữ thêm nhiễu |
| Black-hat ellipse 15×15 + Otsu | 0.078 | 0.101 | Không phù hợp khi thay trực tiếp bước tạo mặt nạ thô, giữ hậu xử lý `P` |
| Sau lọc hình dạng, giao lại với vùng tối `D` | 0.572 | 0.320 | Bớt pixel được phép đóng bù vào nhưng cũng mất pixel nứt |

Đây là kết quả khảo sát, không phải quá trình chọn tham số hợp lệ trên validation. Không có kết luận rằng các hướng này luôn kém hơn; chúng chưa cho bằng chứng đủ để thay mặc định.

## Những sửa đổi có thể kiểm chứng độc lập

- `pipeline(..., steps=True)` tôn trọng `post=False`: mặt nạ trong phần xem từng bước khớp API trả kết quả trực tiếp. Nút bỏ hậu xử lý của app vì thế có tác dụng đúng.
- Hậu xử lý gán nhãn một lần để lọc diện tích và độ dãn, vẫn trả riêng ảnh trung gian `A`, `F` và giữ kết quả như cách lọc hai lần trước đây.
- App giữ ảnh đầu vào và mặt nạ ở kích thước gốc, tránh thu ảnh xuống 1280 trước khi tính điểm. Phần thu về 640 của pipeline vẫn do tham số `size` điều khiển.
- App tìm thư mục mẫu theo vị trí file, nên có thể chạy từ thư mục repo. Notebook PCS import hàm từ module chung để các lần chạy sau dùng cùng logic với app.
- Benchmark ghi cấu hình và số liệu từng ảnh để so sánh có thể chạy lại, thay vì chỉ nhìn ảnh phủ đỏ.

So sánh với module của `origin/mailinh` trên cả 20 ảnh xác nhận cấu hình `P` và các mặt nạ trung gian giữ nguyên từng pixel. Notebook đã kiểm tra cú pháp và import; đầu ra lưu sẵn vẫn được ghi rõ là kết quả lịch sử, chưa chạy lại toàn bộ vì thiếu dataset PCS đầy đủ. Kiểm thử hồi quy và AppTest được chạy cùng mã cuối trước khi đẩy nhánh.

## Chạy lại và đánh giá tiếp

Từ thư mục repo:

```bash
python -m unittest discover -s tests
python code/benchmark.py
```

Lệnh thứ hai chạy các ảnh mẫu, ghi `per_image.csv`, `summary.csv`, `metadata.json` vào `results/benchmark/`. Có thể đổi thư mục bằng `--output`.

Khi có dữ liệu đầy đủ với cấu trúc `{train,val,test}/{images,masks}`:

```bash
python code/benchmark.py --data-root data/Pavement_Crack_Segmentation_Dataset --split val --output results/benchmark-val
```

Chọn thuật toán/tham số bằng train và val, kiểm tra đồng thời Dice/IoU trên ảnh có nứt, ảnh không nứt, Precision/Recall và các nhóm A/B/C. Sau khi cố định cấu hình mới chạy `--split test`. Mặc định của `--split` là `val`.

Các ảnh mẫu thuộc test và đã được quan sát trong khảo sát. Chúng không tạo thành tập validation hay một phép kiểm chứng độc lập; 10 ảnh PCS chỉ có **một ảnh không nứt**, không đủ để kết luận về khả năng tránh báo nhầm. Những kết quả này không cập nhật hay thay thế bảng 231 ảnh test lịch sử trong README.

## Trạng thái

Đã khảo sát hai hướng thuộc học phần và **loại cả hai** sau kiểm chứng ngoài. Nhánh chỉ giữ các sửa lỗi về API/app, chia sẻ logic và tối ưu gán nhãn không đổi mặt nạ baseline; kèm công cụ và bằng chứng thí nghiệm. Không công bố cải thiện chất lượng phân đoạn trên toàn bộ tập test hoặc độ tin cậy ngoài thực tế.
