# Tối ưu thực thi, giữ nguyên kết quả phân đoạn

Thay đổi được giữ ở nhánh `hieugm` là **tối ưu hiệu năng**, không phải tăng độ chính xác. So với snapshot pipeline tại commit `d155f98464dbc9e3d30280efce904d8f6b71bf1e`, phép đo trên 150 ảnh xác nhận các ảnh trung gian, ngưỡng và mặt nạ kích thước gốc trùng giá trị; cấu hình `P` và thứ tự xử lý ảnh không đổi.

Với cấu hình OpenCV mặc định của máy đo (12 luồng), trung vị hệ số tăng tốc theo cặp ảnh là **1,083×**; khi đặt OpenCV một luồng, phép đo riêng cho **1,177×**. Đây là kết quả trên máy và bộ ảnh được ghi trong metadata, không bảo đảm mọi ảnh hoặc mọi máy đều nhanh hơn. Những giới hạn nhận đúng vết nứt của baseline vẫn giữ nguyên; xem [các thử nghiệm chất lượng không được chấp nhận](pipeline-improvement.md) và [kiểm tra nguồn ngoài](external-stress-report.md).

## Thay đổi thực thi

- Bù sáng và tổng gradient Sobel tái sử dụng bộ đệm số thực thay vì tạo nhiều mảng trung gian.
- Lược đồ xám `uint8` dùng `cv2.calcHist` trong miền số pixel mà bộ đếm float32 còn chính xác; trường hợp ngoài miền đó giữ đường `np.bincount`.
- Hộp chữ nhật xoay nhận các điểm đường bao ngoài thay vì toàn bộ pixel vùng; cách lọc diện tích/độ dãn vẫn giữ tiêu chí của `P`.
- Với ảnh `uint8` đủ lớn, cửa sổ trung vị lớn, OpenCV đặt một luồng và đang ở tiến trình chính, có thể chia bộ lọc trung vị thành hai phần có vùng chồng rộng `k//2`. Chỉ lấy phần trong mỗi mảnh để giữ đúng láng giềng của ảnh gốc. Các trường hợp khác dùng trực tiếp `cv2.medianBlur`; không bật thêm pool luồng trong worker của notebook multiprocessing.

Đây là cách thực hiện lại cùng các phép xử lý ảnh đã có trong học phần. Không đổi ngưỡng, tham số hình thái, quy tắc quyết định ảnh hay vị trí pixel đầu ra để lấy tốc độ.

## Cách đo và kiểm tra tương đương

Benchmark dùng đúng danh sách **150 ảnh train PCS đã cố định**, chỉ đọc ảnh đầu vào, **không đọc GT**. Mỗi ảnh được chạy cả snapshot baseline và bản tối ưu; so sánh mọi mục trả về từ `steps=True` cùng mặt nạ cuối ở kích thước gốc trước khi đo. Không bỏ ảnh khó, không thu nhỏ thêm đầu ra và không bỏ bước pipeline để đạt thời gian.

Kiểm tra tương đương mở rộng riêng trên **1.210 ảnh, 2.010 điều kiện**: 1.050 ảnh train PCS đã quan sát (ảnh gốc), 20 ảnh kèm repo, 100 PaveSeg và 40 DeepCrack (mỗi ảnh ngoài/ảnh kèm repo chạy sáu điều kiện). Mọi ảnh trung gian và mặt nạ kích thước gốc đều trùng với snapshot baseline. Phép kiểm tra này không đọc GT và **không mở 158 ảnh cuối**. Đây là kiểm tra đầu ra không đổi, không phải phép đo độ chính xác hay benchmark thời gian trên toàn bộ 1.210 ảnh. Xem [tổng hợp parity](benchmarks/performance/parity/summary.json) và [từng trường hợp](benchmarks/performance/parity/per_case.csv).

Mỗi phương pháp có hai lần khởi động trước, sau đó đo năm lần chạy toàn bộ pipeline cho mỗi ảnh. Thứ tự baseline/tối ưu đảo luân phiên theo ảnh và lần đo. Lấy trung vị thời gian của từng phương pháp trên từng ảnh rồi tổng hợp; thời gian không gồm đọc ảnh hoặc ghi file. "Hệ số tăng tốc theo cặp" là thời gian baseline chia thời gian tối ưu trên cùng ảnh, không phải tỷ số của hai trung vị tổng hợp.

Môi trường lần đo đã lưu: Windows, Python 3.12.9, OpenCV 5.0.0, NumPy 2.3.4, CPU `Intel64 Family 6 Model 151 Stepping 2, GenuineIntel`. Metadata lưu số luồng thực tế, phiên bản, mã băm từng ảnh và mã băm nguồn; không suy diễn kết quả này thành số đo trên máy khác.

## Kết quả

| Chế độ | Ảnh | Trung vị baseline | Trung vị tối ưu | Trung vị tăng tốc theo cặp | Ảnh chạy chậm hơn |
|---|---:|---:|---:|---:|---:|
| OpenCV mặc định, thực tế 12 luồng | 150 | 39,668 ms | 36,969 ms | 1,083× | 2/150 |
| Đặt OpenCV một luồng, phép đo lại riêng | 150 | 47,493 ms | 39,203 ms | 1,177× | 4/150 |

Ở chế độ mặc định, trường hợp chậm hơn nhiều nhất có thời gian **1,058× baseline**; ở chế độ OpenCV một luồng là **1,064×**. Vì vậy không mô tả kết quả là mọi ảnh đều nhanh hơn. Toàn bộ so sánh ảnh trung gian và mặt nạ kích thước gốc trên 150 ảnh đều đạt trong cả hai lần đo.

Bằng chứng chế độ mặc định: [summary](benchmarks/performance/default/summary.json), [thời gian từng ảnh](benchmarks/performance/default/per_image.csv), [môi trường, nguồn và mã băm ảnh](benchmarks/performance/default/metadata.json).

Bằng chứng lần đo một luồng cuối: [summary](benchmarks/performance/one-thread/summary.json), [thời gian từng ảnh](benchmarks/performance/one-thread/per_image.csv), [metadata](benchmarks/performance/one-thread/metadata.json).

Dòng một luồng dùng lần đo lại riêng sau khi dừng tác vụ kiểm tra khác trên máy; không dùng số liệu từ lần chạy bị trùng tác vụ đó. Hai dòng là hai lần đo khác nhau, không dùng chênh lệch thời gian tuyệt đối giữa các dòng để kết luận số luồng nào nhanh hơn; chỉ so sánh baseline/tối ưu trong cùng lần đo và thứ tự luân phiên.

## Tái hiện

Từ thư mục repo, chuẩn bị đúng 150 ảnh theo manifest đã cố định:

```bash
python code/fetch_pcs.py --split train --ids-manifest docs/benchmarks/new-method/train-manifest.json --workers 6
python code/benchmark_performance.py --output results/replay-performance-default
python code/benchmark_performance.py --threads 1 --output results/replay-performance-one-thread
```

Mỗi thư mục đầu ra phải chưa tồn tại; script từ chối ghi đè bằng chứng. Lệnh thứ hai giữ số luồng mặc định của OpenCV đang cài trên máy; không mặc định rằng máy nào cũng dùng 12 luồng. Lệnh thứ ba đặt một luồng OpenCV, còn nhánh chia bộ lọc trung vị có hai worker nội bộ khi đủ điều kiện đã nêu.

Snapshot baseline nằm trong `docs/benchmarks/performance/baseline_pipeline.py`. Mã băm ảnh và nguồn trong metadata dùng để kiểm tra tái hiện; nếu môi trường hay mã thay đổi thì phải báo một lần đo mới. Việc ảnh đầu ra bằng nhau có nghĩa các độ đo Dice/IoU của cùng đầu vào và GT giữ nguyên, **không phải đã cải thiện độ chính xác**.

Nếu đã có đủ các nguồn trong manifest ở đúng thư mục `data/` và ảnh mẫu kèm repo, có thể chạy lại kiểm tra tương đương mở rộng:

```bash
python code/verify_performance.py --output results/replay-performance-parity
```

Script chỉ chọn các ID đã quan sát, kiểm tra không giao với 158 ID cuối rồi mới đọc ảnh; không dùng tập chưa mở để thử thêm.
