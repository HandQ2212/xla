# Tinh chỉnh mặt nạ bằng tương phản cục bộ

**Trạng thái các nhánh tương phản cục bộ: không đạt điều kiện đưa vào pipeline.** Phiên bản ảnh xám tăng Dice và Precision trên 400 ảnh giữ riêng mới nhưng **giảm chất lượng dưới bóng đổ tổng hợp**. Phiên bản sau dùng ảnh bù sáng `R` cũng không đạt ngưỡng tăng Dice và khoảng tin cậy trên **500 ảnh giữ riêng khác**. Không chuyển một trong hai thành cấu hình mặc định. `P` vẫn giữ nguyên để tái lập baseline; các nhánh này chỉ có giá trị nghiên cứu. Báo cáo này không tuyên bố pipeline đã đáng tin cậy trên nguồn ảnh bất kỳ.

## Thuật toán và phạm vi

Phương pháp chỉ dùng kiến thức xử lý ảnh: lọc trung vị và thống kê cục bộ (C3), phân ngưỡng (C5), hình thái và liên thông (C7), lọc hình dạng vùng (C8). Không có học sâu, mô hình huấn luyện sẵn hoặc bộ phân loại được học từ ảnh.

Đầu tiên chạy pipeline gốc `P`, lấy mặt nạ thô cuối cùng ở cạnh dài 640 px. Nhánh bổ sung dùng ảnh xám đã lọc trung vị 3×3, tính trung bình `μ` và độ lệch chuẩn `σ` trong cửa sổ 71×71. Pixel tối thỏa `I < μ − 1,5σ − 1` tạo vùng ứng viên. Sau hậu xử lý, giữ các thành phần ứng viên chạm hạt giống từ `P`; bổ sung phần của `P` nằm gần vùng cục bộ bằng giãn ellipse 3×3.

Mặt nạ được đưa về kích thước gốc. Nếu số pixel vượt số pixel của `P`, chỉ giữ các pixel có tương phản cục bộ chuẩn hóa lớn nhất; các giá trị bằng nhau được xếp theo thứ tự ổn định. Nếu kết quả rỗng trong khi `P` không rỗng, trả lại `P`. Các tham số còn lại giữ từ baseline.

Cấu hình giới hạn chỉ xử lý mới khi **cạnh dài ảnh gốc > 640 px**. Ảnh có cạnh dài ≤ 640 px đi theo đúng `P`, tránh áp dụng thuật toán mới vào chế độ phóng to ảnh mà kết quả trước đó cho thấy suy giảm. Điều kiện này đã cố định trước khi mở ảnh xác nhận, không dựa vào tên dataset hay nhãn GT.

Hai tính chất được kiểm tra ở kích thước gốc:

- Quyết định ảnh rỗng/không rỗng giữ nguyên như `P`.
- Tổng số pixel dự đoán của mỗi ảnh không vượt `P`.

Giới hạn diện tích **không bảo đảm** pixel báo nhầm giảm trên ảnh có nứt, vì mặt nạ mới có thể chuyển sang vị trí khác. Trên ảnh âm, mọi pixel dự đoán đều là báo nhầm, nên diện tích báo nhầm không tăng. Giữ quyết định mức ảnh cũng có nghĩa phương pháp này không tự sửa được các ảnh âm mà `P` đã báo nhầm hay các ảnh dương mà `P` đã trả rỗng.

## Dữ liệu và cách giữ riêng ảnh xác nhận

Nguồn PCS đầy đủ được tìm thấy tại [Zenodo record 20040434](https://zenodo.org/records/20040434): 2.209 ảnh, chia sẵn train 1.480 / val 498 / test 231. Dữ liệu được tải và kiểm tra CRC/mã băm; ảnh gốc, GT và ảnh phủ phục vụ QA nằm trong các thư mục được git bỏ qua. Không phân phối lại ảnh nguồn khi chưa có giấy phép rõ ràng từ tác giả.

Quá trình nghiên cứu ban đầu đã xem train mới, validation và phần test chính thức. Sau đó những tập này được xem là **dữ liệu phát triển đã quan sát**, không dùng lại để gọi là xác nhận độc lập cho phiên bản tiếp theo.

Phép xác nhận mới lấy 400 ảnh từ train gốc, loại cả 150 ảnh đã dùng chọn phương pháp mới và 300 ảnh từng dùng tune `P`. Hai nhóm loại trừ trùng 28 ảnh, tổng cộng loại 422/1.480 ảnh; lấy mẫu 400 từ 1.058 ảnh còn lại bằng seed `20261116`. [Danh sách ảnh](benchmarks/new-method/reserve-manifest.json) được cố định trước khi tải hoặc giải mã pixel. Mẫu gồm nhóm A: 21, B: 108, C: 271; sau đọc GT có **327 ảnh dương và 73 ảnh âm**. Đây là tập giữ riêng mới từ train gốc, **không phải test chính thức của bộ dữ liệu**.

[Protocol triển vọng](../plans/conservative-local-confirmation.md), [cấu hình](benchmarks/new-method/large-config.json) và [mã băm trước khi chạy](benchmarks/new-method/large-freeze.json) cố định mục tiêu trước suy luận: Dice dương tăng ít nhất 0,01 và khoảng bootstrap 95% trên ảnh nằm hoàn toàn trên 0; IoU tăng; Precision tăng ít nhất 0,02; Recall không giảm quá 0,02; giữ quyết định mức ảnh và giới hạn diện tích, kiểm tra các biến đổi đã định trước; trung vị thời gian không quá 2 lần `P`.

Nguồn không có ID session chụp, nên chưa chứng minh các ảnh độc lập về cảnh. Khoảng tin cậy bên dưới bootstrap **theo ảnh**, không phải theo session hay nguồn độc lập; không dùng nó để khẳng định cải thiện qua nhiều nguồn.

## Kết quả ảnh gốc của 400 ảnh giữ riêng

| Độ đo | Baseline `P` | Cấu hình cục bộ giới hạn | Thay đổi |
|---|---:|---:|---:|
| Dice trung bình trên 327 ảnh dương | 0.2740 | 0.2924 | +0.0184 |
| IoU trung bình trên ảnh dương | 0.1954 | 0.2098 | +0.0145 |
| Precision trên ảnh dương | 0.3120 | 0.3726 | +0.0606 |
| Recall trên ảnh dương | 0.2850 | 0.2819 | −0.0032 |
| Dice trên toàn bộ 400 ảnh | 0.3090 | 0.3241 | +0.0151 |
| Số ảnh âm bị báo nhầm | 39/73 | 39/73 | Giữ nguyên |
| Diện tích báo nhầm TB trên ảnh âm | 0.7337% | 0.2562% | Giảm khoảng 65,1% |

Chênh lệch Dice dương có khoảng bootstrap 95% **[+0,00955; +0,02737]**, 10.000 lần lấy mẫu, seed 42. Có 162 ảnh dương tăng điểm, 69 ảnh giảm, 96 ảnh bằng nhau. Không trộn Dice = 1 của ảnh âm trả rỗng vào Dice dương.

Mức tăng trung bình không che được các ca suy giảm: **25 ảnh dương giảm Dice quá 0,05**. Nặng nhất là `B_00302` (−0,5197), `B_00300` (−0,4036), `B_00130` (−0,2778), `B_00127` (−0,2253), `C_01801` (−0,2111). Phương pháp có thể xóa nhầm một phần nứt thật dù vẫn trả mặt nạ không rỗng. [Danh sách đầy đủ 25 ca](benchmarks/new-method/reserve/paired.json) và [số liệu mọi ảnh](benchmarks/new-method/reserve/per_image.csv) được giữ nguyên; không thể coi cải thiện trung bình là bảo đảm cho từng ảnh. Xem thêm [bảng tổng hợp](benchmarks/new-method/reserve/summary.csv) và [metadata](benchmarks/new-method/reserve/metadata.json).

Lần chạy ảnh gốc đầu tiên có trung vị thời gian **17,24 ms (`P`) và 34,87 ms (ứng viên)**, tỷ lệ khoảng **2,022**, không đạt giới hạn 2 lần. [Kết quả đầu tiên](benchmarks/new-method/reserve-original-timing/metadata.json) được giữ lại. Sau đó chỉ tối ưu thực thi: tái sử dụng ảnh đã thu nhỏ, bỏ tính/xếp hạng tương phản khi không vượt giới hạn diện tích, và tái sử dụng phép đếm pixel. Không đổi ngưỡng, tham số, phạm vi hoặc phương pháp chọn pixel.

Đối chiếu đủ **800 mã băm mặt nạ** (`P` và ứng viên trên cả 400 ảnh) cho kết quả trùng từng pixel với bản đã cố định. Lần đo lại cùng cách khởi động một lần, trung vị ba lần cho từng ảnh và một luồng OpenCV cho **16,97 ms (`P`) và 27,48 ms (ứng viên)**, tỷ lệ **1,620**, đạt giới hạn. Xem [kiểm tra tương đương và thời gian](benchmarks/new-method/performance-parity.json), [mã băm nguồn tối ưu](benchmarks/new-method/optimized-freeze.json). Thời gian chưa gồm đọc/ghi ảnh; đây là tối ưu hiệu năng giữ nguyên kết quả, không phải chọn lại thuật toán dựa vào ảnh xác nhận.

## Kiểm tra biến đổi: thất bại dưới bóng đổ

Sau khi cố định thuật toán, áp dụng cùng năm biến đổi đã khai báo cho cả `P` và ứng viên trên đủ 400 ảnh. Đây là các phiên bản của cùng 400 cảnh, không phải thêm 2.000 ảnh độc lập. GT được lật cùng ảnh khi lật ngang; các biến đổi còn lại giữ GT. Phép chạy stress chỉ dùng một lần đo mỗi trường hợp để chấm chất lượng, không dùng nó cho kết luận thời gian bên trên.

| Điều kiện | Dice dương `P` | Dice dương cục bộ | Chênh lệch | Báo nhầm ảnh âm `P` / cục bộ |
|---|---:|---:|---:|---:|
| Lật ngang | 0.2733 | 0.2916 | +0.0184 | 39/73 / 39/73 |
| Gamma 0,75 | 0.2717 | 0.2901 | +0.0183 | 41/73 / 41/73 |
| Gamma 1,25 | 0.2714 | 0.2894 | +0.0180 | 38/73 / 38/73 |
| JPEG chất lượng 60 | 0.2746 | 0.2966 | +0.0220 | 41/73 / 41/73 |
| Bóng đổ chéo tổng hợp | 0.2594 | 0.2392 | **−0.0202** | 42/73 / 42/73 |

Bóng tổng hợp dùng hệ số sáng trơn 0,35–1 như phép kiểm tra nguồn ngoài trước đó. Ở điều kiện này, Recall giảm từ **0,2662 xuống 0,2057**, tức **−0,0605**. Giới hạn diện tích và bảo toàn quyết định ảnh vẫn đúng, nhưng không ngăn được việc xóa nhầm nhiều pixel nứt thật trong bóng. Điều kiện yêu cầu tăng Dice ở mọi stress đã khai báo vì thế **thất bại**.

Không bỏ điều kiện bóng sau khi thấy điểm, không gọi bốn điều kiện còn lại là đủ xác nhận, và không đổi scope sau phép đo để nhận thuật toán vào mặc định. Kết quả này là một hướng có ích cho nghiên cứu tương phản cục bộ, chưa phải cải thiện đã vượt đủ kiểm chứng.

[Số liệu stress mọi ảnh](benchmarks/new-method/reserve-stress/per_image.csv), [bảng tổng hợp](benchmarks/new-method/reserve-stress/summary.csv) và [chênh lệch, khoảng tin cậy, danh sách ca giảm](benchmarks/new-method/reserve-stress/paired.json) được lưu cùng các điều kiện đã chạy.

## Giả thuyết mới trên ảnh bù sáng: vẫn không đạt

Sau thất bại bóng của phiên bản ảnh xám, một [protocol mới](../plans/illumination-local-confirmation.md) đặt giả thuyết rằng thống kê cục bộ nên tính trên **ảnh tỉ số bù sáng `R` đã có của `P`**, thay vì ảnh xám gốc. Thử trước 12 cấu hình trên cùng 150 ảnh train và bóng tổng hợp: cửa sổ 31/71, hệ số độ lệch chuẩn 1/1,5/2 và có/không bảo vệ phần lõi mặt nạ. Chọn duy nhất `ratio_w31_k1.0_c0` theo quy tắc đã đặt trước; không thay cấu hình dựa vào tập giữ riêng mới.

500 ảnh giữ riêng tiếp theo lấy từ 658 ảnh train còn lại sau khi loại historical tune300, train150 và 400 ảnh giữ riêng cũ, seed `20261117`; nhóm A: 21, B: 153, C: 326. [Manifest](benchmarks/new-method/ratio-reserve-manifest.json), [cấu hình](benchmarks/new-method/ratio-config.json) và [mã băm nguồn](benchmarks/new-method/ratio-freeze.json) được cố định trước tải/giải mã. Sau đọc GT có **414 ảnh dương, 86 ảnh âm**. Đây cũng là mẫu giữ riêng từ train gốc, không phải test chính thức; 400 ảnh trước đã trở thành dữ liệu phát triển và không được dùng lại làm xác nhận mới.

| Độ đo trên 500 ảnh mới | `P` | Cục bộ trên `R` |
|---|---:|---:|
| Dice dương | 0.292080 | 0.298638 |
| IoU dương | 0.208309 | 0.215026 |
| Precision dương | 0.324814 | 0.350809 |
| Recall dương | 0.297614 | 0.290314 |
| Báo nhầm ảnh âm | 43/86 | 43/86 |
| Diện tích âm trung bình | 0.663462% | 0.246308% |

Dice dương chỉ tăng **+0,006558**, dưới yêu cầu **+0,01**. Khoảng bootstrap theo ảnh 95% là **[−0,0000925267; +0,0125985423]**, có chứa 0. Không làm tròn cận dưới âm thành 0 hoặc thành bằng chứng thành công. Precision tăng và diện tích âm giảm vẫn không đủ để vượt hai điều kiện chính đã thất bại; phiên bản này **bị loại**. Bất kỳ phát triển tiếp theo nào phải giữ lại kết quả này và không gọi lại 500 ảnh đã quan sát là tập xác nhận chưa xem.

Xem [tổng hợp ratio500](benchmarks/new-method/ratio-reserve/summary.csv), [từng ảnh](benchmarks/new-method/ratio-reserve/per_image.csv), [khoảng tin cậy và mọi ca suy giảm lớn](benchmarks/new-method/ratio-reserve/paired.json), [metadata](benchmarks/new-method/ratio-reserve/metadata.json).

## Đa độ phân giải và khôi phục mặt nạ: bị loại ngay trên train

Sau hai lần xác nhận thất bại, các giả thuyết tiếp theo vẫn chỉ dùng **150 ảnh train đã quan sát** cùng bóng tổng hợp, không được dùng dữ liệu giữ riêng cuối để chọn phương án.

[Thử đa độ phân giải](../plans/multiscale-confirmation.md) giữ `P` ở 640 px làm quyết định mức ảnh, bổ sung mặt nạ ở 960/1280 px, giới hạn không vượt kích thước gốc. Mười hai cấu hình kết hợp tham số hình thái cố định hoặc tỉ lệ theo độ phân giải với ba cách giao nối vùng. Không giữ giới hạn tổng diện tích cũ, vì nó có thể chặn phục hồi Recall; thay vào đó phải đo chi phí báo nhầm trên ảnh âm.

Không cấu hình nào đạt tiêu chí train đã cố định. Điểm Dice dương ảnh gốc cao nhất là **0,297591** (`scale_960_physical_supported`), so với **0,287918** của `P`: tăng **0,009673**, dưới yêu cầu **0,02**. Chính cấu hình này tăng diện tích âm trung bình từ **0,7309% lên 1,2457%**, không đạt yêu cầu không tăng. Những cấu hình khác có thể giảm diện tích âm nhưng mất Recall. Vì vậy không chọn cấu hình để chạy xác nhận.

Toàn bộ [12 cấu hình](benchmarks/new-method/multiscale-train/configs.json), [bảng tổng hợp](benchmarks/new-method/multiscale-train/summary.csv) và [từng ảnh/điều kiện](benchmarks/new-method/multiscale-train/per_image.csv) được giữ lại, gồm cả kết quả giảm điểm.

[Thử khôi phục mặt nạ](../plans/mask-restoration-alignment.md) chỉ đổi nội suy khi đưa mặt nạ 640 px về kích thước gốc: nearest-exact, linear với ngưỡng 0,35/0,45/0,5, cubic với ngưỡng 0,5. Không thay detector, ngưỡng vùng tối hay bộ lọc thành phần liên thông.

Trong năm cấu hình, `linear_0.35` tăng Dice dương ảnh gốc **0,287918 → 0,290905** và bóng **0,269463 → 0,273053**, nhưng Precision ảnh gốc giảm **0,333642 → 0,324703** (−0,008939), vượt mức giảm cho phép 0,005. Diện tích âm trung bình cũng tăng **0,7309% → 0,7731%**, khoảng **5,77% tương đối**, vượt giới hạn tăng 2%; điều kiện bóng cũng không đạt các ràng buộc này. Bốn cấu hình còn lại không đủ mức tăng Dice gốc 0,002 đã đặt trước. Không cấu hình nào được chấp nhận.

Xem [năm cách khôi phục](benchmarks/new-method/restore-train/configs.json), [tổng hợp](benchmarks/new-method/restore-train/summary.csv), [từng ảnh](benchmarks/new-method/restore-train/per_image.csv).

**158 ảnh train còn lại chỉ được tải/kiểm tra CRC theo manifest, chưa được mở pixel, xem hay chấm điểm.** Vì không có ứng viên vượt vòng train, không dùng 158 ảnh này để tìm một kết quả đẹp hơn. Cả đa độ phân giải và khôi phục mặt nạ ở trên đều không đi vào pipeline sử dụng.

## Những phiên bản đã không đạt

Các thử trước được giữ lại để tránh biến kết quả phát triển thành một câu chuyện chỉ có thành công:

- Tái tạo toàn vùng tối và chỉ giảm trung vị đã không đạt kiểm tra PaveSeg, làm tăng báo nhầm; xem [báo cáo nguồn ngoài](external-stress-report.md).
- Lọc cấu trúc đường đối xứng, lọc ảnh xám ở độ phân giải gốc và ngưỡng cục bộ độc lập có đánh đổi Recall hoặc báo nhầm, không được nhận trực tiếp vào sản phẩm.
- Phiên bản xếp hạng tương phản chưa có fallback bảo toàn quyết định cho kết quả tốt trên 150 train và 498 val. Trên 221 ảnh test chính thức còn lại sau loại 10 ảnh mẫu đã xem, Dice dương chỉ tăng **0,3005 → 0,3105**, khoảng tin cậy **[−0,00216; +0,01956]**. Nó **không đạt** ngưỡng tăng 0,02 của protocol cũ, không được gọi là cải tiến đã xác nhận.
- Phiên bản bảo toàn quyết định nhưng chưa giới hạn độ phân giải giảm Dice trên 40 ảnh DeepCrack **0,5796 → 0,5371**, Recall **0,5978 → 0,5044**. Vì vậy đã loại khả năng dùng nó làm thay thế không giới hạn. Các ảnh DeepCrack này đã được quan sát, không còn là xác nhận mới cho cấu hình đổi sau đó.

Protocol mới nhắm một cải thiện Precision nhỏ và có giới hạn, với dữ liệu giữ riêng mới. Nó không làm phiên bản cũ vượt lại tiêu chí đã thất bại. Nhánh bypass độ phân giải thấp trả đúng `P`; sự bằng nhau trên DeepCrack chỉ chứng minh bypass hoạt động, **không xác nhận chất lượng của nhánh tăng cường ảnh lớn**.

## Chạy lại bản đã cố định

Chuẩn bị dữ liệu PCS theo cấu trúc `data/Pavement_Crack_Segmentation_Dataset/train/{images,masks}`. Công cụ HTTP Range/CRC tải đúng 400 cặp trong manifest đã cố định, không chọn ảnh theo kết quả và không cần tải toàn bộ archive:

```bash
python code/fetch_pcs.py --split train --ids-manifest docs/benchmarks/new-method/reserve-manifest.json --workers 6
```

Tái hiện phép đo ảnh gốc của phiên bản ảnh xám, với bản tối ưu đã xác nhận không đổi mặt nạ:

```bash
python code/confirm_local.py --stage reserve --config large-config.json --freeze optimized-freeze.json --source-dir docs/benchmarks/new-method/frozen-optimized --conditions original --output results/replay-reserve
```

Lệnh kiểm tra mã băm nguồn trước khi chạy; thư mục đầu ra phải chưa tồn tại để tránh ghi đè bằng chứng. Bản gốc trước tối ưu cũng được lưu ở `frozen-large/`, tương ứng `large-freeze.json`. Chạy lại sau khi đã biết kết quả là tái hiện, không trở thành một phép xác nhận mới. Thời gian tùy máy, số liệu chất lượng được đối chiếu từ cấu hình, nguồn và mã băm mặt nạ.

Phần [bảo vệ tái hiện](../code/replay_guard.py) còn kiểm tra cấu hình, manifest chọn ảnh và helper tái tạo liên thông trước khi giải mã dữ liệu. Các băm chuẩn hóa xuống dòng LF để hỗ trợ checkout CRLF và băm helper còn thiếu trong sổ cũ là **audit bổ sung sau các lần chấm điểm**; helper được đối chiếu với bản `d155` bất biến. Đây không phải bằng chứng helper đã được ghi đủ băm trước 400/500 ảnh, không phải một bản đóng băng mới trước suy luận và không làm thay đổi trạng thái bị loại. Các bản freeze lịch sử được giữ nguyên; không suy diễn thời điểm audit bổ sung thành thời điểm thiết kế protocol.

Tái hiện phiên bản dùng `R` trên 500 ảnh khác:

```bash
python code/fetch_pcs.py --split train --ids-manifest docs/benchmarks/new-method/ratio-reserve-manifest.json --workers 6
python code/confirm_local.py --stage ratio-reserve --engine ratio_local --config ratio-config.json --freeze ratio-freeze.json --source-dir docs/benchmarks/new-method/frozen-ratio --output results/replay-ratio-reserve
```

## Giới hạn và quyết định

Kết quả 400 ảnh gốc của phiên bản ảnh xám hỗ trợ một giả thuyết cải thiện trung bình trong miền **ảnh lớn của cùng bản phát hành PCS**, nhưng kết quả bóng bác bỏ độ ổn định yêu cầu. Phiên bản dùng `R` tiếp theo cũng không đủ mức tăng Dice và khoảng tin cậy trên 500 ảnh khác. Ảnh giữ riêng mới cùng nguồn với dữ liệu phát triển; cần nguồn ảnh độ phân giải cao khác, cảnh chụp độc lập và nhãn phù hợp trước khi tuyên bố khả năng chuyển miền. Không thể phân biệt vệt trám, khe nối, bóng và nứt thật một cách đáng tin cậy chỉ từ nguyên lý "vùng tối dài"; các trường hợp gây nhầm vẫn phải kiểm tra ngoài thực tế.

Không có thay đổi chất lượng phân đoạn nào trong các hướng trên vượt đủ điều kiện để nhận vào pipeline. Không dùng kết quả trung bình tốt hơn để thay thế điều kiện đã thất bại. Bộ `P` và các kết quả lịch sử vẫn được giữ riêng để so sánh; 400/500 ảnh đã quan sát không được gọi lại là tập xác nhận mới. Các [tối ưu hiệu năng giữ nguyên đầu ra](performance-report.md) được đánh giá riêng, không được trình bày thành tăng độ chính xác.
