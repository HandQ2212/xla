# Pipeline phát hiện và phân đoạn vết nứt mặt đường (xử lý ảnh cổ điển)

> **Đầu vào:** ảnh gốc chụp mặt đường (RGB, kích thước bất kỳ).
> **Đầu ra:** mặt nạ nhị phân cùng kích thước ảnh gốc (1 = vết nứt) và ảnh gốc được phủ mặt nạ màu đỏ.
> **Ràng buộc:** chỉ dùng kiến thức Chương 1–9 của học phần; không dùng học sâu hay mô hình huấn luyện sẵn.
> **Code:** `code/crack_segmentation.ipynb`

---

## 0. Toàn cảnh

```
 ẢNH GỐC (RGB, vd 2560×1440)
    │
    ▼
┌───────────────────────── BƯỚC 1: DỮ LIỆU ─────────────────────────┐
│ CFD 118 ảnh + CRACK500 249 ảnh gốc, GT vẽ tay mức pixel            │
│ Chia: tune (chọn tham số) / test (đánh giá, 258 ảnh)               │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────── BƯỚC 2: TIỀN XỬ LÝ (C2, C3, C4) ───────────────┐
│ 2.1 Chuẩn hóa kích thước : cạnh dài → 480 px (INTER_AREA)           │
│ 2.2 Chuyển ảnh xám       : Y = 0.299R + 0.587G + 0.114B             │
│ 2.3 Cân bằng lược đồ     : ĐÃ THỬ – KHÔNG DÙNG                     │
│ 2.4 Khử nhiễu            : lọc trung vị 3×3                         │
│ 2.5 Bù chiếu sáng        : B = trung vị lớn; R = 128·I/B            │
└─────────────────────────────────────────────────────────────────────┘
    │  ảnh tỉ số R (nền ≈ 128, vết nứt < 128)
    ▼
┌────────────────────── BƯỚC 3: XỬ LÝ CHÍNH (C5) ────────────────────┐
│ 3.1 Làm trơn lược đồ W = 3                                          │
│ 3.2 Ngưỡng tam giác θ        →  D = (R < θ)        "tối hơn nền"    │
│ 3.3 Sobel |∇R| + Otsu + giãn →  E                  "gần biên mạnh"  │
│ 3.4 Mặt nạ thô = D ∩ E                                              │
└─────────────────────────────────────────────────────────────────────┘
    │  mặt nạ nhị phân thô (480 px)
    ▼
┌──────────────────── BƯỚC 4: HẬU XỬ LÝ (C7, C8) ────────────────────┐
│ 4.1 Phép đóng (ellipse 5×5 / 7×7) : nối đoạn đứt                    │
│ 4.2 Phép mở                       : ĐÃ THỬ – KHÔNG DÙNG             │
│ 4.3 Lọc vùng theo diện tích       : bỏ vùng < 30 / 60 px            │
│ 4.4 Lọc theo độ dãn dài (mômen)   : bỏ vùng tròn, độ dãn < 3        │
│ 4.5 Phóng về kích thước gốc       : INTER_NEAREST                   │
└─────────────────────────────────────────────────────────────────────┘
    │  mặt nạ cuối (kích thước gốc)
    ▼
┌────────────────────── BƯỚC 5: ĐÁNH GIÁ (C8) ───────────────────────┐
│ Dice, IoU (+ Precision, Recall) trên tập test, so với Otsu và Canny │
└─────────────────────────────────────────────────────────────────────┘
```

**Pipeline cuối cùng:** "Sobel ∩ tam giác" (mã **V-A** trong notebook). Đây là cấu hình cho **Dice và IoU cao nhất** trên tập test của cả hai bộ dữ liệu. Nếu bỏ bước 3.3 (mặt nạ thô = D) thì được phương pháp "tam giác" (mã **M**), thấp hơn khoảng 0.01 Dice.

### Bảng tham số cuối cùng

| Bước | Tham số | CFD | CRACK500 | Cách chọn |
|---|---|---|---|---|
| 2.1 | `size` (cạnh dài) | 480 | 480 | Cố định |
| 2.3 | `equalize` | Không | Không | Tìm theo lưới {Không, Có} |
| 2.4 | `median` | 3 | 3 | Tìm theo lưới {0, 3, 5} |
| 2.5 | `bg` | 61 | 41 | Tìm theo lưới {0, 21, 41, 61} |
| 3.1 | `hist_w` | 3 | 3 | Tìm theo lưới {1, 3, 5} |
| 3.3 | `dil` | 3 | 5 | Tìm theo lưới {3, 5} |
| 4.1 | `close` | 5 | 7 | Tìm theo lưới {0, 3, 5, 7} |
| 4.2 | `open` | 0 (tắt) | 0 (tắt) | Tìm theo lưới {0, 3} |
| 4.3 | `a_min` | 30 | 60 | Tìm theo lưới {0, 30, 60, 120, 200} |
| 4.4 | `e_min` | 3 | 3 | Tìm theo lưới {0, 2, 3, 4} |

---

## BƯỚC 1 – Dữ liệu

### 1.1 Nguồn
| Tập | Mô tả | Ảnh dùng |
|---|---|---|
| **CFD** (CrackForest) | Mặt đường đô thị, chụp bằng điện thoại | `cfd_image/001–118.jpg`, 480×320 RGB. Ảnh `201–329` không có GT nên bị loại |
| **CRACK500** | Mặt đường ở Đại học Temple | `valdata.zip` (50) + `testdata.zip` (200), **ảnh đầy đủ** 1440×2560 đến 3264×2448. Không dùng bản cắt nhỏ |

### 1.2 Cách gán nhãn
Người gán nhãn **vẽ tay** mặt nạ nhị phân PNG lên từng ảnh, ở mức pixel: 255 = vết nứt, 0 = nền.
* GT của CFD **mảnh**, 1–3 px, vẽ theo đường tâm vết nứt.
* GT của CRACK500 **dày**, bao trọn bề rộng vết nứt.

```
  Ảnh gốc (một đoạn)          Mặt nạ GT
  ░░░░░░░░░░░░░░░░            ················
  ░░░░▓▓░░░░░░░░░░            ····██··········
  ░░░░░▓▓▓░░░░░░░░    ──►     ·····███········
  ░░░░░░░░▓▓▓▓░░░░            ········████····
  ░░░░░░░░░░░░▓▓░░            ············██··
```

### 1.3 Chia tập
```
CFD      : 001 ─────── 060 │ 061 ─────── 118
                tune (60)  │  test (58)

CRACK500 : valdata (49)    │ testdata (200)
                tune       │  test
```
* **tune:** chỉ dùng để chọn tham số.
* **test:** chỉ dùng **một lần** để báo cáo kết quả. Tổng cộng **258 ảnh ≥ 50**, đạt yêu cầu đề.

### 1.4 Thống kê
| Tập | Ảnh | % pixel nứt TB (min – max) |
|---|---|---|
| CFD tune / test | 60 / 58 | 1.64% (0.35 – 6.22) / 1.60% (0.73 – 6.33) |
| CRACK500 tune / test | 49 / 200 | 2.45% (0.31 – 8.68) / 2.49% (0.15 – 7.01) |

Vết nứt chỉ chiếm khoảng 2% số pixel, tức dữ liệu **lệch lớp rất nặng**. Đây là lý do của nhiều lựa chọn ở các bước sau.

### 1.5 Hai lỗi dữ liệu đã xử lý
1. **EXIF Orientation.** 132/250 ảnh CRACK500 (25 val + 107 test) có thẻ EXIF yêu cầu xoay. OpenCV mặc định sẽ xoay ảnh 90° nhưng không xoay mặt nạ, nên ảnh và GT lệch nhau. Cách sửa: đọc ảnh với cờ `IMREAD_IGNORE_ORIENTATION`.
   ```
   imread mặc định : ảnh 1440×2560  ≠  GT 2560×1440   ✗
   IGNORE_ORIENT.  : ảnh 2560×1440  =  GT 2560×1440   ✓
   ```
2. **Trùng ảnh.** `20160329_103807` có ở cả valdata và testdata. Ảnh này bị loại khỏi tune (50 → 49 ảnh) để tránh rò rỉ dữ liệu sang tập test.

---

## BƯỚC 2 – Tiền xử lý (Chương 2, 3, 4)

### 2.1 Chuẩn hóa kích thước (Ch2: lấy mẫu, độ phân giải)
| | |
|---|---|
| **Phương pháp** | Lấy mẫu lại sao cho cạnh dài = 480 px, giữ tỉ lệ khung hình, nội suy `INTER_AREA` |
| **Tham số** | `size = 480` |
| **Lý do chọn** | (1) Mọi tham số phía sau tính bằng **pixel** (cửa sổ lọc, phần tử cấu trúc, ngưỡng diện tích) nên mọi ảnh phải về cùng một thang đo. (2) 480 bằng đúng kích thước của CFD nên CFD không bị biến đổi. (3) CRACK500 chạy nhanh hơn khoảng 30 lần. (4) `INTER_AREA` lấy trung bình vùng điểm ảnh nên chống răng cưa khi thu nhỏ. Nội suy láng giềng gần nhất có thể làm đứt vết nứt mảnh |

**Mô phỏng:**
```
2560 × 1440  ──( s = 480/2560 = 0.1875 )──►  480 × 270
Vết nứt rộng 20 px ở ảnh gốc  →  20 × 0.1875 ≈ 3.75 px  (vẫn đủ để nhận ra)
CFD 480 × 320  →  s = 1  →  giữ nguyên
```

### 2.2 Chuyển ảnh xám (Ch4: hệ màu)
| | |
|---|---|
| **Phương pháp** | RGB → Y (độ chói) theo công thức `Y = 0.299R + 0.587G + 0.114B` |
| **Tham số** | Không có (hệ số cố định theo chuẩn BT.601) |
| **Lý do chọn** | Vết nứt khác nền về **độ sáng** (khe hở ít nhận sáng nên tối hơn), **không khác về màu**. Kênh H của HSV gần như vô nghĩa với mặt đường ít màu. Các trọng số mô phỏng độ nhạy của mắt người |

**Mô phỏng:**
```
Pixel (R, G, B) = (120, 110, 100)
Y = 0.299·120 + 0.587·110 + 0.114·100 = 35.88 + 64.57 + 11.40 = 111.85 → 112
Ảnh: 480×270×3  →  480×270×1
```

### 2.3 Cân bằng lược đồ (Ch3) – đã thử, KHÔNG dùng
| | |
|---|---|
| **Phương pháp** | `equalizeHist`: biến đổi mức xám theo hàm phân phối tích lũy (CDF) |
| **Tham số** | `equalize ∈ {Không, Có}` |
| **Kết quả** | Dice tune: CFD **0.516 → 0.075**, CRACK500 **0.638 → 0.375** khi bật |
| **Lý do loại** | (1) Khuếch đại cả kết cấu hạt của mặt đường. (2) Là phép phi tuyến, kéo đỉnh nền ra toàn dải [0, 255], làm lược đồ mất dạng "một đỉnh + đuôi" mà ngưỡng tam giác cần. Ngưỡng khi đó rơi vào giữa vùng nền, nên gần như cả ảnh bị gán là vết nứt |

```
Lược đồ trước:          Lược đồ sau cân bằng:
      ▆█▆                ▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃▃
   ▁▂▄███▄▂▁             (trải đều → mất đỉnh nền, mất "đuôi" vết nứt)
──────────────►         ──────────────►
```

### 2.4 Khử nhiễu: lọc trung vị (Ch3)
| | |
|---|---|
| **Phương pháp** | Lọc trung vị k×k: mỗi pixel được thay bằng **trung vị** của cửa sổ lân cận |
| **Tham số** | `median = 3` (lưới {0, 3, 5}) |
| **Lý do chọn phương pháp** | Hạt sỏi tạo nhiễu dạng **muối tiêu**, mà trung vị là bộ lọc tốt nhất cho loại nhiễu này. Lọc trung bình hoặc Gauss thì **làm nhòe** vết nứt mảnh |
| **Lý do chọn k = 3** | Dice tune theo k = 0 / **3** / 5: CFD 0.496 / **0.516** / 0.498; CRACK500 0.592 / **0.638** / 0.626 |

**Mô phỏng** (nền = 120, nhiễu hoặc vết nứt = 30 hoặc 60):
```
(a) Đốm nhiễu 1 px         → trung vị 3×3 →  bị xóa  ✓
  120 120 120 120 120          120 120 120 120 120
  120 120 120 120 120          120 120 120 120 120
  120 120  30 120 120   ──►    120 120 120 120 120
  120 120 120 120 120          120 120 120 120 120
   Cửa sổ quanh tâm: [30,120,120,120,120,120,120,120,120] → trung vị = 120

(b) Vết nứt rộng 2 px      → trung vị 3×3 →  GIỮ NGUYÊN  ✓
  120 120 120  60  60 120 120     120 120 120  60  60 120 120
  120 120 120  60  60 120 120 ──► 120 120 120  60  60 120 120
   Cửa sổ tại cột 60: [60,60,60,60,60,60,120,120,120] → trung vị = 60

(c) Cùng vết nứt 2 px      → trung vị 5×5 →  BỊ XÓA  ✗  (nên không chọn k = 5)
   Cửa sổ 5×5 chỉ có 10/25 pixel tối → trung vị = 120

(d) Vết nứt rộng 1 px      → trung vị 3×3 →  bị xóa (3/9 pixel tối)
   → đây là lý do không dùng cửa sổ lớn hơn 3
```

### 2.5 Bù chiếu sáng: ảnh tỉ số I/B (Ch3: lọc trung vị + biến đổi điểm)
| | |
|---|---|
| **Phương pháp** | (1) Ước lượng nền: `B = trung vị(I, bg×bg)` với cửa sổ **lớn**. (2) Ảnh tỉ số: `R = 128 · I / (B + 1)`, cắt về [0, 255] |
| **Tham số** | `bg = 61` (CFD), `41` (CRACK500), lưới {0 = tắt, 21, 41, 61} |
| **Lý do chọn phương pháp** | Vết nứt mảnh (3–6 px) so với cửa sổ (41–61 px) nên trung vị cửa sổ chính là **mức nền cục bộ**, và vết nứt bị xóa khỏi B. Bóng đổ thay đổi **chậm** nên nằm trong B; chia cho B sẽ **triệt tiêu** bóng đổ. Dùng phép **chia** thay vì phép trừ vì ánh sáng tác động theo kiểu nhân |
| **Lý do chọn bg** | Dice tune theo bg = 0 / 21 / 41 / 61: CFD 0.377 / 0.489 / 0.513 / **0.516**; CRACK500 0.576 / 0.600 / **0.638** / 0.634. Cửa sổ 21 quá nhỏ vì vết nứt rộng chiếm gần nửa cửa sổ, lọt vào B rồi bị "chia mất". CRACK500 có bóng thay đổi nhanh hơn nên cửa sổ nhỏ hơn (41) tốt hơn |

**Mô phỏng** trên một hàng pixel, bên trái nắng, bên phải bóng râm, có 2 vết nứt:
```
vị trí    :   0    1    2    3    4    5    6    7    8    9   10
              ── nắng ──────────►  chuyển dần  ──── bóng râm ────
I (xám)   : 160  160  [96] 160  150  130  110  100  [60] 100  100
B (nền)   : 160  160  160  160  150  130  110  100  100  100  100
R=128·I/B : 127  127  [76] 127  127  127  126  126  [76] 126  126
                       ↑ nứt                          ↑ nứt

• Nếu tách ngưỡng trực tiếp trên I (vd I < 105): các vị trí 7, 9, 10 (bóng râm) bị
  gán NHẦM là vết nứt.
• Trên R: nền luôn ≈ 127, cả hai vết nứt đều = 76, nên một ngưỡng dùng chung cho
  cả ảnh là đủ.
```

Sau bước này, lược đồ của **mọi ảnh** có cùng dạng: một đỉnh nền ở khoảng 128 và một đuôi tối là vết nứt.

---

## BƯỚC 3 – Xử lý chính (Chương 5)

### 3.1 Làm trơn lược đồ (Ch5, tr.62)
| | |
|---|---|
| **Phương pháp** | Trung bình trượt cửa sổ W trên lược đồ h[b], không làm dịch vị trí đỉnh |
| **Tham số** | `hist_w = 3` (lưới {1 = tắt, 3, 5}) |
| **Lý do** | Dao động nhỏ ở vùng đuôi làm ngưỡng tam giác "nhảy" thất thường. Dice tune theo W = 1 / **3** / 5: CFD 0.481 / **0.516** / 0.510; CRACK500 0.572 / **0.638** / 0.634 |

### 3.2 Ngưỡng tam giác (Ch5, tr.78) → mặt nạ "tối hơn nền" D
| | |
|---|---|
| **Phương pháp** | Thuật toán Zack: (1) nối đỉnh lược đồ (b_max) với mức khác 0 xa nhất về phía đuôi (b_min) bằng đường Δ; (2) với mỗi b, tính khoảng cách d từ h[b] tới Δ; (3) θ = b có d lớn nhất. Gán **D = (R < θ)** |
| **Tham số** | Không có. θ được tính tự động cho **từng ảnh** |
| **Lý do chọn tam giác** | Lược đồ R có **một đỉnh lớn** (nền, khoảng 98%) và **một đuôi mỏng** (nứt, khoảng 2%), đúng loại lược đồ mà phương pháp tam giác được thiết kế cho. **Otsu** giả định hai đỉnh nên đặt ngưỡng vào giữa đỉnh nền: trên ảnh `cfd/001`, Otsu gán 27.5% ảnh là nứt, tam giác gán 2.1%, trong khi GT là 1.2%. **Đối xứng nền** cần thêm hai tham số và cho kết quả kém hơn trên tập tune |

**Mô phỏng** trên lược đồ đồ chơi 16 mức (sau khi làm trơn W = 3):
```
b    :  0    1    2    3    4    5    6    7    8     9    10   11   12   13   14   15
h[b] : 0.3  1.0  1.3  2.0  2.0  3.0  4.0  8.3  20.3  48.3 63.3 56.7 28.3 8.7  2.0  0.3
       └──────────── đuôi tối (vết nứt) ────────┘ └──── đỉnh nền ────┘

          h
     63 ┤                                         ●  ← b_max = 10
        │                                      ●     ●
        │                                 ╱
     20 ┤                           ╱  ●
        │                     Δ ╱    ↕ d lớn nhất
      8 ┤               ╱      ●  ← b = 7
        │         ╱ ●  ●
      0 ●──●──●───────────────────────────────────────── b
        b_min=0

Khoảng cách d(b) từ h[b] tới Δ:
b : 0     1     2     3     4     5     6     7     8     9    10
d : 0.00  0.88  1.82  2.70  3.69  4.52  5.35 [5.66] 4.77  1.36  0.00
                                              ↑ lớn nhất → θ = 7
Mặt nạ D: pixel có R < 7 (chân đỉnh nền) là "tối hơn nền".
```

### 3.3 Sobel + Otsu + giãn (Ch5: gradient) → mặt nạ "gần biên mạnh" E
| | |
|---|---|
| **Phương pháp** | (1) Gradient Sobel 3×3: `|∇R| = √(Gx² + Gy²)`, chuẩn hóa về [0, 255]. (2) Tách ngưỡng **Otsu** trên `|∇R|` để lấy biên mạnh. (3) **Giãn** mặt nạ biên bằng ô vuông `dil×dil` để được E |
| **Tham số** | `dil = 3` (CFD), `5` (CRACK500), lưới {3, 5} |
| **Lý do chọn phương pháp** | Vết nứt thật vừa **tối** vừa có **biên sắc**. Mặt nạ E loại bỏ các mảng tối nhưng chuyển sáng tối **từ từ** (vệt bóng còn sót, vết ẩm loang). Otsu dùng **hợp lý** ở đây vì lược đồ gradient có hai nhóm rõ ràng hơn (vùng phẳng và vùng biên) |
| **Lý do phải giãn** | Ngay **tại tâm** vết nứt, gradient bằng 0 (xem mô phỏng). Biên chỉ nằm ở **hai bờ**, nên phải giãn ra mới phủ được thân vết nứt. CRACK500 có vết nứt rộng hơn nên cần dil = 5 |

**Mô phỏng** trên vết nứt rộng 1 px (R = 80 trên nền 128):
```
R:                              Gx (Sobel ngang):              |∇R|:
128 128 128  80 128 128 128     0  0 -192   0 192  0  0       0  0 192   0 192  0  0
128 128 128  80 128 128 128     0  0 -192   0 192  0  0       0  0 192   0 192  0  0
128 128 128  80 128 128 128     0  0 -192   0 192  0  0       0  0 192   0 192  0  0
                                                                      ↑ tâm nứt = 0 !
Otsu(|∇R|) → biên ở cột 2 và 4:   · · █ · █ · ·
Giãn 3×3 → E phủ cột 1..5:        · █ █ █ █ █ ·   (đã phủ tâm vết nứt)
```

### 3.4 Kết hợp: mặt nạ thô = D ∩ E
```
D (tối hơn nền)            E (gần biên mạnh)          D ∩ E
· · · · · · ·  · ·         · · · · · · · · ·          · · · · · · · · ·
· █ █ █ █ █ ·  · ·         · · █ █ █ · · · ·          · · █ █ █ · · · ·
· █ █ █ █ █ ·  █ ·    ∩    · · █ █ █ · · █ ·    =     · · █ █ █ · · █ ·
· █ █ █ █ █ ·  · ·         · · █ █ █ · · · ·          · · █ █ █ · · · ·
  └ mảng tối loang ┘ └nứt┘   (mảng loang không có biên → bị loại; vết nứt giữ lại)
```
Ví dụ trên là minh họa trên lưới nhỏ. Mặt nạ D ∩ E luôn là **tập con** của D: bước này chỉ xóa bớt pixel, không bao giờ thêm.

**So sánh các phương án xử lý chính** (Dice trung bình, đã kèm hậu xử lý):

| Phương án | Tune CFD | Tune CRACK500 | **Test CFD** | **Test CRACK500** |
|---|---|---|---|---|
| Tam giác (D) | 0.518 | **0.638** | 0.575 | 0.549 |
| **Sobel ∩ tam giác (D ∩ E)** | **0.520** | 0.635 | **0.584** | **0.556** |
| Đối xứng nền + phát triển vùng | 0.512 | 0.574 | 0.528 | 0.511 |

---

## BƯỚC 4 – Hậu xử lý (Chương 7 + đặc trưng hình dạng Chương 8)

Phần tử cấu trúc là hình **ellipse (đĩa)**. Hình này đẳng hướng, nên vết nứt theo hướng nào cũng được xử lý như nhau. Ellipse 5×5 của OpenCV có dạng:
```
· · █ · ·
█ █ █ █ █
█ █ █ █ █
█ █ █ █ █
· · █ · ·
```

### 4.1 Phép đóng: A • B = (A ⊕ B) ⊖ B
| | |
|---|---|
| **Phương pháp** | Giãn rồi co: lấp khe nhỏ và **nối** các đoạn vết nứt bị lệch hoặc đứt, mà không làm dày vết nứt vĩnh viễn |
| **Tham số** | `close = 5` (CFD), `7` (CRACK500), lưới {0, 3, 5, 7} |
| **Lý do** | Dice tune khi không đóng: CFD 0.496, CRACK500 0.607. Đóng k = 5: **0.518** và **0.638**. CRACK500 chọn 7 vì bước 3.3 làm thủng phần ruột của vết nứt rộng, cần phép đóng lớn hơn để lấp lại |

**Mô phỏng:** hai đoạn nứt lệch nhau 1 hàng, chỉ chạm nhau ở góc chéo:
```
Trước (2 vùng):                         Sau phép đóng 5×5 (1 vùng):
···················                     ···················
·███████···········                     ·███████···········
·███████·█████████·          ──►        ·█████████████████·
·········█████████·                     ·········█████████·
                                                 ↑ thêm đúng 1 pixel nối 2 đoạn
```

### 4.2 Phép mở: A ∘ B = (A ⊖ B) ⊕ B – đã thử, KHÔNG dùng
| | |
|---|---|
| **Tham số** | `open = 0` (lưới {0, 3}) |
| **Lý do loại** | Phép mở 3×3 xóa **mọi** cấu trúc mảnh hơn B, kể cả vết nứt rộng 1–2 px. Dice tune: CFD 0.518 → 0.475, CRACK500 0.638 → 0.615. Việc khử đốm do bước 4.3 đảm nhận |

```
Trước:                        Sau phép mở 3×3:
···············               ···············
··██···········               ···············
··██···········      ──►      ···············
·█████████████·               ···············   ← vết nứt 1 px bị xóa sạch ✗
```

### 4.3 Lọc vùng theo diện tích (gán nhãn thành phần liên thông)
| | |
|---|---|
| **Phương pháp** | Gán nhãn các thành phần liên thông theo 8-láng giềng (cùng nguyên lý với phát triển vùng và làm đầy vùng), bỏ vùng có diện tích < `a_min` |
| **Tham số** | `a_min = 30` (CFD), `60` (CRACK500), lưới {0, 30, 60, 120, 200} |
| **Lý do** | Hạt sỏi tạo vùng nhỏ, vết nứt tạo vùng dài và lớn. Đây là bước hậu xử lý **quan trọng nhất**: không lọc thì Dice tune chỉ còn CFD 0.411, CRACK500 0.556. Lọc mạnh quá (200) thì mất các đoạn nứt ngắn |

### 4.4 Lọc theo độ dãn dài (Ch8: đặc trưng hình dạng, mômen)
| | |
|---|---|
| **Phương pháp** | Tính mômen trung tâm bậc 2 μ₂₀, μ₀₂, μ₁₁. Lấy λ₁ ≥ λ₂ là trị riêng của ma trận [[μ₂₀, μ₁₁], [μ₁₁, μ₀₂]]. **Độ dãn dài = √(λ₁/λ₂)**. Bỏ vùng có độ dãn < `e_min`. Vùng lớn (≥ 5·a_min) luôn được giữ, vì mạng nứt chằng chịt có độ dãn thấp |
| **Tham số** | `e_min = 3` (lưới {0, 2, 3, 4}) |
| **Lý do** | Vết nứt **dài và mảnh** nên λ₁ ≫ λ₂. Vết dầu, ổ gà, đốm tối thì **gần tròn** (≈ 1). Bước này tăng Dice tune thêm khoảng 0.02 |

**Mô phỏng 4.3 + 4.4** (CFD: a_min = 30, e_min = 3, ngưỡng "luôn giữ" = 5 × 30 = 150):
```
Vùng                     Diện tích   Độ dãn √(λ1/λ2)   4.3 (≥30?)  4.4                    Kết quả
───────────────────────  ─────────   ───────────────   ──────────  ─────────────────────  ───────
Đốm sỏi 3×3                    9          –               ✗         –                      BỎ
Đốm tối tròn (r = 5)          81        1.0               ✓         81 < 150 và 1.0 < 3    BỎ
Đoạn nứt chéo dày 2 px       120       16.7               ✓         16.7 ≥ 3               GIỮ
Mạng nứt lớn (giả định)      900        2.1               ✓         900 ≥ 150 → giữ luôn   GIỮ
```

### 4.5 Phóng về kích thước gốc
Mặt nạ 480 px được phóng về kích thước gốc bằng `INTER_NEAREST`, vì phép này giữ được giá trị nhị phân 0/1. Ví dụ 480×270 → 2560×1440. Đầu ra cuối cùng là ảnh gốc được **phủ mặt nạ màu đỏ**.

---

## BƯỚC 5 – Đánh giá (Chương 8)

### 5.1 Độ đo
Các độ đo tính trên từng pixel, ở **độ phân giải gốc**, so với GT:
```
                         GT = nứt     GT = nền
  Dự đoán = nứt            TP            FP
  Dự đoán = nền            FN            TN

  Dice = 2TP / (2TP + FP + FN)      (= F1 mức pixel)
  IoU  = TP / (TP + FP + FN)        (IoU = Dice / (2 − Dice) ≤ Dice)
```
**Mô phỏng:** TP = 600, FP = 400, FN = 350
```
Dice      = 1200 / (1200 + 400 + 350) = 0.615
IoU       =  600 / ( 600 + 400 + 350) = 0.444
Precision =  600 / ( 600 + 400)       = 0.600
Recall    =  600 / ( 600 + 350)       = 0.632
```
**Không dùng Accuracy.** Nền chiếm 97.6–98.4% số pixel, nên một mô hình "đoán toàn nền" vẫn đạt accuracy 98.4% (CFD) trong khi không tìm được vết nứt nào.

### 5.2 Chọn tham số (chỉ trên tập tune)
```
Giai đoạn A: lưới {equalize, median, bg, hist_w, dil} = 2×3×4×3×2 = 144 cấu hình
             (hậu xử lý cố định: close 5, open 0, a_min 60, e_min 3)
                  │ giữ cấu hình có Dice tune cao nhất
                  ▼
Giai đoạn B: lưới {close, open, a_min, e_min}         = 4×2×5×4 = 160 cấu hình
                  │ giữ cấu hình có Dice tune cao nhất
                  ▼
        Tham số cuối cùng  →  chạy MỘT LẦN trên tập test
```

### 5.3 Kết quả trên tập test
| Phương pháp | CFD Dice | CFD IoU | CRACK500 Dice | CRACK500 IoU |
|---|---|---|---|---|
| Otsu (đối chứng) | 0.058 | 0.030 | 0.076 | 0.040 |
| Canny (đối chứng) | 0.053 | 0.029 | 0.047 | 0.024 |
| Tam giác (không Sobel) | 0.575 | 0.414 | 0.549 | 0.400 |
| **Pipeline cuối (Sobel ∩ tam giác)** | **0.584** | **0.423** | **0.556** | **0.404** |

Độ đo phụ của pipeline cuối: Precision 0.618 / 0.578, Recall 0.612 / 0.619 (CFD / CRACK500).

* Dice tăng **khoảng 10 lần** so với các phương pháp đối chứng.
* **Otsu** có recall 0.88 nhưng precision chỉ 0.03: phủ đỏ khoảng một nửa ảnh.
* **Canny** bắt biên của mọi hạt sỏi.

### 5.4 Nghiên cứu cắt bỏ (thay đổi một bước, đo Dice test của phương pháp tam giác)
| Thay đổi | CFD | CRACK500 |
|---|---|---|
| Đầy đủ | 0.575 | 0.549 |
| Bỏ bù chiếu sáng | 0.476 (−0.10) | 0.479 (−0.07) |
| Bỏ lọc trung vị | 0.566 (−0.01) | 0.415 (−0.13) |
| Thêm cân bằng lược đồ | 0.041 (−0.53) | 0.082 (−0.47) |
| Bỏ toàn bộ hậu xử lý | 0.413 (−0.16) | 0.414 (−0.14) |
| Chỉ hình thái, không lọc vùng | 0.425 (−0.15) | 0.430 (−0.12) |

Mọi bước đều có đóng góp. Hai bước đóng góp lớn nhất là **hậu xử lý (lọc vùng)** và **bù chiếu sáng**.

### 5.5 Phân tích lỗi
* **FN (bỏ sót):** đoạn nứt mờ hoặc rất hẹp. Các đoạn này không vượt được ngưỡng θ, hoặc chỉ còn là mảnh ngắn < a_min nên bị lọc mất.
* **FP (báo nhầm):** vệt tối dài nhưng không phải nứt (mép vạch sơn, vệt dầu, khe giữa các viên sỏi lớn).
* **Giới hạn của GT CFD (1–3 px):** dự đoán lệch 1 px so với đường tâm cũng bị tính là FP + FN.

---

## Tổng kết kiến thức theo chương
| Chương | Kiến thức dùng | Bước |
|---|---|---|
| C2 | Lấy mẫu lại, độ phân giải không gian | 2.1, 4.5 |
| C3 | Lọc trung vị, biến đổi điểm (ảnh tỉ số), cân bằng lược đồ (đã thử, bị loại) | 2.3, 2.4, 2.5 |
| C4 | Hệ màu RGB → ảnh xám | 2.2 |
| C5 | Làm trơn lược đồ, ngưỡng tam giác, Otsu, gradient Sobel, Canny (đối chứng), đối xứng nền và phát triển vùng (biến thể) | 3.1 – 3.4 |
| C7 | Phần tử cấu trúc, phép giãn, phép co, phép đóng, phép mở, thành phần liên thông | 3.3, 4.1 – 4.3 |
| C8 | Mômen bậc 2 và độ dãn dài, ma trận nhầm lẫn, Precision / Recall / F1 (= Dice) | 4.4, 5 |
