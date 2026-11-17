# Pipeline phát hiện và phân đoạn vết nứt mặt đường – bộ CFD (xử lý ảnh cổ điển)

> **Đầu vào:** ảnh gốc RGB chụp mặt đường (CFD, 480×320).
> **Đầu ra:** mặt nạ nhị phân cùng kích thước (1 = vết nứt) và ảnh gốc phủ mặt nạ màu đỏ.
> **Ràng buộc:** chỉ dùng kiến thức của học phần; không dùng học sâu hay mô hình huấn luyện sẵn.
> **Code:** `code/crack_segmentation_cfd.ipynb` (bản pipeline trên CFD; bản cho bộ Pavement Crack Segmentation: xem `PIPELINE.md`)

---

## 0. Toàn cảnh

```
 ẢNH GỐC RGB 480×320
    │
    ▼
┌────────────────────── 1. DỮ LIỆU (C9) ─────────────────────────────┐
│ CFD 118 ảnh + nhãn pixel → nhị phân; chia ngẫu nhiên seed 42        │
│ 30 tune (chọn tham số) / 88 test (báo cáo)                          │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────── 2. TIỀN XỬ LÝ (C3, C4) ──────────────────────┐
│ 2.1 Ảnh xám        Y = 0.299R + 0.587G + 0.114B                     │
│ 2.2 Khử nhiễu      trung vị 3×3                                     │
│ 2.3 Bù chiếu sáng  B = trung vị 31×31;  R = 128·I/B, cắt [0, 255]   │
└─────────────────────────────────────────────────────────────────────┘
    │  ảnh tỉ số R (nền ≈ 128, vết nứt < 128)
    ▼
┌────────────────────── 3. XỬ LÝ CHÍNH (C5, C7) ─────────────────────┐
│ 3.1 Làm trơn lược đồ R, W = 3                                       │
│ 3.2 Ngưỡng tam giác θ             →  D = (R < θ)   "tối hơn nền"    │
│ 3.3 Sobel |Gx|+|Gy| → Otsu → giãn →  E             "gần biên mạnh"  │
│ 3.4 M = D ∩ E                                                       │
└─────────────────────────────────────────────────────────────────────┘
    │  mặt nạ thô M
    ▼
┌────────────────────── 4. HẬU XỬ LÝ (C7) ───────────────────────────┐
│ 4.1 Phép đóng ellipse 7×7                                           │
│ 4.2 Bỏ thành phần liên thông < 60 px                                │
│ 4.3 Bỏ vùng có độ dãn (hộp chữ nhật xoay) < 2                       │
└─────────────────────────────────────────────────────────────────────┘
    │  mặt nạ cuối
    ▼
┌────────────────────── 5. ĐÁNH GIÁ (C5) ────────────────────────────┐
│ Dice, IoU, Precision, Recall trên 88 ảnh test; so với Otsu, Canny   │
└─────────────────────────────────────────────────────────────────────┘
```

### Bảng tham số

| Bước | Tham số | Khởi đầu | Dãy thử trên tune | **Chọn** | Dice tune tốt nhất theo từng giá trị |
|---|---|---|---|---|---|
| 2.2 | trung vị | 3×3 | 3, 5 | **3×3** | 3: **0.523** · 5: 0.497 |
| 2.3 | cửa sổ nền | 51×51 | 31, 51, 71 | **31×31** | 31: **0.523** · 51: 0.519 · 71: 0.515 |
| 3.1 | làm trơn W | 3 | 1, 3, 5 | **3** | 1: 0.443 · 3: **0.523** · 5: 0.516 |
| 3.2 | θ | tự động | – | tự động | – |
| 3.3 | giãn biên | ellipse 5×5 | 3, 5, 7 | **7×7** | 3: 0.519 · 5: **0.523** · 7: **0.523** |
| 4.1 | phép đóng | ellipse 5×5 | 3, 5, 7 | **7×7** | 3: 0.489 · 5: 0.515 · 7: **0.523** |
| 4.2 | diện tích tối thiểu | 30 px | 15, 30, 60 | **60 px** | 15: 0.502 · 30: 0.515 · 60: **0.523** |
| 4.3 | độ dãn tối thiểu | 3 | 2, 3, 4 | **2** | 2: **0.523** · 3: 0.507 · 4: 0.462 |

Lưới đầy đủ gồm 2 × 3 × 3 × 3 × 3 × 3 × 3 = **1458 cấu hình**, tất cả đều được thử. Tiêu chí chọn là **Dice trung bình trên 30 ảnh tune**.
* Tham số khởi đầu: Dice tune 0.502, xếp hạng 64/1458.
* Tham số chọn: Dice tune **0.523**.

---

## BƯỚC 1 – Dữ liệu (Chương 9)

| | |
|---|---|
| **Nguồn** | CFD (CrackForest, Shi và cộng sự 2016): mặt đường đô thị Bắc Kinh, chụp bằng điện thoại, RGB 480×320. Lấy từ `cfd_image.zip` + `cfd_gt.zip`, giữ **nguyên byte ảnh gốc** |
| **Số ảnh** | Zip có 155 ảnh nhưng chỉ **118 ảnh (`001`–`118`) có nhãn**. Ảnh `201`–`329` bị loại |
| **Gán nhãn** | Mặt nạ PNG **vẽ tay ở mức pixel**, nét mảnh 1–3 px theo đường tâm vết nứt |
| **Nhãn về nhị phân** | GT = (giá trị > 127): 1 = nứt, 0 = nền. Loại bỏ giá trị trung gian do lưu hoặc nén |
| **Chia tập** | **Ngẫu nhiên, seed = 42**: 30 tune / **88 test**. 88 ≥ 50 nên đạt yêu cầu đề. Danh sách lưu ở `results/split.json` |
| **Lý do chia ngẫu nhiên** | Số thứ tự ảnh gần với thứ tự chụp, nên ảnh liền nhau thường cùng đoạn đường và cùng ánh sáng. Chia ngẫu nhiên giúp tune và test đại diện như nhau |

```
Thống kê:   tập    ảnh   % pixel nứt TB (min – max)
            tune    30   1.40%  (0.64 – 2.59)
            test    88   1.69%  (0.35 – 6.33)
→ Vết nứt chỉ chiếm khoảng 1.6% ảnh: lệch lớp nặng. Accuracy "đoán toàn nền" ≈ 98%,
  nên KHÔNG dùng accuracy mà dùng Dice / IoU.
```

---

## BƯỚC 2 – Tiền xử lý

### 2.1 Chuyển ảnh xám (Chương 4)
| | |
|---|---|
| **Phương pháp** | Y = 0.299R + 0.587G + 0.114B (`cv2.COLOR_BGR2GRAY`) |
| **Tham số** | Không có (hệ số chuẩn BT.601) |
| **Lý do** | Vết nứt khác nền về **độ sáng**, không khác về màu. Kênh H của HSV gần như vô nghĩa với mặt đường ít màu |

```
(R, G, B) = (120, 110, 100)  →  Y = 35.88 + 64.57 + 11.40 = 111.85 → 112
Ảnh 480×320×3  →  480×320×1
```

### 2.2 Khử nhiễu: lọc trung vị (Chương 3)
| | |
|---|---|
| **Phương pháp** | Mỗi pixel được thay bằng **trung vị** của cửa sổ k×k |
| **Tham số** | Khởi đầu 3×3, thử {3, 5}. **Chọn 3×3** (Dice tune 0.523 so với 0.497 khi dùng 5×5) |
| **Lý do chọn trung vị** | Hạt sỏi tạo nhiễu kiểu **muối tiêu**, mà trung vị là bộ lọc tốt nhất cho loại nhiễu này. Lọc trung bình hoặc Gauss làm nhòe vết nứt |
| **Lý do chọn 3×3** | 3×3 giữ được nét nứt rộng ≥ 2 px. 5×5 xóa luôn nét rộng 2 px |

```
(a) Đốm nhiễu 1 px, trung vị 3×3:          (b) Vết nứt rộng 2 px, trung vị 3×3:
  120 120 120       120 120 120              120  60  60 120      120  60  60 120
  120  30 120  ──►  120 120 120              120  60  60 120  ──► 120  60  60 120
  120 120 120       120 120 120              120  60  60 120      120  60  60 120
  cửa sổ [30,120×8] → 120: BỊ XÓA ✓          cửa sổ [60×6,120×3] → 60: GIỮ ✓

(c) Cùng vết nứt 2 px, trung vị 5×5: chỉ 10/25 pixel tối → trung vị = 120: BỊ XÓA ✗
```

### 2.3 Bù chiếu sáng: ảnh tỉ số (Chương 3)
| | |
|---|---|
| **Phương pháp** | B = trung vị cửa sổ **lớn** (ảnh nền); **R = 128·I/(B + 1)**, cắt về [0, 255] |
| **Tham số** | Khởi đầu 51×51, thử {31, 51, 71}. **Chọn 31×31** (Dice tune 0.523 / 0.519 / 0.515, ít nhạy) |
| **Lý do** | Vết nứt mảnh so với cửa sổ nên **không lọt vào B**, tức B chính là mức nền cục bộ. Bóng đổ thay đổi chậm nên nằm trong B, và bị **triệt tiêu** khi chia I cho B. Dùng phép **chia** vì ánh sáng tác động theo kiểu nhân |
| **Đóng góp** | Bỏ bước này thì Dice test giảm **0.121** (0.538 → 0.417) |

```
vị trí    :   0    1    2    3    4    5    6    7    8    9   10
              ── nắng ───────────► chuyển dần ───── bóng râm ─────
I         : 160  160  [96] 160  150  130  110  100  [60] 100  100
B         : 160  160  160  160  150  130  110  100  100  100  100
R=128·I/B : 127  127  [76] 127  127  127  126  126  [76] 126  126
→ Ngưỡng trên I (I < 105) gán nhầm bóng râm (7, 9, 10) là vết nứt.
→ Trên R: nền ≈ 127 ở mọi nơi, cả hai vết nứt đều = 76, nên một ngưỡng duy nhất là đủ.
```

---

## BƯỚC 3 – Xử lý chính

### 3.1 Làm trơn lược đồ của R (Chương 5, tr.62)
| | |
|---|---|
| **Phương pháp** | Trung bình trượt cửa sổ W trên lược đồ: h'[b] = trung bình của h[b−W/2 … b+W/2] |
| **Tham số** | Khởi đầu W = 3, thử {1, 3, 5}. **Chọn W = 3** (Dice tune 0.443 / **0.523** / 0.516) |
| **Lý do** | R được tính từ các số nguyên rồi làm tròn (lượng tử hóa), nên lược đồ có **gai ở 128 ngay cạnh hố ở 127**. Không làm trơn thì thuật toán tam giác chọn đúng hố 127, và D phủ khoảng **50% ảnh** |
| **Đóng góp** | Bỏ bước này (W = 1) thì có **11/88 ảnh test hỏng hoàn toàn** và Dice giảm 0.062 |

```
Lược đồ R thật (ảnh 047, không làm trơn), quanh đỉnh nền:
mức   : 122   123   124    125    126   [127]  [128]   129    130
h     : 7448  9400  11235  12611  13962  5930  17589  12280  10620
                                         ↑ hố    ↑ gai
W = 1 : Δ nối tới gai 128 → điểm xa Δ nhất là hố 127 → θ = 127 → D phủ ~50% ảnh  ✗
W = 3 : (13962 + 5930 + 17589)/3 = 12494 ... → hố biến mất → θ nằm ở chân đỉnh nền  ✓
```

### 3.2 Vùng tối hơn nền D: ngưỡng tam giác (Chương 5, tr.78)
| | |
|---|---|
| **Phương pháp** | (1) Đường Δ nối đỉnh lược đồ (b_max) với mức khác 0 xa nhất phía tối (b_min). (2) Với mỗi b, tính khoảng cách d từ h[b] tới Δ. (3) **θ = b có d lớn nhất**. **D = (R < θ)** |
| **Tham số** | Không có. θ tự động cho **từng ảnh** |
| **Lý do chọn tam giác** | Lược đồ R gồm **một đỉnh lớn** (nền khoảng 98%) và một **đuôi mỏng** (nứt khoảng 2%), đúng loại lược đồ phương pháp tam giác được thiết kế cho |
| **Lý do không dùng Otsu** | Otsu giả định hai đỉnh nên đặt ngưỡng vào giữa đỉnh nền. Trên ảnh mẫu: Otsu gán 7.5% ảnh là nứt, tam giác gán 4.9%, trong khi GT là 1.3% |

```
Ví dụ lược đồ 16 mức (đã làm trơn W = 3):
b    :  0    1    2    3    4    5    6    7    8     9    10   11   12   13   14   15
h[b] : 0.3  1.0  1.3  2.0  2.0  3.0  4.0  8.3  20.3  48.3 63.3 56.7 28.3 8.7  2.0  0.3
       └────────── đuôi tối (nứt) ──────────┘ └──── đỉnh nền ────┘

          h
     63 ┤                                         ●  ← b_max = 10
        │                                      ●     ●
     20 ┤                       Δ ╱        ●
        │                     ╱    ↕ d lớn nhất
      8 ┤               ╱      ●  ← b = 7
      0 ●──●──●──●──●──●───────────────────────────────── b
        b_min = 0

d(b): 0.00 0.88 1.82 2.70 3.69 4.52 5.35 [5.66] 4.77 1.36 0.00   →  θ = 7
```

### 3.3 Vùng gần biên mạnh E: Sobel + Otsu + giãn (Chương 5, 7)
| | |
|---|---|
| **Phương pháp** | Sobel 3×3 trên R, **\|∇R\| = \|Gx\| + \|Gy\|**, chuẩn hóa [0, 255] → **Otsu** để lấy biên mạnh → **giãn ellipse** để được E |
| **Tham số** | Khởi đầu giãn 5×5, thử {3, 5, 7}. **Chọn 7×7** (Dice tune 0.519 / 0.523 / 0.523) |
| **Lý do dùng biên** | Vết nứt thật vừa tối vừa có **biên sắc**. E dùng để loại các vùng tối nhưng chuyển sáng tối **từ từ** (bóng còn sót, vết ẩm loang) |
| **Lý do dùng Otsu ở đây** | Lược đồ \|∇R\| có **hai nhóm** (vùng phẳng và vùng biên), nên hợp giả định của Otsu |
| **Lý do phải giãn** | **Tại tâm** vết nứt mảnh, gradient = 0; biên chỉ nằm ở hai bờ |

```
R:     128 128 128  80 128 128 128
Gx:      0   0 -192   0 192   0   0
|∇R|:    0   0  192   0 192   0   0     ← tâm nứt có |∇R| = 0 !
Otsu:    ·   ·   █    ·  █    ·   ·
giãn →E: ·   █   █    █  █    █   ·     ← phủ kín thân vết nứt
```

**Kết quả đo được (thẳng thắn):** trên CFD, bước này **gần như không có tác dụng**. Otsu trên \|∇R\| lấy cả biên của kết cấu sỏi, nên E phủ phần lớn ảnh:

| Giãn | E phủ (TB test) | Pixel của D bị E loại |
|---|---|---|
| 3×3 | 52.5% ảnh | 6.0% |
| 5×5 | 77.9% ảnh | 1.4% |
| 7×7 | 88.9% ảnh | 0.9% |

Bỏ bước này (M = D) cho 89% ảnh test có kết quả **giống hệt**, và Dice chỉ giảm 0.002.

### 3.4 Mặt nạ thô M = D ∩ E
```
D (tối)              E (gần biên)         M = D ∩ E
· █ █ █ █ · · ·      · · █ █ · · █ ·      · · █ █ · · · ·
· █ █ █ █ · █ ·  ∩   · · █ █ · · █ ·  =   · · █ █ · · █ ·
· █ █ █ █ · · ·      · · █ █ · · · ·      · · █ █ · · · ·
(minh họa: mảng tối loang không có biên ở giữa bị loại)
M luôn ⊆ D: bước này chỉ xóa bớt pixel, không bao giờ thêm.
```

---

## BƯỚC 4 – Hậu xử lý (Chương 7)

Phần tử cấu trúc ellipse 5×5 của OpenCV:
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
| **Phương pháp** | Giãn rồi co với ellipse: lấp khe nhỏ và **nối đoạn nứt đứt**, không làm dày vết nứt vĩnh viễn |
| **Tham số** | Khởi đầu 5×5, thử {3, 5, 7}. **Chọn 7×7** (Dice tune 0.489 / 0.515 / **0.523**) |
| **Lý do** | Vết nứt có chỗ hẹp hoặc sáng hơn nên bị đứt thành nhiều đoạn. Nối lại giúp 4.2 không xóa nhầm các đoạn ngắn |

```
Trước (2 vùng):                       Sau phép đóng 5×5 (1 vùng):
·███████···········                   ·███████···········
·███████·█████████·        ──►        ·█████████████████·
·········█████████·                   ·········█████████·
                                              ↑ thêm 1 pixel nối 2 đoạn
```

### 4.2 Lọc diện tích
| | |
|---|---|
| **Phương pháp** | Gán nhãn thành phần liên thông **8-láng giềng**, bỏ vùng có diện tích < a_min |
| **Tham số** | Khởi đầu 30 px, thử {15, 30, 60}. **Chọn 60 px** (Dice tune 0.502 / 0.515 / **0.523**) |
| **Lý do** | Đốm sỏi là vùng nhỏ, vết nứt là vùng dài và lớn. Sau phép đóng 7×7, vết nứt đã liền thành vùng lớn nên có thể lọc mạnh hơn. Dùng 8-láng giềng để vết nứt xiên không bị tách vùng |

### 4.3 Lọc độ dãn: trục dài / trục ngắn của hộp chữ nhật xoay
| | |
|---|---|
| **Phương pháp** | `cv2.minAreaRect`: hộp chữ nhật **xoay** có diện tích nhỏ nhất bao vùng, có cạnh L ≥ S. **Độ dãn = L / max(S, 1)**. Bỏ vùng có độ dãn < e_min |
| **Tham số** | Khởi đầu 3, thử {2, 3, 4}. **Chọn 2** (Dice tune **0.523** / 0.507 / 0.462) |
| **Lý do** | Vết nứt dài và mảnh (L ≫ S), còn đốm dầu hoặc ổ gà gần tròn (L/S ≈ 1). Hộp **xoay** đo đúng cả vết nứt chéo. Chọn 2 (lỏng) vì vết nứt CFD thường **cong hoặc phân nhánh**, khiến hộp bao phình ra |

```
Hình (đo bằng code)                  Diện tích   L/S hộp xoay   L/S hộp thẳng   e_min = 2   e_min = 3
───────────────────────────────────  ─────────   ────────────   ─────────────   ─────────   ─────────
Đốm tròn r = 5                             81        1.00            1.00          BỎ          BỎ
Đốm vuông 9×9                              81        1.00            1.00          BỎ          BỎ
Vết nứt ngang dày 2, dài 40               125       21.00           14.33          GIỮ         GIỮ
Vết nứt chéo 45° dày 2                    207       20.50            1.00 ✗        GIỮ         GIỮ
Vết nứt cong (cung tròn)                  173        2.75            2.65          GIỮ         BỎ ✗
→ Hộp thẳng sẽ xóa nhầm vết nứt chéo; e_min = 3 xóa nhầm vết nứt cong.
```

---

## BƯỚC 5 – Đánh giá (Chương 5)

### Độ đo
```
                     GT = nứt    GT = nền
  Dự đoán = nứt         TP          FP
  Dự đoán = nền         FN          TN

  Dice = 2TP/(2TP+FP+FN)   IoU = TP/(TP+FP+FN)   Precision = TP/(TP+FP)   Recall = TP/(TP+FN)

  Ví dụ: TP = 600, FP = 400, FN = 350
         Dice = 1200/1950 = 0.615   IoU = 600/1350 = 0.444   P = 0.600   R = 0.632
```

### Kết quả trên tập test (88 ảnh, trung bình theo ảnh)
| Phương pháp | Dice | IoU | Precision | Recall | Ảnh hỏng (Dice < 0.1) |
|---|---|---|---|---|---|
| Đối chứng: Otsu | 0.061 | 0.032 | 0.032 | 0.852 | 79 |
| Đối chứng: Canny | 0.060 | 0.032 | 0.040 | 0.393 | 77 |
| Pipeline – tham số khởi đầu | 0.532 | 0.383 | 0.606 | 0.538 | 3 |
| **Pipeline – tham số đã chọn** | **0.538** | **0.387** | 0.561 | 0.578 | 3 |

* Dice tăng **khoảng 9 lần** so với đối chứng, và pipeline tốt hơn Otsu và Canny trên **98%** số ảnh test.
* Dice tune 0.523 so với Dice test 0.538, nên không có dấu hiệu khớp quá mức.
* Ảnh tốt nhất đạt Dice 0.82; có 12/88 ảnh vượt 0.7.

### Đóng góp của từng bước (bỏ một bước, giữ nguyên tham số đã chọn)
| Thay đổi | Dice | Mức giảm |
|---|---|---|
| Đầy đủ | 0.538 | – |
| Bỏ bước 4 (hậu xử lý) | 0.385 | −0.153 |
| Bỏ 2.3 (bù chiếu sáng) | 0.417 | −0.121 |
| Bỏ 3.1 (làm trơn, W = 1) | 0.476 | −0.062 (11 ảnh hỏng) |
| Bỏ 3.3 (M = D) | 0.536 | −0.002 |

### Phân tích lỗi
* **FN (bỏ sót):** đoạn nứt mờ hoặc rất hẹp không vượt được θ; mảnh nứt ngắn bị lọc ở bước 4.2.
* **FP (báo nhầm):** vệt tối dài không phải nứt (mép vạch sơn, vệt dầu, khe sỏi lớn); phép đóng 7×7 làm vết nứt dày hơn GT.
* **GT mảnh (1–3 px):** lệch 1 px so với đường tâm cũng bị tính là FP + FN.

---

## Tổng kết kiến thức theo chương
| Chương | Kiến thức | Bước |
|---|---|---|
| C3 | Lọc trung vị (khử nhiễu và ước lượng nền), biến đổi điểm (ảnh tỉ số) | 2.2, 2.3 |
| C4 | Hệ màu RGB → ảnh xám | 2.1 |
| C5 | Làm trơn lược đồ, ngưỡng tam giác, Sobel, Otsu, Canny, độ đo đánh giá | 3.1–3.3, 5 |
| C7 | Phép giãn, phép đóng, thành phần liên thông, độ dãn (hộp chữ nhật xoay) | 3.3, 4.1–4.3 |
| C9 | Nguồn dữ liệu, gán nhãn, chia tập có seed | 1 |
