# Pipeline phát hiện và phân đoạn vết nứt mặt đường – bộ Pavement Crack Segmentation Dataset (xử lý ảnh cổ điển)

> **Đầu vào:** ảnh gốc RGB chụp mặt đường (1702×866 hoặc 1080×760).
> **Đầu ra:** mặt nạ nhị phân cùng kích thước ảnh gốc (1 = vết nứt; ảnh không nứt cho mặt nạ rỗng) và ảnh gốc phủ mặt nạ đỏ.
> **Ràng buộc:** chỉ dùng kiến thức của học phần; không dùng học sâu hay mô hình huấn luyện sẵn.
> **Code:** `code/crack_segmentation.ipynb`. Pipeline gốc trên CFD: `PIPELINE_CFD.md`, `code/crack_segmentation_cfd.ipynb`.

---

## 0. Pipeline CFD phải sửa gì cho bộ mới

| Khác biệt của bộ mới (đo được) | Hậu quả nếu giữ pipeline CFD | Sửa | Bằng chứng (đo) |
|---|---|---|---|
| Ảnh lớn gấp 4–10 lần CFD; vết nứt GT rộng 7–12 px (CFD 1–3 px) | Các cửa sổ cố định (nền 31, đóng 7, a_min 60) sai tỉ lệ | **Thêm 2.0: thu nhỏ cạnh dài về 640 px** | Bỏ 2.0: Dice test 0.339 → 0.267 |
| Mặt nạ lưu dạng xám / RGB / RGBA; 27/2209 mặt nạ có mép khử răng cưa | Đọc sai kênh, giữ mức trung gian | **Nhãn về nhị phân:** đọc xám, > 127 | – |
| **45/231 ảnh test (19%) không có vết nứt** (CFD: 0) | Pipeline luôn tìm ra "vết nứt", nên mọi ảnh lành bị Dice = 0 | **Thêm 4.4: quyết định mức ảnh** theo tổng diện tích | Bỏ 4.4: Dice 0.339 → 0.274 |
| 3 nhóm ảnh A, B, C rất khác nhau; nền sỏi thô | Tham số CFD xếp hạng 3842/4320 | **Chọn lại tham số** (tune trên train, xác nhận trên val) | Val: CFD 0.253 → chọn lại 0.364 |
| (Đã thử, **không** dùng) nền bằng phép đóng thay trung vị | – | Giữ trung vị | Đóng 7–21: Dice 0.06–0.20 so với trung vị 0.24 |

## 1. Toàn cảnh

```
 ẢNH GỐC RGB (1702×866 / 1080×760)
    │
    ▼
┌──────────────────────── 1. DỮ LIỆU (C9) ─────────────────────────────┐
│ Chia sẵn: train 1480 / val 498 / test 231; nhãn = (xám > 127)          │
│ tune = 300 ảnh train phân tầng (A 30, B 90, C 180, seed 42)            │
└───────────────────────────────────────────────────────────────────────┘
    ▼
┌──────────────────────── 2. TIỀN XỬ LÝ (C2, C3, C4) ──────────────────┐
│ 2.0 Thu nhỏ   cạnh dài = 640 px, INTER_AREA                    (MỚI) │
│ 2.1 Ảnh xám   Y = 0.299R + 0.587G + 0.114B                           │
│ 2.2 Khử nhiễu trung vị 5×5                                            │
│ 2.3 Bù sáng   B = trung vị 71×71;  R = 128·I/(B+1), cắt [0, 255]      │
└───────────────────────────────────────────────────────────────────────┘
    ▼  ảnh tỉ số R (nền ≈ 128, vết nứt < 128)
┌──────────────────────── 3. XỬ LÝ CHÍNH (C5, C7) ─────────────────────┐
│ 3.1 Làm trơn lược đồ R, W = 3                                         │
│ 3.2 Ngưỡng tam giác θ              →  D = (R < θ)                     │
│ 3.3 Sobel |Gx|+|Gy| → Otsu → giãn 3×3 →  E                            │
│ 3.4 M = D ∩ E                                                         │
└───────────────────────────────────────────────────────────────────────┘
    ▼
┌──────────────────────── 4. HẬU XỬ LÝ (C7) ───────────────────────────┐
│ 4.1 Đóng ellipse 7×7                                                  │
│ 4.2 Bỏ thành phần liên thông < 30 px                                  │
│ 4.3 Bỏ vùng có độ dãn L/S (hộp chữ nhật xoay) < 3                     │
│ 4.4 Tổng diện tích < 0.3% ảnh  →  mặt nạ RỖNG                  (MỚI) │
│     Phóng mặt nạ về kích thước gốc (láng giềng gần nhất)              │
└───────────────────────────────────────────────────────────────────────┘
    ▼
┌──────────────────────── 5. ĐÁNH GIÁ (C5) ────────────────────────────┐
│ Dice, IoU, P, R trên 231 ảnh test, ở kích thước gốc;                  │
│ so với Otsu, Canny, pipeline CFD                                      │
└───────────────────────────────────────────────────────────────────────┘
```

### Bảng tham số

| Bước | Tham số | CFD (= khởi đầu) | Dãy thử | **Chọn** | Dice tune tốt nhất theo từng giá trị |
|---|---|---|---|---|---|
| 2.0 | cạnh dài | 640 | 480, 640 | **640** | 480: 0.328 · 640: 0.326 |
| 2.2 | trung vị | 3×3 | 3, 5 | **5×5** | 3: 0.309 · 5: **0.328** |
| 2.3 | cửa sổ nền | 31 | 31, 51, 71 | **71** | 31: **0.328** · 51: 0.326 · 71: 0.319 |
| 3.1 | làm trơn W | 3 | 3, 5 | **3** | 3: **0.328** · 5: 0.317 |
| 3.3 | giãn biên | 7×7 | 3, 5, 7 | **3×3** | 3: **0.328** · 5: 0.320 · 7: 0.318 |
| 4.1 | phép đóng | 7×7 | 5, 7 | **7×7** | 5: 0.317 · 7: **0.328** |
| 4.2 | a_min | 60 | 30, 60, 120 | **30** | 30: **0.328** · 60: **0.328** · 120: 0.314 |
| 4.3 | e_min | 2 | 2, 3 | **3** | 2: **0.328** · 3: **0.328** |
| 4.4 | t_min | – (P0: 0.2%) | 0, 0.1, 0.2, 0.3, 0.4% | **0.3%** | 0: 0.285 · 0.1: 0.300 · 0.2: 0.325 · 0.3: **0.328** · 0.4: **0.328** |

* **Lưới:** 2 × 2 × 3 × 2 × 3 × 2 × 3 × 2 × 5 = **4320 cấu hình**, thử toàn bộ trên 300 ảnh tune (khoảng 2 phút, 10 tiến trình).
* **Chọn:** 10 cấu hình có Dice tune cao nhất được chạy trên **toàn bộ 498 ảnh val** ở kích thước gốc. Cấu hình có Dice val cao nhất (0.364) được chọn; trên tune nó xếp hạng 10 (0.319). Mười cấu hình đầu chỉ chênh nhau 0.009 trên tune.
* **Tham số CFD** xếp hạng 3842/4320 trên tune (0.237).

---

## BƯỚC 1 – Dữ liệu (Chương 9)

| | train | val | test |
|---|---|---|---|
| Số ảnh (A / B / C) | 1480 (81 / 440 / 959) | 498 (27 / 155 / 316) | **231** (13 / 82 / 136) |
| Ảnh không nứt | 261 | 115 | **45** (44 thuộc nhóm C) |
| % pixel nứt trung bình | 0.86 | 0.82 | 0.93 |
| Kích thước | A, B: 1702×866 · C: 1080×760 | | |

* **Nguồn:** `data/Pavement_Crack_Segmentation_Dataset/` (từ `Pavement_Crack_Segmentation_Dataset.zip`), đã chia sẵn `train/val/test`, mỗi tập có `images/*.jpg` và `masks/*.png`. Tiền tố A_, B_, C_ là 3 nhóm nguồn. Pipeline không dùng thông tin nhóm khi dự đoán.
* **Nhãn:** mặt nạ mức pixel, tô **cả bề rộng** vết nứt. Mặt nạ lưu ở 3 dạng: xám (160 ảnh C), RGB (1744 ảnh), RGBA (305 ảnh C). 27 mặt nạ có mép khử răng cưa. Nhãn được đưa về nhị phân bằng **GT = đọc xám > 127**.
* **Vai trò các tập:** tune = 300 ảnh train phân tầng (thử lưới) → val (chọn 1 trong 10 cấu hình) → test (báo cáo **một lần**).

---

## BƯỚC 2 – Tiền xử lý

### 2.0 Thu nhỏ (MỚI, Chương 2: lấy mẫu)
| | |
|---|---|
| **Phương pháp** | Cạnh dài = 640 px, giữ tỉ lệ, **INTER_AREA** (mỗi pixel mới là trung bình vùng nó phủ, tương đương lọc thông thấp trước khi lấy mẫu lại nên chống răng cưa). Mặt nạ kết quả được phóng về kích thước gốc bằng **láng giềng gần nhất** |
| **Lý do** | Đưa bề rộng vết nứt về dải mà các cửa sổ của pipeline được thiết kế cho. Đồng thời ít pixel hơn 4–7 lần nên thử được 4320 cấu hình |
| **Đo** | Bề rộng GT (px): gốc A 8.4 / B 12.1 / C 7.4; ở 640 px còn 3.2 / 4.7 / 4.5; ở 480 px còn 2.4 / 3.5 / 3.4. Với tham số CFD, Dice tune: 480 px 0.258, 640 px 0.246, 800 px 0.228. Vì vậy 800 bị loại; 480 và 640 vào lưới |

```
1702×866 ──INTER_AREA──► 640×326     1 pixel mới ≈ trung bình khối 2.66×2.66 pixel cũ
vết nứt rộng 12 px     ──►  ≈ 4.5 px
                                      ┌ 10 20 ┐
Ví dụ thu nhỏ 2×: khối 2×2 pixel      └ 30 40 ┘  ──► 25 (trung bình)
Láng giềng gần nhất sẽ lấy 10 và bỏ 3 pixel còn lại: vết nứt 1 px có thể bị mất hẳn.
```

### 2.1 Ảnh xám (Chương 4)
Y = 0.299R + 0.587G + 0.114B. Vết nứt khác nền ở **độ sáng**, không ở màu; phần có màu trong bộ mới là vạch sơn và vỉa hè. Ví dụ: (120, 110, 100) → 111.85 ≈ **112**.

### 2.2 Lọc trung vị 5×5 (Chương 3) – khác CFD (3×3)
Hạt sỏi tạo nhiễu kiểu **muối tiêu**, và lọc trung vị là bộ lọc phù hợp nhất cho loại nhiễu này. Trên CFD, 5×5 xóa nét 2 px. Ở bộ mới sau khi thu nhỏ, vết nứt rộng 3–5 px nên 5×5 vẫn giữ được, và khử mạnh hơn nền sỏi thô (nhóm B).

```
Vết nứt rộng 3 px, cửa sổ 5×5:            Đốm nhiễu 2×2, cửa sổ 5×5:
120  60  60  60 120                         4/25 pixel tối → trung vị = 120: BỊ XÓA ✓
120  60  60  60 120   15/25 pixel tối
120  60  60  60 120   → trung vị = 60: GIỮ ✓
```

### 2.3 Bù chiếu sáng: R = 128·I/(B + 1), B = trung vị 71×71 (Chương 3)
Bóng đổ và độ sáng giảm dần theo chiều sâu (ảnh chụp xiên) thay đổi chậm nên nằm trong B và bị triệt tiêu khi chia. Dùng phép **chia** vì ánh sáng tác động theo kiểu nhân. Bỏ bước này thì Dice ảnh nứt giảm từ 0.303 xuống 0.206.

```
I         : 160  160  [96] 160  150  130  110  100  [60] 100  100     (nắng → bóng râm)
B         : 160  160  160  160  150  130  110  100  100  100  100
R=128·I/B : 127  127  [76] 127  127  127  126  126  [76] 126  126     → một ngưỡng duy nhất là đủ
```
Đã thử thay B bằng **phép đóng** (cực đại cục bộ rồi cực tiểu): nền bị nâng theo các hạt sỏi sáng, nên cả mặt đường thành "tối hơn nền". Dice chỉ đạt 0.06–0.20, thấp hơn trung vị (0.24), nên không dùng.

---

## BƯỚC 3 – Xử lý chính

### 3.1 Làm trơn lược đồ R, W = 3 (Chương 5, tr.62)
R được làm tròn từ số nguyên nên lược đồ có **gai ở 128 cạnh hố ở 127**. Trung bình trượt h'[b] = (h[b−1] + h[b] + h[b+1]) / 3 xóa răng cưa này trước khi tìm ngưỡng.

```
b      : 125   126   127   128   129
h      : 900  1000   300  2600  1100     ← hố 127 cạnh gai 128
h' (W=3):  …    733  1300  1333   …       ← trơn, đỉnh vẫn ở 128
```
Trên bộ mới, bỏ bước này **không làm giảm** Dice (0.340 so với 0.339): với trung vị 5×5, R ít bị lượng tử hóa hơn. Bước này được giữ để an toàn (trên CFD nó ngăn 11/88 ảnh hỏng).

### 3.2 Ngưỡng tam giác → D = (R < θ) (Chương 5, tr.78)
Lược đồ R có **một đỉnh lớn** (nền khoảng 99%) và **đuôi mỏng** về phía tối (vết nứt khoảng 1%). Thuật toán: nối đỉnh b_max với mức khác 0 tối nhất b_min bằng đường Δ; **θ = b có khoảng cách tới Δ lớn nhất**. Otsu cần lược đồ hai đỉnh nên thất bại ở đây. Ví dụ ảnh mẫu B: Otsu gán **56.9%** ảnh là nứt; tam giác gán **1.5%**; GT 0.8%.

```
 h │                 █ ← b_max (nền)
   │                ██
   │               ███
   │  Δ        . ' ███
   │     . '   ↕d max ███
   │. '  ▂▃▄▅▆▇█████████▇▅▃
   └──┬──────────┬────┬──── b
    b_min        θ  128
```

### 3.3 Vùng gần biên E: Sobel L1 → Otsu → giãn 3×3 (Chương 5, 7) – khác CFD (7×7)
Vết nứt có biên sắc; vùng tối loang thì không. Tại tâm vết nứt gradient = 0, nên phải giãn để phủ thân vết nứt. Trên CFD, giãn 7×7 làm E phủ gần hết ảnh nên bước này vô tác dụng. Ở bộ mới, giãn **3×3** cho E chặt hơn: bỏ 3.3 thì Dice giảm 0.339 → 0.329, và số ảnh lành đoán đúng giảm 22 → 16.

```
R:     128 128 128  80 128 128 128
|∇R|:    0   0 192   0 192   0   0      (tâm nứt = 0)
Otsu:    ·   ·   █   ·   █   ·   ·
giãn:    ·   █   █   █   █   █   ·  = E (phủ thân vết nứt)
```

### 3.4 M = D ∩ E
Pixel là vết nứt khi **vừa tối hơn nền vừa gần biên mạnh**.

---

## BƯỚC 4 – Hậu xử lý (Chương 7)

### 4.1 Phép đóng ellipse 7×7: (A ⊕ B) ⊖ B
Nối các đoạn nứt đứt, không làm dày vĩnh viễn.
```
·███████···········          ·███████···········
·███████·█████████·   ──►    ·█████████████████·
·········█████████·          ·········█████████·
```

### 4.2 Lọc diện tích < 30 px (thành phần liên thông 8-láng giềng)
Xóa đốm sỏi rời rạc. Ở bộ mới chọn **30** (CFD 60): quyết định "ảnh có nứt hay không" đã chuyển sang 4.4, nên 4.2 chỉ cần xóa vụn và không nên xóa đoạn nứt ngắn.

### 4.3 Lọc độ dãn L/S < 3 (hộp chữ nhật xoay `minAreaRect`) – CFD dùng 2
| Hình | L/S hộp xoay | L/S hộp thẳng | Giữ (e_min = 3)? |
|---|---|---|---|
| đốm tròn r = 5 | 1.00 | 1.00 | không |
| đốm vuông 9×9 | 1.00 | 1.00 | không |
| nứt ngang dày 2, dài 40 | 21.0 | 14.3 | giữ |
| nứt chéo 45° dày 2 | 20.5 | **1.00** (hộp thẳng xóa nhầm) | giữ |
| nứt cong (cung tròn) | 2.75 | 2.65 | **không** |

Vết nứt ở bộ mới **thẳng** hơn CFD nên chịu được e_min = 3, và lọc thêm được các vết dầu, nắp cống gần tròn.

### 4.4 Quyết định mức ảnh: s = diện tích / diện tích ảnh < 0.3% → mặt nạ rỗng (MỚI)
```
ảnh lành   : sau 4.3 còn vài khe nối, vết bẩn → s = 0.05% < 0.3% → RỖNG   (Dice 0 → 1)
ảnh có nứt : s = 0.9% ≥ 0.3%                                  → giữ nguyên
ảnh nứt nhỏ: s = 0.2% < 0.3%                                  → RỖNG   (mất vết nứt: cái giá)
```
* **Vì sao tổng diện tích:** đây là đặc trưng tách ảnh nứt / không nứt tốt nhất trong các đặc trưng đơn giản đã thử. AUC trên tune: tổng diện tích **0.72**, diện tích vùng lớn nhất 0.71, độ dài vùng dài nhất 0.67, số vùng 0.59, độ tương phản trung bình 0.46.
* **Đo trên test:** 22/45 ảnh lành được trả rỗng đúng (không có 4.4: 2/45), đổi lại 45/186 ảnh có nứt bị trả rỗng (C 30, B 14, A 1).

---

## BƯỚC 5 – Đánh giá trên test (231 ảnh, đánh giá ở kích thước gốc)

**Độ đo:** Dice = 2TP/(2TP+FP+FN), IoU = TP/(TP+FP+FN). **Ảnh không nứt:** dự đoán rỗng thì Dice = IoU = 1, có pixel nào thì 0. Precision và Recall chỉ tính trên ảnh có nứt. **Gộp** = cộng TP, FP, FN mọi ảnh rồi mới tính.

| Phương pháp | Dice | IoU | Dice ảnh nứt | P | R | Dice gộp | IoU gộp | Ảnh lành đúng | Ảnh nứt bị trả rỗng |
|---|---|---|---|---|---|---|---|---|---|
| Otsu | 0.024 | 0.012 | 0.030 | 0.015 | 0.723 | 0.027 | 0.014 | 0/45 | 0 |
| Canny | 0.018 | 0.009 | 0.022 | 0.012 | 0.352 | 0.019 | 0.010 | 0/45 | 0 |
| Pipeline CFD (tham số CFD, không 4.4) | 0.271 | 0.197 | **0.337** | 0.460 | 0.325 | 0.349 | 0.211 | 0/45 | 4 |
| Pipeline mới – khởi đầu P0 | 0.296 | 0.227 | 0.319 | 0.399 | 0.314 | 0.345 | 0.209 | 9/45 | 32 |
| **Pipeline mới – đã chọn** | **0.339** | **0.274** | 0.303 | 0.328 | 0.312 | **0.426** | **0.271** | **22/45** | 45 |

**Theo nhóm** (Dice theo ảnh; trong ngoặc là Dice ảnh nứt): CFD A 0.183 (0.183) · B 0.587 (0.595) · C 0.089 (0.132). Pipeline mới A 0.126 (0.126) · B 0.554 (0.548) · **C 0.231** (0.113).

**Bỏ từng bước** (giữ tham số đã chọn):
| Thay đổi | Dice | Dice ảnh nứt | Ảnh lành đúng |
|---|---|---|---|
| (đầy đủ) | 0.339 | 0.303 | 22/45 |
| Bỏ bước 4 | 0.214 | 0.265 | 0/45 |
| Bỏ 2.0 (kích thước gốc) | 0.267 | 0.197 | 25/45 |
| Bỏ 4.4 | 0.274 | 0.329 | 2/45 |
| Bỏ 2.3 | 0.283 | 0.206 | 27/45 |
| Bỏ 3.3 | 0.329 | 0.323 | 16/45 |
| Bỏ 3.1 | 0.340 | 0.304 | 22/45 |

### Đọc kết quả
1. Pipeline mới tốt hơn pipeline CFD: Dice **0.271 → 0.339**, IoU **0.197 → 0.274**, Dice gộp **0.349 → 0.426**. Tune → val → test: 0.32 → 0.36 → 0.34, nên không có dấu hiệu khớp quá mức.
2. Mức tăng chủ yếu đến từ **ảnh lành** (bước 4.4) và từ việc **báo nhầm ít hơn** (Dice gộp tăng). Trên ảnh có nứt, Dice **giảm nhẹ** (0.337 → 0.303) vì 45 ảnh nứt nhỏ bị trả rỗng. Đây là đánh đổi của 4.4 và cần nêu rõ khi báo cáo.
3. Theo nhóm: C tăng mạnh nhờ 4.4; B giảm nhẹ (do 4.4: bỏ 4.4 thì B đạt 0.582); A giảm do tham số chung hợp với B, C hơn (bỏ 4.4 thì A vẫn là 0.128). A chỉ có 13 ảnh nên con số dao động lớn.
4. **Lỗi còn lại:** FN do vết nứt mờ ở xa trong ảnh chụp xiên, và do GT tô cả bề rộng trong khi D chỉ lấy lõi tối. FP do vạch sơn bị nứt, mép vỉa hè, nắp cống, khe nối và vết dầu dài: chúng tối, dài và có biên giống vết nứt.

---

## Tổng kết kiến thức theo chương
| Chương | Kiến thức | Bước |
|---|---|---|
| C2 | Lấy mẫu lại (INTER_AREA, láng giềng gần nhất), lượng tử hóa | 2.0, 3.1 |
| C3 | Lọc trung vị (khử nhiễu, ước lượng nền), biến đổi điểm (ảnh tỉ số) | 2.2, 2.3 |
| C4 | RGB → xám | 1, 2.1 |
| C5 | Làm trơn lược đồ, tam giác, Sobel, Otsu, Canny, độ đo | 3.1–3.3, 5 |
| C7 | Giãn, đóng, thành phần liên thông, diện tích, độ dãn, quyết định mức ảnh | 3.3, 4.1–4.4 |
| C9 | Nguồn dữ liệu, nhãn nhị phân, train/val/test, tránh rò rỉ | 1 |
