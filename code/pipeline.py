# -*- coding: utf-8 -*-
"""Pipeline phát hiện vết nứt mặt đường (xử lý ảnh cổ điển) – bản module dùng chung.

Tách nguyên văn từ `crack_segmentation.ipynb` (bộ Pavement Crack Segmentation Dataset)
để notebook và app Streamlit import chung một nguồn: KHÔNG đổi logic nào.
Chi tiết từng bước và lý do chọn tham số: xem PIPELINE.md.

Dùng nhanh:
    import cv2, pipeline
    mask = pipeline.pipeline(img, pipeline.P)        # ảnh BGR -> mặt nạ bool kích thước gốc
    vis  = pipeline.overlay_red(img, mask)           # ảnh gốc phủ đỏ kết quả
"""
import numpy as np
import cv2

# Tham số ĐÃ CHỌN (khớp bảng "Bảng tham số" trong PIPELINE.md và best_params.json
# mà notebook sinh ra; notebook có assert kiểm tra lại khớp cấu hình chọn trên val).
P = dict(size=640, median=5, bg=71, hist_w=3, dil=3, close=7, a_min=30, e_min=3, tmin=0.3)

# Tham số khởi đầu trước khi tune (để so sánh trong app, giống P0 trong notebook)
P0 = dict(size=640, median=3, bg=31, hist_w=3, dil=7, close=7, a_min=60, e_min=2, tmin=0.2)


# ---- Tiện ích ảnh ----
def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

def u8(mask):
    return mask.astype(np.uint8) * 255

def ellipse(k):
    """Phần tử cấu trúc hình ellipse k×k (đẳng hướng: vết nứt theo hướng nào cũng được xử lý như nhau)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

def ncc(m):
    """Số thành phần liên thông 8-láng giềng."""
    return cv2.connectedComponents(m.astype(np.uint8), connectivity=8)[0] - 1

def resize_long(img, size, interp=cv2.INTER_AREA):
    h, w = img.shape[:2]; s = size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)

def to_orig(mask, shape):
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0

def small(img, size=640, nearest=False):
    """Thu nhỏ chỉ để HIỂN THỊ."""
    h, w = img.shape[:2]; s = size / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

def overlay_red(img, mask, alpha=0.6):
    out = img.copy()
    out[mask] = (alpha * np.array([0, 0, 255]) + (1 - alpha) * img[mask]).astype(np.uint8)
    return out

def error_map(pred, gt):
    """Xanh lá = TP, đỏ = FP (báo nhầm), xanh dương = FN (bỏ sót)."""
    m = np.zeros(gt.shape + (3,), np.uint8)
    m[pred & gt] = (0, 200, 0); m[pred & ~gt] = (0, 0, 255); m[~pred & gt] = (255, 0, 0)
    return m


# ---- Bước 2: tiền xử lý ----
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def denoise(gray, p):
    k = p["median"]
    if p.get("denoise", "median") == "gauss":
        return cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.medianBlur(gray, k) if k > 1 else gray

def illumination_ratio(img, k, mode="median", op="div"):
    bg = cv2.medianBlur(img, k) if mode == "median" else cv2.morphologyEx(img, cv2.MORPH_CLOSE, ellipse(k))
    I, B = img.astype(np.float32), bg.astype(np.float32)
    R = 128.0 + I - B if op == "sub" else 128.0 * I / (B + 1.0)
    return np.clip(R, 0, 255).astype(np.uint8), bg


# ---- Bước 3: xử lý chính ----
def histogram(img, w=1):
    """Lược đồ xám, làm trơn bằng trung bình trượt cửa sổ w (Ch5 tr.62); w = 1: không làm trơn."""
    h = np.bincount(img.ravel(), minlength=256).astype(np.float64)
    if w > 1:
        h = np.convolve(np.pad(h, w // 2, mode="edge"), np.ones(w) / w, mode="valid")
    return h

def triangle_threshold(img, w=3):
    """3.2 – Zack: Δ nối (b_min, h[b_min]) – (b_max, h[b_max]); θ = b có khoảng cách tới Δ lớn nhất.
    Đuôi đối tượng nằm ở phía TỐI (mức thấp) của đỉnh nền."""
    h = histogram(img, w)
    bmax = int(np.argmax(h)); bmin = int(np.nonzero(h)[0][0])
    if bmax == bmin:
        return bmax
    b = np.arange(bmin, bmax + 1)
    x1, y1, x2, y2 = bmin, h[bmin], bmax, h[bmax]
    d = np.abs((y2 - y1) * b - (x2 - x1) * h[b] + x2 * y1 - y2 * x1) / np.hypot(y2 - y1, x2 - x1)
    return int(b[np.argmax(d)])

def dark_region(ratio, w=3):
    t = triangle_threshold(ratio, w)
    return ratio < t, t

def sobel_l1(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.normalize(np.abs(gx) + np.abs(gy), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def edge_region(ratio, dil):
    g = sobel_l1(ratio)
    t, edge = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(edge, ellipse(dil)) > 0, g, edge > 0, t


# ---- Bước 4: hậu xử lý ----
def closing(mask, k):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(k)) > 0

def opening(mask, k):
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, ellipse(k)) > 0

def area_filter(mask, a_min):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = st[:, cv2.CC_STAT_AREA] >= a_min
    keep[0] = False
    return keep[lab]

def elongation(comp, rect="rotated"):
    """Trục dài / trục ngắn của hộp chữ nhật bao vùng: xoay nhỏ nhất (minAreaRect) hoặc thẳng (boundingRect)."""
    pts = cv2.findNonZero(comp.astype(np.uint8))
    if rect == "axis":
        _, _, w, h = cv2.boundingRect(pts)
    else:
        (_, _), (w, h), _ = cv2.minAreaRect(pts)
    return max(w, h) / max(min(w, h), 1.0)

def elongation_filter(mask, e_min, rect="rotated"):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h = st[i, :4]
        keep[i] = elongation(lab[y:y + h, x:x + w] == i, rect) >= e_min
    return keep[lab]

def image_decision(mask, tmin):
    s = mask.mean() * 100
    return (mask if s >= tmin else np.zeros_like(mask)), s


# ---- Toàn bộ pipeline. Các khóa tùy chọn (denoise, bg_mode, op, thresh, use_edge, morph, rect, post)
# ---- chỉ dùng cho các thí nghiệm kiểm chứng; mặc định là phương pháp đã chọn.
def raw_mask(img, p):
    """Bước 2 + 3: ảnh gốc -> mặt nạ thô M ở kích thước đã thu nhỏ (và các ảnh trung gian)."""
    small_ = resize_long(img, p["size"]) if p.get("size") else img
    gray = to_gray(small_)
    den = denoise(gray, p)
    ratio = (illumination_ratio(den, p["bg"], p.get("bg_mode", "median"), p.get("op", "div"))[0]
             if p.get("bg", 0) > 1 else den)
    if p.get("thresh", "triangle") == "otsu":
        theta, _ = cv2.threshold(ratio, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); D = ratio < theta
    else:
        D, theta = dark_region(ratio, p["hist_w"])
    M = D & edge_region(ratio, p["dil"])[0] if p.get("use_edge", True) else D
    return M, dict(small=small_, gray=gray, den=den, ratio=ratio, D=D, theta=theta)

def post(M, p):
    """Bước 4: đóng -> lọc diện tích -> lọc độ dãn -> quyết định mức ảnh."""
    morph = p.get("morph", "close")
    C = closing(M, p["close"]) if morph == "close" else opening(M, p["close"]) if morph == "open" else M
    A = area_filter(C, p["a_min"])
    F = elongation_filter(A, p["e_min"], p.get("rect", "rotated"))
    Z, s = image_decision(F, p["tmin"])
    return Z, C, A, F

def pipeline(img, p, steps=False):
    """Ảnh gốc -> mặt nạ nhị phân ở KÍCH THƯỚC GỐC."""
    M, st = raw_mask(img, p)
    if steps:
        Z, C, A, F = post(M, p)
        return dict(st, M=M, C=C, A=A, F=F, Z=Z)
    Z = post(M, p)[0] if p.get("post", True) else M
    return to_orig(Z, img.shape[:2])


# ---- Đánh giá & đối chứng ----
def scores(pred, gt):
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt)); fn = int(np.count_nonzero(~pred & gt))
    gpos, ppos = bool(gt.any()), bool(pred.any())
    return dict(tp=tp, fp=fp, fn=fn, gt_pos=gpos, pred_pos=ppos,
                dice=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0,
                iou=tp / (tp + fp + fn) if tp + fp + fn else 1.0,
                precision=(tp / (tp + fp) if tp + fp else 0.0) if gpos else np.nan,
                recall=tp / (tp + fn) if gpos else np.nan)

def baseline_otsu(img):
    _, m = cv2.threshold(to_gray(img), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return m > 0

def baseline_canny(img):
    return cv2.Canny(to_gray(img), 50, 150) > 0
