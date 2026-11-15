# -*- coding: utf-8 -*-
"""App Streamlit demo pipeline phát hiện vết nứt mặt đường.

Chạy:
    python -m streamlit run code/app.py

Mọi tính toán gọi qua pipeline.py.
"""
import os
from pathlib import Path

import numpy as np
import cv2
import streamlit as st

import pipeline as pl

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samples_pcs"

st.set_page_config(page_title="Phát hiện vết nứt mặt đường", page_icon="🛣️", layout="wide")


# ---------------- Tính toán (cache theo nội dung ảnh + tham số) ----------------

def load_image(data: bytes) -> np.ndarray:
    """Bytes file upload -> BGR ở kích thước gốc; pipeline tự thu nhỏ để xử lý."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Không đọc được file ảnh.")
    return img

@st.cache_data(show_spinner=False)
def run_pipeline(data: bytes, p_key: tuple):
    """p_key: tuple (k, v) đã sort – hashable để cache; bên trong khôi phục thành dict."""
    p = dict(p_key)
    img = load_image(data)
    st_ = pl.pipeline(img, p, steps=True)
    mask = pl.to_orig(st_["Z"], img.shape[:2])
    return img, mask, st_

@st.cache_data(show_spinner=False)
def run_baseline(data: bytes, kind: str):
    img = load_image(data)
    return pl.baseline_otsu(img) if kind == "otsu" else pl.baseline_canny(img)


# ---------------- Trạng thái ----------------
if "img_data" not in st.session_state:
    st.session_state.img_data = None
    st.session_state.img_name = ""
    st.session_state.is_sample = False

def set_image(data: bytes, name: str, is_sample=False):
    st.session_state.img_data = data
    st.session_state.img_name = name
    st.session_state.is_sample = is_sample

def find_gt(name: str):
    """Ảnh mẫu từ tập test có file GT cùng tên; ảnh upload thì không có."""
    if st.session_state.is_sample and name.endswith(".jpg"):
        cand = os.path.join(SAMPLE_DIR, name.replace(".jpg", "_gt.png"))
        if os.path.isfile(cand):
            gt = cv2.imread(cand, cv2.IMREAD_GRAYSCALE)
            return gt > 127
    return None


# ---------------- Thanh bên: chọn ảnh + tham số ----------------
with st.sidebar:
    st.header("🖼️ Ảnh đầu vào")
    up = st.file_uploader("Tải ảnh lên (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is not None and up.getvalue() != st.session_state.img_data:
        set_image(up.getvalue(), up.name)

    st.divider()
    st.caption("Hoặc chọn ảnh mẫu từ tập test:")
    for f in sorted(x for x in os.listdir(SAMPLE_DIR) if x.endswith(".jpg") and "_pred" not in x):
        if st.button(f, use_container_width=True, key=f"sample_{f}"):
            with open(os.path.join(SAMPLE_DIR, f), "rb") as fh:
                set_image(fh.read(), f, is_sample=True)

    st.divider()
    presets = {"P đã xác nhận": pl.P, "Tái tạo vùng (thử nghiệm)": pl.P_RECON, "Trung vị 3×3 (thử nghiệm)": pl.P_THIN}
    profile = st.selectbox("Phương án xử lý", list(presets))
    chosen = presets[profile]
    st.caption("Hai phương án thử nghiệm cần đánh giá trên train/val trước khi thay P.")
    st.header("⚙️ Bước xử lý (ablation)")
    st.caption("Bỏ tick một bước = chạy lại thí nghiệm \"bỏ từng bước\" của PIPELINE.md.")
    use_size = st.checkbox("2.0 Thu nhỏ về 640 px", value=True)
    use_bg = st.checkbox("2.3 Bù chiếu sáng", value=True)
    use_edge = st.checkbox("3.3 Vùng biên E (M = D ∩ E)", value=True)
    use_post = st.checkbox("4. Hậu xử lý (đóng, lọc)", value=True)
    use_tmin = st.checkbox("4.4 Quyết định mức ảnh (0.3%)", value=True)

    params = dict(chosen)
    if not use_size:
        params["size"] = None
    if not use_bg:
        params["bg"] = 0
    if not use_edge:
        params["use_edge"] = False
    if not use_post:
        params["post"] = False
    if use_post and not use_tmin:
        params["tmin"] = 0.0
    params_key = tuple(sorted(params.items()))          # hashable cho cache


# ---------------- Nội dung chính ----------------
st.title("🛣️ Phát hiện & phân đoạn vết nứt mặt đường")
st.caption("Xử lý ảnh cổ điển – không học sâu. Tham số đã chọn trên tập val "
           "P: test 231 ảnh Dice 0.339, IoU 0.274. Điểm lịch sử này chỉ áp dụng cho P nguyên bản.")
st.caption(f"Đang chạy: {profile}" + (" · đã thay đổi bước xử lý" if params != chosen else ""))

if st.session_state.img_data is None:
    st.info("⬅️ Tải một ảnh lên hoặc chọn ảnh mẫu ở thanh bên để bắt đầu.")
    st.stop()

with st.spinner("Đang chạy pipeline..."):
    img, mask, st_ = run_pipeline(st.session_state.img_data, params_key)

# Kết quả: 3 khung
c1, c2, c3 = st.columns(3)
c1.image(pl.rgb(img), caption=f"Ảnh gốc ({img.shape[1]}×{img.shape[0]})", use_container_width=True)
c2.image(pl.u8(mask), caption=f"Mặt nạ dự đoán ({int(mask.sum())} px nứt)", use_container_width=True, clamp=True)
c3.image(pl.rgb(pl.overlay_red(img, mask)), caption="Kết quả phủ đỏ", use_container_width=True)

if not mask.any():
    st.info("Không có vùng nào được giữ lại với cấu hình hiện tại; vết nứt mảnh có thể bị bỏ sót.")
else:
    st.info(f"🔴 Phát hiện **{pl.ncc(mask)} vùng** nghi vết nứt, "
            f"chiếm **{mask.mean()*100:.2f}%** diện tích ảnh.")

# Điểm số nếu có GT (ảnh mẫu từ tập test)
gt = find_gt(st.session_state.img_name)
if gt is not None:
    if gt.shape != mask.shape:
        gt = cv2.resize(gt.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    sc = pl.scores(mask, gt)
    st.subheader("📊 Đánh giá với Ground Truth")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Dice", f"{sc['dice']:.3f}")
    m2.metric("IoU", f"{sc['iou']:.3f}")
    m3.metric("Precision", f"{sc['precision']:.3f}" if sc["gt_pos"] else "–")
    m4.metric("Recall", f"{sc['recall']:.3f}" if sc["gt_pos"] else "–")
    m5.metric("GT", "Không nứt" if not sc["gt_pos"] else f"{int(gt.sum())} px nứt")
    st.image(pl.rgb(pl.error_map(mask, gt)),
             caption="Bản đồ lỗi: xanh lá = TP · đỏ = FP (báo nhầm) · xanh dương = FN (bỏ sót)",
             use_container_width=True)

# Từng bước biến đổi
with st.expander("🔬 Xem từng bước của pipeline", expanded=False):
    E = st_["E"]
    steps = [
        ("2.0+2.1 · Thu nhỏ + xám", st_["gray"],
         f"Cạnh dài = {params['size'] or 'gốc'}; chuyển xám Y = 0.299R + 0.587G + 0.114B"),
        ("2.2 · Lọc trung vị", st_["den"],
         f"Kernel {params['median']}×{params['median']}: xoá nhiễu hạt mà giữ bờ vết nứt"),
        ("2.3 · Bù chiếu sáng R = 128·I/B", st_["ratio"],
         f"Nền B = trung vị {params['bg']}×{params['bg']}" if use_bg else "Đã bỏ qua bù chiếu sáng"),
        (f"3.2 · Vùng tối D (θ = {st_['theta']})", pl.u8(st_["D"]),
         f"Ngưỡng tam giác (lược đồ làm trơn W = {params['hist_w']}): D = R < θ – đuôi tối của lược đồ"),
        ("3.3 · Vùng biên E (Sobel → Otsu → giãn)", pl.u8(E),
         f"Sobel → Otsu → giãn {params['dil']}×{params['dil']}" if use_edge else "Đã bỏ qua biên: E = toàn bộ ảnh"),
        ("3.4 · Mặt nạ thô M", pl.u8(st_["M"]),
         "Tái tạo từ D ∩ E trong D" if params.get("edge_mode") == "reconstruct" else "Giao vùng tối và vùng gần biên: D ∩ E"),
        ("4.1 · Phép đóng", pl.u8(st_["C"]),
         f"Ellipse {params['close']}×{params['close']}: nối các đoạn nứt bị đứt gãy"),
        ("4.2 · Lọc diện tích", pl.u8(st_["A"]),
         f"Bỏ thành phần liên thông < {params['a_min']} px"),
        ("4.3 · Lọc độ dãn", pl.u8(st_["F"]),
         f"Giữ vùng L/S ≥ {params['e_min']} (hộp chữ nhật xoay minAreaRect): bỏ đốm tròn, giữ vết dài"),
        ("4.4 · KẾT QUẢ Z", pl.u8(st_["Z"]),
         f"Ngưỡng diện tích {params['tmin']}%" if use_post else "Đã bỏ qua toàn bộ hậu xử lý; Z = M"),
    ]
    for title, im, note in steps:
        colA, colB = st.columns([1, 2])
        colA.markdown(f"**{title}**")
        colB.image(pl.rgb(pl.small(im, 900)), use_container_width=True)
        colB.caption(note)
        st.divider()
    if not use_post:
        st.caption("Các khung 4.1–4.3 giữ nguyên M vì hậu xử lý đang tắt.")

# So sánh với đối chứng
with st.expander("⚖️ So sánh với phương pháp đối chứng", expanded=False):
    b1, b2, b3 = st.columns(3)
    b1.image(pl.rgb(pl.overlay_red(img, run_baseline(st.session_state.img_data, "otsu"))),
             caption="Đối chứng: Otsu đơn lẻ", use_container_width=True)
    b2.image(pl.rgb(pl.overlay_red(img, run_baseline(st.session_state.img_data, "canny"))),
             caption="Đối chứng: Canny đơn lẻ", use_container_width=True)
    b3.image(pl.rgb(pl.overlay_red(img, mask)), caption=f"Pipeline: {profile}", use_container_width=True)
    st.caption("Kết quả lịch sử trên test 231 ảnh, P nguyên bản: Otsu Dice 0.024 · Canny 0.018 · P 0.339. "
               "Otsu/Canny không có bước quyết định mức ảnh nên luôn báo nứt, cả trên ảnh lành.")

st.divider()
st.caption("Đồ án Xử lý ảnh – PTIT · Pipeline & tham số: `PIPELINE.md` · Code gốc: `crack_segmentation.ipynb` · "
           "Module dùng chung: `pipeline.py`. Ảnh ngoài dataset không có chuẩn đúng nên không hiện điểm số.")
