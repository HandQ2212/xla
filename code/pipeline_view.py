"""Step visualization for the classical pipeline's actual intermediates."""
import streamlit as st

import pipeline as pl


def legacy_steps(stages, params):
    return [
        ("2.0+2.1 · Thu nhỏ + xám", stages["gray"],
         f"Cạnh dài = {params['size'] or 'gốc'}; chuyển xám Y = 0.299R + 0.587G + 0.114B"),
        ("2.2 · Lọc trung vị", stages["den"],
         f"Kernel {params['median']}×{params['median']}: giảm nhiễu hạt"),
        ("2.3 · Bù chiếu sáng R = 128·I/B", stages["ratio"],
         f"Nền B = trung vị {params['bg']}×{params['bg']}" if params.get("bg", 0) > 1 else "Đã bỏ qua bù sáng"),
        (f"3.2 · Vùng tối D (θ = {stages['theta']})", pl.u8(stages["D"]),
         f"Ngưỡng tam giác; lược đồ làm trơn W = {params['hist_w']}"),
        ("3.3 · Vùng biên E", pl.u8(stages["E"]),
         f"Sobel → Otsu → giãn {params['dil']}×{params['dil']}" if params.get("use_edge", True) else "Đã bỏ qua biên"),
        ("3.4 · Mặt nạ thô M", pl.u8(stages["M"]), "Giao vùng tối và vùng gần biên: D ∩ E"),
        ("4.1 · Phép đóng", pl.u8(stages["C"]), f"Ellipse {params['close']}×{params['close']}"),
        ("4.2 · Lọc diện tích", pl.u8(stages["A"]), f"Bỏ vùng < {params['a_min']} px"),
        ("4.3 · Lọc độ dãn", pl.u8(stages["F"]), f"Giữ L/S ≥ {params['e_min']}"),
        ("4.4 · Kết quả P", pl.u8(stages["Z"]),
         f"Ngưỡng diện tích {params['tmin']}%" if params.get("post", True) else "Bỏ hậu xử lý: Z = M"),
    ]


def render(stages, params):
    for title, picture, note in legacy_steps(stages, params):
        left, right = st.columns([1, 2])
        left.markdown(f"**{title}**")
        right.image(pl.rgb(pl.small(picture, 900)), use_container_width=True)
        right.caption(note)
        st.divider()
    if not params.get("post", True):
        st.caption("Các khung 4.1–4.3 giữ nguyên M vì hậu xử lý đang tắt.")
