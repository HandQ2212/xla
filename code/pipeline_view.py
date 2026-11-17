"""Step visualization for the classical pipeline's actual intermediates."""
import streamlit as st

import pipeline as pl


def legacy_steps(stages, params):
    return [
        ("2.0+2.1 · Resize + grayscale", stages["gray"],
         f"Cạnh dài = {params['size'] or 'gốc'}; chuyển xám Y = 0.299R + 0.587G + 0.114B"),
        ("2.2 · Median filter", stages["den"],
         f"Kernel {params['median']}×{params['median']}: giảm nhiễu hạt sỏi"),
        ("2.3 · Ratio image R = 128·I/B", stages["ratio"],
         f"Background B = median {params['bg']}×{params['bg']}" if params.get("bg", 0) > 1 else "Đã bỏ qua bù sáng"),
        (f"3.2 · Vùng tối D (θ = {stages['theta']})", pl.u8(stages["D"]),
         f"Triangle threshold; histogram smoothing W = {params['hist_w']}"),
        ("3.3 · Vùng gần edge E", pl.u8(stages["E"]),
         f"Sobel → Otsu → dilation {params['dil']}×{params['dil']}" if params.get("use_edge", True) else "Đã bỏ qua điều kiện edge"),
        ("3.4 · Mask thô M", pl.u8(stages["M"]), "Giao vùng tối và vùng gần edge: D ∩ E"),
        ("4.1 · Closing", pl.u8(stages["C"]), f"Ellipse {params['close']}×{params['close']}"),
        ("4.2 · Area filter", pl.u8(stages["A"]), f"Bỏ vùng < {params['a_min']} px"),
        ("4.3 · Elongation filter", pl.u8(stages["F"]),
         f"Giữ vùng có L/S ≥ {params['e_min']}" if params.get("e_min", 0) > 0 else "Đã tắt lọc độ dãn"),
        ("4.4 · Kết quả", pl.u8(stages["Z"]),
         f"Area < {params['tmin']}% ảnh thì trả mask rỗng" if params.get("post", True) else "Bỏ hậu xử lý: kết quả = M"),
    ]


def render(stages, params):
    """Trình xem từng bước: kéo thanh trượt để đi qua pipeline, bên dưới là toàn cảnh."""
    steps = legacy_steps(stages, params)
    titles = [t for t, _, _ in steps]
    pick = st.select_slider("Kéo để đi qua từng bước của pipeline", options=titles,
                            value=titles[-1], key="step_slider")
    title, picture, note = steps[titles.index(pick)]
    big, info = st.columns([3, 1])
    big.image(pl.rgb(pl.small(picture, 1100)), width="stretch")
    i = titles.index(pick)
    info.html(f"""
      <div class="step-card">
        <div class="step-no">BƯỚC {i + 1}/{len(steps)}</div>
        <div class="step-title">{title}</div>
        <div class="step-note">{note}</div>
      </div>""")
    if not params.get("post", True):
        info.caption("Các khung 4.1–4.3 giữ nguyên M vì hậu xử lý đang tắt.")

    st.html('<div class="section-label">Toàn cảnh 10 bước</div>')
    for row in range(0, len(steps), 5):
        cols = st.columns(5)
        for col, (t, pic, _) in zip(cols, steps[row:row + 5]):
            col.image(pl.rgb(pl.small(pic, 420)), caption=t, width="stretch")
