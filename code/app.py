# -*- coding: utf-8 -*-
"""App Streamlit demo pipeline phát hiện vết nứt mặt đường.

Chạy (từ thư mục gốc repo để nhận theme trong .streamlit/config.toml):
    python -m streamlit run code/app.py

Mọi tính toán gọi qua pipeline.py; app chỉ hiển thị.
"""
import base64
import os
import time
from pathlib import Path

import numpy as np
import cv2
import streamlit as st
from skimage.morphology import skeletonize

import pipeline as pl
import pipeline_view

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "samples_pcs"

st.set_page_config(page_title="Phát hiện vết nứt mặt đường", page_icon="🛣️", layout="wide")

# ---------------- Giao diện ----------------
st.html("""
<style>
:root { --asphalt:#14171A; --panel:#1E2226; --line:#2C3238; --ink:#ECEAE4; --soft:#A7ADB4;
        --yellow:#F2B824; --red:#FF453A; --green:#30D158; }
.block-container { padding-top: 1.6rem; max-width: 1400px; }
.hero { position:relative; overflow:hidden; border-radius:20px; padding:34px 38px 30px;
        background:
          radial-gradient(1200px 300px at 85% -40%, rgba(242,184,36,.22), transparent 60%),
          radial-gradient(600px 260px at 0% 120%, rgba(255,69,58,.16), transparent 60%),
          linear-gradient(135deg, #1B1F23 0%, #111417 100%);
        border:1px solid var(--line); margin-bottom: 18px; }
.hero:before { content:""; position:absolute; inset:0; opacity:.07; pointer-events:none;
        background-image: radial-gradient(#fff 1px, transparent 1px); background-size: 6px 6px; }
.hero .eyebrow { font: 600 13px/1.4 ui-monospace, Menlo, monospace; letter-spacing:3px;
        color:var(--yellow); text-transform:uppercase; }
.hero h1 { margin:8px 0 6px; font-size:44px; line-height:1.1; font-weight:800; letter-spacing:-1px; color:var(--ink); }
.hero h1 .crack { background: linear-gradient(90deg, var(--yellow), var(--red)); -webkit-background-clip:text;
        background-clip:text; color:transparent; }
.hero .sub { color:var(--soft); font-size:17px; margin-bottom:22px; }
.hero .stats { display:flex; gap:14px; flex-wrap:wrap; }
.hero .stat { background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:14px; padding:12px 18px; min-width:150px; }
.hero .stat b { display:block; font-size:28px; color:var(--ink); font-weight:800; }
.hero .stat span { color:var(--soft); font-size:13px; }
.flow { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:6px 0 4px; }
.flow .chip { background:var(--panel); border:1px solid var(--line); border-radius:999px; padding:7px 14px;
        color:var(--ink); font-size:14px; white-space:nowrap; }
.flow .chip.hot { border-color:var(--yellow); color:var(--yellow); }
.flow .arr { color:var(--soft); }
.kpis { display:grid; grid-template-columns: 1.6fr repeat(4, 1fr); gap:12px; margin: 4px 0 14px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px 18px; }
.kpi .lab { color:var(--soft); font-size:13px; text-transform:uppercase; letter-spacing:1px; }
.kpi .val { color:var(--ink); font-size:30px; font-weight:800; margin-top:4px; line-height:1.15; }
.kpi .hint { color:var(--soft); font-size:12.5px; margin-top:2px; }
.kpi.verdict.yes { border-color:rgba(255,69,58,.6); background:linear-gradient(135deg, rgba(255,69,58,.18), var(--panel)); }
.kpi.verdict.no  { border-color:rgba(48,209,88,.5); background:linear-gradient(135deg, rgba(48,209,88,.14), var(--panel)); }
.kpi.verdict .val { font-size:22px; white-space:nowrap; }
.kpi.verdict.yes .val { color:var(--red); } .kpi.verdict.no .val { color:var(--green); }
.cards { display:grid; grid-template-columns: repeat(3, 1fr); gap:14px; margin-top: 14px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px 20px; color:var(--soft); }
.card h4 { color:var(--ink); margin:0 0 6px; font-size:17px; }
.step-card { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px; }
.step-no { font: 600 12px ui-monospace, Menlo, monospace; letter-spacing:2px; color:var(--yellow); }
.step-title { color:var(--ink); font-size:20px; font-weight:700; margin:6px 0 8px; }
.step-note { color:var(--soft); font-size:14.5px; line-height:1.5; }
.section-label { font: 600 12px ui-monospace, Menlo, monospace; letter-spacing:2px; color:var(--yellow);
        text-transform:uppercase; margin: 18px 0 6px; }
.thumb { width:100%; aspect-ratio: 16/10; object-fit:cover; border-radius:10px; border:2px solid transparent; display:block; }
.thumb.on { border-color: var(--yellow); }
.legend { display:flex; gap:18px; flex-wrap:wrap; color:var(--soft); font-size:14px; margin:6px 0 2px; }
.legend i { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.bars { display:flex; flex-direction:column; gap:10px; margin-top:6px; }
.bar-row { display:grid; grid-template-columns: 170px 1fr 70px; gap:12px; align-items:center; color:var(--ink); font-size:15px; }
.bar-track { background:#2A3036; border-radius:6px; height:16px; overflow:hidden; }
.bar-fill { height:100%; background:#6B747C; border-radius:6px; }
.bar-fill.hl { background: linear-gradient(90deg, var(--yellow), #FFD866); }
.foot { color:var(--soft); font-size:13px; text-align:center; margin-top:26px; }
.st-key-input_panel { background: linear-gradient(180deg, #1B1F23, #16191C); border:1px solid var(--line);
        border-radius:20px; padding:22px 24px 18px; }
.panel-title { display:flex; align-items:center; gap:10px; color:var(--ink); font-size:19px; font-weight:700; }
.panel-title .ico { width:24px; height:24px; flex:none; }
.panel-sub { color:var(--soft); font-size:14px; margin:3px 0 10px 34px; }
[data-testid="stFileUploaderDropzone"] { display:flex; flex-direction:column; align-items:center; justify-content:center;
        gap:10px; min-height:236px; padding:22px; border:2px dashed rgba(242,184,36,.5); border-radius:18px;
        background: radial-gradient(420px 180px at 50% 0%, rgba(242,184,36,.10), transparent 70%), #191D21;
        transition: border-color .2s, background-color .2s, transform .2s; cursor:pointer; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--yellow); border-style: solid; background-color:#1E2328; }
[data-testid="stFileUploaderDropzone"]::before { content:""; order:0; width:64px; height:64px; border-radius:18px;
        background: rgba(242,184,36,.12) center / 34px 34px no-repeat url("data:image/svg+xml;utf8,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23F2B824' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M16 5h6'/%3E%3Cpath d='M19 2v6'/%3E%3Cpath d='M21 11.5V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7.5'/%3E%3Cpath d='m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21'/%3E%3Ccircle cx='9' cy='9' r='2'/%3E%3C/svg%3E"); }
[data-testid="stFileUploaderDropzoneInstructions"] { order:1; text-align:center; margin:0; display:flex !important;
        flex-direction:column !important; align-items:center; }
[data-testid="stFileUploaderDropzoneInstructions"] span { display:none; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content:"Kéo thả ảnh mặt đường vào đây"; display:block;
        color:var(--ink); font-size:18px; font-weight:700; }
[data-testid="stFileUploaderDropzoneInstructions"]::after { content:"JPG hoặc PNG · tối đa 200 MB"; display:block;
        color:var(--soft); font-size:13px; margin-top:4px; }
[data-testid="stFileUploaderDropzone"] > span { order:2; }
[data-testid="stFileUploaderDropzone"] button { background:var(--yellow); color:#14171A; border:none; border-radius:999px;
        padding:8px 20px; font-weight:700; }
[data-testid="stFileUploaderDropzone"] button:hover { background:#FFCB45; color:#14171A; }
[data-testid="stFileUploaderDropzone"] button p { display:none; }
[data-testid="stFileUploaderDropzone"] button > div::after { content:"Chọn ảnh từ máy"; font-weight:700; }
[class*="st-key-sample_"] button { height:104px; width:100%; background-color:#22272C; background-size:cover;
        background-position:center; border:2px solid #2C3238 !important; border-radius:14px; display:flex;
        align-items:flex-end; justify-content:flex-start; padding:8px 10px; position:relative; overflow:hidden;
        transition: transform .15s ease, border-color .15s, box-shadow .15s; }
[class*="st-key-sample_"] button:hover { transform: translateY(-3px); border-color: var(--yellow) !important;
        box-shadow: 0 10px 24px rgba(0,0,0,.45); }
[class*="st-key-sample_"] button p { color:#fff; font: 600 13px ui-monospace, Menlo, monospace;
        text-shadow: 0 1px 3px rgba(0,0,0,.9); }
.st-key-change_panel [data-testid="stExpander"] details { border-radius:16px; border-color: var(--line); }
.verdict .ico { width:22px; height:22px; vertical-align:-4px; margin-right:6px; }
@media (max-width: 900px) { .kpis { grid-template-columns: repeat(2, 1fr); } .cards { grid-template-columns: 1fr; }
        .hero h1 { font-size:32px; } }
</style>
""")


# ---------------- Tính toán (cache theo nội dung ảnh + tham số) ----------------

@st.cache_data(show_spinner=False, max_entries=4)
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
    t0 = time.perf_counter()
    st_ = pl.pipeline(img, p, steps=True)
    mask = pl.to_orig(st_["Z"], img.shape[:2])
    ms = (time.perf_counter() - t0) * 1000
    return img, mask, st_, ms

@st.cache_data(show_spinner=False)
def run_baseline(data: bytes, kind: str):
    img = load_image(data)
    return pl.baseline_otsu(img) if kind == "otsu" else pl.baseline_canny(img)

@st.cache_data(show_spinner=False)
def crack_length(mask: np.ndarray) -> int:
    """Chiều dài ước tính = số pixel của khung xương (skeleton) mask, ở kích thước gốc."""
    return int(skeletonize(mask).sum()) if mask.any() else 0

def b64_jpeg(img_bgr: np.ndarray, quality=88) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()

def png_bytes(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()

def display_pair(img: np.ndarray, mask: np.ndarray, max_w=1400):
    """Ảnh gốc và ảnh 'kết quả' thu nhỏ CHỈ để hiển thị. Nét nứt mảnh được giữ bằng cách
    coi mọi pixel hiển thị có phủ một phần mask là nứt (INTER_AREA > 0), không làm dày mask gốc."""
    h, w = img.shape[:2]
    s = min(1.0, max_w / w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_AREA) > 0
    after = (small * 0.45).astype(np.uint8)
    after[m] = (58, 69, 255)                         # BGR của #FF453A
    return small, after

@st.cache_resource(show_spinner=False)
def sample_thumbs():
    out = {}
    for f in sorted(x for x in os.listdir(SAMPLE_DIR) if x.endswith(".jpg") and "_pred" not in x):
        im = cv2.imread(str(SAMPLE_DIR / f))
        out[f] = b64_jpeg(pl.small(im, 260), 80)
    return out


def compare_slider(before_b64: str, after_b64: str, w: int, h: int, height=600):
    """Thanh kéo so sánh trước/sau (iframe tĩnh, không gửi dữ liệu ra ngoài)."""
    st.iframe(f"""
<!doctype html><html><head><style>
  html,body {{ margin:0; background:transparent; }}
  .wrap {{ display:flex; justify-content:center; padding-bottom:2px; }}
  .cmp {{ position:relative; width:min(100%, {height * w / h:.0f}px); aspect-ratio:{w}/{h};
          border-radius:16px; overflow:hidden; border:1px solid #2C3238; user-select:none; }}
  .cmp img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:cover; display:block; }}
  .top {{ position:absolute; inset:0; clip-path: inset(0 50% 0 0); }}
  .handle {{ position:absolute; top:0; bottom:0; left:50%; width:3px; margin-left:-1.5px;
             background:#F2B824; box-shadow:0 0 18px rgba(242,184,36,.8); pointer-events:none; }}
  .knob {{ position:absolute; top:50%; left:50%; width:44px; height:44px; margin:-22px 0 0 -22px; border-radius:50%;
           background:#F2B824; color:#14171A; font:700 18px/44px system-ui, sans-serif; text-align:center;
           box-shadow:0 4px 16px rgba(0,0,0,.5); }}
  .lbl {{ position:absolute; top:14px; padding:6px 12px; border-radius:999px; font:600 13px system-ui, sans-serif;
          background:rgba(20,23,26,.75); color:#ECEAE4; backdrop-filter: blur(6px); pointer-events:none; }}
  .lbl.l {{ left:14px; }} .lbl.r {{ right:14px; color:#FF6B61; }}
  input {{ position:absolute; inset:0; width:100%; height:100%; margin:0; opacity:0; cursor:ew-resize; }}
</style></head><body><div class="wrap"><div class="cmp" id="c">
  <img src="data:image/jpeg;base64,{after_b64}" alt="Kết quả">
  <div class="top" id="t"><img src="data:image/jpeg;base64,{before_b64}" alt="Ảnh gốc"></div>
  <div class="handle" id="h"><div class="knob">⟷</div></div>
  <span class="lbl l">Ảnh gốc</span><span class="lbl r">Vùng nghi nứt</span>
  <input type="range" id="r" min="0" max="100" step="0.1" value="50" aria-label="Kéo để so sánh">
</div></div>
<script>
  const r = document.getElementById('r'), t = document.getElementById('t'), h = document.getElementById('h');
  const set = v => {{ t.style.clipPath = 'inset(0 ' + (100 - v) + '% 0 0)'; h.style.left = v + '%'; }};
  r.addEventListener('input', e => {{ stop = true; set(+e.target.value); }});
  let stop = false; const t0 = performance.now();
  const seq = [[0, 50, 92], [700, 92, 8], [1500, 8, 50], [2100, 50, 50]];
  (function anim(now) {{
    if (stop) return; const dt = now - t0;
    for (let i = 0; i < seq.length - 1; i++) {{
      const [a, from, to] = seq[i], b = seq[i + 1][0];
      if (dt >= a && dt < b) {{ const k = (dt - a) / (b - a), e = k < .5 ? 2*k*k : 1 - Math.pow(-2*k + 2, 2) / 2;
        set(from + (to - from) * e); r.value = from + (to - from) * e; break; }}
    }}
    if (dt < seq[seq.length - 1][0]) requestAnimationFrame(anim);
  }})(t0);
</script></body></html>""", height="content")


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


# ---------------- Khu vực chọn ảnh ----------------
ICON_UP = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
           'stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/>'
           '<line x1="12" y1="3" x2="12" y2="15"/></svg>')
ICON_IMGS = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
             'stroke-linejoin="round"><path d="M18 22H4a2 2 0 0 1-2-2V6"/><path d="m22 13-1.296-1.296a2.41 2.41 0 0 0-3.408 0L11 18"/>'
             '<circle cx="12" cy="8" r="2"/><rect width="16" height="16" x="6" y="2" rx="2"/></svg>')
ICON_ALERT = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
              'stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
              '<path d="M12 9v4"/><path d="M12 17h.01"/></svg>')
ICON_OK = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" '
           'stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>')


def icon(svg: str, color: str, cls="ico") -> str:
    data = base64.b64encode(svg.replace("currentColor", color).replace("<svg ", "<svg xmlns='http://www.w3.org/2000/svg' ", 1)
                            .encode()).decode()
    return f'<img class="{cls}" alt="" src="data:image/svg+xml;base64,{data}">'


def on_upload():
    f = st.session_state.get("uploader")
    if f is not None:
        set_image(f.getvalue(), f.name)


def pick_sample(f):
    with open(SAMPLE_DIR / f, "rb") as fh:
        set_image(fh.read(), f, is_sample=True)


def render_input():
    """Ô kéo thả ảnh + thư viện ảnh mẫu dạng thẻ ảnh bấm được."""
    thumbs = sample_thumbs()
    css = "".join(
        f'.st-key-sample_{f.replace(".", "-")} button {{ background-image: linear-gradient(to top, '
        f'rgba(10,12,14,.92), rgba(10,12,14,0) 62%), url(data:image/jpeg;base64,{b64}); }}'
        for f, b64 in thumbs.items())
    cur = st.session_state.img_name if st.session_state.is_sample else None
    if cur in thumbs:
        k = f'.st-key-sample_{cur.replace(".", "-")} button'
        css += (f'{k} {{ border-color: var(--yellow) !important; box-shadow: 0 0 0 3px rgba(242,184,36,.28); }}'
                f'{k}::after {{ content:"Đang dùng"; position:absolute; top:7px; right:7px; padding:2px 9px; '
                f'border-radius:999px; background:var(--yellow); color:#14171A; font:700 11px system-ui, sans-serif; white-space:nowrap; }}')
    st.html(f"<style>{css}</style>")
    with st.container(key="input_panel"):
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.html(f'<div class="panel-title">{icon(ICON_UP, "#F2B824")}Tải ảnh của bạn</div>'
                    '<div class="panel-sub">Ảnh mặt đường, chụp càng thẳng từ trên xuống càng tốt.</div>')
            st.file_uploader("Tải ảnh mặt đường lên", type=["jpg", "jpeg", "png"], key="uploader",
                             on_change=on_upload, label_visibility="collapsed")
        with right:
            st.html(f'<div class="panel-title">{icon(ICON_IMGS, "#F2B824")}Hoặc thử ngay một ảnh mẫu</div>'
                    '<div class="panel-sub">10 ảnh từ tập test, có ground truth để chấm điểm tự động.</div>')
            files = list(thumbs)
            for row in range(0, len(files), 5):
                for col, f in zip(st.columns(5, gap="small"), files[row:row + 5]):
                    col.button(f.replace(".jpg", ""), key=f"sample_{f}", width="stretch",
                               on_click=pick_sample, args=(f,))


# ---------------- Thanh bên: cấu hình ----------------
with st.sidebar:
    st.markdown("### :material/tune: Bật / tắt từng bước")
    st.caption("Bỏ tick một bước để xem pipeline thay đổi ra sao (ablation).")
    use_size = st.checkbox("2.0 Resize về 640 px", value=True)
    use_bg = st.checkbox("2.3 Bù chiếu sáng (ratio image)", value=True)
    use_edge = st.checkbox("3.3 Điều kiện edge (M = D ∩ E)", value=True)
    use_post = st.checkbox("4. Hậu xử lý", value=True)
    use_elong = st.checkbox("4.3 Elongation filter (L/S ≥ 3)", value=True, disabled=not use_post)
    use_tmin = st.checkbox("4.4 Image-level decision (0.3%)", value=True, disabled=not use_post)

    params = dict(pl.P)
    if not use_size:
        params["size"] = None
    if not use_bg:
        params["bg"] = 0
    if not use_edge:
        params["use_edge"] = False
    if not use_post:
        params["post"] = False
    if use_post and not use_elong:
        params["e_min"] = 0
    if use_post and not use_tmin:
        params["tmin"] = 0.0
    params_key = tuple(sorted(params.items()))          # hashable cho cache
    is_default = params == dict(pl.P)

    with st.expander("Tham số đang dùng", icon=":material/data_table:"):
        st.markdown("\n".join([
            "| Tham số | Giá trị |", "|---|---|",
            f"| Resize | {params['size'] or 'gốc'} px |",
            f"| Median | {params['median']}×{params['median']} |",
            f"| Background kernel | {params['bg'] or 'tắt'} |",
            f"| Histogram W | {params['hist_w']} |",
            f"| Edge dilation | {params['dil']}×{params['dil']} |",
            f"| Closing | {params['close']}×{params['close']} |",
            f"| Min area | {params['a_min']} px |",
            f"| Min L/S | {params['e_min'] or 'tắt'} |",
            f"| Image-level | {params['tmin']}% |",
        ]))


# ---------------- Banner ----------------
st.html("""
<div class="hero">
  <div class="eyebrow">Đồ án Xử lý ảnh · PTIT · Nhóm 06</div>
  <h1>Phát hiện &amp; phân đoạn <span class="crack">vết nứt mặt đường</span></h1>
  <div class="sub">Pipeline xử lý ảnh cổ điển, không dùng học sâu: bù chiếu sáng, triangle threshold, Sobel và hình thái học.</div>
  <div class="stats">
    <div class="stat"><b>0.339</b><span>Dice trên 231 ảnh test</span></div>
    <div class="stat"><b>×14</b><span>so với baseline Otsu (0.024)</span></div>
    <div class="stat"><b>22/45</b><span>ảnh không nứt nhận đúng</span></div>
    <div class="stat"><b>4 320</b><span>cấu hình trong grid search</span></div>
  </div>
</div>
""")

if st.session_state.img_data is None:
    render_input()
    st.html("""
    <div class="section-label">Pipeline</div>
    <div class="flow">
      <span class="chip">Ảnh RGB</span><span class="arr">→</span>
      <span class="chip">Resize 640</span><span class="arr">→</span>
      <span class="chip">Grayscale</span><span class="arr">→</span>
      <span class="chip">Median 5×5</span><span class="arr">→</span>
      <span class="chip hot">R = 128·I / B</span><span class="arr">→</span>
      <span class="chip hot">Triangle θ ∩ Sobel edge</span><span class="arr">→</span>
      <span class="chip">Closing</span><span class="arr">→</span>
      <span class="chip">Area + elongation</span><span class="arr">→</span>
      <span class="chip hot">Image-level 0.3%</span><span class="arr">→</span>
      <span class="chip">Binary mask</span>
    </div>
    <div class="cards">
      <div class="card"><h4>① Chọn ảnh</h4>Kéo thả ảnh của bạn vào ô bên trên, hoặc bấm một ảnh mẫu (có ground truth để chấm điểm).</div>
      <div class="card"><h4>② Xem kết quả</h4>Kéo thanh so sánh trước/sau, xem từng bước trong 10 bước của pipeline và so với Otsu, Canny.</div>
      <div class="card"><h4>③ Thử ablation</h4>Tắt từng bước ở thanh bên để thấy vì sao bước đó cần thiết, ví dụ tắt bù chiếu sáng hoặc elongation filter.</div>
    </div>
    """)
    st.stop()


with st.container(key="change_panel"):
    with st.expander(f"Ảnh đang dùng: **{st.session_state.img_name}** · bấm để đổi ảnh khác",
                     icon=":material/photo_camera:"):
        render_input()

# ---------------- Chạy pipeline ----------------
with st.spinner("Đang chạy pipeline..."):
    img, mask, st_, ms = run_pipeline(st.session_state.img_data, params_key)
gt = find_gt(st.session_state.img_name)
if gt is not None and gt.shape != mask.shape:
    gt = cv2.resize(gt.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

n_regions = pl.ncc(mask)
area_pct = mask.mean() * 100
length = crack_length(mask)
found = bool(mask.any())
fmt = lambda n: f"{n:,}".replace(",", " ")

st.html(f"""
<div class="kpis">
  <div class="kpi verdict {'yes' if found else 'no'}">
    <div class="lab">Kết luận</div>
    <div class="val">{(icon(ICON_ALERT, '#FF453A') + 'Có vùng nghi nứt') if found else (icon(ICON_OK, '#30D158') + 'Không phát hiện')}</div>
    <div class="hint">{st.session_state.img_name} · {img.shape[1]}×{img.shape[0]}</div>
  </div>
  <div class="kpi"><div class="lab">Số vùng</div><div class="val">{n_regions}</div><div class="hint">thành phần liên thông</div></div>
  <div class="kpi"><div class="lab">Diện tích</div><div class="val">{area_pct:.2f}%</div><div class="hint">{fmt(int(mask.sum()))} px trên ảnh gốc</div></div>
  <div class="kpi"><div class="lab">Chiều dài ước tính</div><div class="val">{fmt(length)} px</div><div class="hint">theo skeleton của mask</div></div>
  <div class="kpi"><div class="lab">Thời gian</div><div class="val">{ms:.0f} ms</div><div class="hint">chạy pipeline trên máy này</div></div>
</div>
""")
if not is_default:
    st.warning("Đang chạy cấu hình đã tắt bớt bước. Các con số trên banner chỉ đúng với cấu hình đầy đủ.")

T_CMP, T_RES, T_GT, T_STEP, T_BASE = (":material/compare: Trước / sau", ":material/grid_view: Kết quả",
                                    ":material/fact_check: Chấm điểm với GT", ":material/account_tree: Từng bước",
                                    ":material/balance: So với baseline")
tab_names = [T_CMP, T_RES, T_STEP, T_BASE]
if gt is not None:
    tab_names.insert(2, T_GT)
tabs = dict(zip(tab_names, st.tabs(tab_names)))

with tabs[T_CMP]:
    before, after = display_pair(img, mask)
    compare_slider(b64_jpeg(before), b64_jpeg(after), before.shape[1], before.shape[0])
    st.caption("Kéo thanh vàng để so sánh. Ảnh được thu nhỏ để hiển thị; mask gốc vẫn ở độ phân giải đầy đủ.")
    d1, d2, _ = st.columns([1, 1, 2])
    stem = Path(st.session_state.img_name).stem or "image"
    d1.download_button("Tải mask (PNG)", png_bytes(pl.u8(mask)), f"{stem}_mask.png", "image/png", width="stretch",
                       icon=":material/download:")
    d2.download_button("Tải ảnh phủ đỏ (PNG)", png_bytes(pl.overlay_red(img, mask)), f"{stem}_overlay.png",
                       "image/png", width="stretch", icon=":material/download:")

with tabs[T_RES]:
    c1, c2, c3 = st.columns(3)
    c1.image(pl.rgb(img), caption=f"Ảnh gốc ({img.shape[1]}×{img.shape[0]})", width="stretch")
    c2.image(pl.u8(mask), caption=f"Mask dự đoán ({int(mask.sum())} px nứt)", width="stretch", clamp=True)
    c3.image(pl.rgb(pl.overlay_red(img, mask)), caption="Kết quả phủ đỏ", width="stretch")
    if not found:
        st.info("Không có vùng nào được giữ lại với cấu hình hiện tại; vết nứt mảnh có thể bị bỏ sót.")
    st.caption("Vệt trám, khe nối, vạch sơn và bóng đổ có thể bị nhận nhầm. Kết quả là vùng nghi nứt để kiểm tra lại.")

if gt is not None:
    with tabs[T_GT]:
        sc = pl.scores(mask, gt)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Dice", f"{sc['dice']:.3f}")
        m2.metric("IoU", f"{sc['iou']:.3f}")
        m3.metric("Precision", f"{sc['precision']:.3f}" if sc["gt_pos"] else "–")
        m4.metric("Recall", f"{sc['recall']:.3f}" if sc["gt_pos"] else "–")
        m5.metric("GT", "Không nứt" if not sc["gt_pos"] else f"{int(gt.sum())} px nứt")
        st.html("""<div class="legend"><span><i style="background:#00C800"></i>True positive</span>
                <span><i style="background:#FF0000"></i>False positive (báo nhầm)</span>
                <span><i style="background:#0000FF"></i>False negative (bỏ sót)</span></div>""")
        g1, g2 = st.columns(2)
        g1.image(pl.rgb(pl.overlay_red(img, gt)), caption="Ground truth phủ đỏ", width="stretch")
        g2.image(pl.rgb(pl.error_map(mask, gt)), caption="Error map của pipeline", width="stretch")

with tabs[T_STEP]:
    pipeline_view.render(st_, params)

with tabs[T_BASE]:
    otsu = run_baseline(st.session_state.img_data, "otsu")
    canny = run_baseline(st.session_state.img_data, "canny")
    b1, b2, b3 = st.columns(3)
    b1.image(pl.rgb(pl.overlay_red(img, otsu)), caption="Baseline: Otsu", width="stretch")
    b2.image(pl.rgb(pl.overlay_red(img, canny)), caption="Baseline: Canny", width="stretch")
    b3.image(pl.rgb(pl.overlay_red(img, mask)), caption="Pipeline của nhóm", width="stretch")
    if gt is not None:
        rows = [("Otsu", pl.scores(otsu, gt)["dice"], False), ("Canny", pl.scores(canny, gt)["dice"], False),
                ("Pipeline của nhóm", pl.scores(mask, gt)["dice"], True)]
        title = "Dice trên ảnh này"
    else:
        rows = [("Otsu", 0.024, False), ("Canny", 0.018, False), ("Pipeline của nhóm", 0.339, True)]
        title = "Dice trung bình trên 231 ảnh test (cấu hình đầy đủ)"
    top = max(0.4, max(v for _, v, _ in rows))
    bars = "".join(f'<div class="bar-row"><span>{n}</span><div class="bar-track"><div class="bar-fill{" hl" if hl else ""}" '
                   f'style="width:{v / top * 100:.1f}%"></div></div><b>{v:.3f}</b></div>' for n, v, hl in rows)
    st.html(f'<div class="section-label">{title}</div><div class="bars">{bars}</div>')
    st.caption("Otsu và Canny không có bước quyết định mức ảnh nên luôn báo nứt, kể cả trên ảnh lành.")

st.html('<div class="foot">Đồ án Xử lý ảnh – PTIT · Nhóm 06 · Thuật toán: <code>code/pipeline.py</code> · '
        'Ảnh ngoài dataset không có ground truth nên không hiện điểm số.</div>')
