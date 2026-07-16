"""
Sistem Rekomendasi Destinasi Wisata Indonesia
Content-Based Filtering · TF-IDF + Cosine Similarity
"""

import difflib
import html
import math
import re
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import st_folium

# ─────────────────────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH_CANDIDATES = [
    BASE_DIR / "data" / "DATASET_WISATA_READY_MODELING_FINAL.csv",
    BASE_DIR / "DATASET_WISATA_READY_MODELING_FINAL.csv",
]

PLACEHOLDER_IMG = "https://placehold.co/640x400/e8f5e9/388e3c?text=Gambar+Tidak+Tersedia"

CATEGORY_EMOJI = {
    "Budaya": "🏛️",
    "Taman Hiburan": "🎡",
    "Cagar Alam": "🌿",
    "Bahari": "🌊",
    "Pusat Perbelanjaan": "🛍️",
    "Tempat Ibadah": "🕌",
}

CITY_COORDS = {
    "Jakarta": (-6.2, 106.816),
    "Bandung": (-6.917, 107.619),
    "Yogyakarta": (-7.795, 110.369),
    "Semarang": (-6.966, 110.416),
    "Surabaya": (-7.257, 112.752),
    "Bogor": (-6.597, 106.806),
    "Malang": (-7.966, 112.632),
    "Surakarta": (-7.575, 110.824),
}

st.set_page_config(
    page_title="WisataJawa · Rekomendasi Wisata",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Lora:ital,wght@0,500;0,600;1,500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #f1f5f1 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1360px !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: #0d2818 !important; }
[data-testid="stSidebar"] section > div { padding-top: 1rem; }
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stTextInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p { color: #81c784 !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #1b3d22 !important; border-color: #2e5e35 !important;
    border-radius: 10px !important; color: #e8f5e9 !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #2e7d32, #43a047) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; font-weight: 700 !important;
    font-size: 0.92rem !important; padding: 0.7rem 1rem !important;
    width: 100% !important; transition: opacity .2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover { opacity: .88 !important; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d2818 0%, #1b5e20 55%, #2e7d32 100%);
    border-radius: 22px; padding: 2.6rem 3rem 2.4rem;
    margin-bottom: 1.6rem; overflow: hidden; position: relative;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:280px; height:280px; border-radius:50%;
    background:rgba(255,255,255,.04);
}
.hero-title {
    font-family:'Lora',serif; font-size:2.4rem; font-weight:600;
    color:#f1f8e9; line-height:1.18; position:relative;
}
.hero-title em { color:#a5d6a7; font-style:italic; }
.hero-sub { color:#a5d6a7; font-size:.96rem; margin-top:.5rem; position:relative; }
.hero-pills { display:flex; flex-wrap:wrap; gap:.6rem; margin-top:1.6rem; position:relative; }
.hero-pill {
    background:rgba(255,255,255,.1); color:#c8e6c9;
    border-radius:999px; padding:5px 14px; font-size:.77rem; font-weight:600;
}

/* Section labels */
.sec-tag { font-size:.67rem; font-weight:700; letter-spacing:.12em;
           text-transform:uppercase; color:#2e7d32; margin-bottom:4px; }
.sec-h { font-family:'Lora',serif; font-size:1.42rem; font-weight:600;
         color:#1b2e1c; margin-bottom:.9rem; }

/* Search box */
.search-wrap {
    background:#fff; border:1.5px solid #c8e6c9; border-radius:16px;
    padding:1.2rem 1.4rem; margin-bottom:1.4rem;
    box-shadow:0 2px 12px rgba(0,0,0,.05);
}

/* Cards */
.card {
    background:#fff; border-radius:18px; overflow:hidden;
    border:1px solid #dcedc8; height:100%;
    display:flex; flex-direction:column;
    box-shadow:0 2px 10px rgba(0,0,0,.05);
    transition:transform .18s, box-shadow .18s;
    cursor:pointer;
}
.card:hover { transform:translateY(-4px); box-shadow:0 10px 28px rgba(0,0,0,.1); }
.card-img-wrap { position:relative; aspect-ratio:4/3; overflow:hidden; background:#e8f5e9; }
.card-img-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
.pill-kat {
    position:absolute; top:10px; left:10px;
    background:rgba(13,40,24,.82); color:#a5d6a7;
    font-size:.63rem; font-weight:700; text-transform:uppercase;
    letter-spacing:.06em; padding:4px 10px; border-radius:999px;
}
.pill-score {
    position:absolute; top:10px; right:10px;
    background:rgba(46,125,50,.9); color:#fff;
    font-size:.7rem; font-weight:800; padding:4px 10px; border-radius:999px;
}
.pill-rank {
    position:absolute; bottom:10px; left:10px;
    background:rgba(0,0,0,.55); color:#fff;
    font-size:.72rem; font-weight:800; padding:4px 11px; border-radius:999px;
}
.card-body { padding:13px 15px 15px; flex:1; display:flex; flex-direction:column; gap:6px; }
.card-nama {
    font-family:'Lora',serif; font-size:.98rem; font-weight:600;
    color:#1b2e1c; line-height:1.3;
}
.card-tags { display:flex; flex-wrap:wrap; gap:4px; }
.tag {
    display:inline-flex; align-items:center; gap:3px;
    border-radius:999px; padding:3px 9px;
    font-size:.7rem; font-weight:600; line-height:1;
}
.t-kota  { background:#e8f5e9; color:#1b5e20; }
.t-rate  { background:#fff8e1; color:#e65100; }
.t-hrg   { background:#f3e5f5; color:#6a1b9a; }
.card-desc {
    font-size:.77rem; color:#4a6741; line-height:1.55; flex:1;
    display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical; overflow:hidden;
}

/* Detail panel */
.det-wrap {
    background:#fff; border-radius:22px; overflow:hidden;
    border:1px solid #dcedc8; box-shadow:0 4px 24px rgba(0,0,0,.07);
    margin-bottom:1.8rem;
}
.det-img { width:100%; aspect-ratio:21/8; object-fit:cover; display:block; background:#e8f5e9; }
.det-body { padding:1.8rem 2.2rem; }
.det-name {
    font-family:'Lora',serif; font-size:1.9rem; font-weight:600;
    color:#1b2e1c; line-height:1.15; margin-bottom:.5rem;
}
.det-tags { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1.3rem; }
.det-tag { display:inline-flex; align-items:center; gap:5px; border-radius:999px;
           padding:6px 14px; font-size:.85rem; font-weight:600; }
.info-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr));
             gap:10px; margin-bottom:1.6rem; }
.info-box { background:#f9fbe7; border:1px solid #dcedc8; border-radius:12px; padding:11px 13px; }
.info-lbl { font-size:.63rem; font-weight:700; letter-spacing:.08em;
            text-transform:uppercase; color:#558b2f; margin-bottom:4px; }
.info-val { font-size:.95rem; font-weight:700; color:#1b2e1c; }
.det-desc { font-size:.92rem; color:#33502e; line-height:1.8; }

/* Empty */
.empty-box { text-align:center; padding:3rem 1rem; }
.empty-icon { font-size:3rem; margin-bottom:.8rem; }
.empty-txt { color:#6d8e6e; font-size:.95rem; }

/* Divider */
.divider { height:1px; background:#dcedc8; margin:1.5rem 0; }

/* Result header */
.result-header {
    background:linear-gradient(90deg,#e8f5e9,#f1f8e9);
    border:1px solid #c8e6c9; border-radius:16px;
    padding:1.2rem 1.6rem; margin-bottom:1.2rem;
    display:flex; align-items:center; gap:1rem; flex-wrap:wrap;
}
.result-count { font-family:'Lora',serif; font-size:1.5rem; font-weight:600; color:#1b5e20; }
.result-sub { font-size:.82rem; color:#558b2f; }

/* Metric override */
div[data-testid="metric-container"] {
    background:#fff; border:1px solid #dcedc8; border-radius:13px; padding:12px 16px;
}

/* Tabs */
[data-baseweb="tab-list"] { background:#f1f8e9 !important; border-radius:12px !important; padding:4px !important; }
[data-baseweb="tab"] { border-radius:9px !important; font-weight:600 !important; font-size:.82rem !important; }
[aria-selected="true"][data-baseweb="tab"] { background:#fff !important; color:#1b5e20 !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-thumb { background:#a5d6a7; border-radius:5px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def fmt_rupiah(v) -> str:
    try:
        n = int(float(v))
        return "Gratis" if n == 0 else f"Rp {n:,.0f}".replace(",", ".")
    except Exception:
        return "Gratis"


def short_desc(t: str, w: int = 45) -> str:
    t = str(t).strip()
    if t in {"BELUM DIISI", "BELUM_DIISI", "nan", ""}:
        return "Deskripsi belum tersedia."
    ws = t.split()
    return " ".join(ws[:w]) + ("…" if len(ws) > w else "")


IMG_ROOT = BASE_DIR / "koleksi_gambar"


@st.cache_data(show_spinner=False)
def _list_img_folders() -> list:
    """Nama folder gambar yang benar-benar ada di disk."""
    if not IMG_ROOT.exists():
        return []
    return sorted(p.name for p in IMG_ROOT.iterdir() if p.is_dir())


@st.cache_data(show_spinner=False)
def _known_cities() -> list:
    """Daftar nama kota, dipakai untuk membersihkan akhiran nama folder yang salah."""
    for p in DATA_PATH_CANDIDATES:
        if p.exists():
            try:
                kota = pd.read_csv(p, usecols=["kota"])["kota"].dropna().unique().tolist()
                return sorted({str(k) for k in kota}, key=len, reverse=True)
            except Exception:
                return []
    return []


def _strip_city_suffix(folder: str) -> str:
    """
    Sebagian baris di CSV menulis nama folder dengan akhiran kota yang
    dobel, misalnya "Taman Pelangi Yogyakarta Yogyakarta wisata" padahal
    folder aslinya cuma "Taman Pelangi". Fungsi ini membuang akhiran itu.
    """
    cities = _known_cities()
    if not cities:
        return folder
    pattern = "|".join(re.escape(c) for c in cities)
    suffix_re = re.compile(rf"(\s(?:{pattern})){{1,2}}\s*wisata\s*$", re.IGNORECASE)
    return suffix_re.sub("", folder).strip()


def resolve_img(val: str) -> str:
    s = str(val).strip()
    if s.startswith(("http://", "https://")):
        return s
    if not s or s in {"BELUM_DIISI", "BELUM DIISI", "nan", ""}:
        return PLACEHOLDER_IMG

    # 1) Coba path apa adanya dulu.
    for base in [BASE_DIR, IMG_ROOT]:
        p = base / s
        if p.exists():
            return str(p)

    # 2) Nama folder di CSV kadang tidak persis sama dengan nama folder di
    #    disk (akhiran kota dobel, typo kecil, dsb), dan ekstensi file yang
    #    tertulis di CSV kadang juga tidak sama dengan file yang benar-benar
    #    ada (CSV bilang .webp, filenya ternyata .jpg). Coba cocokkan ulang.
    m = re.match(r"koleksi_gambar[\\/](.+)[\\/](Image_\d+)\.\w+$", s)
    if m:
        folder_in_csv, fstem = m.groups()
        stripped = _strip_city_suffix(folder_in_csv)

        candidates = [stripped, folder_in_csv]
        close = difflib.get_close_matches(stripped, _list_img_folders(), n=1, cutoff=0.55)
        if close:
            candidates.append(close[0])

        for candidate in candidates:
            folder_path = IMG_ROOT / candidate
            if not folder_path.is_dir():
                continue
            # Cocokkan nama file (Image_1) tanpa peduli ekstensinya.
            hits = sorted(folder_path.glob(f"{fstem}.*"))
            if hits:
                return str(hits[0])
            # Fallback terakhir: ambil gambar pertama yang ada di folder itu.
            any_img = sorted(
                q for q in folder_path.iterdir()
                if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}
            )
            if any_img:
                return str(any_img[0])

    return PLACEHOLDER_IMG


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def stars(r: float) -> str:
    r = round(float(r), 1)
    full = int(r); half = 1 if r-full >= .5 else 0; empty = 5-full-half
    return "★"*full + ("⯨" if half else "") + "☆"*empty


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
for k, v in {
    "view": "home",        # home | detail | hasil
    "detail_id": None,
    "hasil_df": None,
    "ref_label": None,
    "clicked_lat": -6.2,
    "clicked_lon": 106.816,
    "run_rec": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# LOAD DATA & MODEL
# ─────────────────────────────────────────────────────────────
def find_data() -> Path:
    for p in DATA_PATH_CANDIDATES:
        if p.exists():
            return p
    return DATA_PATH_CANDIDATES[-1]


@st.cache_data(show_spinner=False)
def load_dataset(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    df["harga"]  = pd.to_numeric(df["harga"],  errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).round(1)
    df["lat"]    = pd.to_numeric(df["lat"],    errors="coerce")
    df["long"]   = pd.to_numeric(df["long"],   errors="coerce")
    for c in ["nama_wisata", "deskripsi", "kategori", "kota", "gambar", "fitur_gabungan"]:
        df[c] = df[c].fillna("").astype(str)
    df = df.dropna(subset=["lat", "long"]).reset_index(drop=True)
    return df


@st.cache_resource(show_spinner=False)
def build_model(texts: tuple):
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=1)
    mat = vec.fit_transform(texts)
    sim = cosine_similarity(mat, mat)
    return sim


data_path = find_data()
if not data_path.exists():
    st.error(f"Dataset tidak ditemukan. Letakkan `DATASET_WISATA_READY_MODELING_FINAL.csv` sefolder dengan `app.py`.")
    st.stop()

with st.spinner("Memuat data…"):
    df = load_dataset(str(data_path))

with st.spinner("Membangun model rekomendasi…"):
    SIM = build_model(tuple(df["fitur_gabungan"].tolist()))

# Normalisasi skor
df["rating_norm"] = df["rating"] / 5.0
h_max = df["harga"].max() if df["harga"].max() > 0 else 1
df["harga_norm"]  = df["harga"] / h_max


# ─────────────────────────────────────────────────────────────
# ENGINE REKOMENDASI
# ─────────────────────────────────────────────────────────────
def recommend(
    ref_id=None,
    kota_list=None,
    kat_list=None,
    keyword="",
    max_harga=None,
    min_rating=0.0,
    top_n=10,
    use_loc=False,
    user_lat=None,
    user_lon=None,
    radius_km=50.0,
) -> pd.DataFrame:

    cand = df.copy()

    # Filter
    if kota_list:
        cand = cand[cand["kota"].isin(kota_list)]
    if kat_list:
        cand = cand[cand["kategori"].isin(kat_list)]
    if keyword.strip():
        kw = keyword.strip().lower()
        mask = (
            cand["nama_wisata"].str.lower().str.contains(kw, na=False) |
            cand["deskripsi"].str.lower().str.contains(kw, na=False) |
            cand["kota"].str.lower().str.contains(kw, na=False) |
            cand["kategori"].str.lower().str.contains(kw, na=False)
        )
        cand = cand[mask]
    if max_harga is not None:
        cand = cand[(cand["harga"] <= max_harga) | (cand["harga"] == 0)]
    cand = cand[cand["rating"] >= min_rating]
    if ref_id is not None:
        cand = cand[cand["place_id"] != ref_id]
    if cand.empty:
        return pd.DataFrame()

    # Cosine similarity
    if ref_id is not None:
        idx = df[df["place_id"] == ref_id].index
        if len(idx):
            cand = cand.copy()
            cand["cos_sim"] = cand.index.map(lambda i: float(SIM[idx[0]][i]))
        else:
            cand["cos_sim"] = 0.0
    else:
        cand["cos_sim"] = 0.0

    # Lokasi + Haversine
    has_loc = (
        use_loc
        and user_lat is not None
        and user_lon is not None
        and not math.isnan(float(user_lat))
        and not math.isnan(float(user_lon))
        and radius_km > 0
    )

    if has_loc:
        cand = cand.copy()

        cand["jarak_km"] = cand.apply(
            lambda row: haversine(
                float(user_lat),
                float(user_lon),
                float(row["lat"]),
                float(row["long"])
            ),
            axis=1
        )

        cand = cand[cand["jarak_km"] <= radius_km]

        if cand.empty:
            return pd.DataFrame()

        cand["dist_score"] = (
            1 - cand["jarak_km"] / radius_km
        ).clip(0, 1)
    else:
        cand = cand.copy()
        cand["jarak_km"] = np.nan
        cand["dist_score"] = 0.0

    # Weighted scoring (Bab 3)
    has_ref = ref_id is not None

    if has_ref and has_loc:
        cand["final_score"] = (
            0.5 * cand["cos_sim"]
            + 0.3 * cand["dist_score"]
            + 0.2 * cand["rating_norm"]
        )
        cand["skenario"] = "CBF + Lokasi + Rating"

    elif has_ref and not has_loc:
        cand["final_score"] = (
            0.7 * cand["cos_sim"]
            + 0.3 * cand["rating_norm"]
        )
        cand["skenario"] = "CBF + Rating"

    elif not has_ref and has_loc:
        cand["final_score"] = (
            0.6 * cand["dist_score"]
            + 0.4 * cand["rating_norm"]
        )
        cand["skenario"] = "Lokasi + Rating"

    else:
        cand["final_score"] = cand["rating_norm"]
        cand["skenario"] = "Rating Tertinggi"

    cand["skor"] = (cand["final_score"] * 100).round(1)

    return cand.sort_values(
        ["final_score", "rating"],
        ascending=[False, False]
    ).head(top_n)


# ─────────────────────────────────────────────────────────────
# UI KOMPONEN
# ─────────────────────────────────────────────────────────────
def card_html(row, rank=None, show_score=True) -> str:
    img     = resolve_img(row["gambar"])
    emoji   = CATEGORY_EMOJI.get(row["kategori"], "📍")
    score   = row.get("skor", 0)
    rank_html   = f"<div class='pill-rank'>#{rank}</div>" if rank else ""
    score_html  = f"<div class='pill-score'>{score:.0f} pts</div>" if show_score and score > 0 else ""
    desc    = short_desc(row["deskripsi"])
    return f"""
    <div class='card'>
      <div class='card-img-wrap'>
        <img src='{img}' onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'" loading='lazy'/>
        <div class='pill-kat'>{emoji} {row['kategori']}</div>
        {score_html}{rank_html}
      </div>
      <div class='card-body'>
        <div class='card-nama'>{html.escape(str(row['nama_wisata']))}</div>
        <div class='card-tags'>
          <span class='tag t-kota'>📍 {html.escape(str(row['kota']))}</span>
          <span class='tag t-rate'>⭐ {float(row['rating']):.1f}</span>
          <span class='tag t-hrg'>💰 {fmt_rupiah(row['harga'])}</span>
        </div>
        <div class='card-desc'>{html.escape(desc)}</div>
      </div>
    </div>"""


def render_grid(data_: pd.DataFrame, n_cols=3, show_score=True, show_rank=False):
    if data_.empty:
        st.markdown("""<div class='empty-box'>
            <div class='empty-icon'>🔍</div>
            <div class='empty-txt'>Tidak ada destinasi yang cocok.</div>
        </div>""", unsafe_allow_html=True)
        return
    rows = [data_.iloc[i:i+n_cols] for i in range(0, len(data_), n_cols)]
    for rg in rows:
        cols = st.columns(n_cols)
        for col, (_, rec) in zip(cols, rg.iterrows()):
            with col:
                rank_num = int(rec.name) + 1 if show_rank else None
                st.markdown(card_html(rec, rank=rank_num, show_score=show_score),
                            unsafe_allow_html=True)
                if st.button("Lihat Detail →",
                             key=f"card_{rec['place_id']}_{rec.get('skor',0):.0f}",
                             use_container_width=True):
                    st.session_state.detail_id = int(rec["place_id"])
                    st.session_state.view = "detail"
                    st.rerun()


def render_detail(place_id: int):
    r_df = df[df["place_id"] == place_id]
    if r_df.empty:
        st.error("Data tidak ditemukan.")
        return
    r = r_df.iloc[0]
    emoji = CATEGORY_EMOJI.get(r["kategori"], "📍")

    # Nav
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("← Kembali", key="back_det", use_container_width=True):
            st.session_state.view = "home" if st.session_state.hasil_df is None else "hasil"
            st.rerun()
    with c2:
        if st.button(f"🔗 Wisata Serupa dengan ini", key="serupa_det",
                     type="primary", use_container_width=True):
            sim = recommend(ref_id=place_id, top_n=12)
            st.session_state.hasil_df = sim
            st.session_state.ref_label = r["nama_wisata"]
            st.session_state.view = "hasil"
            st.rerun()

    st.markdown("<div style='height:.6rem'/>", unsafe_allow_html=True)

    img_url = resolve_img(r["gambar"])
    deskripsi = (r["deskripsi"] if str(r["deskripsi"]).strip()
                 not in {"BELUM DIISI", "BELUM_DIISI", "nan", ""}
                 else "Deskripsi belum tersedia.")

    st.markdown(f"""
    <div class='det-wrap'>
      <img class='det-img' src='{img_url}'
           onerror="this.onerror=null;this.src='{PLACEHOLDER_IMG}'"/>
      <div class='det-body'>
        <div class='sec-tag'>{emoji} {r['kategori']} &nbsp;·&nbsp; 📍 {r['kota']}</div>
        <div class='det-name'>{html.escape(str(r['nama_wisata']))}</div>
        <div class='det-tags'>
          <span class='det-tag t-rate'>⭐ {float(r['rating']):.1f} &nbsp; {stars(r['rating'])}</span>
          <span class='det-tag t-hrg'>💰 {fmt_rupiah(r['harga'])}</span>
          <span class='det-tag t-kota'>📍 {html.escape(str(r['kota']))}</span>
          <span class='det-tag' style='background:#e8eaf6;color:#283593;'>
            {emoji} {html.escape(str(r['kategori']))}
          </span>
        </div>
        <div class='info-grid'>
          <div class='info-box'>
            <div class='info-lbl'>Kategori</div>
            <div class='info-val'>{html.escape(str(r['kategori']))}</div>
          </div>
          <div class='info-box'>
            <div class='info-lbl'>Harga Tiket</div>
            <div class='info-val'>{fmt_rupiah(r['harga'])}</div>
          </div>
          <div class='info-box'>
            <div class='info-lbl'>Rating</div>
            <div class='info-val'>{float(r['rating']):.1f} / 5.0</div>
          </div>
          <div class='info-box'>
            <div class='info-lbl'>Koordinat</div>
            <div class='info-val' style='font-size:.8rem;'>{float(r['lat']):.5f}, {float(r['long']):.5f}</div>
          </div>
        </div>
        <div class='det-desc'>{html.escape(deskripsi)}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab_map, tab_sim = st.tabs(["🗺️ Lokasi di Peta", "🔗 5 Wisata Serupa"])

    with tab_map:
        m = folium.Map(location=[float(r["lat"]), float(r["long"])],
                       zoom_start=15, tiles="CartoDB Positron")
        folium.Marker(
            [float(r["lat"]), float(r["long"])],
            popup=folium.Popup(
                f"<b>{html.escape(str(r['nama_wisata']))}</b><br>"
                f"⭐ {float(r['rating']):.1f} · {fmt_rupiah(r['harga'])}",
                max_width=240,
            ),
            tooltip=str(r["nama_wisata"]),
            icon=folium.Icon(color="green", icon="map-marker"),
        ).add_to(m)
        st_folium(m, height=380, key=f"map_det_{place_id}", returned_objects=[])

    with tab_sim:
        sim5 = recommend(ref_id=place_id, top_n=5)
        sim5 = sim5[sim5["place_id"] != place_id] if not sim5.empty else sim5
        render_grid(sim5, n_cols=5, show_score=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.6rem 0 .8rem;'>
      <div style='font-family:Lora,serif;font-size:1.35rem;font-weight:600;
                  color:#81c784;'>🌿 WisataJawa</div>
      <div style='font-size:.73rem;color:#4caf50;margin-top:2px;'>Rekomendasi Wisata Pulau Jawa</div>
    </div>
    <hr style='border-color:#1b3d22;margin:.2rem 0 1rem;'/>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:.75rem;font-weight:700;color:#81c784;"
                "letter-spacing:.08em;text-transform:uppercase;margin-bottom:.4rem;'>"
                "Preferensi Wisata</p>", unsafe_allow_html=True)

    # Referensi
    ref_options = ["— Tanpa referensi —"] + sorted(df["nama_wisata"].tolist())
    ref_name = st.selectbox("🏛️ Wisata Referensi",
                            ref_options, help="Pilih destinasi acuan untuk CBF")

    st.markdown("<hr style='border-color:#1b3d22;margin:.6rem 0;'/>", unsafe_allow_html=True)

    # Filter
    kota_opts = sorted(df["kota"].unique().tolist())
    kat_opts  = sorted(df["kategori"].unique().tolist())
    kw        = st.text_input("🔎 Kata Kunci", placeholder="Contoh: pantai, museum…")
    pilih_kat  = st.multiselect("🎯 Kategori", kat_opts, placeholder="Semua kategori")
    pilih_kota = st.multiselect("🏙️ Kota", kota_opts, placeholder="Semua kota")
    max_h      = st.slider("💰 Budget Maksimal (Rp)",
                            0, int(df["harga"].max()), int(df["harga"].max()),
                            step=5_000, format="Rp %d")
    min_r      = st.slider("⭐ Rating Minimum", 0.0, 5.0, 0.0, step=0.1)
    top_n_val  = st.slider("📋 Jumlah Rekomendasi", 5, 20, 10)

    st.markdown("<hr style='border-color:#1b3d22;margin:.6rem 0;'/>", unsafe_allow_html=True)

    # Lokasi
    use_loc = st.checkbox("📍 Gunakan Lokasi Saya",
                          help="Aktifkan untuk filter berdasarkan jarak")
    u_lat = u_lon = None
    radius_km = 50.0

    if use_loc:
        loc_method = st.radio("Metode Lokasi",
                              ["Pilih kota asal", "Koordinat manual", "Klik peta"])
        radius_km = float(st.slider("Radius (km)", 5, 300, 50, step=5))

        if loc_method == "Pilih kota asal":
            origin = st.selectbox("Kota asal", list(CITY_COORDS.keys()))
            u_lat, u_lon = CITY_COORDS[origin]
        elif loc_method == "Koordinat manual":
            u_lat = st.number_input("Latitude",  value=-6.2,  format="%.6f")
            u_lon = st.number_input("Longitude", value=106.816, format="%.6f")
        else:
            st.info("Klik peta di halaman utama untuk set lokasi.")
            u_lat = st.session_state.clicked_lat
            u_lon = st.session_state.clicked_lon
            st.caption(f"Lokasi: {u_lat:.5f}, {u_lon:.5f}")

    st.markdown("<div style='height:.4rem'/>", unsafe_allow_html=True)
    cari_btn = st.button("🔍 Cari Rekomendasi", use_container_width=True)

    st.markdown("<hr style='border-color:#1b3d22;margin:.6rem 0;'/>", unsafe_allow_html=True)
    st.markdown(
        f"<p style='font-size:.69rem;color:#388e3c;text-align:center;line-height:1.6;'>"
        f"{len(df):,} destinasi &nbsp;·&nbsp; {df['kota'].nunique()} kota<br>"
        f"<span style='opacity:.55;'>TF-IDF · Cosine Similarity</span></p>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────
# PROSES TOMBOL
# ─────────────────────────────────────────────────────────────
if cari_btn:
    _ref_id = None
    if ref_name != "— Tanpa referensi —":
        m_ = df[df["nama_wisata"] == ref_name]
        if not m_.empty:
            _ref_id = int(m_.iloc[0]["place_id"])

    _hasil = recommend(
        ref_id=_ref_id,
        kota_list=pilih_kota or None,
        kat_list=pilih_kat   or None,
        keyword=kw,
        max_harga=max_h,
        min_rating=min_r,
        top_n=top_n_val,
        use_loc=use_loc,
        user_lat=u_lat,
        user_lon=u_lon,
        radius_km=radius_km,
    )
    # tambah rank index
    _hasil = _hasil.reset_index(drop=True)
    st.session_state.hasil_df  = _hasil
    st.session_state.ref_label = ref_name if ref_name != "— Tanpa referensi —" else None
    st.session_state.view = "hasil"
    st.rerun()


# ─────────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────────

# ── DETAIL ───────────────────────────────────────────────────
if st.session_state.view == "detail" and st.session_state.detail_id:
    render_detail(st.session_state.detail_id)
    st.stop()


# ── HASIL ────────────────────────────────────────────────────
if st.session_state.view == "hasil" and st.session_state.hasil_df is not None:
    hasil = st.session_state.hasil_df
    ref_lbl = st.session_state.ref_label

    sub = f"Serupa dengan <em>{html.escape(str(ref_lbl))}</em>" if ref_lbl \
          else "Berdasarkan preferensi kamu"
    skenario = hasil["skenario"].iloc[0] if not hasil.empty and "skenario" in hasil.columns else ""

    st.markdown(f"""
    <div class='hero' style='padding:1.8rem 2.5rem;'>
      <div class='sec-tag' style='color:#a5d6a7;'>Hasil Rekomendasi</div>
      <div class='hero-title' style='font-size:1.8rem;'>
        Ditemukan <em>{len(hasil)}</em> Destinasi Wisata
      </div>
      <div class='hero-sub'>{sub}</div>
      <div class='hero-pills'>
        <span class='hero-pill'>🎯 {skenario}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    cback, _ = st.columns([1, 5])
    with cback:
        if st.button("← Kembali ke Beranda", key="back_home_hasil"):
            st.session_state.view     = "home"
            st.session_state.hasil_df = None
            st.rerun()

    if hasil.empty:
        st.markdown("""<div class='empty-box'>
            <div class='empty-icon'>😕</div>
            <div class='empty-txt'><b>Tidak ada destinasi yang cocok.</b><br>
            Coba longgarkan filter.</div></div>""", unsafe_allow_html=True)
        st.stop()

    # Tabs hasil
    tab_kartu, tab_peta, tab_tabel = st.tabs(
        ["🃏 Kartu Rekomendasi", "🗺️ Peta Lokasi", "📊 Tabel Skor"]
    )

    with tab_kartu:
        render_grid(hasil, n_cols=3, show_score=True, show_rank=True)

    with tab_peta:
        lat_c = hasil["lat"].mean(); lon_c = hasil["long"].mean()
        m_h = folium.Map(location=[lat_c, lon_c], zoom_start=8, tiles="CartoDB Positron")
        colors_ = ["green","blue","purple","orange","red","darkgreen",
                   "cadetblue","darkblue","lightred","pink"]
        for i, (_, rec) in enumerate(hasil.iterrows()):
            pop = (
                f"<b>#{i+1} {html.escape(str(rec['nama_wisata']))}</b><br>"
                f"{html.escape(str(rec['kota']))} · {html.escape(str(rec['kategori']))}<br>"
                f"⭐ {float(rec['rating']):.1f} · {fmt_rupiah(rec['harga'])}<br>"
                f"Skor: {rec['skor']:.1f}"
            )
            if pd.notna(rec.get("jarak_km")):
                pop += f"<br>Jarak: {rec['jarak_km']:.1f} km"
            folium.Marker(
                [float(rec["lat"]), float(rec["long"])],
                popup=folium.Popup(pop, max_width=260),
                tooltip=f"#{i+1} {rec['nama_wisata']}",
                icon=folium.Icon(color=colors_[i % len(colors_)], icon="map-marker"),
            ).add_to(m_h)
        st_folium(m_h, height=480, key="peta_hasil", returned_objects=[])

    with tab_tabel:
        cols_tbl = ["nama_wisata","kategori","kota","harga","rating","skor"]
        if "jarak_km" in hasil.columns and hasil["jarak_km"].notna().any():
            cols_tbl.insert(5, "jarak_km")
        tbl = hasil[cols_tbl].copy()
        tbl.index = range(1, len(tbl)+1)
        tbl.columns = [c.replace("_"," ").title() for c in tbl.columns]
        st.dataframe(tbl, use_container_width=True)

        csv = hasil[cols_tbl].to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", data=csv,
                           file_name="rekomendasi_wisata.csv",
                           mime="text/csv", use_container_width=True)

    st.stop()


# ─────────────────────────────────────────────────────────────
# HOME VIEW
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero'>
  <div class='hero-title'>
    Temukan <em>Destinasi Wisata</em><br>Pulau Jawa Impianmu
  </div>
  <div class='hero-sub'>
    Content-Based Filtering &nbsp;·&nbsp; {len(df):,} destinasi &nbsp;·&nbsp; {df['kota'].nunique()} kota
  </div>
  <div class='hero-pills'>
    <span class='hero-pill'>🏛️ {(df['kategori']=='Budaya').sum()} Budaya</span>
    <span class='hero-pill'>🌿 {(df['kategori']=='Cagar Alam').sum()} Alam</span>
    <span class='hero-pill'>🎡 {(df['kategori']=='Taman Hiburan').sum()} Hiburan</span>
    <span class='hero-pill'>🌊 {(df['kategori']=='Bahari').sum()} Bahari</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Metrik ────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Destinasi", f"{len(df):,}")
m2.metric("Kota",            df["kota"].nunique())
m3.metric("Kategori",        df["kategori"].nunique())
m4.metric("Avg Rating",      f"{df['rating'].mean():.2f} ⭐")

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Search ────────────────────────────────────────────────────
st.markdown("""
<div class='search-wrap'>
  <div class='sec-tag'>Cari Destinasi</div>
  <div class='sec-h' style='margin-bottom:.4rem;'>Ketik nama tempat wisata</div>
</div>
""", unsafe_allow_html=True)

c_srch, c_go = st.columns([5, 1])
with c_srch:
    cari_nama = st.selectbox(
        "cari", options=[""] + sorted(df["nama_wisata"].tolist()),
        label_visibility="collapsed", placeholder="Contoh: Monas, Prambanan, Pantai Parangtritis…"
    )
with c_go:
    if st.button("Lihat Detail", type="primary", use_container_width=True):
        if cari_nama:
            m_ = df[df["nama_wisata"] == cari_nama]
            if not m_.empty:
                st.session_state.detail_id = int(m_.iloc[0]["place_id"])
                st.session_state.view = "detail"
                st.rerun()

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Peta Eksplorasi ───────────────────────────────────────────
st.markdown("<div class='sec-tag'>Peta Interaktif</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-h'>Eksplorasi Semua Destinasi</div>", unsafe_allow_html=True)
st.caption("Klik titik pada peta untuk info destinasi. Klik peta kosong untuk set lokasi kamu.")

col_map, col_dist = st.columns([3, 1])

with col_map:
    m_home = folium.Map(location=[df["lat"].mean(), df["long"].mean()],
                        zoom_start=7, tiles="CartoDB Positron")
    from folium.plugins import MarkerCluster as MC
    mc = MC(options={"maxClusterRadius": 40}).add_to(m_home)

    kat_f = {"Budaya":"orange","Taman Hiburan":"green","Cagar Alam":"darkgreen",
             "Bahari":"blue","Pusat Perbelanjaan":"purple","Tempat Ibadah":"red"}
    for _, r in df.iterrows():
        wf = kat_f.get(r["kategori"], "gray")
        pop_txt = (
            "<b>" + html.escape(str(r["nama_wisata"])) + "</b><br>"
            + html.escape(str(r["kategori"])) + " · " + html.escape(str(r["kota"])) + "<br>"
            + "⭐ " + str(round(float(r["rating"]),1))
            + " | " + fmt_rupiah(r["harga"])
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["long"])],
            radius=5, color=wf, fill=True,
            fill_color=wf, fill_opacity=.75, weight=1.5,
            popup=folium.Popup(pop_txt, max_width=250),
            tooltip=html.escape(str(r["nama_wisata"])),
        ).add_to(mc)

    # Lokasi klik pengguna
    folium.Marker(
        [st.session_state.clicked_lat, st.session_state.clicked_lon],
        tooltip="Lokasi kamu",
        icon=folium.Icon(color="red", icon="user", prefix="fa"),
    ).add_to(m_home)

    leg = """<div style='position:fixed;bottom:22px;right:22px;background:#fff;border-radius:12px;
    padding:10px 14px;box-shadow:0 2px 10px rgba(0,0,0,.13);font-size:11px;z-index:9999;'>
    <b style='color:#1b5e20;display:block;margin-bottom:6px;'>Kategori</b>
    <span style='color:orange;'>●</span> Budaya &nbsp;
    <span style='color:green;'>●</span> Taman Hiburan<br>
    <span style='color:darkgreen;'>●</span> Cagar Alam &nbsp;
    <span style='color:blue;'>●</span> Bahari<br>
    <span style='color:purple;'>●</span> Perbelanjaan &nbsp;
    <span style='color:red;'>●</span> Tempat Ibadah</div>"""
    m_home.get_root().html.add_child(folium.Element(leg))

    map_out = st_folium(m_home, height=520, key="peta_home",
                        returned_objects=["last_clicked"])

    if map_out and map_out.get("last_clicked"):
        st.session_state.clicked_lat = map_out["last_clicked"]["lat"]
        st.session_state.clicked_lon = map_out["last_clicked"]["lng"]
        st.toast(f"📍 Lokasi diset: {st.session_state.clicked_lat:.5f}, {st.session_state.clicked_lon:.5f}")

with col_dist:
    st.markdown("<div class='sec-tag'>Kategori</div>", unsafe_allow_html=True)
    kat_c = {"Budaya":"#e88a00","Taman Hiburan":"#2e7d32","Cagar Alam":"#1b5e20",
             "Bahari":"#1565c0","Pusat Perbelanjaan":"#6a1b9a","Tempat Ibadah":"#b71c1c"}
    for kat, cnt in df["kategori"].value_counts().items():
        pct = cnt / len(df) * 100
        w = kat_c.get(kat, "#888")
        emoji = CATEGORY_EMOJI.get(kat, "")
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:7px;margin-bottom:4px;'>
          <div style='width:8px;height:8px;border-radius:50%;background:{w};flex-shrink:0;'></div>
          <div style='flex:1;font-size:.75rem;color:#33502e;'>{emoji} {kat}</div>
          <div style='font-size:.75rem;font-weight:700;color:#1b2e1c;'>{cnt}</div>
        </div>
        <div style='height:4px;background:#e8f5e9;border-radius:4px;margin-bottom:9px;'>
          <div style='width:{pct:.0f}%;height:100%;background:{w};border-radius:4px;'></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    st.markdown("<div class='sec-tag'>Top Kota</div>", unsafe_allow_html=True)
    for kota, cnt in df["kota"].value_counts().head(8).items():
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             font-size:.78rem;padding:4px 0;border-bottom:1px solid #e8f5e9;'>
          <span style='color:#33502e;'>📍 {html.escape(str(kota))}</span>
          <span style='font-weight:700;color:#1b2e1c;background:#e8f5e9;
                padding:1px 7px;border-radius:999px;font-size:.72rem;'>{cnt}</span>
        </div>""", unsafe_allow_html=True)

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Populer ───────────────────────────────────────────────────
st.markdown("<div class='sec-tag'>Populer</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-h'>Destinasi Terpopuler</div>", unsafe_allow_html=True)

pop9 = df.sort_values("rating", ascending=False).head(9).copy()
pop9["skor"] = 0.0
render_grid(pop9, n_cols=3, show_score=False)

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Per Kategori ──────────────────────────────────────────────
st.markdown("<div class='sec-tag'>Jelajahi</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-h'>Jelajahi per Kategori</div>", unsafe_allow_html=True)

TAB_L = ["🏛️ Budaya","🎡 Taman Hiburan","🌿 Cagar Alam",
         "🌊 Bahari","🛍️ Perbelanjaan","🕌 Tempat Ibadah"]
KAT_K = ["Budaya","Taman Hiburan","Cagar Alam",
         "Bahari","Pusat Perbelanjaan","Tempat Ibadah"]

for tab, kat in zip(st.tabs(TAB_L), KAT_K):
    with tab:
        sub_kat = df[df["kategori"] == kat].sort_values("rating", ascending=False).head(6).copy()
        sub_kat["skor"] = 0.0
        render_grid(sub_kat, n_cols=3, show_score=False)
