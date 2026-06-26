"""
app.py ─ Sistem Rekomendasi Destinasi Wisata Indonesia
Content-Based Filtering · TF-IDF + Cosine Similarity
Streamlit · Folium
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# ══════════════════════════════════════════════════════════════════════════════
#  KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="WisataJawa · Temukan Destinasimu",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [data-testid="stAppViewContainer"] {
    background: #f0f4f0 !important;
    font-family: 'Plus Jakarta Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background: #f0f4f0 !important; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1400px !important; }
h1,h2,h3,h4 { font-family: 'Lora', serif; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f2d1a !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #d4e8d8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p { color: #a8c9ae !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="input"] > div {
    background: #1a3d25 !important;
    border-color: #2d5c3a !important;
    color: #e8f5eb !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: #2d9e5a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    font-size: 0.92rem !important;
    transition: background .2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #24834a !important;
}

/* ── Hero Header ── */
.hero-wrap {
    background: linear-gradient(135deg, #0f2d1a 0%, #1a5c33 60%, #2d9e5a 100%);
    border-radius: 24px;
    padding: 2.8rem 3rem 2.4rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.04'%3E%3Ccircle cx='30' cy='30' r='28'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E") repeat;
    opacity: .6;
}
.hero-title {
    font-family: 'Lora', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #f0faf2;
    line-height: 1.2;
    position: relative;
}
.hero-title span { color: #7edaa0; font-style: italic; }
.hero-sub {
    color: #a8d8b8;
    font-size: 1rem;
    margin-top: 0.5rem;
    font-weight: 400;
    position: relative;
}
.hero-stats {
    display: flex; gap: 2rem;
    margin-top: 1.8rem;
    position: relative;
}
.hstat { text-align: left; }
.hstat-num { font-size: 1.9rem; font-weight: 800; color: #7edaa0; line-height: 1; }
.hstat-label { font-size: 0.72rem; color: #a8d8b8; text-transform: uppercase; letter-spacing: .08em; margin-top: 2px; }

/* ── Section label ── */
.sec-label {
    font-size: 0.7rem; font-weight: 700; letter-spacing: .1em;
    text-transform: uppercase; color: #2d9e5a;
    margin-bottom: 6px;
}
.sec-title {
    font-family: 'Lora', serif;
    font-size: 1.45rem; font-weight: 600;
    color: #0f2d1a; margin-bottom: 1rem;
}

/* ── Card wisata ── */
.card {
    background: #fff;
    border-radius: 20px;
    overflow: hidden;
    border: 1px solid #dde8de;
    box-shadow: 0 2px 12px rgba(15,45,26,.06);
    transition: transform .18s, box-shadow .18s;
    height: 100%;
    display: flex; flex-direction: column;
}
.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(15,45,26,.13);
}
.card-img-wrap { position: relative; aspect-ratio: 4/3; overflow: hidden; background: #e6f0e8; }
.card-img-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
.card-kat-badge {
    position: absolute; top: 10px; left: 10px;
    background: rgba(15,45,26,.78); color: #7edaa0;
    font-size: 0.67rem; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; padding: 4px 10px; border-radius: 999px;
    backdrop-filter: blur(4px);
}
.card-score-badge {
    position: absolute; top: 10px; right: 10px;
    background: rgba(45,158,90,.9); color: #fff;
    font-size: 0.72rem; font-weight: 800;
    padding: 4px 10px; border-radius: 999px;
}
.card-body { padding: 14px 16px 16px; flex: 1; display: flex; flex-direction: column; gap: 6px; }
.card-nama {
    font-family: 'Lora', serif;
    font-size: 1rem; font-weight: 600; color: #0f2d1a;
    line-height: 1.35; margin-bottom: 4px;
}
.card-meta { display: flex; flex-wrap: wrap; gap: 5px; }
.badge {
    display: inline-flex; align-items: center; gap: 3px;
    border-radius: 999px; padding: 3px 10px;
    font-size: 0.73rem; font-weight: 600;
}
.badge-kota { background: #e8f5eb; color: #1a5c33; }
.badge-harga { background: #fff8e6; color: #8a6200; }
.badge-rating { background: #fff3e0; color: #c05c00; }
.card-desc {
    font-size: 0.79rem; color: #5a7060; line-height: 1.55;
    flex: 1;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden;
}
.card-btn {
    margin-top: auto;
    display: block; width: 100%; text-align: center;
    background: #f0f9f3; color: #1a5c33;
    border: 1.5px solid #b8ddc0; border-radius: 10px;
    padding: 8px; font-size: 0.82rem; font-weight: 700;
    cursor: pointer; transition: background .15s, color .15s;
}
.card-btn:hover { background: #2d9e5a; color: #fff; border-color: #2d9e5a; }

/* ── Detail Panel ── */
.detail-wrap {
    background: #fff; border-radius: 24px;
    border: 1px solid #dde8de;
    box-shadow: 0 4px 24px rgba(15,45,26,.08);
    overflow: hidden; margin-bottom: 2rem;
}
.detail-img { width: 100%; aspect-ratio: 16/7; object-fit: cover; display: block; background: #e6f0e8; }
.detail-body { padding: 2rem 2.5rem; }
.detail-nama {
    font-family: 'Lora', serif; font-size: 1.9rem;
    font-weight: 600; color: #0f2d1a; margin-bottom: .6rem; line-height: 1.2;
}
.detail-badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.2rem; }
.info-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px,1fr)); gap: 12px; margin-bottom: 1.6rem; }
.info-box { background: #f5faf6; border: 1px solid #dde8de; border-radius: 12px; padding: 12px 14px; }
.info-box-label { font-size: 0.68rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #6a9070; margin-bottom: 4px; }
.info-box-value { font-size: 1rem; font-weight: 700; color: #0f2d1a; }
.detail-desc { font-size: 0.93rem; color: #3d5c42; line-height: 1.75; }

/* ── Empty state ── */
.empty-state {
    text-align: center; padding: 3rem 1rem;
    color: #7a9a80; font-size: 0.95rem;
}
.empty-icon { font-size: 3rem; margin-bottom: .8rem; }

/* ── Divider ── */
.divider { height: 1px; background: #dde8de; margin: 1.5rem 0; }

/* ── Tab custom ── */
[data-baseweb="tab-list"] { background: #f0f9f3 !important; border-radius: 12px !important; padding: 4px !important; gap: 4px; }
[data-baseweb="tab"] { border-radius: 9px !important; font-weight: 600 !important; font-size: 0.84rem !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #fff !important; color: #1a5c33 !important; }

/* ── Metric override ── */
div[data-testid="metric-container"] {
    background: #fff; border: 1px solid #dde8de;
    border-radius: 14px; padding: 14px 18px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0f4f0; }
::-webkit-scrollbar-thumb { background: #b8ddc0; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
PLACEHOLDER = "https://placehold.co/640x480/e6f0e8/6a9070?text=Gambar+Tidak+Tersedia"

def fmt_rupiah(x):
    try:
        v = int(float(x))
        return "Gratis" if v == 0 else f"Rp {v:,.0f}".replace(",", ".")
    except Exception:
        return "Rp 0"

def stars(r):
    try:
        r = float(r)
        full = int(r); half = 1 if r - full >= 0.5 else 0; empty = 5 - full - half
        return "★" * full + ("½" if half else "") + "☆" * empty
    except Exception:
        return "☆☆☆☆☆"

def get_img(val):
    s = str(val).strip()
    if s.startswith("http://") or s.startswith("https://"):
        return s
    if s.startswith("koleksi_gambar") or (s and s not in ["BELUM_DIISI", "nan", ""]):
        return s          # path lokal — Streamlit bisa serve jika file ada
    return PLACEHOLDER

def short_desc(text, words=50):
    txt = str(text).strip()
    if txt in ["BELUM DIISI", "", "nan"]:
        return "Deskripsi belum tersedia."
    ws = txt.split()
    return " ".join(ws[:words]) + ("…" if len(ws) > words else "")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    la1, lo1, la2, lo2 = map(math.radians, [lat1, lon1, lat2, lon2])
    a = math.sin((la2-la1)/2)**2 + math.cos(la1)*math.cos(la2)*math.sin((lo2-lo1)/2)**2
    return R * 2 * math.asin(math.sqrt(a))


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "view": "home",        # home | detail | hasil
    "detail_id": None,
    "hasil_df": None,
    "map_klik_lat": -7.0,
    "map_klik_lon": 110.4,
    "use_map_loc": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
#  LOAD DATA + MODEL  (di-cache)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="⏳ Memuat data dan membangun model rekomendasi…")
def load_model():
    try:
        df = pd.read_csv("DATASET_WISATA_FINAL_READY.csv")
    except FileNotFoundError:
        # fallback nama lain
        import glob, os
        candidates = glob.glob("DATASET_WISATA*.csv")
        if candidates:
            df = pd.read_csv(candidates[0])
        else:
            st.error("❌ File dataset tidak ditemukan. Letakkan DATASET_WISATA_FINAL_READY.csv sefolder dengan app.py.")
            st.stop()

    # Bersihkan tipe
    for c in ["nama_wisata", "deskripsi", "kategori", "kota", "gambar", "fitur_gabungan"]:
        df[c] = df[c].fillna("").astype(str)
    df["harga"]  = pd.to_numeric(df["harga"],  errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0.0).round(1)
    df["lat"]    = pd.to_numeric(df["lat"],    errors="coerce")
    df["long"]   = pd.to_numeric(df["long"],   errors="coerce")
    df = df.dropna(subset=["lat","long"]).reset_index(drop=True)

    # ── Modeling: TF-IDF + Cosine Similarity ──────────────────────────────────
    tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,2), min_df=1)
    mat   = tfidf.fit_transform(df["fitur_gabungan"])
    cos   = cosine_similarity(mat, mat)

    # ── Normalisasi skor untuk weighted scoring ────────────────────────────────
    df["rating_norm"] = df["rating"] / 5.0
    h_max = df["harga"].max() if df["harga"].max() > 0 else 1
    df["harga_norm"]  = df["harga"] / h_max   # 0=gratis, 1=termahal

    return df, cos

df, cos_sim = load_model()


# ══════════════════════════════════════════════════════════════════════════════
#  ENGINE REKOMENDASI  (Bab 3 — alur sistem)
# ══════════════════════════════════════════════════════════════════════════════
def get_rekomendasi(
    ref_id=None,           # place_id referensi (int|None)
    kategori_list=None,    # list kategori filter
    kota_list=None,        # list kota filter
    max_harga=None,        # int
    min_rating=0.0,        # float
    top_n=10,
    exclude_id=None,       # place_id yang tidak ditampilkan (referensi itu sendiri)
) -> pd.DataFrame:
    """
    Skor akhir = 0.6 × cosine_sim + 0.3 × rating_norm + 0.1 × (1 - harga_norm)
    Harga dibalik: murah = lebih disukai
    """
    cand = df.copy()

    # ── Step 1: Filter preferensi ────────────────────────────────────────────
    if kategori_list:
        cand = cand[cand["kategori"].isin(kategori_list)]
    if kota_list:
        cand = cand[cand["kota"].isin(kota_list)]
    if max_harga is not None:
        cand = cand[(cand["harga"] <= max_harga) | (cand["harga"] == 0)]
    cand = cand[cand["rating"] >= min_rating]
    if exclude_id is not None:
        cand = cand[cand["place_id"] != exclude_id]

    if cand.empty:
        return pd.DataFrame()

    # ── Step 2: Cosine Similarity ────────────────────────────────────────────
    if ref_id is not None:
        idx = df[df["place_id"] == ref_id].index
        if len(idx):
            sim_scores = cos_sim[idx[0]]
            cand = cand.copy()
            cand["cos_sim"] = cand.index.map(lambda i: sim_scores[i])
        else:
            cand["cos_sim"] = 0.0
    else:
        cand["cos_sim"] = 0.0

    # ── Step 3: Weighted Score (Bab 3 - Rumus Final) ─────────────────────────
    cand["final_score"] = (
        0.6 * cand["cos_sim"] +
        0.3 * cand["rating_norm"] +
        0.1 * (1 - cand["harga_norm"])
    )
    cand["final_score"] = (cand["final_score"] * 100).round(1)

    return cand.sort_values("final_score", ascending=False).head(top_n)


# ══════════════════════════════════════════════════════════════════════════════
#  KOMPONEN UI
# ══════════════════════════════════════════════════════════════════════════════
def render_card(row, cols_per_row=3):
    """Render satu kartu wisata (HTML)."""
    img = get_img(row["gambar"])
    score = row.get("final_score", 0)
    score_html = f"<div class='card-score-badge'>{score:.0f} pts</div>" if score > 0 else ""
    desc50 = short_desc(row["deskripsi"])
    harga_fmt = fmt_rupiah(row["harga"])

    html = f"""
    <div class='card'>
      <div class='card-img-wrap'>
        <img src='{img}' onerror="this.src='{PLACEHOLDER}'" loading='lazy'/>
        <div class='card-kat-badge'>{row['kategori']}</div>
        {score_html}
      </div>
      <div class='card-body'>
        <div class='card-nama'>{row['nama_wisata']}</div>
        <div class='card-meta'>
          <span class='badge badge-kota'>📍 {row['kota']}</span>
          <span class='badge badge-rating'>⭐ {float(row['rating']):.1f}</span>
          <span class='badge badge-harga'>💰 {harga_fmt}</span>
        </div>
        <div class='card-desc'>{desc50}</div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    if st.button("Lihat Detail →", key=f"btn_detail_{row['place_id']}_{score}",
                 use_container_width=True):
        st.session_state.detail_id = int(row["place_id"])
        st.session_state.view = "detail"
        st.rerun()


def render_grid(hasil: pd.DataFrame, n_cols=3):
    """Grid kartu rekomendasi."""
    rows_data = [hasil.iloc[i:i+n_cols] for i in range(0, len(hasil), n_cols)]
    for row_group in rows_data:
        cols = st.columns(n_cols)
        for col, (_, rec) in zip(cols, row_group.iterrows()):
            with col:
                render_card(rec, n_cols)


def render_detail(place_id: int, show_back=True):
    """Halaman detail destinasi wisata."""
    row_df = df[df["place_id"] == place_id]
    if row_df.empty:
        st.error("Data tidak ditemukan.")
        return
    r = row_df.iloc[0]

    if show_back:
        if st.button("← Kembali", key="back_btn"):
            st.session_state.view = "home" if st.session_state.hasil_df is None else "hasil"
            st.rerun()

    # ── Gambar & nama ──────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='detail-wrap'>
      <img class='detail-img' src='{get_img(r["gambar"])}'
           onerror="this.src='{PLACEHOLDER}'"/>
      <div class='detail-body'>
        <div class='sec-label'>{r['kategori']} · {r['kota']}</div>
        <div class='detail-nama'>{r['nama_wisata']}</div>
        <div class='detail-badges'>
          <span class='badge badge-rating' style='font-size:.9rem;padding:5px 14px;'>
            ⭐ {float(r['rating']):.1f}
          </span>
          <span class='badge badge-harga' style='font-size:.9rem;padding:5px 14px;'>
            💰 {fmt_rupiah(r['harga'])}
          </span>
          <span class='badge badge-kota' style='font-size:.9rem;padding:5px 14px;'>
            📍 {r['kota']}
          </span>
        </div>
        <div class='info-grid'>
          <div class='info-box'>
            <div class='info-box-label'>Kategori</div>
            <div class='info-box-value'>{r['kategori']}</div>
          </div>
          <div class='info-box'>
            <div class='info-box-label'>Harga Tiket</div>
            <div class='info-box-value'>{fmt_rupiah(r['harga'])}</div>
          </div>
          <div class='info-box'>
            <div class='info-box-label'>Rating</div>
            <div class='info-box-value'>{float(r['rating']):.1f} ⭐ <small style='color:#6a9070;font-size:.75rem;'>{stars(r['rating'])}</small></div>
          </div>
          <div class='info-box'>
            <div class='info-box-label'>Koordinat</div>
            <div class='info-box-value' style='font-size:.85rem;'>{float(r['lat']):.5f}, {float(r['long']):.5f}</div>
          </div>
        </div>
        <div class='detail-desc'>{r['deskripsi'] if str(r['deskripsi']).strip() not in ['BELUM DIISI','','nan'] else 'Deskripsi belum tersedia.'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs: Peta + Serupa ────────────────────────────────────────────────────
    tab_map, tab_sim = st.tabs(["🗺️ Peta Lokasi", "🔗 Wisata Serupa"])

    with tab_map:
        m = folium.Map(location=[float(r["lat"]), float(r["long"])], zoom_start=15)
        folium.Marker(
            [float(r["lat"]), float(r["long"])],
            popup=folium.Popup(
                f"<b>{r['nama_wisata']}</b><br>⭐ {float(r['rating']):.1f} · {fmt_rupiah(r['harga'])}",
                max_width=240
            ),
            tooltip=r["nama_wisata"],
            icon=folium.Icon(color="green", icon="map-marker"),
        ).add_to(m)
        st_folium(m, height=360, key=f"peta_detail_{place_id}", returned_objects=[])

    with tab_sim:
        sim_df = get_rekomendasi(ref_id=place_id, top_n=5, exclude_id=place_id)
        if sim_df.empty:
            st.markdown("<div class='empty-state'><div class='empty-icon'>🔍</div>Tidak ada wisata serupa ditemukan.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown("<p style='color:#6a9070;font-size:.85rem;margin-bottom:.8rem;'>5 destinasi paling mirip berdasarkan konten</p>",
                        unsafe_allow_html=True)
            render_grid(sim_df, n_cols=5)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — PREFERENSI
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:1.2rem 0 .8rem;'>
      <div style='font-family:Lora,serif;font-size:1.35rem;font-weight:600;color:#7edaa0;line-height:1.2;'>
        🌿 WisataJawa
      </div>
      <div style='font-size:.76rem;color:#6a9070;margin-top:3px;'>Sistem Rekomendasi Wisata</div>
    </div>
    <hr style='border-color:#1a3d25;margin:.5rem 0 1.2rem;'/>
    """, unsafe_allow_html=True)

    # Navigasi
    nav = st.radio("Menu", ["🏠 Beranda & Peta", "🔍 Cari Rekomendasi"],
                   label_visibility="collapsed")

    st.markdown("<hr style='border-color:#1a3d25;margin:.8rem 0;'/>", unsafe_allow_html=True)

    # ── Preferensi Rekomendasi ──────────────────────────────────────────────
    st.markdown("<p style='font-size:.8rem;font-weight:700;color:#7edaa0;letter-spacing:.05em;text-transform:uppercase;'>Preferensi</p>",
                unsafe_allow_html=True)

    semua_kat   = sorted(df["kategori"].unique().tolist())
    semua_kota  = sorted(df["kota"].unique().tolist())

    ref_options = ["— Tidak Ada —"] + sorted(df["nama_wisata"].tolist())
    ref_name = st.selectbox("🏛️ Wisata Referensi", ref_options,
                            help="Cari wisata serupa dengan tempat ini")

    pilih_kat  = st.multiselect("🎯 Kategori", semua_kat,
                                placeholder="Semua kategori")
    pilih_kota = st.multiselect("🏙️ Kota", semua_kota,
                                placeholder="Semua kota")

    harga_max  = st.slider("💰 Harga Maksimal (Rp)", 0,
                           int(df["harga"].max()), int(df["harga"].max()),
                           step=10_000, format="Rp %d")

    min_rat    = st.slider("⭐ Rating Minimum", 1.0, 5.0, 1.0, 0.1)
    top_n      = st.slider("📋 Jumlah Rekomendasi", 5, 20, 10)

    st.markdown("<div style='height:.6rem'/>", unsafe_allow_html=True)
    cari_btn = st.button("🔍 Cari Rekomendasi", use_container_width=True)

    st.markdown("<hr style='border-color:#1a3d25;margin:.8rem 0;'/>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:.72rem;color:#4a7050;text-align:center;'>{len(df)} destinasi wisata · Pulau Jawa</p>",
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PROSES KLIK REKOMENDASI
# ══════════════════════════════════════════════════════════════════════════════
if cari_btn:
    ref_id = None
    if ref_name != "— Tidak Ada —":
        match = df[df["nama_wisata"] == ref_name]
        if not match.empty:
            ref_id = int(match.iloc[0]["place_id"])

    hasil = get_rekomendasi(
        ref_id=ref_id,
        kategori_list=pilih_kat if pilih_kat else None,
        kota_list=pilih_kota if pilih_kota else None,
        max_harga=harga_max,
        min_rating=min_rat,
        top_n=top_n,
        exclude_id=ref_id,
    )
    st.session_state.hasil_df = hasil
    st.session_state.view = "hasil"
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING VIEW
# ══════════════════════════════════════════════════════════════════════════════

# ── DETAIL VIEW ───────────────────────────────────────────────────────────────
if st.session_state.view == "detail" and st.session_state.detail_id:
    render_detail(st.session_state.detail_id)
    st.stop()

# ── HASIL VIEW ────────────────────────────────────────────────────────────────
if st.session_state.view == "hasil" and st.session_state.hasil_df is not None:
    hasil = st.session_state.hasil_df

    # Hero kecil
    st.markdown(f"""
    <div class='hero-wrap' style='padding:1.8rem 2.5rem;'>
      <div style='display:flex;align-items:center;gap:1.5rem;flex-wrap:wrap;'>
        <div>
          <div class='sec-label' style='color:#7edaa0;'>Hasil Rekomendasi</div>
          <div class='hero-title' style='font-size:1.8rem;'>
            Ditemukan <span>{len(hasil)}</span> Destinasi
          </div>
          <div class='hero-sub'>Diurutkan berdasarkan skor kesesuaian</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Kembali ke Beranda", key="back_home"):
        st.session_state.view = "home"
        st.session_state.hasil_df = None
        st.rerun()

    if hasil.empty:
        st.markdown("""
        <div class='empty-state'>
          <div class='empty-icon'>😕</div>
          <b>Tidak ada destinasi yang cocok.</b><br>
          Coba longgarkan filter kategori, kota, atau budget.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Peta hasil ──────────────────────────────────────────────────────────────
    with st.expander("🗺️ Lihat Semua Lokasi di Peta", expanded=False):
        m_hasil = folium.Map(
            location=[hasil["lat"].mean(), hasil["long"].mean()],
            zoom_start=8
        )
        colors = ["green","blue","purple","orange","red","darkgreen","cadetblue",
                  "darkblue","lightred","pink","darkred","lightblue","lightgreen"]
        for i, (_, rec) in enumerate(hasil.iterrows()):
            folium.Marker(
                [float(rec["lat"]), float(rec["long"])],
                popup=folium.Popup(
                    f"<b>#{i+1} {rec['nama_wisata']}</b><br>"
                    f"{rec['kota']} · {rec['kategori']}<br>"
                    f"⭐ {float(rec['rating']):.1f} · {fmt_rupiah(rec['harga'])}<br>"
                    f"Skor: {rec['final_score']:.1f}",
                    max_width=250
                ),
                tooltip=f"#{i+1} {rec['nama_wisata']}",
                icon=folium.Icon(color=colors[i % len(colors)], icon="map-marker"),
            ).add_to(m_hasil)
        st_folium(m_hasil, height=400, key="peta_hasil", returned_objects=[])

    st.markdown("<div class='divider'/>", unsafe_allow_html=True)

    # ── Grid kartu ──────────────────────────────────────────────────────────────
    render_grid(hasil, n_cols=3)

    # ── Tabel ringkasan ─────────────────────────────────────────────────────────
    st.markdown("<div class='divider'/>", unsafe_allow_html=True)
    with st.expander("📊 Tabel Ringkasan", expanded=False):
        tbl = hasil[["nama_wisata","kategori","kota","harga","rating","final_score"]].copy()
        tbl.index = range(1, len(tbl)+1)
        tbl.columns = ["Nama","Kategori","Kota","Harga (Rp)","Rating","Skor"]
        st.dataframe(tbl, use_container_width=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
#  HOME VIEW
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='hero-wrap'>
  <div class='hero-title'>
    Temukan <span>Destinasi Wisata</span><br>Pulau Jawa Impianmu
  </div>
  <div class='hero-sub'>
    Content-Based Filtering · {len(df):,} destinasi di seluruh Pulau Jawa
  </div>
  <div class='hero-stats'>
    <div class='hstat'>
      <div class='hstat-num'>{len(df):,}</div>
      <div class='hstat-label'>Destinasi</div>
    </div>
    <div class='hstat'>
      <div class='hstat-num'>{df['kota'].nunique()}</div>
      <div class='hstat-label'>Kota</div>
    </div>
    <div class='hstat'>
      <div class='hstat-num'>{df['kategori'].nunique()}</div>
      <div class='hstat-label'>Kategori</div>
    </div>
    <div class='hstat'>
      <div class='hstat-num'>{df['rating'].mean():.1f}⭐</div>
      <div class='hstat-label'>Rata-rata Rating</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Search Bar ────────────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Cari Destinasi</div>", unsafe_allow_html=True)
col_search, col_go = st.columns([5, 1])
with col_search:
    cari_nama = st.selectbox(
        "Ketik nama tempat wisata…",
        options=[""] + sorted(df["nama_wisata"].tolist()),
        label_visibility="collapsed",
        key="search_box",
    )
with col_go:
    go_btn = st.button("Lihat Detail", type="primary", use_container_width=True)

if go_btn and cari_nama:
    match = df[df["nama_wisata"] == cari_nama]
    if not match.empty:
        st.session_state.detail_id = int(match.iloc[0]["place_id"])
        st.session_state.view = "detail"
        st.rerun()

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Peta Eksplorasi ───────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Peta Interaktif</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-title'>Eksplorasi Semua Destinasi</div>", unsafe_allow_html=True)
st.caption("Klik titik pada peta untuk melihat lokasi destinasi. Klik nama pada popup untuk membuka detail.")

col_peta, col_info = st.columns([3, 1])

with col_peta:
    lat_c = df["lat"].mean(); lon_c = df["long"].mean()
    m_home = folium.Map(location=[lat_c, lon_c], zoom_start=7,
                        tiles="CartoDB Positron")

    kat_color = {
        "Budaya": "orange", "Taman Hiburan": "green", "Cagar Alam": "darkgreen",
        "Bahari": "blue", "Pusat Perbelanjaan": "purple", "Tempat Ibadah": "red",
    }
    mc = MarkerCluster(
        options={"maxClusterRadius": 40, "spiderfyOnMaxZoom": True}
    ).add_to(m_home)

    for _, r in df.iterrows():
        warna = kat_color.get(r["kategori"], "gray")
        pid = int(r["place_id"])
        nm  = r["nama_wisata"].replace("'", "&#39;")
        popup_html = (
            "<b style='font-size:1rem;'>" + r["nama_wisata"] + "</b><br>"
            "<span style='color:#6a9070;'>" + r["kategori"] + " · " + r["kota"] + "</span><br>"
            "⭐ " + str(round(float(r["rating"]),1)) + " | 💰 " + fmt_rupiah(r["harga"])
        )
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["long"])],
            radius=5, color=warna, fill=True,
            fill_color=warna, fill_opacity=0.75, weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{r['nama_wisata']} ⭐{float(r['rating']):.1f}",
        ).add_to(mc)

    # Legenda warna
    legend_html = """
    <div style='position:fixed;bottom:20px;right:20px;background:white;
         border-radius:12px;padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,.15);
         font-size:11px;z-index:1000;'>
      <b style='display:block;margin-bottom:6px;color:#0f2d1a;'>Kategori</b>
      <span style='color:orange;'>●</span> Budaya &nbsp;
      <span style='color:green;'>●</span> Taman Hiburan<br>
      <span style='color:darkgreen;'>●</span> Cagar Alam &nbsp;
      <span style='color:blue;'>●</span> Bahari<br>
      <span style='color:purple;'>●</span> Perbelanjaan &nbsp;
      <span style='color:red;'>●</span> Tempat Ibadah
    </div>
    """
    m_home.get_root().html.add_child(folium.Element(legend_html))

    hasil_peta = st_folium(m_home, height=500, key="peta_home",
                           returned_objects=["last_clicked"])

with col_info:
    st.markdown("<div class='sec-label'>Distribusi</div>", unsafe_allow_html=True)

    # Distribusi kategori
    for kat, cnt in df["kategori"].value_counts().items():
        pct = cnt / len(df) * 100
        warna_css = {
            "Budaya":"#e88a00","Taman Hiburan":"#2d9e5a","Cagar Alam":"#1a5c33",
            "Bahari":"#1e6fa8","Pusat Perbelanjaan":"#7c4dbd","Tempat Ibadah":"#c0392b",
        }.get(kat,"#888")
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
          <div style='width:10px;height:10px;border-radius:50%;background:{warna_css};flex-shrink:0;'></div>
          <div style='flex:1;font-size:.78rem;color:#3d5c42;'>{kat}</div>
          <div style='font-size:.78rem;font-weight:700;color:#0f2d1a;'>{cnt}</div>
        </div>
        <div style='height:4px;background:#edf4ee;border-radius:4px;margin-bottom:10px;'>
          <div style='width:{pct:.0f}%;height:100%;background:{warna_css};border-radius:4px;'></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:.5rem'/>", unsafe_allow_html=True)
    st.markdown("<div class='sec-label'>Top Kota</div>", unsafe_allow_html=True)
    for kota, cnt in df["kota"].value_counts().head(6).items():
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;font-size:.8rem;
             padding:4px 0;border-bottom:1px solid #edf4ee;'>
          <span style='color:#3d5c42;'>📍 {kota}</span>
          <span style='font-weight:700;color:#0f2d1a;'>{cnt}</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Destinasi Populer (Home default cards) ────────────────────────────────────
st.markdown("<div class='sec-label'>Populer</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-title'>Destinasi Terpopuler</div>", unsafe_allow_html=True)

populer = df.sort_values("rating", ascending=False).head(9).copy()
populer["final_score"] = 0.0
render_grid(populer, n_cols=3)

st.markdown("<div class='divider'/>", unsafe_allow_html=True)

# ── Per Kategori ──────────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>Jelajahi</div>", unsafe_allow_html=True)
st.markdown("<div class='sec-title'>Jelajahi per Kategori</div>", unsafe_allow_html=True)

tab_labels = ["🏛️ Budaya", "🎢 Taman Hiburan", "🌿 Cagar Alam",
              "🌊 Bahari", "🛍️ Perbelanjaan", "🕌 Tempat Ibadah"]
kat_keys   = ["Budaya", "Taman Hiburan", "Cagar Alam",
              "Bahari", "Pusat Perbelanjaan", "Tempat Ibadah"]

tabs = st.tabs(tab_labels)
for tab, kat in zip(tabs, kat_keys):
    with tab:
        sub = df[df["kategori"] == kat].sort_values("rating", ascending=False).head(6).copy()
        sub["final_score"] = 0.0
        if sub.empty:
            st.markdown("<div class='empty-state'><div class='empty-icon'>🏝️</div>Data belum tersedia.</div>",
                        unsafe_allow_html=True)
        else:
            render_grid(sub, n_cols=3)

