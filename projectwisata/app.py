import html
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_folium import st_folium


# =========================================================
# KONFIGURASI DASAR
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATA_PATH_CANDIDATES = [
    BASE_DIR / "data" / "DATASET_WISATA_READY_MODELING_FINAL.csv",
    BASE_DIR / "DATASET_WISATA_READY_MODELING_FINAL.csv",
]

NO_REFERENCE = "Tanpa wisata referensi"
ALL_CITY = "Semua Kota"
ALL_CATEGORY = "Semua Kategori"

REQUIRED_COLUMNS = [
    "place_id",
    "nama_wisata",
    "deskripsi",
    "kategori",
    "kota",
    "harga",
    "rating",
    "lat",
    "long",
    "gambar",
    "fitur_gabungan",
]

CITY_COORDINATES = {
    "Jakarta": (-6.200000, 106.816666),
    "Bandung": (-6.917464, 107.619125),
    "Yogyakarta": (-7.795580, 110.369490),
    "Semarang": (-6.966667, 110.416664),
    "Surabaya": (-7.257472, 112.752090),
    "Bogor": (-6.597147, 106.806039),
    "Malang": (-7.966620, 112.632629),
    "Surakarta": (-7.575488, 110.824327),
}


st.set_page_config(
    page_title="Sistem Rekomendasi Wisata Indonesia",
    page_icon="🧭",
    layout="wide",
)


# =========================================================
# LOAD DATA DAN MODEL
# =========================================================
def find_dataset_path() -> Path:
    for path in DATA_PATH_CANDIDATES:
        if path.exists():
            return path
    return DATA_PATH_CANDIDATES[0]


@st.cache_data(show_spinner=False)
def load_dataset(path_string: str) -> pd.DataFrame:
    data = pd.read_csv(path_string)
    data = data.reset_index(drop=True)
    return data


@st.cache_resource(show_spinner=False)
def build_tfidf_model(text_data: tuple[str, ...]):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(text_data)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return vectorizer, tfidf_matrix, similarity_matrix


# =========================================================
# FUNGSI PERHITUNGAN
# =========================================================
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Menghitung jarak dua titik koordinat dalam kilometer.
    Formula ini sesuai dengan tahap Context-Aware Re-ranking pada Bab 3.
    """
    earth_radius = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return earth_radius * c


def format_rupiah(value):
    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return "Rp 0"
    return f"Rp {value:,.0f}".replace(",", ".")


def build_reference_labels(data: pd.DataFrame):
    option_map = {}
    labels = []
    for _, row in data.iterrows():
        label = (
            f"{row['nama_wisata']} | {row['kota']} | "
            f"{row['kategori']} | ID {row['place_id']}"
        )
        labels.append(label)
        option_map[label] = row["place_id"]
    return labels, option_map


def resolve_image_source(image_value):
    """
    Mendukung dua jenis gambar:
    1. URL online, misalnya https://...
    2. File lokal, misalnya koleksi_gambar/Nama Wisata/Image_1.jpg
    """
    if image_value is None:
        return None

    image_text = str(image_value).strip()

    if not image_text or image_text.upper() in {"BELUM_DIISI", "NAN", "NONE", "-"}:
        return None

    if image_text.startswith(("http://", "https://")):
        return image_text

    possible_paths = [
        BASE_DIR / image_text,
        BASE_DIR / "koleksi_gambar" / image_text,
        BASE_DIR / "images" / image_text,
        BASE_DIR / "assets" / image_text,
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def create_recommendation(
    data: pd.DataFrame,
    similarity_matrix: np.ndarray,
    reference_place_id,
    selected_city: str,
    selected_category: str,
    keyword: str,
    max_budget: int,
    min_rating: float,
    top_n: int,
    use_location: bool,
    user_lat: float | None,
    user_long: float | None,
    radius_km: float,
) -> pd.DataFrame:
    result = data.copy()

    has_reference = reference_place_id is not None

    if has_reference:
        reference_rows = data.index[data["place_id"] == reference_place_id].tolist()

        if not reference_rows:
            return pd.DataFrame()

        reference_index = reference_rows[0]
        result["similarity_score"] = similarity_matrix[reference_index]
        result = result[result.index != reference_index]
    else:
        result["similarity_score"] = 0.0

    if selected_city != ALL_CITY:
        result = result[result["kota"] == selected_city]

    if selected_category != ALL_CATEGORY:
        result = result[result["kategori"] == selected_category]

    if keyword.strip():
        keyword_lower = keyword.strip().lower()
        result = result[
            result["nama_wisata"].astype(str).str.lower().str.contains(keyword_lower, na=False)
            | result["deskripsi"].astype(str).str.lower().str.contains(keyword_lower, na=False)
            | result["kategori"].astype(str).str.lower().str.contains(keyword_lower, na=False)
            | result["kota"].astype(str).str.lower().str.contains(keyword_lower, na=False)
        ]

    result = result[result["harga"] <= max_budget]
    result = result[result["rating"] >= min_rating]

    max_rating = float(data["rating"].max())
    if max_rating <= 0:
        max_rating = 5.0

    result["rating_score"] = result["rating"] / max_rating

    has_location = (
        use_location
        and user_lat is not None
        and user_long is not None
        and not math.isnan(float(user_lat))
        and not math.isnan(float(user_long))
        and radius_km > 0
    )

    if has_location:
        result["distance_km"] = haversine_distance(
            float(user_lat),
            float(user_long),
            result["lat"].astype(float),
            result["long"].astype(float),
        )

        result = result[result["distance_km"] <= radius_km]
        result["distance_score"] = 1 - (result["distance_km"] / radius_km)
        result["distance_score"] = result["distance_score"].clip(lower=0, upper=1)
    else:
        result["distance_km"] = np.nan
        result["distance_score"] = 0.0

    if result.empty:
        return result

    if has_reference and has_location:
        result["final_score"] = (
            0.5 * result["similarity_score"]
            + 0.3 * result["distance_score"]
            + 0.2 * result["rating_score"]
        )
        scenario = "Content-Based Filtering + Context-Aware Re-ranking"
    elif has_reference and not has_location:
        result["final_score"] = (
            0.7 * result["similarity_score"]
            + 0.3 * result["rating_score"]
        )
        scenario = "Content-Based Filtering + Rating Re-ranking"
    elif not has_reference and has_location:
        result["final_score"] = (
            0.6 * result["distance_score"]
            + 0.4 * result["rating_score"]
        )
        scenario = "Context-Aware Re-ranking berbasis lokasi dan rating"
    else:
        result["final_score"] = result["rating_score"]
        scenario = "Rekomendasi berbasis rating"

    result["scenario"] = scenario
    result["skor_rekomendasi"] = (result["final_score"] * 100).round(2)

    result = result.sort_values(
        by=["final_score", "rating"],
        ascending=[False, False],
    ).head(top_n)

    return result


def show_recommendation_card(row, rank: int, use_location: bool):
    with st.container(border=True):
        st.subheader(f"{rank}. {row['nama_wisata']}")

        image_col, info_col = st.columns([1, 2], vertical_alignment="top")

        with image_col:
            image_source = resolve_image_source(row.get("gambar"))
            if image_source:
                st.image(image_source, use_container_width=True)
            else:
                st.info("Gambar tidak tersedia.")

        with info_col:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Skor", f"{row['skor_rekomendasi']:.2f}")
            metric_cols[1].metric("Rating", f"{row['rating']:.1f}")
            metric_cols[2].metric("Harga", format_rupiah(row["harga"]))

            if use_location and pd.notna(row.get("distance_km")):
                metric_cols[3].metric("Jarak", f"{row['distance_km']:.2f} km")
            else:
                metric_cols[3].metric("Kota", str(row["kota"]))

            st.markdown(
                f"""
                **Kategori:** {row['kategori']}  
                **Kota:** {row['kota']}  
                **Similarity Score:** {row['similarity_score']:.4f}  
                **Distance Score:** {row['distance_score']:.4f}  
                **Rating Score:** {row['rating_score']:.4f}
                """
            )

            with st.expander("Lihat deskripsi"):
                st.write(row["deskripsi"])


def create_map(result: pd.DataFrame, use_location: bool, user_lat=None, user_long=None):
    if result.empty:
        return None

    if use_location and user_lat is not None and user_long is not None:
        center = [float(user_lat), float(user_long)]
        zoom_start = 11
    else:
        center = [float(result["lat"].mean()), float(result["long"].mean())]
        zoom_start = 8

    recommendation_map = folium.Map(location=center, zoom_start=zoom_start)

    if use_location and user_lat is not None and user_long is not None:
        folium.Marker(
            location=[float(user_lat), float(user_long)],
            popup="Lokasi Pengguna",
            tooltip="Lokasi Pengguna",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(recommendation_map)

    for _, row in result.iterrows():
        popup_html = f"""
        <b>{html.escape(str(row['nama_wisata']))}</b><br>
        Kategori: {html.escape(str(row['kategori']))}<br>
        Kota: {html.escape(str(row['kota']))}<br>
        Rating: {row['rating']}<br>
        Harga: {format_rupiah(row['harga'])}<br>
        Skor: {row['skor_rekomendasi']:.2f}
        """

        if pd.notna(row.get("distance_km")):
            popup_html += f"<br>Jarak: {row['distance_km']:.2f} km"

        folium.Marker(
            location=[float(row["lat"]), float(row["long"])],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=str(row["nama_wisata"]),
            icon=folium.Icon(color="blue", icon="map-marker", prefix="fa"),
        ).add_to(recommendation_map)

    return recommendation_map


# =========================================================
# LOAD APLIKASI
# =========================================================
st.title("🧭 Sistem Rekomendasi Destinasi Wisata di Indonesia")
st.caption(
    "Content-Based Filtering menggunakan TF-IDF dan Cosine Similarity, "
    "dengan Context-Aware Re-ranking berdasarkan lokasi, radius, budget, rating, kategori, dan kota."
)

dataset_path = find_dataset_path()

if not dataset_path.exists():
    st.error(
        "Dataset tidak ditemukan. Letakkan file "
        "`DATASET_WISATA_READY_MODELING_FINAL.csv` di folder `data/`."
    )
    st.stop()

data = load_dataset(str(dataset_path))

missing_columns = [col for col in REQUIRED_COLUMNS if col not in data.columns]
if missing_columns:
    st.error(f"Kolom wajib belum tersedia: {', '.join(missing_columns)}")
    st.stop()

# Dataset sudah siap modeling. Bagian ini hanya memastikan index stabil untuk model.
data = data.reset_index(drop=True)

with st.spinner("Membangun model TF-IDF dan Cosine Similarity..."):
    _, tfidf_matrix, similarity_matrix = build_tfidf_model(
        tuple(data["fitur_gabungan"].astype(str).tolist())
    )


# =========================================================
# SIDEBAR INPUT
# =========================================================
st.sidebar.header("Input Preferensi Pengguna")

reference_labels, reference_map = build_reference_labels(data)
reference_choice = st.sidebar.selectbox(
    "Wisata referensi",
    [NO_REFERENCE] + reference_labels,
    help="Pilih destinasi acuan untuk mencari wisata lain yang mirip.",
)

reference_place_id = None
if reference_choice != NO_REFERENCE:
    reference_place_id = reference_map[reference_choice]

city_options = [ALL_CITY] + sorted(data["kota"].dropna().unique().tolist())
selected_city = st.sidebar.selectbox(
    "Kota tujuan",
    city_options,
    help="Pilih kota tujuan. Gunakan 'Semua Kota' jika tidak ingin membatasi kota.",
)

category_options = [ALL_CATEGORY] + sorted(data["kategori"].dropna().unique().tolist())
selected_category = st.sidebar.selectbox(
    "Kategori wisata",
    category_options,
    help="Pilih kategori wisata. Gunakan 'Semua Kategori' jika tidak ingin membatasi kategori.",
)

keyword = st.sidebar.text_input(
    "Kata kunci pencarian",
    placeholder="Contoh: pantai, museum, taman",
    help="Opsional. Digunakan untuk menyaring nama, deskripsi, kategori, atau kota.",
)

max_price = int(data["harga"].max())
max_budget = st.sidebar.number_input(
    "Budget maksimal",
    min_value=0,
    max_value=max_price,
    value=max_price,
    step=5000,
    help="Sistem hanya menampilkan wisata dengan harga tiket tidak melebihi budget.",
)

min_rating = st.sidebar.slider(
    "Rating minimum",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.1,
    help="Sistem hanya menampilkan wisata dengan rating minimal sesuai input.",
)

top_n = st.sidebar.slider(
    "Jumlah rekomendasi",
    min_value=1,
    max_value=20,
    value=5,
    step=1,
    help="Menentukan jumlah Top-N Recommendation.",
)

st.sidebar.divider()
use_location = st.sidebar.checkbox(
    "Gunakan lokasi pengguna",
    value=False,
    help="Aktifkan untuk menghitung jarak dengan Formula Haversine.",
)

user_lat = None
user_long = None
radius_km = 50.0

if use_location:
    location_method = st.sidebar.radio(
        "Metode input lokasi",
        ["Pilih kota asal", "Input koordinat manual", "Klik lokasi pada peta"],
    )

    radius_km = st.sidebar.slider(
        "Radius pencarian (km)",
        min_value=1,
        max_value=500,
        value=50,
        step=1,
    )

    if location_method == "Pilih kota asal":
        origin_city = st.sidebar.selectbox(
            "Kota asal pengguna",
            list(CITY_COORDINATES.keys()),
        )
        user_lat, user_long = CITY_COORDINATES[origin_city]

    elif location_method == "Input koordinat manual":
        user_lat = st.sidebar.number_input(
            "Latitude pengguna",
            min_value=-11.5,
            max_value=6.5,
            value=-6.200000,
            step=0.000001,
            format="%.6f",
        )
        user_long = st.sidebar.number_input(
            "Longitude pengguna",
            min_value=95.0,
            max_value=141.0,
            value=106.816666,
            step=0.000001,
            format="%.6f",
        )

    else:
        st.info("Klik titik lokasi pengguna pada peta di bawah, lalu tekan tombol rekomendasi di sidebar.")

        if "clicked_lat" not in st.session_state:
            st.session_state.clicked_lat = -6.200000
        if "clicked_long" not in st.session_state:
            st.session_state.clicked_long = 106.816666

        click_map = folium.Map(
            location=[st.session_state.clicked_lat, st.session_state.clicked_long],
            zoom_start=10,
        )
        folium.Marker(
            [st.session_state.clicked_lat, st.session_state.clicked_long],
            tooltip="Lokasi pengguna saat ini",
            icon=folium.Icon(color="red", icon="user", prefix="fa"),
        ).add_to(click_map)

        clicked_data = st_folium(
            click_map,
            height=380,
            width=1200,
            key="location_picker_map",
        )

        if clicked_data and clicked_data.get("last_clicked"):
            st.session_state.clicked_lat = clicked_data["last_clicked"]["lat"]
            st.session_state.clicked_long = clicked_data["last_clicked"]["lng"]

        user_lat = st.session_state.clicked_lat
        user_long = st.session_state.clicked_long

        st.success(f"Lokasi terpilih: {user_lat:.6f}, {user_long:.6f}")

run_button = st.sidebar.button("Cari Rekomendasi", type="primary", use_container_width=True)

if run_button:
    st.session_state["run_recommendation"] = True


# =========================================================
# INFORMASI DATASET
# =========================================================
summary_cols = st.columns(4)
summary_cols[0].metric("Jumlah Destinasi", f"{len(data):,}".replace(",", "."))
summary_cols[1].metric("Jumlah Kota", data["kota"].nunique())
summary_cols[2].metric("Jumlah Kategori", data["kategori"].nunique())
summary_cols[3].metric("Model", "TF-IDF + Cosine")

with st.expander("Lihat alur input dan proses sistem"):
    st.markdown(
        """
        **Alur sistem:**

        1. Aplikasi membaca dataset akhir yang sudah siap modeling.
        2. Sistem menggunakan kolom `fitur_gabungan` sebagai input TF-IDF.
        3. TF-IDF mengubah teks destinasi menjadi representasi numerik.
        4. Cosine Similarity menghitung kemiripan antar destinasi.
        5. Pengguna mengisi wisata referensi, kota, kategori, budget, rating, lokasi, radius, dan jumlah rekomendasi.
        6. Sistem menyaring kandidat berdasarkan preferensi pengguna.
        7. Jika lokasi aktif, sistem menghitung jarak dengan Formula Haversine.
        8. Sistem menghitung `Similarity Score`, `Distance Score`, `Rating Score`, dan `Final Score`.
        9. Sistem menampilkan Top-N Recommendation dan peta lokasi wisata.
        """
    )

with st.expander("Preview dataset"):
    st.dataframe(
        data[
            [
                "nama_wisata",
                "kategori",
                "kota",
                "harga",
                "rating",
                "lat",
                "long",
            ]
        ].head(20),
        use_container_width=True,
    )


# =========================================================
# HASIL REKOMENDASI
# =========================================================
if not st.session_state.get("run_recommendation", False):
    st.info("Isi preferensi di sidebar, lalu klik **Cari Rekomendasi**.")
    st.stop()

recommendations = create_recommendation(
    data=data,
    similarity_matrix=similarity_matrix,
    reference_place_id=reference_place_id,
    selected_city=selected_city,
    selected_category=selected_category,
    keyword=keyword,
    max_budget=max_budget,
    min_rating=min_rating,
    top_n=top_n,
    use_location=use_location,
    user_lat=user_lat,
    user_long=user_long,
    radius_km=radius_km,
)

st.divider()
st.header("Hasil Rekomendasi Wisata")

if recommendations.empty:
    st.warning(
        "Tidak ada destinasi yang memenuhi seluruh preferensi. "
        "Coba naikkan budget, turunkan rating minimum, perluas radius, "
        "atau pilih Semua Kota/Semua Kategori."
    )
    st.stop()

scenario = recommendations["scenario"].iloc[0]
st.success(f"Skenario rekomendasi: {scenario}")

selected_summary = {
    "Wisata referensi": "Tidak digunakan" if reference_choice == NO_REFERENCE else reference_choice,
    "Kota": selected_city,
    "Kategori": selected_category,
    "Budget maksimal": format_rupiah(max_budget),
    "Rating minimum": min_rating,
    "Lokasi aktif": "Ya" if use_location else "Tidak",
    "Radius": f"{radius_km} km" if use_location else "-",
    "Jumlah rekomendasi": top_n,
}

with st.expander("Ringkasan input pengguna", expanded=True):
    st.json(selected_summary)

tab_list, tab_map, tab_table = st.tabs(["Daftar Rekomendasi", "Peta Rekomendasi", "Tabel Skor"])

with tab_list:
    for number, (_, row) in enumerate(recommendations.iterrows(), start=1):
        show_recommendation_card(row, number, use_location)

with tab_map:
    recommendation_map = create_map(recommendations, use_location, user_lat, user_long)
    if recommendation_map is not None:
        st_folium(recommendation_map, height=520, width=1200, key="recommendation_result_map")

with tab_table:
    table_columns = [
        "nama_wisata",
        "kategori",
        "kota",
        "harga",
        "rating",
        "similarity_score",
        "distance_score",
        "rating_score",
        "final_score",
        "skor_rekomendasi",
    ]

    if use_location:
        table_columns.insert(5, "distance_km")

    st.dataframe(
        recommendations[table_columns],
        use_container_width=True,
    )

    csv_output = recommendations[table_columns].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download hasil rekomendasi CSV",
        data=csv_output,
        file_name="hasil_rekomendasi_wisata.csv",
        mime="text/csv",
        use_container_width=True,
    )
