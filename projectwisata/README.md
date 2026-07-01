# Sistem Rekomendasi Destinasi Wisata di Indonesia

Aplikasi ini merupakan implementasi tahap **Modeling** dan **Deployment** dari skripsi:

**Sistem Rekomendasi Destinasi Wisata di Indonesia Menggunakan Content-Based Filtering**

Aplikasi dibuat menggunakan **Streamlit**, **TF-IDF**, **Cosine Similarity**, **Formula Haversine**, dan **Context-Aware Re-ranking**.

## Struktur Folder

```text
.
├── app.py
├── requirements.txt
├── data/
│   └── DATASET_WISATA_READY_MODELING_FINAL.csv
├── koleksi_gambar/
│   └── folder gambar lokal sesuai path pada dataset
└── .streamlit/
    └── config.toml
```

## Input Aplikasi

Aplikasi menyediakan input berikut:

1. Wisata referensi
2. Kota tujuan
3. Kategori wisata
4. Kata kunci pencarian
5. Budget maksimal
6. Rating minimum
7. Jumlah rekomendasi
8. Penggunaan lokasi pengguna
9. Metode input lokasi
   - Pilih kota asal
   - Input koordinat manual
   - Klik lokasi pada peta
10. Radius pencarian

## Alur Sistem

1. Aplikasi membaca dataset akhir yang sudah siap modeling.
2. Sistem menggunakan kolom `fitur_gabungan` sebagai input model.
3. TF-IDF mengubah teks destinasi menjadi vektor numerik.
4. Cosine Similarity menghitung kemiripan antar destinasi wisata.
5. Sistem mengambil kandidat berdasarkan wisata referensi.
6. Kandidat difilter berdasarkan kota, kategori, budget, rating, kata kunci, dan radius.
7. Jika lokasi pengguna aktif, sistem menghitung jarak menggunakan Formula Haversine.
8. Sistem menghitung:
   - Similarity Score
   - Distance Score
   - Rating Score
   - Final Score
9. Sistem mengurutkan destinasi berdasarkan Final Score.
10. Sistem menampilkan Top-N Recommendation dan peta Folium.

## Skenario Weighted Scoring

| Kondisi Input | Rumus |
|---|---|
| Ada wisata referensi dan lokasi aktif | 0.5 Similarity + 0.3 Distance + 0.2 Rating |
| Ada wisata referensi dan lokasi tidak aktif | 0.7 Similarity + 0.3 Rating |
| Tidak ada wisata referensi dan lokasi aktif | 0.6 Distance + 0.4 Rating |
| Tidak ada wisata referensi dan lokasi tidak aktif | Rating Score |

## Cara Menjalankan di Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cara Deploy melalui GitHub dan Streamlit Community Cloud

1. Buat repository baru di GitHub.
2. Upload semua file dan folder berikut:
   - `app.py`
   - `requirements.txt`
   - folder `data`
   - folder `.streamlit`
   - folder `koleksi_gambar` jika memakai gambar lokal
3. Pastikan nama dataset berada di:
   ```text
   data/DATASET_WISATA_READY_MODELING_FINAL.csv
   ```
4. Buka Streamlit Community Cloud.
5. Pilih repository GitHub.
6. Atur main file path:
   ```text
   app.py
   ```
7. Deploy aplikasi.

## Catatan Gambar Lokal

Kolom `gambar` pada dataset mendukung dua format:

1. URL online:
   ```text
   https://contoh.com/gambar.jpg
   ```

2. Path lokal:
   ```text
   koleksi_gambar/Nama Wisata/Image_1.jpg
   ```

Jika memakai path lokal, pastikan folder dan nama file di GitHub sama persis dengan isi kolom `gambar`.
