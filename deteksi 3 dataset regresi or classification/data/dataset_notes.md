# Dataset Notes — Identifikasi Tipe Tugas

Pull date: 2026-11-08
Source: Kaggle API (anonymous download)

| # | Dataset | File | Baris | Kolom | Target | Tipe tugas |
|---|---------|------|-------|-------|--------|-----------|
| 1 | Telco Customer Churn | `telco/WA_Fn-UseC_-Telco-Customer-Churn.csv` | 7.043 | 21 | `Churn` | Binary classification |
| 2 | Wine Quality | `wine/WineQT.csv` | 1.143 | 13 | `quality` | Multiclass classification (opsional ordinal regression) |
| 3 | Medical Cost (Insurance) | `insurance/insurance.csv` | 1.338 | 7 | `charges` | Regression |

## 1. Telco — Binary classification

- `Churn`: `No` = 5.174 (73,5%), `Yes` = 1.869 (26,5%). 2 nilai unik, kategorikal.
- 11 baris `TotalCharges` kosong (string kosong, bukan NaN — dtype terbaca `str` di pandas 3.x). Setelah `pd.to_numeric(errors='coerce')` jadi NaN.
- Cek isi: **semua 11 baris itu punya `tenure = 0`** → pelanggan baru yang belum pernah ditagih. Isi `0`, jangan drop, jangan pakai mean.
- Fitur campuran: kategorikal (gender, Contract, PaymentMethod, dll) + numerik (tenure, MonthlyCharges, TotalCharges, SeniorCitizen).
- Metrik: ROC-AUC, F1 kelas `Yes`. Accuracy menyesatkan karena imbalance 73/27.

## 2. Wine Quality — Multiclass classification

- `quality`: 6 nilai unik — 3 (6), 4 (33), 5 (483), 6 (462), 7 (143), 8 (16).
- **Sangat imbalanced**: kelas 5 dan 6 mendominasi (82,7%), kelas 3 hanya 6 sampel (0,5%).
- Kolom `Id` = identifier, harus dibuang.
- Default: multiclass classification.
- Alternatif: **ordinal regression** (karena 3 < 4 < ... < 8 punya urutan), atau regresi biasa kalau target diperlakukan kontinu.
- Metrik: macro-F1 (bukan accuracy), confusion matrix. Kalau ordinal: MAE / QWK.

## 3. Medical Cost (Insurance) — Regression

- 1.338 baris × 7 kolom, tanpa missing value.
- Fitur: `age`, `bmi`, `children` (numerik); `sex`, `smoker`, `region` (kategorikal).
- Target `charges`: 1.337 nilai unik dari 1.338 baris → kontinu, bukan kelas.
- Statistik: min 1.121,87 | median 9.382,03 | mean 13.270,42 | max 63.770,43 | std 12.110,01.
- **Skew 1,516** (ekor kanan) → pertimbangkan `log1p(charges)` sebelum training.
- `smoker` adalah prediktor terkuat: mean 32.050 (smoker) vs 8.434 (non-smoker).
- Metrik: RMSE, MAE, R². Bukan akurasi.

## Cara menentukan regresi vs klasifikasi (dipakai di sini)

1. Lihat dtype target: object/category/bool → klasifikasi.
2. Numerik + 2 nilai unik → binary classification.
3. Numerik + <= ~15 nilai unik dan semuanya bilangan bulat → multiclass (cek juga ordinal).
4. Numerik kontinu, banyak nilai unik → regresi.
5. Beberapa kolom 0/1 yang bisa aktif bersamaan → multilabel classification.

## Catatan environment

- Python 3.14.4 di venv `.venv` (pandas 3.1.1, scikit-learn 1.9.1, numpy 2.5.4, matplotlib 3.11.2, seaborn 0.13.2).
- `requirements.txt` sudah pinned.
- Unduhan asli: `data/raw/*.zip`, hasil ekstrak: `data/raw/<nama>/`.
