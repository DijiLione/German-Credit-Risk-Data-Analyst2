# Identifikasi Tipe Tugas — Regresi vs Klasifikasi

Digenerate oleh `scripts/detect_task.py` (aturan deterministik, tanpa LLM).

| Dataset | Tipe | Subtipe | Keyakinan | Bukti utama |
|---------|------|---------|-----------|-------------|
| Telco Customer Churn | classification | binary | high | target bertipe str (kategorikal/teks), bukan besaran kontinu |
| Wine Quality | classification | multiclass | high | target numerik bulat dengan 6 nilai unik (<= 15) |
| Medical Cost (Insurance) | regression | continuous | high | target numerik kontinu: 1337 nilai unik dari 1338 baris |

## Telco Customer Churn

**Verdict: classification / binary**

Alasan:
- target bertipe str (kategorikal/teks), bukan besaran kontinu

Distribusi target: `{"No": 5174, "Yes": 1869}`

Loss: `BinaryCrossentropy / log loss`
Metrik: ROC-AUC, PR-AUC, F1 (kelas positif), confusion matrix
Hindari: accuracy

## Wine Quality

**Verdict: classification / multiclass**

Alasan:
- target numerik bulat dengan 6 nilai unik (<= 15)
- nilai [3, 4, 5, 6, 7, 8] berurutan/kontigu -> label punya urutan, ordinal regression layak dipertimbangkan
- kelas minoritas hanya 0.5% -> pakai class_weight / resampling

Distribusi target: `{"3": 6, "4": 33, "5": 483, "6": 462, "7": 143, "8": 16}`

Statistik target: `{"min": 3.0, "max": 8.0, "mean": 5.657, "median": 6.0, "std": 0.8058, "skew": 0.2868}`

Loss: `CategoricalCrossentropy / multinomial log loss`
Metrik: macro-F1, weighted-F1, confusion matrix
Hindari: accuracy
Alternatif: **ordinal regression**

## Medical Cost (Insurance)

**Verdict: regression / continuous**

Alasan:
- target numerik kontinu: 1337 nilai unik dari 1338 baris
- distribusi target miring (skew=1.52)

Statistik target: `{"min": 1121.8739, "max": 63770.42801, "mean": 13270.4223, "median": 9382.033, "std": 12110.0112, "skew": 1.5159}`

Loss: `MSE / MAE / Huber`
Metrik: RMSE, MAE, R2
Hindari: akurasi/klasifikasi

## Kesimpulan

- Telco -> binary classification (target kategorikal 2 kelas).
- Wine -> multiclass classification; alternatif ordinal regression karena label berurutan.
- Medical Cost (Insurance) -> regression murni (target `charges` kontinu, 1.337 nilai unik).
- Komposisi akhir: 2 klasifikasi + 1 regresi.
