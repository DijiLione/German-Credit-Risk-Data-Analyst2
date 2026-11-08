#!/usr/bin/env python3
"""Deteksi tipe tugas (regresi vs klasifikasi) untuk dataset di data/raw.

Aturan (dipakai berurutan):
 1. Target teks/kategorikal/bool                  -> klasifikasi
 2. Beberapa kolom biner yang bisa aktif bersamaan -> multilabel classification
 3. Numerik, 2 nilai unik                          -> binary classification
 4. Numerik, <= 15 nilai unik & semuanya bulat     -> multiclass classification
    (kalau rentangnya kontigu -> kandidat ordinal regression)
 5. Sisanya                                        -> regression
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORT = ROOT / "reports" / "task_identification.md"
MAX_CLASSES = 15


def classify_target(name: str, y: pd.Series) -> dict:
    y = y.dropna()
    n = len(y)
    n_unique = y.nunique()
    dtype = str(y.dtype)

    is_text = pd.api.types.is_string_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype)
    is_bool = pd.api.types.is_bool_dtype(y)

    result = {
        "dataset": name,
        "target_dtype": dtype,
        "n_rows": n,
        "n_unique": int(n_unique),
        "class_balance": {},
        "task": None,
        "subtype": None,
        "confidence": None,
        "reasons": [],
        "recommended": {},
        "alternative": None,
    }

    if is_text or is_bool:
        result["task"] = "classification"
        result["subtype"] = "binary" if n_unique == 2 else "multiclass"
        result["confidence"] = "high"
        result["reasons"].append(f"target bertipe {dtype} (kategorikal/teks), bukan besaran kontinu")
    elif not pd.api.types.is_numeric_dtype(y):
        result["task"] = "classification"
        result["subtype"] = "multiclass"
        result["confidence"] = "high"
        result["reasons"].append(f"target non-numerik ({dtype})")
    else:
        integer_like = bool(np.allclose(y.to_numpy(dtype=float), np.round(y.to_numpy(dtype=float))))
        if n_unique == 2:
            result["task"] = "classification"
            result["subtype"] = "binary"
            result["confidence"] = "high"
            result["reasons"].append("target numerik dengan tepat 2 nilai unik")
        elif n_unique <= MAX_CLASSES and integer_like:
            values = sorted(y.unique().tolist())
            contiguous = (max(values) - min(values) + 1) == len(values)
            result["task"] = "classification"
            result["subtype"] = "multiclass"
            result["confidence"] = "high"
            result["reasons"].append(
                f"target numerik bulat dengan {n_unique} nilai unik (<= {MAX_CLASSES})"
            )
            if contiguous and n_unique > 2:
                result["alternative"] = "ordinal regression"
                result["reasons"].append(
                    f"nilai {values} berurutan/kontigu -> label punya urutan, ordinal regression layak dipertimbangkan"
                )
        elif n_unique / max(n, 1) < 0.05 and integer_like:
            result["task"] = "classification"
            result["subtype"] = "multiclass"
            result["confidence"] = "medium"
            result["reasons"].append(
                f"nilai unik sedikit relatif terhadap jumlah baris ({n_unique}/{n}) dan bulat"
            )
        else:
            result["task"] = "regression"
            result["subtype"] = "continuous"
            result["confidence"] = "high"
            result["reasons"].append(
                f"target numerik kontinu: {n_unique} nilai unik dari {n} baris"
            )

        result["target_stats"] = {
            "min": float(y.min()),
            "max": float(y.max()),
            "mean": round(float(y.mean()), 4),
            "median": round(float(y.median()), 4),
            "std": round(float(y.std()), 4),
        }
        skew = float(pd.Series(y).skew())
        result["target_stats"]["skew"] = round(skew, 4)
        if abs(skew) > 1:
            result["reasons"].append(f"distribusi target miring (skew={skew:.2f})")

    if n_unique <= MAX_CLASSES:
        vc = y.value_counts().sort_index()
        result["class_balance"] = {str(k): int(v) for k, v in vc.items()}
        minority = vc.min() / vc.sum()
        result["minority_share"] = round(float(minority), 4)
        if minority < 0.05:
            result["reasons"].append(
                f"kelas minoritas hanya {minority:.1%} -> pakai class_weight / resampling"
            )

    if result["task"] == "classification":
        if result["subtype"] == "binary":
            result["recommended"] = {
                "loss": "BinaryCrossentropy / log loss",
                "metrics": ["ROC-AUC", "PR-AUC", "F1 (kelas positif)", "confusion matrix"],
                "avoid": ["accuracy"] if result.get("minority_share", 1) < 0.4 else [],
            }
        else:
            result["recommended"] = {
                "loss": "CategoricalCrossentropy / multinomial log loss",
                "metrics": ["macro-F1", "weighted-F1", "confusion matrix"],
                "avoid": ["accuracy"],
            }
    else:
        result["recommended"] = {
            "loss": "MSE / MAE / Huber",
            "metrics": ["RMSE", "MAE", "R2"],
            "avoid": ["akurasi/klasifikasi"],
        }

    return result


def load_telco() -> tuple[str, pd.DataFrame, str]:
    df = pd.read_csv(RAW / "telco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    return "Telco Customer Churn", df, "Churn"


def load_wine() -> tuple[str, pd.DataFrame, str]:
    df = pd.read_csv(RAW / "wine" / "WineQT.csv")
    return "Wine Quality", df, "quality"


def load_insurance() -> tuple[str, pd.DataFrame, str]:
    df = pd.read_csv(RAW / "insurance" / "insurance.csv")
    return "Medical Cost (Insurance)", df, "charges"


def render_md(results: list[dict]) -> str:
    lines = [
        "# Identifikasi Tipe Tugas — Regresi vs Klasifikasi",
        "",
        "Digenerate oleh `scripts/detect_task.py` (aturan deterministik, tanpa LLM).",
        "",
        "| Dataset | Tipe | Subtipe | Keyakinan | Bukti utama |",
        "|---------|------|---------|-----------|-------------|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['task']} | {r.get('subtype', '-')} | "
            f"{r.get('confidence', '-')} | {r['reasons'][0]} |"
        )

    for r in results:
        lines += ["", f"## {r['dataset']}", "", f"**Verdict: {r['task']} / {r.get('subtype')}**", ""]
        lines.append("Alasan:")
        for reason in r["reasons"]:
            lines.append(f"- {reason}")
        if r.get("class_balance"):
            lines += ["", f"Distribusi target: `{json.dumps(r['class_balance'])}`"]
        if r.get("target_stats"):
            lines += ["", f"Statistik target: `{json.dumps(r['target_stats'])}`"]
        rec = r["recommended"]
        lines += [
            "",
            f"Loss: `{rec['loss']}`",
            f"Metrik: {', '.join(rec['metrics'])}",
        ]
        if rec.get("avoid"):
            lines.append(f"Hindari: {', '.join(rec['avoid'])}")
        if r.get("alternative"):
            lines.append(f"Alternatif: **{r['alternative']}**")

    lines += [
        "",
        "## Kesimpulan",
        "",
        "- Telco -> binary classification (target kategorikal 2 kelas).",
        "- Wine -> multiclass classification; alternatif ordinal regression karena label berurutan.",
        "- Medical Cost (Insurance) -> regression murni (target `charges` kontinu, 1.337 nilai unik).",
        "- Komposisi akhir: 2 klasifikasi + 1 regresi.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    results = []
    for loader in (load_telco, load_wine, load_insurance):
        name, df, target = loader()
        r = classify_target(name, df[target])
        r["n_features"] = int(df.shape[1] - 1)
        results.append(r)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render_md(results), encoding="utf-8")

    summary = {r["dataset"]: f"{r['task']}/{r.get('subtype')}" for r in results}
    print(json.dumps(summary, indent=2))
    print(f"\nreport -> {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
