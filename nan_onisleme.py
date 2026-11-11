"""
TEKNOFEST Sağlıkta Yapay Zeka — NaN Ön-İşleme
-----------------------------------------------
Strateji (MNAR / bilgilendirici eksiklik için):
  1) Eksiklik gösterge kolonları  -> _isNaN bayrakları
  2) Alan-bilgili doldurma:
       - frekans (gnomAD/AllofUs)  -> 0
       - sayısal skorlar           -> medyan
       - kategorikler              -> "Missing"
  3) Çıktı LightGBM/XGBoost'a hazır (ham NaN de bırakılabilir; bu script doldurur
     ama indicator'lar sayesinde sinyal korunur).

Kullanım:
  python nan_onisleme.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("veri/EĞİTİM (TRAIN) SETLERİ")
FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "CFTR":   "YARISMA_TRAIN_CFTR.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH":    "YARISMA_TRAIN_PAH.csv",
}

ID_COL, LABEL_COL = "Variant_ID", "Label"

# Hangi kolonlar "frekans" sayılacak? gnomAD/AllofUs frekans kolonlarını
# içeriğe göre tespit ediyoruz: değer aralığı [0,1] içinde kalan AL_* kolonları
# büyük olasılıkla allel frekansı -> eksikse 0 ile doldur.
def is_frequency_col(s: pd.Series) -> bool:
    v = s.dropna()
    if len(v) == 0 or not np.issubdtype(v.dtype, np.number):
        return False
    return v.min() >= 0.0 and v.max() <= 1.0


def preprocess(df: pd.DataFrame, indicator_min_missing=0.01, drop_thresh=0.98):
    df = df.copy()
    ids = df[ID_COL] if ID_COL in df else None
    y = df[LABEL_COL] if LABEL_COL in df else None
    X = df.drop(columns=[c for c in (ID_COL, LABEL_COL) if c in df])

    # --- Adım 0: neredeyse tamamen boş kolonları at (sinyalsiz) ---
    nan_rate = X.isna().mean()
    drop_cols = nan_rate[nan_rate >= drop_thresh].index.tolist()
    X = X.drop(columns=drop_cols)

    num_cols = X.select_dtypes("number").columns
    cat_cols = X.select_dtypes("object").columns

    # --- Adım 1: eksiklik gösterge kolonları (MNAR sinyali) ---
    miss = X.isna().mean()
    flag_cols = miss[miss >= indicator_min_missing].index
    ind = X[flag_cols].isna().astype("int8")
    ind.columns = [f"{c}_isNaN" for c in flag_cols]

    # --- Adım 2: alan-bilgili doldurma ---
    freq_cols = [c for c in num_cols if is_frequency_col(X[c])]
    score_cols = [c for c in num_cols if c not in freq_cols]

    X[freq_cols] = X[freq_cols].fillna(0.0)              # gözlenmemiş = 0
    X[score_cols] = X[score_cols].fillna(X[score_cols].median())  # skorlar -> medyan
    for c in cat_cols:
        X[c] = X[c].astype("object").fillna("Missing").astype("category")

    # --- birleştir ---
    out = pd.concat([ind, X], axis=1)
    if ids is not None:
        out.insert(0, ID_COL, ids.values)
    if y is not None:
        out[LABEL_COL] = y.values

    info = dict(dropped=drop_cols, n_indicator=ind.shape[1],
                n_freq=len(freq_cols), n_score=len(score_cols),
                n_cat=len(cat_cols), final_shape=out.shape)
    return out, info


if __name__ == "__main__":
    for name, fname in FILES.items():
        path = DATA_DIR / fname
        if not path.exists():
            print(f"[atla] {path} bulunamadi"); continue
        df = pd.read_csv(path)
        out, info = preprocess(df)
        out_path = f"{name}_islenmis.csv"
        out.to_csv(out_path, index=False)
        print(f"[{name}] {df.shape} -> {info['final_shape']} | "
              f"gosterge:{info['n_indicator']} frekans:{info['n_freq']} "
              f"skor:{info['n_score']} kategorik:{info['n_cat']} "
              f"atilan:{len(info['dropped'])} -> {out_path}")

    # --- Hızlı doğrulama: LightGBM ile baseline (opsiyonel) ---
    try:
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        d = pd.read_csv("MASTER_islenmis.csv")
        y = d[LABEL_COL]; X = d.drop(columns=[ID_COL, LABEL_COL], errors="ignore")
        for c in X.select_dtypes("object").columns:
            X[c] = X[c].astype("category")
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                 class_weight="balanced", verbosity=-1)
        auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
        print(f"\nLightGBM 5-fold ROC-AUC: {auc.mean():.4f} +/- {auc.std():.4f}")
    except ImportError:
        print("\n(lightgbm kurulu degil; baseline atlandi: pip install lightgbm)")
