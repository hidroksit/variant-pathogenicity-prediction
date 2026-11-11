# ============================================================
# TEKNOFEST Sağlıkta Yapay Zeka — Sızıntısız NaN Pipeline (Colab)
# Genetik varyant patojenite sınıflandırması
# Strateji: frekans->0 (sabit) + kalan NaN'i LightGBM doğal işler
#           + _isNaN gösterge kolonları (MNAR sinyali)
#           + tüm doldurma CV içinde => VERİ SIZINTISI YOK
# ============================================================

# --- 0) Kurulum (Colab'da bir kez) ---
# !pip -q install lightgbm scikit-learn pandas numpy joblib

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from lightgbm import LGBMClassifier
import joblib

# --- 1) Veri yolu ---
# Google Drive kullanıyorsan:
# from google.colab import drive; drive.mount('/content/drive')
# DATA_DIR = Path('/content/drive/MyDrive/train_data')
DATA_DIR = Path('train_data')          # dosyaları Colab'a yüklediysen
ID_COL, LABEL_COL = 'Variant_ID', 'Label'

FILES = {
    'GENERAL': 'train_general.csv',
    'CFTR':    'train_cftr.csv',
    'KANSER':  'train_kanser.csv',
    'PAH':     'train_pah.csv',
}

# --- 2) Sızıntısız NaN dönüştürücü ---
def _is_frequency_col(s: pd.Series) -> bool:
    """Allel frekansı kolonu mu? Sayısal ve değerleri [0,1] aralığında."""
    v = s.dropna()
    if len(v) == 0 or not np.issubdtype(v.dtype, np.number):
        return False
    return (v.min() >= 0.0) and (v.max() <= 1.0)

class GenomicNaNHandler(BaseEstimator, TransformerMixin):
    """
    fit(): SADECE train fold'undan öğrenir (sızıntısız):
      - >drop_thresh oranında boş kolonları atar
      - gösterge (_isNaN) eklenecek kolonları belirler
      - frekans kolonlarını tespit eder
    transform():
      - frekans NaN -> 0 (sabit, gözlenmedi=nadir)
      - kalan NaN   -> bırakılır (LightGBM doğal işler)
      - _isNaN bayrakları eklenir
    """
    def __init__(self, drop_thresh=0.98, indicator_min_missing=0.01, freq_fill=0.0):
        self.drop_thresh = drop_thresh
        self.indicator_min_missing = indicator_min_missing
        self.freq_fill = freq_fill

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        miss = X.isna().mean()
        self.drop_cols_ = miss[miss >= self.drop_thresh].index.tolist()
        Xk = X.drop(columns=self.drop_cols_)
        miss2 = Xk.isna().mean()
        self.flag_cols_ = miss2[miss2 >= self.indicator_min_missing].index.tolist()
        self.freq_cols_ = [c for c in Xk.columns if _is_frequency_col(Xk[c])]
        self.feature_names_ = Xk.columns.tolist()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.reindex(columns=self.feature_names_)          # train kolon düzenine hizala
        ind = X[self.flag_cols_].isna().astype('int8')
        ind.columns = [f'{c}_isNaN' for c in self.flag_cols_]
        X[self.freq_cols_] = X[self.freq_cols_].fillna(self.freq_fill)  # frekans -> 0
        # kalan NaN'ler LightGBM'e ham bırakılır
        return pd.concat([X, ind], axis=1)

def build_pipeline():
    clf = LGBMClassifier(
        n_estimators=600, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced',        # sınıf dengesizliği için
        random_state=42, n_jobs=-1, verbosity=-1,
    )
    return Pipeline([('nan', GenomicNaNHandler()), ('clf', clf)])

# --- 3) Sızıntısız çapraz doğrulama (ROC-AUC + PR-AUC) ---
def evaluate(name, path):
    df = pd.read_csv(path)
    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[c for c in (ID_COL, LABEL_COL) if c in df])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    res = cross_validate(
        build_pipeline(), X, y, cv=cv,
        scoring={'roc_auc': 'roc_auc', 'pr_auc': 'average_precision'},
        n_jobs=-1,
    )
    print(f"[{name:8s}] n={len(df):4d}  poz%={y.mean():.2f}  "
          f"ROC-AUC={res['test_roc_auc'].mean():.4f}±{res['test_roc_auc'].std():.3f}  "
          f"PR-AUC={res['test_pr_auc'].mean():.4f}±{res['test_pr_auc'].std():.3f}")

print('=== Sızıntısız 5-fold CV ===')
for name, fname in FILES.items():
    p = DATA_DIR / fname
    if p.exists():
        evaluate(name, p)
    else:
        print(f'[atla] {p} bulunamadı')

# --- 4) Tüm veriyle final modeli eğit ve kaydet ---
def fit_and_save(train_path, model_out='model_general.joblib'):
    df = pd.read_csv(train_path)
    y = df[LABEL_COL].astype(int)
    X = df.drop(columns=[c for c in (ID_COL, LABEL_COL) if c in df])
    pipe = build_pipeline().fit(X, y)
    joblib.dump(pipe, model_out)        # ön-işleme + model birlikte saklanır
    print(f'Model kaydedildi -> {model_out}')
    return pipe

pipe = fit_and_save(DATA_DIR / FILES['GENERAL'], 'model_general.joblib')

# --- 5) Yarışma test setine uygulama (örnek) ---
# test = pd.read_csv('test_general.csv')
# ids  = test[ID_COL]
# Xte  = test.drop(columns=[c for c in (ID_COL, LABEL_COL) if c in test])
# proba = pipe.predict_proba(Xte)[:, 1]     # patojenik olasılığı
# pred  = (proba >= 0.5).astype(int)
# pd.DataFrame({ID_COL: ids, 'Label': pred, 'proba': proba}).to_csv('submission.csv', index=False)
