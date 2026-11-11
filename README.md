# Missense Varyant Patojenite Tahmini

Tek bir amino asit değişimine yol açan nokta mutasyonlarının **patojenik** mi **benign** mi olduğunu tahmin eden, sızıntısız doğrulama mimarisi üzerine kurulu bir makine öğrenmesi hattı.

> **English:** Machine-learning pipeline for classifying missense variants as pathogenic or benign across four disease panels. Built around a leakage-free validation architecture, MNAR-aware imputation of population-frequency features, and transfer learning for panels with very few samples.

TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması için geliştirilmiştir — Proje Detay Raporu değerlendirmesi **89/100**, final barajı 3 puan farkla kaçırıldı.

---

## Problem

Patojenik varyantlar doğaları gereği nadirdir; veri setleri ağır biçimde dengesizdir. Bu, modellerin patojenik sınıfı kaçırmasına (yanlış negatif) yol açar — klinik kararlarda en pahalı hata türü. İkinci zorluk, öznitelik matrisindeki **%30–58 aralığında eksikliktir** ve bu eksiklik rastgele değildir.

## Yöntem

**MNAR farkındalıklı impütasyon.** Popülasyon frekansı (`AL_`) sütunlarındaki boşluk, varyantın gnomAD'de *hiç gözlenmediğini* — yani gerçekten nadir olduğunu — gösterir. Standart medyan doldurma bu biyolojik sinyali yok eder. Bunun yerine eksik değerler sıfıra yakın, sola kaydırılmış bir Gauss dağılımından (μ−1.8σ, σ→0.3σ) örneklenir; ayrıca eksiklik deseninin kendisi (`_isNaN` bayrakları, `n_al_missing`, `block_missing`) modele öznitelik olarak verilir.

**Sızıntısız doğrulama.** Her panelde önce %15'lik kilitli holdout ayrılır, kalanda 5 katlı stratified CV uygulanır. İmputasyon ve hedef kodlama dahil **tüm ön işleme her fold'un yalnızca eğitim parçasına fit edilir**; holdout, karar eşiği belirlenene kadar hiç görülmez.

**Panel bazlı model stratejisi.** Veri yeterliyse beş üyeli soft-voting topluluğu (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees); örneklem çok küçükse MASTER panelinde eğitilmiş ağaçların düşük öğrenme oranıyla (lr=0.01) ince ayarlandığı transfer öğrenme.

**Dengesizlik.** SMOTE ve SMOTEENN denendi, hiçbir panelde anlamlı iyileşme sağlamadı ve terk edildi. Yerine `scale_pos_weight` ağırlıklandırması ve OOF tahminler üzerinde macro-F1'e göre panel bazlı eşik optimizasyonu kullanıldı.

## Sonuçlar

| Panel | n (toplam / holdout) | Yaklaşım | OOF macro-F1 | Holdout macro-F1 | AUC-PR | Eşik | MCC |
|---|---|---|---|---|---|---|---|
| MASTER | 2931 / 352 | Bağımsız topluluk | 0,7811 | 0,7640 | 0,9463 | 0,550 | 0,5313 |
| KANSER | 388 / 59 | Bağımsız topluluk | 0,8453 | 0,8578 | 0,9563 | 0,490 | 0,7164 |
| PAH | 372 / 56 | MASTER'dan transfer | 0,6953 | 0,7573 | 0,9354 | 0,580 | 0,5161 |
| CFTR | 111 / 17 | MASTER'dan transfer | 0,8771 | 0,9101 | 0,9908 | 0,610 | 0,8345 |

Öne çıkan bulgu: **transfer öğrenmenin katkısı veri kıtlığıyla doğru orantılı.** Yalnızca 111 örnekli CFTR panelinde holdout macro-F1, sıfırdan eğitime kıyasla **0,7733 → 0,9101** yükseldi. 388 örnekli KANSER panelinde ise katkı ihmal edilebilir kaldı ve bu panel bağımsız eğitildi.

En belirleyici öznitelik, evrimsel korunmuşluğu temsil eden `EK_7` (%11,86 önem). Amino asit değişiminin fizikokimyasal etkisinden türetilen `delta_volume` ve `aa1_mol_weight` ek ayırt edicilik sağladı.

### Dürüstlük notu

CFTR (n=17) ve PAH (n=56) holdout kümeleri küçüktür; bu panellerdeki yüksek skorlar geniş güven aralığına sahiptir ve temkinli yorumlanmalıdır. CFTR'de benign duyarlılığının 1,000 olması yalnızca 3 örneğe dayanır.

## Kullanım

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy joblib

python nan_onisleme.py                      # MNAR farkındalıklı ön işleme
python TEKNOFEST_nan_pipeline_colab.py      # sızıntısız CV hattı
```

`TEKNOFEST_NaN_model_colab.ipynb` aynı hattın Colab üzerinde çalıştırılabilir hâlidir.

## Veri

Eğitim verisi TEKNOFEST tarafından sağlanmıştır ve **bu depoya dahil değildir**. Kod, `train_data/` altında `Variant_ID` ve `Label` sütunlarını içeren panel CSV'leri bekler. Çalışmada hiçbir dış veri kaynağı kullanılmamıştır.

## Lisans

MIT — bkz. [LICENSE](LICENSE).
