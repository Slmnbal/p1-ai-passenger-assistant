# Model Kartı: P1 Intent Sınıflandırıcı

## Model açıklaması

**Temel model:** `dbmdz/bert-base-turkish-cased` (Türkçe BERT, ~110M parametre)
**Görev:** 6 sınıflı intent sınıflandırma (bkz. `DATASET_CARD.md`)
**Seçilen versiyon:** Full fine-tune, 10 epoch (`models/intent_full_10ep/`)

## Neden bu versiyon seçildi

4 yaklaşım denendi, aynı 114 (train) / 51 (test) bölünmesinde ölçüldü:

| Yaklaşım | Accuracy | Macro F1 | ROC-AUC | Kritik sınıf* |
|---|---|---|---|---|
| Kural tabanlı (baseline) | 0.725 | 0.737 | — | 0.889 |
| Frozen-probe (BERT dondu, sadece üst katman eğitildi) | 0.412 | 0.312 | 0.717 | 1.000 |
| **Full fine-tune, 10 epoch (seçilen)** | **0.745** | **0.722** | **0.907** | **1.000** |
| Full fine-tune, 20 epoch | 0.686 | 0.648 | 0.917 | 1.000 |
| Doğrudan LLM routing (llama3.1:8b, prompt ile) | 0.667 | 0.631 | — | 0.889 |

*Kritik sınıf = `rezervasyon_islem_talebi` (iptal/iade/değişiklik — insan onayı gerektiren işlemler)

**Gerekçe:** 20 epoch'a göre daha az aşırı öğrenme (eval_loss 1.03 vs 1.77), frozen-probe'a göre çok daha yüksek genel performans, ve kritik sınıfta hem kural tabanlı hem LLM routing'den daha güvenilir (%100 vs %88.9).

## Metrikler — ne ölçüyor, neden seçildi (İlke 2)

- **Accuracy:** Genel doğru oranı. Sezgisel ama sınıflar dengesizken yanıltıcı olabilir (büyük sınıfa kayar). Tek başına yeterli değil.
- **Macro F1:** Her sınıfın F1'ini eşit ağırlıklandırıp ortalar — küçük/nadir sınıflardaki (örn. `belirsiz_acikliga_kavusturma`) başarısızlığı gizlemez. Bu yüzden asıl karar metriği bu, accuracy değil.
- **ROC-AUC (one-vs-rest):** Modelin doğru sınıfı, olası tüm eşiklerde ne kadar iyi "üstte" sıraladığını ölçer — varsayılan eşikten (argmax) bağımsızdır. 20 epoch'ta ROC-AUC'nin accuracy'den daha iyi çıkması, modelin sıralama gücünün karar eşiğinden ayrı bir şey olduğunu gösterdi (bkz. aşağıdaki kalibrasyon bulgusu).
- **Kritik sınıf doğruluğu:** İş etkisi en yüksek hata türü — insan onayına giden bir işlemin yanlış sınıflandırılması, ya gereksiz onay yüküne ya da güvenlik açığına yol açar. Bu yüzden ayrı raporlanıyor.

## Sınıf bazlı sonuçlar (seçilen model, test seti n=51)

| Sınıf | Precision | Recall | F1 |
|---|---|---|---|
| checkin_talebi | 0.80 | 1.00 | 0.89 |
| rezervasyon_islem_talebi | 0.75 | 1.00 | 0.86 |
| kapsam_disi | 0.70 | 0.88 | 0.78 |
| ucus_sorgulama | 0.78 | 0.78 | 0.78 |
| politika_bilgi_sorgusu | 0.80 | 0.44 | 0.57 |
| belirsiz_acikliga_kavusturma | 0.60 | 0.38 | 0.46 |

**En zayıf iki sınıf, EDA'nın öngördüğü gibi çıktı:** `belirsiz_acikliga_kavusturma` (en yüksek kelime örtüşmesine sahipti) ve `politika_bilgi_sorgusu` (en büyük/en heterojen kategori) — hatalar bu iki sınıfta diğer tüm sınıflara dağılıyor, tek bir karışan çift yok.

## Bilinen ve ölçülmüş sınırlama: kalibrasyon sorunu

Modelin softmax güven skoru, gerçek doğruluğu güvenilir şekilde yansıtmıyor:

| Güven aralığı | n | Ortalama güven | Gerçek doğruluk |
|---|---|---|---|
| 0.7–0.9 | 15 | 0.812 | **0.533** |
| 0.9–1.0 | 29 | 0.956 | 0.897 |

0.7-0.9 aralığında model "%81 eminim" derken gerçekte yarıdan biraz fazla doğru çıkıyor — **ham güven skoruna dayalı bir "eşik üstündeyse güven" kuralı burada güvenilir değil.** Bu, RAG tarafında ADR-0002'de bulunan sorunla aynı aile: iki alt sistemde de model güveni, doğrudan güvenilirlik anlamına gelmiyor. Sonuç: Adım 7'deki guardrail, ham confidence skoruna değil (İlke 4'ün öngördüğü gibi) ikinci bir doğrulama adımına dayanmalı.

## Eğitim detayları

- Base model: `dbmdz/bert-base-turkish-cased`
- Epochs: 10, learning rate: 5e-5, batch size: 8, cihaz: Apple Silicon MPS
- Train/test: 114/51 (stratified, sabit seed=42)
- Kod: `app/intent/train_classifier.py --mode full --epochs 10`

## Kullanım amacı ve sınırları

- **Amaçlanan kullanım:** P1'in FastAPI `/chat` ucunda, gelen mesajı RAG/tool-call/human-in-loop/netleştirme akışlarından birine yönlendirmek.
- **Amaçlanmayan kullanım:** Üretim ölçeğinde, gerçek kullanıcı verisiyle yeniden eğitilmeden doğrudan kullanılmamalı (bkz. dataset card — cold-start veri).
- **Bilinen zayıf noktalar:** Belirsiz/kısa mesajlar (%38 recall), genel politika soruları (%44 recall), kalibre olmayan güven skoru.

## Gelecek iyileştirmeler (bilinçli olarak ertelendi)

Aşağıdaki iyileştirmeler tanımlandı ama **şimdi uygulanmayacak** — MVP mantığı gereği
önce Adım 5-11 tamamlanıp uçtan uca sistem çalışır hale gelecek, "ne kadar iyi yeterli"
sorusunun gerçek cevabı Adım 11'in segment analizi ve bilinen hata kalıpları regresyon
setiyle netleşecek. O zaman bu listeye geri dönülecek (bkz. proje hafızası
"professional-rigor-mentality" — RAG'deki kapsam-dışı güven kararıyla aynı gerekçe).

1. **Kalibrasyon düzeltmesi (temperature scaling)** — ham güven skorunun gerçek doğruluğu yansıtmaması sorununu doğrudan hedefler.
2. **K-fold cross-validation** — test seti küçük (n=51, sınıf başına 8-9); tek bir yanlış tahmin bile bir sınıfın recall'ını %11-12 oynatıyor, mevcut sayılar olduğundan daha "kesin" görünüyor.
3. **Zayıf sınıflara (belirsiz_acikliga_kavusturma, politika_bilgi_sorgusu) daha fazla/çeşitli örnek eklemek** — muhtemelen en yüksek getirili tek değişiklik.
4. **Erken durdurma (early stopping)** — epoch sayısını elle denemek yerine validation loss'a göre otomatik durdurma.
5. **Hibrit yaklaşım** — kural tabanlının güçlü olduğu yerlerle (kapsam_disi precision) fine-tune modelin güçlü olduğu yerleri (kritik sınıf) birleştiren bir sistem.
