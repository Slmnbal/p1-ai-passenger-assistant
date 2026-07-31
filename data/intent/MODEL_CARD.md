# Model Kartı: P1 Intent Sınıflandırıcı

## Model açıklaması

**Temel model:** `dbmdz/bert-base-turkish-cased` (Türkçe BERT, ~110M parametre)
**Görev:** 6 sınıflı intent sınıflandırma (bkz. `DATASET_CARD.md`)
**Seçilen versiyon:** Full fine-tune, 10 epoch (`models/intent_full_10ep/`)

## Adım 11 sonrası yeniden eğitim (30-31 Temmuz 2026)

Adım 11/ADR-0004'ün bulduğu darboğazı (hataların %77'si intent sınıflandırıcıdan,
koşullu cümle yapıları + İngilizce girdi zayıflığı) doğrudan hedefleyerek veri seti
165→254 örneğe genişletildi (bkz. `DATASET_CARD.md`, "Adım 11 sonrası genişletme") ve
model **aynı hiperparametrelerle** (full fine-tune, 10 epoch, lr=5e-5) yeniden eğitildi.
Aşağıdaki tablo ve sınıf bazlı sonuçlar bu yeni eğitime aittir; eski (165 örnek) sonuçlar
karşılaştırma için satır başlarında parantez içinde bırakıldı.

## Neden bu versiyon seçildi

4 yaklaşım, orijinal olarak aynı 114 (train) / 51 (test) bölünmesinde karşılaştırıldı
(bu karşılaştırma tekrarlanmadı — sadece seçilen yaklaşım yeni veriyle yeniden eğitildi):

| Yaklaşım | Accuracy | Macro F1 | ROC-AUC | Kritik sınıf* |
|---|---|---|---|---|
| Kural tabanlı (baseline) | 0.725 | 0.737 | — | 0.889 |
| Frozen-probe (BERT dondu, sadece üst katman eğitildi) | 0.412 | 0.312 | 0.717 | 1.000 |
| Full fine-tune, 10 epoch (165 örnek, eski) | 0.745 | 0.722 | 0.907 | 1.000 |
| **Full fine-tune, 10 epoch (254 örnek, seçilen/güncel)** | **0.763** | **0.764** | **0.923** | **0.923** |
| Full fine-tune, 20 epoch (165 örnek) | 0.686 | 0.648 | 0.917 | 1.000 |
| Doğrudan LLM routing (llama3.1:8b, prompt ile) | 0.667 | 0.631 | — | 0.889 |

*Kritik sınıf = `rezervasyon_islem_talebi` (iptal/iade/değişiklik — insan onayı gerektiren işlemler)

**Gerekçe:** Genişletilmiş veriyle Macro F1 ve ROC-AUC her ikisi de arttı (asıl karar
metriği Macro F1 — accuracy değil, bkz. aşağıdaki metrik açıklamaları). **Dürüstçe
belirtilmesi gereken bir trade-off:** kritik sınıf doğruluğu %100'den %92.3'e (13
örnekten 1'i yanlış) düştü — test seti küçük olduğu için (n=13) bu tek bir örnek demek;
yine de bu bir regresyon olarak burada saklanmadan raporlanıyor. Genel iyileşmenin kritik
sınıftaki küçük gerilemeye değip değmediği, gerçek trafikte kritik sınıfın hata maliyeti
(insan onayına yanlış yönlendirme) ile tartılmalı — bu proje ölçeğinde kabul edildi çünkü
asıl darboğaz (politika/belirsiz karışıklığı) çok daha büyük bir kazanımla düzeldi.

## Metrikler — ne ölçüyor, neden seçildi (İlke 2)

- **Accuracy:** Genel doğru oranı. Sezgisel ama sınıflar dengesizken yanıltıcı olabilir (büyük sınıfa kayar). Tek başına yeterli değil.
- **Macro F1:** Her sınıfın F1'ini eşit ağırlıklandırıp ortalar — küçük/nadir sınıflardaki (örn. `belirsiz_acikliga_kavusturma`) başarısızlığı gizlemez. Bu yüzden asıl karar metriği bu, accuracy değil.
- **ROC-AUC (one-vs-rest):** Modelin doğru sınıfı, olası tüm eşiklerde ne kadar iyi "üstte" sıraladığını ölçer — varsayılan eşikten (argmax) bağımsızdır. 20 epoch'ta ROC-AUC'nin accuracy'den daha iyi çıkması, modelin sıralama gücünün karar eşiğinden ayrı bir şey olduğunu gösterdi (bkz. aşağıdaki kalibrasyon bulgusu).
- **Kritik sınıf doğruluğu:** İş etkisi en yüksek hata türü — insan onayına giden bir işlemin yanlış sınıflandırılması, ya gereksiz onay yüküne ya da güvenlik açığına yol açar. Bu yüzden ayrı raporlanıyor.

## Sınıf bazlı sonuçlar (seçilen/güncel model, test seti n=76, 254 örneklik genişletilmiş veriyle)

| Sınıf | Precision | Recall | F1 | (eski F1, 165 örnek) |
|---|---|---|---|---|
| kapsam_disi | 0.889 | 0.727 | 0.800 | (0.78) |
| checkin_talebi | 0.750 | 0.900 | 0.818 | (0.89) |
| rezervasyon_islem_talebi | 0.706 | 0.923 | 0.800 | (0.86) |
| ucus_sorgulama | 0.750 | 0.692 | 0.720 | (0.78) |
| belirsiz_acikliga_kavusturma | 0.727 | 0.667 | 0.696 | (0.46) |
| politika_bilgi_sorgusu | 0.800 | 0.706 | 0.750 | (0.57) |

**Hedeflenen iki zayıf sınıf belirgin iyileşti:** `belirsiz_acikliga_kavusturma` F1
0.46→0.696, `politika_bilgi_sorgusu` F1 0.57→0.750 — bu ikisi tam olarak ADR-0004'ün
işaret ettiği, koşullu cümle yapılarında karışan çift. Confusion matrix'te bu iki sınıf
hâlâ birbirine en çok karışan çift (12 örnekten 2'si `belirsiz`→`politika` ve 2'si
`politika`→`belirsiz`/`checkin` karışıyor) — EDA'nın Jaccard bulgusuyla tutarlı, tamamen
çözülmedi ama kazanç net.

**Confusion matrix (satır=gerçek, sütun=tahmin):**

```
                              kapsam checkin rezerv ucus  belirsiz politika
kapsam_disi                     8      0       1      0      1        1
checkin_talebi                  0      9       0      1      0        0
rezervasyon_islem_talebi         0      0      12      0      0        1
ucus_sorgulama                   1      1       2      9      0        0
belirsiz_acikliga_kavusturma      0      0       2      1      8        1
politika_bilgi_sorgusu           0      2       0      1      2       12
```

## Bilinen ve ölçülmüş sınırlama: kalibrasyon sorunu

Modelin softmax güven skoru, gerçek doğruluğu güvenilir şekilde yansıtmıyor — genişletilmiş
veriyle yeniden ölçüldü, sorun **iyileşmedi, hatta bir aralıkta belirginleşti**:

| Güven aralığı | n | Ortalama güven | Gerçek doğruluk | (eski, 165 örnek) |
|---|---|---|---|---|
| 0.5–0.7 | 6 | 0.624 | 0.833 | — |
| 0.7–0.9 | 12 | 0.798 | **0.417** | (0.533) |
| 0.9–1.0 | 57 | 0.977 | 0.825 | (0.897) |

0.7-0.9 aralığında model "%80 eminim" derken gerçekte %41.7 doğru çıkıyor — kalibrasyon
sorunu genişletilmiş veriyle **daha da kötüleşti** (bu dürüstçe raporlanıyor, veri
genişletmesinin her metriği iyileştirmediğinin bir kanıtı). Hatta 0.9-1.0 gibi çok yüksek
güven aralığında bile gerçek doğruluk %82.5'te kalıyor (eskiden %89.7). **Ham güven
skoruna dayalı bir "eşik üstündeyse güven" kuralı hâlâ güvenilir değil, önceki
sonuçtan daha da güçlü bir kanıtla.** Bu, RAG tarafında ADR-0002'de bulunan sorunla aynı
aile: iki alt sistemde de model güveni, doğrudan güvenilirlik anlamına gelmiyor. Sonuç
değişmedi: Adım 7'deki guardrail, ham confidence skoruna değil (İlke 4'ün öngördüğü gibi)
ikinci bir doğrulama adımına dayanmalı.

## Eğitim detayları

- Base model: `dbmdz/bert-base-turkish-cased`
- Epochs: 10, learning rate: 5e-5, batch size: 8, cihaz: Apple Silicon MPS
- Train/test: 178/76 (stratified, sabit seed=42) — Adım 11 sonrası 254 örneğe genişletilmiş veriyle
- Kod: `app/intent/train_classifier.py --mode full --epochs 10 --output models/intent_full_10ep`

## Kullanım amacı ve sınırları

- **Amaçlanan kullanım:** P1'in FastAPI `/chat` ucunda, gelen mesajı RAG/tool-call/human-in-loop/netleştirme akışlarından birine yönlendirmek.
- **Amaçlanmayan kullanım:** Üretim ölçeğinde, gerçek kullanıcı verisiyle yeniden eğitilmeden doğrudan kullanılmamalı (bkz. dataset card — cold-start veri).
- **Bilinen zayıf noktalar (güncellendi):** Belirsiz/politika sınıfları arasındaki karışıklık
  azaldı ama sürmekte (bkz. confusion matrix); kritik sınıfta küçük bir gerileme (%100→%92.3);
  kalibrasyon sorunu düzelmedi, aksine 0.7-0.9 aralığında belirginleşti (%53.3→%41.7 gerçek
  doğruluk); test seti hâlâ küçük (n=76).

## Nitel doğrulama (görülmemiş, eğitim setinde olmayan cümlelerle manuel test)

Eğitim setinde olmayan yeni koşullu/İngilizce cümlelerle hızlı bir manuel kontrol yapıldı
(bu resmi bir değerlendirme seti değil, sadece hedeflenen darboğazın gerçekten iyileşip
iyileşmediğine dair bir sağlık kontrolü):

| Cümle | Tahmin | Değerlendirme |
|---|---|---|
| "Bagajım havaalanında kaybolursa kime başvururum?" | politika_bilgi_sorgusu (0.75) | Doğru |
| "Do you offer compensation for cancelled flights?" | politika_bilgi_sorgusu (0.95) | Doğru — EN artık çalışıyor |
| "Is there free wifi on board?" | politika_bilgi_sorgusu (0.99) | Doğru |
| "Uçağım 2 saat gecikirse ne yapmalıyım?" | ucus_sorgulama (0.76) | Tartışmalı — gerçekten belirsiz bir cümle |
| "What if my connecting flight is missed due to a delay?" | ucus_sorgulama (0.98) | Tartışmalı |
| "Rezervasyonumu iptal edersem ceza öder miyim?" | rezervasyon_islem_talebi (0.90) | Tartışmalı — aslında politika sorusu olabilirdi |

Sonuç dürüstçe karışık: en net EN/politika örnekleri artık doğru gidiyor, ama bazı
koşullu cümleler hâlâ gerçekten belirsiz (insan için de net olmayan) — bu, sınıflandırma
sorununun tamamen "veri eksikliği" değil kısmen dilin doğasında var olan belirsizlik
olduğunu gösteriyor.

## Gelecek iyileştirmeler (bilinçli olarak ertelendi)

Madde 3 (zayıf sınıflara örnek ekleme) bu turda uygulandı ve ölçülen kazanımı yukarıda
raporlandı. Kalan maddeler hâlâ ertelendi — aynı gerekçeyle (proje hafızası
"professional-rigor-mentality"):

1. **Kalibrasyon düzeltmesi (temperature scaling)** — ham güven skorunun gerçek doğruluğu yansıtmaması sorununu doğrudan hedefler; bu turda daha da öncelikli hale geldi çünkü sorun kötüleşti.
2. **K-fold cross-validation** — test seti hâlâ küçük (n=76, sınıf başına 10-17); tek bir yanlış tahmin bir sınıfın recall'ını %8-10 oynatabiliyor.
3. ~~Zayıf sınıflara daha fazla/çeşitli örnek eklemek~~ — **YAPILDI** (bu ADR/tur), ölçülen kazanım yukarıda.
4. **Erken durdurma (early stopping)** — epoch sayısını elle denemek yerine validation loss'a göre otomatik durdurma.
5. **Hibrit yaklaşım** — kural tabanlının güçlü olduğu yerlerle (kapsam_disi precision) fine-tune modelin güçlü olduğu yerleri (kritik sınıf) birleştiren bir sistem.
