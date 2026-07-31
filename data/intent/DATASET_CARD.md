# Veri Kartı: P1 Intent Sınıflandırma Veri Seti

## Özet

254 örnek mesaj, 6 intent sınıfına etiketlenmiş (Adım 11'in bulduğu intent-sınıflandırıcı
darboğazı üzerine, Adım 12 sonrası 165'ten genişletildi — bkz. "Adım 11 sonrası genişletme"
bölümü). Türk Hava Yolları yolcu asistanı sisteminde, gelen kullanıcı mesajının hangi alt
sisteme (RAG, tool-call, human-in-the-loop, netleştirme, red) yönlendirileceğini belirlemek
için kullanılır.

## Önemli: Bu veri gerçek kullanıcı trafiği DEĞİLDİR

P1 henüz canlıya alınmadığı için (bkz. proje planı), gerçek kullanıcı mesajı yoktur. Bu
veri seti, projenin gerçek 51 belgelik RAG korpusundaki konulara ve gerçek sistem
mimarisindeki routing kararlarına dayanarak **yazar tarafından oluşturulmuş, gerçekçi
örnek cümlelerdir** — NLU sistemlerinde "cold-start" (soğuk başlangıç) olarak bilinen,
standart bir bootstrapping yöntemidir. Bu gizlenmemektedir: model card'da ve mülakat
hazırlık belgesinde de aynı şekilde belirtilecektir. Gerçek kullanıcı verisi toplandığında
(veya toplanabilseydi), bu veri seti onunla değiştirilip model yeniden eğitilmelidir.

## Sınıf tanımları ve dağılım

| Intent | Sayı | Yönlendiği yer | Açıklama |
|---|---|---|---|
| `politika_bilgi_sorgusu` | 57 | RAG retrieval | Politika/kural bilgisi isteyen soru |
| `ucus_sorgulama` | 43 | Tool call (salt okunur) | Uçuş durumu/detayı sorgusu |
| `rezervasyon_islem_talebi` | 43 | Tool call + human-in-loop | İptal/değişiklik/iade gibi kritik işlem talebi |
| `checkin_talebi` | 33 | Tool call | Check-in işlemi talebi |
| `belirsiz_acikliga_kavusturma` | 40 | Netleştirici soru | Birden fazla yoruma açık, eksik bağlamlı soru |
| `kapsam_disi` | 38 | Nazik red/yönlendirme | Sistemin kapsamı dışındaki istek |

**Toplam: 254**

## Toplama yöntemi

Yazar tarafından, `data/policies/` altındaki 51 gerçek THY kaynak belgesinin konu
başlıklarına (bagaj, gecikme, çocuk yolcu, evcil hayvan, Miles&Smiles, ödeme vb.) ve
projenin mimari şemasındaki (README) routing kararlarına dayanarak yazılmıştır. Türkçe,
İngilizce ve karışık (aynı cümlede iki dilin unsurlarını taşıyan, örn. "check-in" gibi
yerleşmiş İngilizce terimler içeren Türkçe cümleler) örnekler kasıtlı olarak dahil
edilmiştir — gerçek kullanıcıların da böyle yazması beklenir.

## Adım 11 sonrası genişletme (165 → 254 örnek)

Adım 11'in uçtan uca değerlendirmesi ve ADR-0004, hataların %77'sinin intent
sınıflandırıcıdan kaynaklandığını, ve iki somut kalıbı gösterdi: (1) koşullu cümle
yapıları ("X yaparsa/ise ne olur?") modeli özellikle `politika_bilgi_sorgusu` ve
`belirsiz_acikliga_kavusturma` arasında kararsız bırakıyordu (örn. "Uçağım rötar yaparsa
tazminat alabilir miyim?" yanlışlıkla `belirsiz_acikliga_kavusturma`'ya gitmişti), (2)
İngilizce girdide doğruluk belirgin şekilde düşüktü (%75 TR vs %50 EN, n=4 küçük
örneklem). Bu iki bulguyu doğrudan hedefleyerek:

- Her 6 sınıfa da koşullu cümle yapılı örnekler eklendi (yalnızca `politika_bilgi_sorgusu`'na
  değil — amaç "koşullu cümle = politika" gibi yeni bir yanlış kısayol öğretmemekti).
- İngilizce örnek sayısı 17'den 71'e çıkarıldı (%10.3 → %28.0), her sınıfa dengeli dağıtıldı.

**Çıktı (bkz. MODEL_CARD.md):** Macro F1 0.722→0.764, en zayıf iki sınıfın recall'ı
belirgin arttı (`politika_bilgi_sorgusu` %44→%70.6, `belirsiz_acikliga_kavusturma`
%38→%66.7). Kritik sınıf (`rezervasyon_islem_talebi`) doğruluğu %100'den %92.3'e (13'te 1
hata) hafifçe düştü — bu dürüstçe raporlanıyor, gizlenmiyor.

## Bilinen sınırlamalar

- **Küçük örneklem** (165) — gerçek bir üretim sınıflandırıcısı için düşük; bu bir
  portföy/kanıt projesidir, üretim ölçeği değildir.
- **Yazar önyargısı:** Örnekler tek bir kişi tarafından yazıldığı için gerçek kullanıcı
  çeşitliliğini (yazım hataları, argo, bölgesel farklar) tam yansıtmaz.
- **Sınıf dengesi kasıtlı yaklaşık eşit** tutuldu (25-30 arası) — gerçek trafikte muhtemelen
  `politika_bilgi_sorgusu` ve `ucus_sorgulama` çok daha baskın olurdu; bu dengesizlik burada
  simüle edilmedi çünkü gerçek oranı bilmeden yapay bir dengesizlik uydurmak yanıltıcı olur.
- **`belirsiz_acikliga_kavusturma` sınıfı en kırılgan olanı** — tanım gereği diğer
  sınıflarla örtüşmeye en yatkın kategori (bkz. EDA).

## EDA bulguları (bkz. `eda.py`, `eda_results.json`, Adım 11 sonrası genişletmeyle güncellendi)

- **Dil karışımı:** Genel %61.0 Türkçe, %11.0 karışık, %28.0 İngilizce (genişletme
  öncesi %72.7/%17.0/%10.3 idi) — ama `checkin_talebi` sınıfı hâlâ belirgin şekilde
  farklı: çoğunluğu karışık (İngilizce "check-in" teriminin Türkçe cümlelere yerleşmesi
  nedeniyle).
- **Mesaj uzunluğu:** `belirsiz_acikliga_kavusturma` en kısa (ort. 4.0 kelime), diğerleri
  5.3-6.4 kelime arası — belirsiz mesajların kısalığı EDA'da önceden tespit edildi, koşullu
  cümle örnekleri eklendikten sonra da bu sıralama korundu.
- **Kelime örtüşmesi (Jaccard):** `belirsiz_acikliga_kavusturma` ve `politika_bilgi_sorgusu`
  birbirleriyle en yüksek örtüşmeye sahip çift (Jaccard≈0.13) — bu, tam da Adım 11'in
  bulduğu karışıklık yönüyle tutarlı ve genişletmeden sonra da tamamen ortadan kalkmadı
  (beklenen: koşullu cümleler doğaları gereği bu iki sınıfın kesişiminde kalıyor).

## Train/test bölünmesi

178 train / 76 test, sınıf oranları korunarak (stratified), sabit seed=42
(`split_dataset.py`). Test oranı ~%30 — küçük veri setinde daha stabil metrik için
standart %20'den yüksek tutuldu.

## Üretim scripti

`data/intent/build_dataset.py` — tüm örnekler bu dosyada kaynak kodu olarak okunabilir,
gizli/rastgele üretim yok.
