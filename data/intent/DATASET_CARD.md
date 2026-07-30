# Veri Kartı: P1 Intent Sınıflandırma Veri Seti

## Özet

165 örnek mesaj, 6 intent sınıfına etiketlenmiş. Türk Hava Yolları yolcu asistanı
sisteminde, gelen kullanıcı mesajının hangi alt sisteme (RAG, tool-call, human-in-the-loop,
netleştirme, red) yönlendirileceğini belirlemek için kullanılır.

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
| `politika_bilgi_sorgusu` | 30 | RAG retrieval | Politika/kural bilgisi isteyen soru |
| `ucus_sorgulama` | 30 | Tool call (salt okunur) | Uçuş durumu/detayı sorgusu |
| `rezervasyon_islem_talebi` | 30 | Tool call + human-in-loop | İptal/değişiklik/iade gibi kritik işlem talebi |
| `checkin_talebi` | 25 | Tool call | Check-in işlemi talebi |
| `belirsiz_acikliga_kavusturma` | 25 | Netleştirici soru | Birden fazla yoruma açık, eksik bağlamlı soru |
| `kapsam_disi` | 25 | Nazik red/yönlendirme | Sistemin kapsamı dışındaki istek |

**Toplam: 165**

## Toplama yöntemi

Yazar tarafından, `data/policies/` altındaki 51 gerçek THY kaynak belgesinin konu
başlıklarına (bagaj, gecikme, çocuk yolcu, evcil hayvan, Miles&Smiles, ödeme vb.) ve
projenin mimari şemasındaki (README) routing kararlarına dayanarak yazılmıştır. Türkçe,
İngilizce ve karışık (aynı cümlede iki dilin unsurlarını taşıyan, örn. "check-in" gibi
yerleşmiş İngilizce terimler içeren Türkçe cümleler) örnekler kasıtlı olarak dahil
edilmiştir — gerçek kullanıcıların da böyle yazması beklenir.

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

## EDA bulguları (bkz. `eda.py`, `eda_results.json`)

- **Dil karışımı:** Genel %72.7 Türkçe, %17.0 karışık, %10.3 İngilizce — ama `checkin_talebi`
  sınıfı belirgin şekilde farklı: %76'sı karışık (İngilizce "check-in" teriminin Türkçe
  cümlelere yerleşmesi nedeniyle).
- **Mesaj uzunluğu:** `belirsiz_acikliga_kavusturma` en kısa (ort. 3.8 kelime), diğerleri
  4.8-5.6 kelime arası — belirsiz mesajların kısalığı EDA'da önceden tespit edildi.
- **Kelime örtüşmesi (Jaccard):** `belirsiz_acikliga_kavusturma`, diğer sınıfların çoğuyla
  en yüksek örtüşmeye sahip sınıf — bu EDA bulgusu, hem kural tabanlı router'da hem
  fine-tune edilmiş modelde bu sınıfın en düşük recall'a sahip çıkmasıyla doğrulandı
  (bkz. MODEL_CARD.md).

## Train/test bölünmesi

114 train / 51 test, sınıf oranları korunarak (stratified), sabit seed=42
(`split_dataset.py`). Test oranı ~%30 — küçük veri setinde daha stabil metrik için
standart %20'den yüksek tutuldu.

## Üretim scripti

`data/intent/build_dataset.py` — tüm örnekler bu dosyada kaynak kodu olarak okunabilir,
gizli/rastgele üretim yok.
