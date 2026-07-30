# ADR 0001: Embedding modeli olarak çok dilli (multilingual) model seçimi

**Durum:** Kabul edildi (27-28 Temmuz 2026)

## Bağlam

`app/rag/embeddings.py`, RAG retrieval için Hugging Face sentence-transformers
tabanlı bir embedding modeli kullanıyor. İlk yazıldığında varsayılan model
`sentence-transformers/all-MiniLM-L6-v2` idi — küçük, hızlı, yaygın kullanılan
bir model, ama ağırlıklı olarak İngilizce metin çiftleriyle eğitilmiş. Bu
projenin kaynak belgeleri ve kullanıcı soruları Türkçe/İngilizce karışık.

Varsayım "embedding modeli, TF-IDF baseline'a göre Türkçe eş anlamlı/morfolojik
varyasyonlarda daha iyi performans gösterir" şeklindeydi (bkz. `tfidf_retriever.py`
docstring'i). p1_proje_plani.md İlke 2/3 gereği bu varsayım test edilmeden karara
bağlanmadı.

## Deney

`evaluation/tfidf_vs_embedding_demo.py`, `data/policies/` altındaki 3 gerçek
belgeden üretilen 26 chunk üzerinde, TF-IDF baseline'ı ile iki embedding modelini
(`all-MiniLM-L6-v2` ve `paraphrase-multilingual-MiniLM-L12-v2`) 5 soruda karşılaştırdı:
2 kontrol sorusu (belgedeki kelimeyle birebir eşleşen), 2 eş anlamlı/parafraz sorusu,
1 kapsam dışı negatif kontrol (evcil hayvan — corpus'ta yok).

**Sonuçlar (top-1 skor ve doğru/yanlış):**

| Soru | TF-IDF | Embedding (en) | Embedding (multi) |
|---|---|---|---|
| Gecikme (kontrol) | 0.392 ✓ | 0.663 ✗ | 0.614 ✓ |
| Rötar (eş anlamlı) | 0.252 ✗ | 0.653 ✗ | 0.513 ✗ |
| Valiz + parafraz (eş anlamlı) | 0.176 ✗ | 0.591 ✗ | 0.506 ✓ |
| Check-in (kontrol) | 0.314 ✓ | 0.717 ✓ | 0.508 ✓ |
| Evcil hayvan (kapsam dışı) | 0.134 (düşük, doğru) | 0.533 (yüksek, **yanlış**) | 0.198 (düşük, doğru) |

## Karar

Varsayılan embedding modeli `paraphrase-multilingual-MiniLM-L12-v2` olarak
değiştirildi. Gerekçe:

1. İngilizce modelin kapsam dışı soruya verdiği yüksek skor (0.533), İlke 4'teki
   "güven eşiği altında tahmin üretme" güvenlik kontrolünü bu model üzerine
   kurmayı riskli hale getiriyor — model, alakasız bir soruda bile kendinden
   emin görünüyor.
2. Çok dilli model, aynı negatif kontrolde TF-IDF'e yakın, doğru şekilde düşük
   bir skor (0.198) verdi.
3. Çok dilli model, İngilizce modelin yanlış bulduğu 2 soruyu (gecikme kontrolü,
   valiz+parafraz) doğru buldu.

## Kısıtlar / henüz kanıtlanmayanlar

- Örneklem küçük: 26 chunk, 5 soru. Bu, Adım 3'teki resmi RAG değerlendirmesinin
  (20-30 belge / 150-300 chunk, 50-100 etiketli soru, Recall@k ölçümü) yerine
  geçmez — sadece model seçimi kararını erken ve ucuza yönlendirmek için yapıldı.
- Çok dilli model "rötar" sorusunda hâlâ yanlış bölümü buldu (tamamen alakasız
  bir bölüm: "Kısıtlanmış malzemeler") — küçük corpus'ta beklenebilir, ama Adım 3'te
  corpus büyüyünce yeniden ölçülecek.
- Bu karar Adım 3'te resmi eval seti ile yeniden doğrulanacak; sonuç değişirse bu
  ADR güncellenecek.

## İlgili dosyalar

- `app/rag/embeddings.py` (DEFAULT_MODEL_NAME)
- `evaluation/tfidf_vs_embedding_demo.py` (deney scripti)
