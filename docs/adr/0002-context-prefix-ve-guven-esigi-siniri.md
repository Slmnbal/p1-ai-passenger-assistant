# ADR 0002: Context-prefixed embedding kararı ve güven-eşiği probleminin ertelenmesi

**Durum:** Kabul edildi (28 Temmuz 2026)

## Bağlam

Adım 3'te 302 chunk'lık korpus Qdrant'a taşındıktan sonra, 72 soruluk etiketli bir
değerlendirme seti (`evaluation/eval_questions.json`) oluşturuldu: 51 kontrol, 8 kafa
karıştırıcı çift, 8 kapsam dışı (negatif), 5 eş anlamlı/parafraz sorusu. Bu, projenin
"önce ölç, tek değişkenle dene, dürüst raporla" ilkesinin (bkz. proje hafızası
"professional-rigor-mentality") ilk uygulaması oldu.

## Baseline ölçüm

`thy_policies` collection'ında (chunk metni doğrudan embed edilmiş):

| Kategori | Recall@1 | Recall@3 |
|---|---|---|
| Genel (n=64) | %56.2 | %79.7 |
| Kontrol (n=51) | %62.7 | %84.3 |
| Kafa karıştırıcı (n=8) | %37.5 | %50.0 |
| Eş anlamlı (n=5) | %20.0 | %80.0 |

Kapsam dışı 8 soruda ortalama top-1 güven skoru: **0.590** (5/8 soru 0.6'yı aşıyor).

## Deney 1: Context-prefixed embedding

**Hipotez:** Kısa chunk'lar tek başına embed edildiğinde konu bağlamını kaybediyor.
**Değişken:** Embed edilen metin `chunk.text` yerine `"{belge_başlığı}. {bölüm_başlığı}. {chunk.text}"` oldu (payload/gösterim metni değişmedi). Ayrı bir collection'da (`thy_policies_ctx`) test edildi.

**Sonuç:**

| Kategori | Baseline R@1 | Deney R@1 | Baseline R@3 | Deney R@3 |
|---|---|---|---|---|
| Genel | %56.2 | **%71.9** | %79.7 | **%89.1** |
| Kontrol | %62.7 | **%80.4** | %84.3 | **%92.2** |
| Kafa karıştırıcı | %37.5 | **%50.0** | %50.0 | **%75.0** |
| Eş anlamlı | %20.0 | %20.0 (değişmedi) | %80.0 | %80.0 (değişmedi) |

Kapsam dışı ortalama skor: 0.575 (baseline'dan neredeyse değişmedi).

## Deney 2: Ortak kelime sayısı, kapsam-dışı tespiti için sinyal olabilir mi?

**Hipotez:** Embedding'in top-1 sonucu ile sorgu arasında ortak anlamlı kelime yoksa, bu kapsam dışı olduğunun işareti olabilir.

**Sonuç:** Kısmen doğru, ama önemli bir confound var:

| Kategori | Ort. ortak kelime | %0 ortak kelime |
|---|---|---|
| Kontrol | 2.90 | %8 |
| Kafa karıştırıcı | 3.62 | %0 |
| Kapsam dışı | 0.38 | %75 |
| Eş anlamlı | 0.20 | %80 |

**Kapsam dışı ve eş anlamlı kategoriler bu metrikte neredeyse ayırt edilemiyor** — ikisi de doğası gereği sorguyla kaynak metin arasında kelime paylaşmıyor (biri gerçekten cevapsız olduğu için, diğeri aynı şeyi farklı kelimeyle sorduğu için). Basit bir "ortak kelime yoksa reddet" kuralı, geçerli ama farklı ifade edilmiş soruları da yanlışlıkla reddederdi.

## Kararlar

1. **`thy_policies_ctx` (context-prefixed embedding), bundan sonraki adımlarda (Adım 6+) kullanılacak varsayılan collection olarak kabul edildi.** Ölçülebilir, tek değişkenli bir iyileşme kanıtlandı; `thy_policies` (baseline) karşılaştırma amacıyla silinmedi.
2. **Kapsam-dışı güven sorunu, basit bir skor eşiği veya kelime-örtüşme kuralıyla ŞİMDİ çözülmeyecek — bilinçli olarak Adım 7'ye ertelendi.** Gerekçe: Deney 2, bu problemin retrieval-skoru seviyesinde çözülemeyeceğini kanıtladı (eş anlamlı ile kapsam dışı istatistiksel olarak ayırt edilemiyor). Planın Adım 7'de zaten öngördüğü "ikinci bir LLM çağrısıyla faithfulness/groundedness kontrolü" (İlke 4), artık varsayım değil, bu deneyle **kanıtlanmış bir gereklilik**.

## Henüz kanıtlanmayanlar / açık noktalar

- Kafa karıştırıcı kategoride hâlâ %50 Recall@1 — deney bunu iyileştirdi ama çözmedi, ayrı bir çalışma gerektiriyor (muhtemelen hybrid arama veya re-ranking, henüz denenmedi).
- Örneklem küçük (kafa karıştırıcı n=8, kapsam dışı n=8, eş anlamlı n=5) — kesin hüküm için büyütülmeli.
- Adım 7'ye kadar sistemin "bilmiyorum" davranışı retrieval katmanında garanti edilemez; bu, Adım 6'da LLM entegrasyonuna kadar bilinen bir sınırlama olarak kabul edildi.

## İlgili dosyalar

- `app/rag/qdrant_retriever.py` (`embed_text_fn` parametresi)
- `evaluation/build_context_prefixed_collection.py`
- `evaluation/analyze_keyword_overlap_signal.py`
- `evaluation/eval_questions.json`, `evaluation/eval_results.json`, `evaluation/eval_results_ctx.json`, `evaluation/keyword_overlap_analysis.json`
