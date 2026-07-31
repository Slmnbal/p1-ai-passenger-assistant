# ADR 0004: Uçtan uca değerlendirme — asıl darboğaz RAG değil, intent sınıflandırıcı

**Durum:** Kabul edildi (30 Temmuz 2026)

## Bağlam

Adım 3 ve Adım 4, RAG'i ve intent sınıflandırıcıyı BİREBİR kendi bileşenleri olarak
ölçmüştü (retrieval Recall@k, intent test-set accuracy %74.5). Adım 11'in amacı farklı:
bu bileşenleri TEK TEK değil, gerçek `graph.run()` üzerinden UÇTAN UCA — bir kullanıcı
mesajının planlama → RAG/tool → guardrail zincirinin tamamından geçtiğinde ne olduğunu
ölçmek. Bir bileşen izole test edildiğinde iyi görünse bile, ondan ÖNCEKİ bir adım
(planlama) onu hiç çalıştırmayabilir — bu ADR tam olarak bu etkiyi ölçüyor.

`evaluation/e2e_scenarios.json`: 48 senaryo, 10 kategori (policy, flight_status,
route_search, checkin, reservation_cancel/change_date/add_baggage, ambiguous,
conflicting_source, out_of_scope), segment etiketli (dil, mesaj uzunluğu — bkz.
`evaluation/run_e2e_evaluation.py` + `analyze_e2e_results.py`).

## Deney: 48 senaryo, gerçek Qdrant+Ollama+fine-tune model+guardrail zinciriyle

**Genel sonuç: 35/48 (%72.9).**

| Kategori | n | Geçen | Oran |
|---|---|---|---|
| ambiguous | 5 | 5 | %100 |
| checkin | 3 | 3 | %100 |
| out_of_scope | 5 | 5 | %100 |
| reservation_cancel | 3 | 3 | %100 |
| reservation_change_date | 2 | 2 | %100 |
| reservation_add_baggage | 2 | 2 | %100 |
| flight_status | 6 | 6 | %100 |
| route_search | 4 | 3 | %75 |
| **policy** | 12 | **4** | **%33.3** |
| **conflicting_source** | 6 | **2** | **%33.3** |

**Hata kategorisi dağılımı: `yanlis_intent` 10, `sayisal_veya_icerik_hatasi` 2,
`yanlis_outcome` 1.** Yani 13 hatanın 10'u — %77'si — RAG'in veya guardrail'in değil,
**planlama (intent sınıflandırma) adımının** hatası: mesaj hiç `politika_bilgi_sorgusu`
veya `ucus_sorgulama`'ya yönlendirilmiyor, bu yüzden RAG/tool hiç çalışma şansı bulmuyor.

**Somut örnekler (tekrarlanabilir, iki ayrı koşuda da aynı sonuç):**
- "Uçağım rötar yaparsa tazminat alabilir miyim?" → `belirsiz_acikliga_kavusturma`
  (olması gereken: `politika_bilgi_sorgusu`)
- "Valizim standarttan büyükse ne kadar ödeme yaparım?" → `belirsiz_acikliga_kavusturma`
- "Mil kullanarak yaptığım kabin yükseltmede bagaj hakkım Business'a geçer mi?" →
  `rezervasyon_islem_talebi` (kafa karıştırıcı/RAG-zor sorular, intent seviyesinde de zor)
- "Do infants need a separate ticket?" (İngilizce) → `kapsam_disi`
- "Are there flights from IST to LHR?" (İngilizce) → `politika_bilgi_sorgusu`

**Dil bazlı segment analizi (bias/fairness kontrolü):**

| Dil | n | Geçen | Oran |
|---|---|---|---|
| Türkçe | 44 | 33 | %75.0 |
| İngilizce | 4 | 2 | %50.0 |

Örneklem küçük (n=4 İngilizce) ama yön net ve tutarlı: model, eğitim setinin ağırlıklı
Türkçe/karışık olması nedeniyle SAF İngilizce girdide daha zayıf. Bu, kesin bir istatistik
değil ama izlenmesi gereken bir sinyal — daha büyük bir İngilizce test örneklemi gerekir.

**Mesaj uzunluğu segmenti:** kısa mesajlarda (≤4 kelime) %91.7 başarı, uzun mesajlarda
(>10 kelime) %33.3. Koşullu/dolaylı cümle yapıları ("X yaparsa", "X ise") ve daha uzun
cümleler modeli zorluyor — 165 örneklik eğitim setinin bu yapıları yeterince temsil
etmediğini gösteriyor.

**İş KPI'ları:**

| KPI | Değer |
|---|---|
| İlk temasta çözüm oranı | %41.7 |
| Yanlış yönlendirme oranı | %20.8 |
| İnsan devri oranı | %39.6 |
| İşlem tamamlama oranı (n=10 işlem senaryosu) | %70.0 |

%20.8'lik yanlış yönlendirme oranı, Adım 4'ün ölçtüğü test-set doğruluğuyla (%74.5,
yani ~%25.5 hata) makul ölçüde tutarlı — bağımsız bir örneklemle çapraz doğrulama.

## Bulunan ve düzeltilen GERÇEK bir bug (RAG/intent'ten bağımsız)

Değerlendirme sırasında `tool_agent.py`'nin PNR regex'i (`\b[A-Z]{3}[A-Z0-9]{4}\b`)
sıradan 7 harfli BÜYÜK HARFE ÇEVRİLMİŞ Türkçe kelimeleri de eşlediği ortaya çıktı:
"KAPANIR" (check-in ile ilgili bir soruda) ve "YOLCUYA" PNR sanılıp
`reservation.get_reservation()`'a gönderiliyordu. Bu mock sistemdeki TÜM PNR'ler "SYN"
ile başladığı için kalıp `\bSYN[A-Z0-9]{4}\b`'ye daraltıldı — artık sıradan kelimelerle
çakışmıyor (bkz. `app/agent/tool_agent.py`). Bu, izole birim testlerinde hiç
YAKALANAMAMIŞTI çünkü o testler hep temiz, PNR içermeyen ya da gerçek PNR içeren
cümlelerle yazılmıştı — gerçek/çeşitli cümlelerle uçtan uca test etmenin somut kanıtı.

## Kararlar

1. **RAG katmanını "iyileştirmeye" çalışmak yanlış öncelik olurdu.** Adım 3'te retrieval
   zaten Recall@3 %89 ölçülmüştü; burada RAG'in kendisi çalıştığında (12 policy
   senaryosundan 4'ünde) her seferinde doğru, kaynaklı cevap üretti. Asıl kaldıraç
   noktası intent sınıflandırıcı — Adım 4'ün zaten belgelediği %74.5 test doğruluğunun
   pratik sonucu, bu ADR'de somut örneklerle gösterildi.
2. **Bu, şu an için MODELİ YENİDEN EĞİTMEYİ gerektirmiyor** (kapsam dışı, Adım 4'ün işiydi)
   — bunun yerine bulgular dürüstçe belgelendi ve regresyon testine bağlandı
   (`tests/test_e2e_scenarios.py`), gelecekte model iyileştirilirse bu testler
   ilerlemeyi somut olarak gösterecek.
3. **PNR regex bug'ı düzeltildi** — düşük maliyetli, yüksek değerli bir düzeltme.
4. **Dil önyargısı sinyali izlenmeye değer ama küçük örneklemle kesinleştirilemez** —
   gelecekte İngilizce test örneklemi büyütülmeli (bkz. açık noktalar).

## Henüz kanıtlanmayanlar / açık noktalar

- İngilizce segment n=4 çok küçük — dil önyargısı bulgusu yönü net ama büyüklüğü
  belirsiz, daha büyük bir örneklemle doğrulanmalı.
- `conflicting_source` kategorisindeki RAG/faithfulness-judge başarısızlıkları
  (%33.3) ADR-0002/0003'ün zaten belgelediği bilinen sınırlamaların uçtan uca
  düzeyde yeniden teyidi — yeni bir bulgu değil, ama artık somut sayılarla bağlanmış.
- Intent sınıflandırıcının koşullu cümle yapılarında (`X yaparsa/ise`) ve İngilizce
  girdide neden zayıf olduğu kök nedeni (eğitim verisi dağılımı) doğrulandı ama
  düzeltilmedi — bu, veri setini genişletme gerektiren ayrı bir çalışma.
- **Sonradan bulunan ek bulgu (regresyon testleri yazılırken):** RAG+claim-decomposition
  zincirinin non-determinizmi, ilk ölçülenden DAHA GENİŞ kapsamlı çıktı. `e041`
  (conflicting_source, iki değerlendirmede de FAIL) `tests/test_e2e_scenarios.py`
  yazılırken yapılan ÜÇÜNCÜ bir koşuda beklenmedik şekilde PASS oldu; `e005` (policy,
  iki değerlendirmede de PASS — "kolay" kategoride) aynı üçüncü koşuda beklenmedik
  şekilde fallback'e düştü. Yani bu non-determinizm sadece "zor/kafa karıştırıcı"
  sorularla sınırlı değil — basit, tek kaynaklı sorularda bile ara sıra ortaya çıkabiliyor.
  Kök neden büyük olasılıkla: (a) Qdrant'ın top-k sıralamasında çok yakın skorlu
  chunk'lar arasında çalıştırmalar arası küçük ondalık farklar, ve/veya (b) claim
  decomposition'ın "cevabı kaç/nasıl iddiaya böleceği" kararının kendisi tam
  deterministik olmayabilir (LLM'in metin bölme kararı, sayısal örnekleme kadar
  temperature=0.0'a duyarlı olmayabilir). Regresyon testleri bu gerçekliği YOK SAYMADI —
  `test_e2e_scenarios.py`'deki ilgili testler "ya doğru cevap ya dürüst bulamadım, ama
  asla halüsinasyon yok" invaryantını koruyacak şekilde YENİDEN yazıldı (sert "her zaman
  grounded olmalı" iddiaları yerine).

## Yapılacaklar (Selman'ın arayüz denemesi + geri bildirimi sonrası, 30 Temmuz 2026)

Öncelik sırasıyla:

1. **Intent sınıflandırıcıyı iyileştir (asıl darboğaz).** `data/intent/` eğitim setini
   genişlet — özellikle koşullu cümle yapıları ("X yaparsa/ise ne olur?") ve İngilizce
   girdi örnekleri ekle (bkz. bu ADR'nin ölçtüğü %75 TR / %50 EN farkı). 165 örnek
   yetersiz kaldı.
2. **Çok turlu konuşma hafızası ekle.** `/chat` şu an tamamen stateless —
   `app/agent/graph.py::run()` sadece tek bir mesaj alıyor, önceki turu hiç bilmiyor.
   Takip soruları ("peki bunun ücreti ne?") şu an çalışmaz. `ConversationState`'teki
   `session_preferences`/`history` alanları zaten var ama `/chat` bir session id'yle
   state'i kalıcı tutmuyor.
3. **Retrieval iyileştirmeleri (RAG çalıştığında kaliteyi daha da artırmak için,
   ikincil öncelik — asıl semptomu çözmez ama "conflicting_source" %33 gibi durumları
   iyileştirebilir):**
   - Chunk boyutunu büyüt: şu an 600 karakter/100 overlap (~120-150 token) —
     400-800 token aralığı denenmedi.
   - Hybrid search (BM25/keyword + embedding, Reciprocal Rank Fusion).
   - Reranker (30-50 aday getirip en iyi 5-10'u seç).

Denenip REDDEDİLDİĞİ İÇİN öncelik listesinde OLMAYAN: sabit benzerlik-skoru eşiği
(ADR-0002/0003'te tamamen iç içe geçen dağılımlarla ölçülüp terk edildi).

## İlgili dosyalar

- `evaluation/e2e_scenarios.json`, `run_e2e_evaluation.py`, `analyze_e2e_results.py`
- `evaluation/e2e_results.json`, `e2e_analysis.json` (ham veri + analiz çıktısı)
- `app/agent/tool_agent.py` (PNR regex düzeltmesi)
- `tests/test_e2e_scenarios.py`
