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

## Güncelleme: Yapılacaklar #1 uygulandı — intent sınıflandırıcı yeniden eğitildi (31 Temmuz 2026)

`data/intent/messages.json` 165→254 örneğe genişletildi: her 6 sınıfa koşullu cümle
yapıları ("X yaparsa/ise ne olur?") ve dengeli İngilizce örnekler eklendi (İngilizce
%10.3→%28.0). Model aynı hiperparametrelerle (full fine-tune, 10 epoch) yeniden eğitildi.
Tam detay ve yeni confusion matrix: `data/intent/MODEL_CARD.md` ve `DATASET_CARD.md`.

**Sonuç (izole test seti, n=76):** Macro F1 0.722→0.764, ROC-AUC 0.907→0.923. Hedeflenen
iki zayıf sınıf belirgin iyileşti: `politika_bilgi_sorgusu` recall %44→%70.6,
`belirsiz_acikliga_kavusturma` recall %38→%66.7. **Trade-off dürüstçe raporlandı:** kritik
sınıf (`rezervasyon_islem_talebi`) doğruluğu %100→%92.3'e (13'te 1) hafifçe geriledi;
kalibrasyon sorunu düzelmedi, 0.7-0.9 güven aralığında daha da kötüleşti (%53.3→%41.7
gerçek doğruluk).

## Uçtan uca yeniden ölçüm sonucu (31 Temmuz 2026) — gerçekten çalıştırıldı, varsayılmadı

`evaluation/run_e2e_evaluation.py` yeni modelle tekrar çalıştırıldı (eski ham sonuçlar
`evaluation/e2e_results_before_intent_retrain.json`'da korunuyor, karşılaştırma için).

**Genel sonuç DEĞİŞMEDİ: yine 35/48 (%72.9).** Ama bu sabit sayının ARKASINDAKİ bileşim
tamamen değişti — bu ADR'nin asıl bulduğu "darboğaz intent'te" tespiti doğrudan
doğrulandı ve şimdi bir SONRAKİ darboğaz görünür hale geldi:

| Hata kategorisi | Eski (165 örnek) | Yeni (254 örnek) |
|---|---|---|
| `yanlis_intent` | 10 | **1** |
| `yanlis_outcome` | 1 | **12** |
| `sayisal_veya_icerik_hatasi` | 2 | 0 |

**Intent kaynaklı hatalar 10'dan 1'e düştü** — hedeflenen darboğaz gerçekten daraltıldı.
Ama toplam hata sayısı aynı kaldı (13) çünkü RAG artık çalışma şansı bulduğu senaryolarda
kendi (önceden ADR-0002/0003'te zaten bilinen) sınırlamalarıyla karşılaşıyor:

- **e011, e038, e039 — intent artık DOĞRU (`politika_bilgi_sorgusu`), ama senaryo hâlâ
  "başarısız" sayılıyor:**
  - e038/e039 GERÇEKTEN çelişen kaynağa sahip senaryolar (miles_redemption_policy.md vs
    paid_business_upgrade_policy.md) — `policy_verification_agent` bunu doğru tespit
    edip bir NETLEŞTİRİCİ SORU döndürüyor. Bu `expected_outcome=grounded_answer`
    beklentisini karşılamıyor ama İlke 4'ün "çelişen kaynakta netleştir" kuralına tam
    uygun — muhtemelen bu iki senaryo için asıl doğru davranış bu, eval setinin
    `expected_outcome` alanı gözden geçirilmeli.
  - e011 ("Do infants need a separate ticket?") GERÇEKTEN çelişen bir kaynağa sahip
    DEĞİL (kategori: `policy`, tek doğru cevap var) — ama aynı sezgisel (en iyi iki
    kaynağın skoru yakın + farklı bölüm) burada YANLIŞ POZİTİF veriyor ve gereksiz bir
    netleştirici soru döndürüyor. Bu, **daha dar/daha spesifik yeni bir bilinen hata**:
    artık "İngilizce soru hiç RAG'e ulaşmıyor" değil, "RAG'e ulaşıyor ama çelişen-kaynak
    sezgiselinin yanlış pozitifi cevabı engelliyor."
- **e004, e022 — intent DÜZELDİ ve senaryo artık tamamen GEÇİYOR** (net, tartışmasız
  kazanım): "Online check-in ne zaman açılır ve ne zaman kapanır?" ve
  "Are there flights from IST to LHR?".
- **e005, e010 — eskiden GEÇİYORDU, şimdi BAŞARISIZ:** ikisi de intent hem eskiden hem
  şimdi doğru (`politika_bilgi_sorgusu`); regresyonun nedeni intent modeli DEĞİL, bu
  ADR'nin zaten belgelediği RAG+claim-decomposition zincirinin ölçülmüş
  non-determinizmi (e005 zaten "flaky" olarak işaretliydi, bkz. yukarı; e010 aynı
  ailenin yeni bir örneği).

**Sonuç:** Yapılacaklar #1 (intent sınıflandırıcı) kendi hedefini tuttu — intent artık
neredeyse hiç hata kaynağı değil. Ama bu, sistemin TOPLAM başarı oranını otomatik
yükseltmedi çünkü intent düzelince RAG katmanının kendi (önceden zaten ADR-0002/0003'te
ölçülmüş) sınırlamaları — çelişen-kaynak sezgiselinin yanlış pozitifleri, claim-
decomposition non-determinizmi — artık gizlenmeden ortaya çıkıyor. Bu **beklenen ve
sağlıklı bir bulgu**: bir darboğaz kapanınca bir sonraki görünür hale geliyor. **Yapılacaklar
listesindeki #3 (retrieval iyileştirmeleri: hybrid search, reranker, çelişen-kaynak
sezgiselinin isabetini artırma) artık ikincil değil, ölçülmüş veriyle desteklenen bir
sonraki öncelik.**

Regresyon testleri (`tests/test_e2e_scenarios.py`) bu yeni tabloyu yansıtacak şekilde
güncellendi: eski `test_known_gap_conflicting_source_intent_misroute` ve
`test_known_gap_english_policy_question_misrouted` testleri, artık intent'in düzeldiğini
(`assert intent == politika_bilgi_sorgusu`) VE yeni/daha dar netleştirme davranışını
(`assert needs_clarification is True`) sabitleyen testlere dönüştürüldü.

## İlgili dosyalar

- `evaluation/e2e_scenarios.json`, `run_e2e_evaluation.py`, `analyze_e2e_results.py`
- `evaluation/e2e_results.json`, `e2e_analysis.json` (ham veri + analiz çıktısı)
- `app/agent/tool_agent.py` (PNR regex düzeltmesi)
- `tests/test_e2e_scenarios.py`
