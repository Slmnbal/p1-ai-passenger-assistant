# ADR 0005: RAG/retrieval deneyleri — chunk boyutu ve hibrit arama reddedildi, çelişen-kaynak sezgiseli düzeltildi

**Durum:** Kabul edildi (1 Ağustos 2026)

## Bağlam

ADR-0004'ün güncellemesi, intent sınıflandırıcı düzeltildikten sonra 48 senaryolu
uçtan uca değerlendirmenin genel başarısının %72.9'da sabit kaldığını, ama darboğazın
RAG katmanının kendi bilinen sınırlamalarına (çelişen-kaynak sezgiselinin yanlış
pozitifleri, RAG+claim-decomposition non-determinizmi) kaydığını göstermişti. Bu ADR,
Selman'ın orijinal RAG-iyileştirme kontrol listesindeki (ADR-0004'ün "Yapılacaklar #3")
üç fikri TEK TEK, tek değişkenli deneylerle ölçüyor — ADR-0002/0003'ün kurduğu
"varsayma, ölç" disipliniyle aynı.

## Deney 1: Chunk boyutunu büyütmek (600 → 2500 karakter)

**Hipotez:** 600 karakter/100 overlap (~120-150 token) çok küçük kalıyor olabilir;
400-800 token aralığına (~1700-3400 karakter) çıkarmak Recall@1/3'ü artırabilir.

**Yöntem:** `evaluation/build_bigger_chunk_collection.py` ile aynı context-prefixed embed
stratejisi (ADR-0002'nin kabul ettiği varsayılan) korunarak, tek değişken (chunk boyutu)
değiştirilip ayrı bir Qdrant collection'ı (`thy_policies_bigchunk`) oluşturuldu, orijinal
`thy_policies` collection'ına dokunulmadı. 72 soruluk etiketli sette (`evaluation/
run_eval.py`) ölçüldü.

**Sonuç — beklenmedik bir keşif:** Chunk boyutunu 600→2500 karaktere çıkarmak chunk
sayısını neredeyse hiç değiştirmedi (302→285) ve ortalama chunk uzunluğu SABİT kaldı
(289→304 karakter). Neden: `app/rag/chunking.py`'nin chunking mantığı ÖNCELİKLE markdown
başlıklarına (`##`/`###`) göre bölüyor; `max_chars` yalnızca nadir/uzun bölümleri
ikiye ayırıyor. Bu korpustaki 51 belgenin ortalama uzunluğu zaten yalnızca ~1900 karakter
ve çoğu bölüm 600 karakterden kısa — yani chunk boyutu limiti pratikte neredeyse hiç
devreye girmiyor.

| Metrik | 600 karakter (baseline) | 2500 karakter |
|---|---|---|
| Chunk sayısı | 302 | 285 |
| Recall@1 (genel) | %71.9 | %73.4 |
| Recall@3 (genel) | %89.1 | %89.1 |
| Recall@1 (kafa_karıştırıcı) | %50.0 | %50.0 |
| Recall@1 (eş_anlamlı) | %20.0 | %20.0 |

**Karar: REDDEDİLDİ / üretime alınmadı.** Bu korpus için chunk boyutu, chunking'in
ASIL belirleyicisi (başlık yapısı) değil — bu fikir "mantıklı görünen ama bu korpusta
etkisiz" bir örnek, dürüstçe raporlanıyor. Üretime alınırsa hem karmaşıklık ekler hem de
ölçülebilir bir kazanım getirmez.

## Deney 2: Hibrit arama (BM25 + embedding, Reciprocal Rank Fusion)

**Hipotez:** Salt embedding araması özellikle `kafa_karistirici` (%50) ve `es_anlamli`
(%20) kategorilerinde zayıf — bu kategoriler genelde anahtar kelimenin kendisinin
(economy/business, iç hat/dış hat) ayırt edici olduğu durumlar, tam da BM25'in (kelime
eşleşmesi) güçlü olabileceği bir senaryo.

**Yöntem:** `evaluation/hybrid_search_experiment.py` — `rank_bm25` ile chunk metinleri
üzerinde bir BM25 indeksi kuruldu, embedding tarafı (mevcut `thy_policies` collection,
top-50 aday) ile Reciprocal Rank Fusion (k=60) ile birleştirildi. Tek değişken: sıralama
stratejisi.

**Sonuç — hipotezin TERSİ çıktı:**

| Metrik | Embedding-only (baseline) | Hibrit (BM25+embedding RRF) |
|---|---|---|
| Recall@1 (genel) | %71.9 | %65.6 |
| Recall@3 (genel) | %89.1 | %87.5 |
| Recall@1 (kontrol) | %80.4 | %74.5 |
| Recall@1 (kafa_karıştırıcı) | %50.0 | %50.0 (değişmedi) |
| Recall@1 (eş_anlamlı) | **%20.0** | **%0.0** |

Hibrit arama GENEL performansı düşürdü, hedeflenen iki kategoriden birinde (kafa
karıştırıcı) hiçbir kazanım sağlamadı, diğerinde (eş anlamlı) SIFIRA indirdi. Kök neden:
`es_anlamli` kategorisi TANIM GEREĞİ, sorunun kaynak metinle farklı kelimeler kullandığı
durumları test ediyor — BM25 (saf kelime eşleşmesi) bu kategoride yapısal olarak zayıf
olmak ZORUNDA, çünkü ölçtüğü şey tam olarak "kelime örtüşmesi yokken doğru chunk'ı
bulma" yeteneği. Ayrıca Türkçe'nin eklemeli/çekim yapısı (bagaj/bagajım/bagajını gibi
gövde aynı ama ek farklı) kök bulma (stemming) olmadan BM25'i zaten zayıflatıyor. Eşit
ağırlıklı RRF, bu zayıf sinyali güçlü embedding sinyaliyle karıştırarak top-1 kalitesini
düşürdü.

**Karar: REDDEDİLDİ.** Bu, "mantıklı görünen ama ölçülünce ters etki yapan" bir fikir —
gizlenmeden raporlanıyor. Türkçe'ye özel bir stemmer (örn. Zemberek) veya BM25'e daha
düşük bir ağırlık verilmiş bir füzyon denenirse sonuç değişebilir ama bu, kapsam dışı
bırakılan ayrı bir gelecek çalışma.

## Deney 3: Çelişen-kaynak sezgiselinin kök nedeni ve düzeltmesi

ADR-0004'ün intent-düzeltmesi sonrası e011 ("Do infants need a separate ticket?")
senaryosunda RAG'e artık doğru yönlendiriliyor ama `policy_verification_agent.py`'nin
"çelişen kaynak" sezgiseli gereksiz bir netleştirme sorusu döndürüyordu. Kök neden
araştırıldı:

```
0.7578 | infant_child_passenger_policy.md | Birden fazla bebek / tek refakatçi
0.7517 | infant_child_passenger_policy.md | Koltuk ve puset kuralları
```

Eski `_sources_conflict` mantığı yalnızca "en iyi iki sonucun skoru yakın VE bölüm
başlığı farklı" bakıyordu. Ama burada top-1 ve top-2 AYNI belgenin (`infant_child_
passenger_policy.md`) iki farklı ama ALAKASIZ alt-bölümü — gerçek bir politika çakışması
değil, düşük hassasiyetli bir retrieval sonucu. Sezgisel bunu yanlışlıkla "çakışma"
sanıyordu.

**Düzeltme (`app/agent/policy_verification_agent.py::_sources_conflict`):** "farklı bölüm
başlığı" şartı "farklı KAYNAK DOSYA" ile değiştirildi — gerçek bir politika çakışması
tanım gereği iki AYRI belge arasında olur, aynı belgenin iki alt-başlığı arasında değil.

**Ölçülen etki (manuel doğrulama + tam 48 senaryolu yeniden koşum):**

| Senaryo | Düzeltmeden önce | Düzeltmeden sonra |
|---|---|---|
| e011 (Do infants need a separate ticket?) | Yanlış "çakışma" → gereksiz netleştirme | Düzeldi: artık dürüst "bulamadım" (top-1/top-2 aynı dosya, çakışma YOK) |
| e038 (mil ile yükseltme) | Yanlış "çakışma" → gereksiz netleştirme | Düzeldi: dürüst "bulamadım" (top-1/top-2 aynı dosya, ikisi de alakasız) |
| e039 (ücretli yükseltme) | Doğru "çakışma" → netleştirme | DEĞİŞMEDİ: hâlâ netleştirme (top-1 gerçekten alakalı `paid_business_upgrade_policy.md`, top-2 alakasız ama TESADÜFEN farklı dosyadan — bkz. aşağı) |

48 senaryolu değerlendirmede (`evaluation/run_e2e_evaluation.py` yeniden çalıştırıldı,
ham sonuçlar `e2e_results_before_conflict_fix.json`'da karşılaştırma için korunuyor):
**genel başarı %72.9'da DEĞİŞMEDİ** (id bazında pass/fail/error_category karşılaştırması
sıfır fark gösterdi) — çünkü değerlendirme scripti hem "yanlış netleştirme" hem "dürüst
bulamadım" çıktısını `expected_outcome=grounded_answer` beklentisine göre aynı şekilde
`yanlis_outcome` sayıyor; bu ikisi arasındaki fark (biri kullanıcıyı gereksiz yere
bekletiyor, diğeri dürüstçe teslim oluyor) mevcut ikili başarı/başarısızlık ölçütünde
görünmüyor. Ama **insan devri oranı** ölçülebilir şekilde düştü (%35.4→%31.2, bkz.
`analyze_e2e_results.py`'nin `_HANDOFF_OUTCOMES`'u "clarification"ı sayıyor,
"not_found"ı saymıyor) — e011 ve e038 artık kullanıcıyı gereksiz bir netleştirme
turuna sokmuyor, doğrudan dürüst cevap veriyor. Bu, ikili pass/fail metriğinin
yakalayamadığı ama gerçek bir kullanıcı-deneyimi kazanımı.

**Kalan sınırlama (e039):** Düzeltme, "aynı dosya + farklı bölüm" yanlış pozitifini
giderdi ama "farklı dosya + tesadüfen yakın skor" durumunu gidermedi — e039'da top-1
gerçekten doğru kaynak (`paid_business_upgrade_policy.md`), top-2 tamamen alakasız bir
"Gelinlik" (kabin bagajı istisnası, gelinlik/damatlık taşıma) bölümü ama FARKLI dosyadan
geldiği için hâlâ "çakışma" tetikleniyor. Bu, sezgiselin temelinde yatan varsayımın
(iki kaynağın skoru yakınsa ikisi de "meşru aday"dır) hâlâ kırılgan olduğunu gösteriyor;
tam çözüm muhtemelen bir relevanslık eşiği veya reranker gerektirir (bkz. aşağıdaki
"sonraki adımlar").

**Ayrıca ölçülen, ayrı bir bulgu (e038):** Beklenen kaynak (`miles_redemption_policy.md`)
bu sorgu için ilk 5 retrieval sonucunda HİÇ görünmüyor — bu, çakışma-sezgiseli
düzeltmesiyle ilgisiz, ayrı bir RETRIEVAL RECALL sorunu. Sistem bunu doğru şekilde
halüsinasyon yapmadan "bulamadım" ile karşılıyor ama asıl soruyu cevaplayamıyor.

**Karar: KABUL EDİLDİ ve üretime alındı.** Kök nedene dayalı, dar kapsamlı, ölçülmüş bir
düzeltme — bir yanlış-pozitif kalıbını gidermek için tasarlandı ve tam olarak bunu yaptı;
insan-devri oranında ölçülebilir bir iyileşme sağladı; ikili başarı metriğinde görünmeyen
ama gerçek bir kazanım olduğu için gizlenmeden ayrıca belgelendi. `tests/
test_e2e_scenarios.py`'deki ilgili testler (e011, e038, e039) yeni gerçek davranışı
sabitleyecek şekilde güncellendi (hepsi geçiyor).

## Genel sonuç ve sonraki adımlar

Üç fikirden ikisi (chunk boyutu, hibrit arama) dürüstçe ölçülüp REDDEDİLDİ — ikisi de
"mantıklı görünen ama bu korpusta/dilde işe yaramayan" fikirler oldu, bu bir başarısızlık
değil, ADR-0002/0003'ün kurduğu "ölç, varsayma" disiplininin doğal bir sonucu. Üçüncü
fikir (çelişen-kaynak sezgiseli düzeltmesi) kabul edildi ve KABUL EDİLDİ.

Kalan, çözülmemiş sorunlar (gelecek çalışma için):
- e038 tipi RETRIEVAL RECALL boşlukları (beklenen kaynağın ilk 5'te hiç görünmemesi) —
  bir reranker (30-50 aday getirip en iyi 5-10'u seç) bu sorunu hedefleyebilir ama bu
  ADR'de denenmedi (zaman/kapsam kısıtı, iki negatif deney sonrası).
- e039 tipi "tesadüfen farklı dosyadan gelen alakasız chunk" yanlış pozitifi —
  relevanslık eşiği veya reranker gerektirebilir.
- RAG+claim-decomposition zincirinin ADR-0004'te belgelenen non-determinizmi
  (e005/e010 gibi) bu ADR'nin kapsamı dışında, ayrı bir sorun ailesi.

## İlgili dosyalar

- `evaluation/build_bigger_chunk_collection.py`, `evaluation/hybrid_search_experiment.py`
- `evaluation/eval_results_bigchunk.json`, `evaluation/eval_results_hybrid.json`,
  `evaluation/eval_results_baseline_check.json`
- `app/agent/policy_verification_agent.py::_sources_conflict`
- `evaluation/e2e_results_before_conflict_fix.json`,
  `evaluation/e2e_analysis_before_conflict_fix.json` (düzeltmeden önceki ham sonuçlar)
- `tests/test_e2e_scenarios.py` (e011/e038/e039 testleri güncellendi)
