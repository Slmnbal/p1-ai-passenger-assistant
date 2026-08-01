# ADR 0006: `belirsiz_acikliga_kavusturma` artık RAG'e de uğruyor — intent-gate güvenlik ağı

**Durum:** Kabul edildi (1 Ağustos 2026)

## Bağlam

Selman, önceki turlardaki iyileştirmelere rağmen "model hâlâ vasat çalışıyor" geri
bildirimini verdi ve profesyonel bir RAG sisteminin hata analizini nasıl ikiye
ayırması gerektiğini (1) doğru bilgi modele ulaşıyor mu, (2) doğru bilgi geldiği hâlde
model yanlış mı cevaplıyor — özetleyen kapsamlı bir çerçeve paylaştı.

Bu çerçevenin "hata analizi" tablosu kullanılarak canlı sunucuda (Docker, güncel
model + guardrail zinciriyle) dört gerçek soru test edildi:

| Soru | Beklenen | Gerçekte olan |
|---|---|---|
| "Hamileyken uçağa binebilir miyim?" | politika_bilgi_sorgusu | ✅ Doğru, iyi cevap |
| "Evcil hayvanımla nasıl seyahat ederim?" | politika_bilgi_sorgusu | ✅ Doğru yönlendi (gerçekten çakışan konu: evcil hayvan vs rehber köpek) |
| **"Kayıp bagajım için ne yapmalıyım?"** | politika_bilgi_sorgusu | ❌ `belirsiz_acikliga_kavusturma` — RAG'e HİÇ ulaşmadı |
| **"Öğrenci indirimi var mı?"** | politika_bilgi_sorgusu | ❌ `belirsiz_acikliga_kavusturma` — RAG'e HİÇ ulaşmadı |

En çarpıcı bulgu: **"Kayıp bagajım için ne yapmalıyım?" kelimesi kelimesine
`data/intent/test.json`'da `politika_bilgi_sorgusu` olarak etiketli** (id: m0012) —
model kendi test setindeki bir örneği bile yanlış sınıflandırıyor
(`models/intent_full_10ep/eval_results.json`'da doğrulandı; bu, MODEL_CARD.md'nin
zaten raporladığı %76.3 test doğruluğunun/~%24 hata oranının somut bir örneği, yeni
bir bug değil).

**Kritik gözlem (çerçevenin "hata analizi" tablosuna göre sınıflandırma):** Bu dört
örnekten HİÇBİRİ "doğru chunk geldi, model yanlış cevapladı" (generation problemi)
değil. İkisi doğru çalıştı, ikisi ise retrieval'a HİÇ ULAŞAMADAN, mimarideki bir
intent-gate tarafından engellendi — çerçevenin varsaymadığı bir ek darboğaz katmanı
(çerçeve, her sorunun retrieval'a ulaştığı bir mimari varsayıyor; P1'de intent
sınıflandırıcı retrieval'dan ÖNCE bir kapı görevi görüyor).

## Karar

`planning_agent.py::route_after_planning`, `belirsiz_acikliga_kavusturma`
sınıflandığında artık akışı doğrudan bitirmiyor — `politika_bilgi_sorgusu` ile AYNI
şekilde `retrieval`'a yönlendiriyor. `policy_verification_agent.py::verify_node`'un
iki "gerçekten grounded değil" dalı (LLM'in kendi "bulamadım" cevabı ve retry
tükendiğinde düşük kelime-örtüşmesi) artık orijinal intent
`belirsiz_acikliga_kavusturma` ise generik "Bu konuda kaynaklarımda net bir bilgi
bulamadım" yerine `plan_node`'un zaten ürettiği netleştirme sorusuna dönüyor.

Bu, Adım 6'nın zaten kanıtladığı bir prensibin (bkz. proje planı: "Filodaki uçak
tiplerinin teknik özellikleri nedir?" sorusu fine-tune modeli kandırmıştı ama
retrieval+verification katmanı bunu yakalayıp doğru şekilde reddetmişti) TERSİNE
uygulanması: orada RAG, intent'in KAÇIRDIĞI bir kapsam-dışı soruyu yakalamıştı; burada
RAG, intent'in YANLIŞ ENGELLEDİĞİ bir kapsam-İÇİ soruyu kurtarıyor.

**Neden risksiz:** RAG grounded bir cevap bulamazsa kullanıcı AYNI netleştirme
mesajını görüyor (davranış değişmiyor) — tek fark, arka planda bir RAG denemesi daha
yapılıyor (gecikme maliyeti var, doğruluk maliyeti yok).

## Ölçülen etki

**Birim/entegrasyon testleri (43 canlı test, hepsi geçiyor):** En kritik doğrulama,
`tests/test_e2e_scenarios.py::test_ambiguous_message_always_asks_for_clarification`
(e033-e037: "Bir sorum var", "Yardım lazım", "Ne yapmalıyım?", "Merhaba", "Bilgi almak
istiyorum") — bu GERÇEKTEN belirsiz mesajlar artık RAG'e de uğruyor, YANLIŞLIKLA
grounded bir cevap dönme riski vardı. Ölçüldü: risk gerçekleşmedi, hepsi hâlâ doğru
şekilde netleştirme istiyor.

**48 senaryolu uçtan uca değerlendirme:** 35/48 (%72.9) → 36/48 (%75.0). Ama bu +1'in
gerçek nedeni bu değişiklik DEĞİL — id bazında diff alındığında tek fark `e008`
(zaten `politika_bilgi_sorgusu` olarak doğru sınıflandırılmıştı, ADR-0004'ün belgelediği
RAG+claim-decomposition non-determinizmi yüzünden bu koşuda geçti). **Dürüstçe
belirtilmesi gereken bulgu: mevcut 48 senaryonun HİÇBİRİ orijinal olarak
`belirsiz_acikliga_kavusturma`'ya yanlış sınıflandırılmamıştı** — bu eval seti (Adım 11'de
kurulmuştu) bu spesifik hata kalıbını (bariz politika sorusu → yanlış "belirsiz")
örneklemiyor. Bu, eval setinin bir kapsam boşluğu, düzeltmenin etkisiz olduğu anlamına
gelmiyor.

**Doğrudan, canlı doğrulama (asıl kanıt):** Docker'daki gerçek sunucu üzerinden
`"Kayıp bagajım için ne yapmalıyım?"` tekrar test edildi. Sonuç: intent hâlâ
`belirsiz_acikliga_kavusturma` (değişmedi, beklenmiyordu — intent modeli
değiştirilmedi), AMA artık `retrieval` düğümü GERÇEKTEN çalışıyor (log'da doğrulandı:
`retrieval başladı`/`retrieval bitti` bu istek için artık var, düzeltmeden önce hiç
yoktu). RAG yine de grounded bir cevap bulamadı — ama bu artık AYRI, dürüstçe
tanılanmış bir problem:

```
POLICY ANSWER: Bu konuda kaynaklarımda net bir bilgi bulamadım.
0.610  nonstandard_cabin_baggage_fees_policy.md — Standart dışı kabin bagajı (check-in)
0.594  nonstandard_cabin_baggage_fees_policy.md — Standart dışı kabin bagajı (boarding)
0.561  baggage_interline_transfer_policy.md — Aktarma sırasında bagaj
```

`lost_damaged_baggage_policy.md` (asıl doğru kaynak) ilk 3 sonuca HİÇ girmiyor — bu,
çerçevenin hata tablosundaki **"Doğru belge var ama bulunmuyor → Retrieval problemi"**
satırına birebir uyan, YENİ ve ayrı bir bulgu (ADR-0005'in reddettiği hibrit
arama/chunk boyutu denemelerinden farklı, spesifik bir sorgu-belge eşleşmesi zayıflığı).
Bu düzeltmenin KENDİSİ doğru çalışıyor (RAG'e artık ulaşıyor); ulaştığında bulduğu şey
yetersizse bu retrieval katmanının ayrı bir sınırlaması.

## Kararın gerekçesi

1. Değişiklik dar kapsamlı, tek yönlü (yalnızca `belirsiz_acikliga_kavusturma`'nın
   son-çare fallback metnini değiştiriyor) ve ölçülerek RİSKSİZ bulundu (genuinely
   ambiguous mesajlarda regresyon yok).
2. "Kayıp bagajım için ne yapmalıyım?" gibi somut, gerçek örnekler artık en azından
   RAG'e ULAŞIYOR — daha önce hiçbir şansı yoktu. RAG'in kendisi bulamıyorsa bu artık
   ayrı, tanılanmış bir retrieval-kalite sorunu (bkz. "Sonraki adımlar").
3. 48 senaryolu eval setinin bu değişikliği "göstermemesi" kabul edilebilir — set bu
   hata kalıbını örneklemiyor, bu eval setinin genişletilmesi gereken bir kapsam
   boşluğu olarak ayrıca not edildi (aşağı bkz.).

## Sonraki adımlar (bu ADR'nin AÇIK bıraktığı)

- **Retrieval recall sorunu (`lost_damaged_baggage_policy.md` gibi):** Belirli
  sorgu-belge çiftlerinde ilgili belge ilk-k'ya hiç girmiyor. Reranker (ADR-0005'te
  denenmeyen tek fikir) veya bu belgenin context-prefix embedding'inin gözden
  geçirilmesi gerekebilir.
- **Eval seti kapsam boşluğu:** `evaluation/e2e_scenarios.json`'a, intent
  sınıflandırıcının bariz politika sorularını "belirsiz" sandığı örnekler (bu ADR'nin
  bulduğu somut örnekler dahil) eklenmeli — Adım 11'de bu hata kalıbı hiç
  örneklenmemişti.
- **Query rewriting (ADR/tartışma önceki turda ertelendi):** Takip sorularının
  bağımsız arama sorgusuna çevrilmesi hâlâ yapılmadı.

## İlgili dosyalar

- `app/agent/planning_agent.py::route_after_planning`
- `app/agent/policy_verification_agent.py::verify_node`
- `tests/test_agent.py::test_route_after_planning`
- `evaluation/e2e_results_before_ambiguous_fallback.json`,
  `evaluation/e2e_analysis_before_ambiguous_fallback.json` (düzeltmeden önceki ham sonuçlar)
