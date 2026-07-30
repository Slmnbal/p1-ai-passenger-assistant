# P1 — AI Passenger Experience Assistant

[![CI](https://github.com/Slmnbal/p1-ai-passenger-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Slmnbal/p1-ai-passenger-assistant/actions/workflows/ci.yml)

Bagaj hakkı, gecikme/iptal, check-in ve rezervasyon sorularını gerçek kaynaklara (SHY-YOLCU
yönetmeliği + Turkish Airlines'ın yayınladığı politika sayfaları) dayanarak yanıtlayan,
kritik işlemleri insan onayına gönderen bir agentic RAG sistemi.

**Bağımsız bir portföy projesidir.** Turkish Airlines / Turkish Technology ile resmi bir
bağlantısı yoktur; ilgili iş ilanlarındaki teknik yeterlilikleri göstermek amacıyla
hazırlanmıştır. Mock uçuş verisi Turkish Airlines'ın gerçek, kamuya açık rota ağına dayanır;
PNR/rezervasyon/ödeme gibi kişisel veriler sentetiktir (bkz. `data/mock_flights.json`).

Durum: proje iskeleti hazır; `data/policies/` altında 51 gerçek THY/SHY-YOLCU kaynak
belgesi (**302 chunk** — plan aralığının üst sınırı olan 300'ün üzerinde) var — bagaj
(genel/kabin/ekstra/kayıp-hasarlı/ABD/kısıtlamalar/müzik-spor ekipmanı/aktarma-interline/
standart dışı ücretlendirme), gecikme/iptal (SHY-YOLCU + THY operasyonel süreçleri),
check-in/rezervasyon (temel + detaylı işlemler + isim düzeltme + biniş kapısı kuralları),
engelli/hasta yolcu, bebek/çocuk yolcu, hamile yolcu, şehit yakınları/gaziler, öğrenci,
asker indirimi, vefat/cenaze, transfer/transit + vize/APIS, ücret koşulları, ödeme,
Miles&Smiles (genel/üyelik/mil kazanma-harcama-satın alma/kredi kartı/TK Cüzdan/Star
Alliance ortaklıkları), uçuş paketleri (iç/dış hat), fiyat sabitleme, mobil uygulama,
koltuk seçimi, özel yemek/ekstra diz mesafesi, uçak içi wifi/eğlence, lounge erişimi,
e-vize (BAE), seyahat sigortası (XCover), hediye kart, Corporate Club (avantaj/üyelik/
başvuru/Elite kart/portal), araç kiralama/otel ortaklıkları, tatil paketleri, ücretli
Business Upgrade konularını kapsıyor (bkz. `docs/adr/`). Bu hâlâ gerçek üretim ölçeğinin
(THY'nin tüm ülke bazlı vize sayfaları, kargo/Turkish Cargo, vb. dahil olduğu çok daha
geniş bir bilgi tabanının) altında — amaç THY'yi birebir
kopyalamak değil, retrieval pipeline'ının metodolojisini (chunking, embedding seçimi,
negatif kontrol, ölçüm) temsili ve çeşitli bir kaynak kümesinde kanıtlamak.

**Adım 3 tamamlandı** (28 Temmuz 2026): Qdrant Docker'da çalışıyor, 302 chunk yüklü,
72 soruluk etiketli değerlendirme setiyle (`evaluation/eval_questions.json`) ölçüldü.
Context-prefixed embedding kararıyla Recall@1 %56→%72, Recall@3 %80→%89'a çıkarıldı
(`docs/adr/0002-context-prefix-ve-guven-esigi-siniri.md`). Kapsam dışı sorularda güvenilir
"bilmiyorum" davranışı retrieval katmanında çözülemedi — bilinçli olarak Adım 7'ye
(LLM tabanlı groundedness kontrolü) ertelendi, bu bir eksiklik değil kanıtlanmış bir karar.

**Adım 4 tamamlandı** (29-30 Temmuz 2026): 6 sınıflı intent taksonomisi (`data/intent/`),
165 örnekli veri seti + EDA, 5 yaklaşımın (kural tabanlı, frozen-probe BERT, full
fine-tune 10/20 epoch, doğrudan LLM routing) tam metriklerle (confusion matrix,
precision/recall/F1, Macro F1, ROC-AUC, kalibrasyon) karşılaştırılması. Hipotez
doğrulandı: fine-tune edilmiş model kritik sınıfta (`rezervasyon_islem_talebi`) %100,
doğrudan LLM routing %88.9 doğruluk. Kalibrasyon sorunu yine tespit edildi (RAG'deki
ADR-0002 bulgusuyla aynı aile). Veri/model kartı ve MLflow'da (`localhost:5001`) kayıtlı
5 deney mevcut (`data/intent/DATASET_CARD.md`, `MODEL_CARD.md`).

**Adım 5 tamamlandı** (30 Temmuz 2026): Mock tool/API katmanı — uçuş arama/durum,
rezervasyon (iptal/tarih değişikliği/ekstra bagaj), check-in, politika sorgulama
(RAG'e köprü). Her tool Pydantic şemasıyla tanımlı (`app/tools/schemas.py`), HTTP
katmanından bağımsız (`app/main.py` sadece `None`→404 çevirisi yapar). 27 test geçiyor;
canlı sunucuda da (uçuş arama, check-in, 404, gerçek Qdrant sorgusu) doğrulandı.

**Adım 6 tamamlandı** (30 Temmuz 2026): LangGraph multi-agent orkestrasyon (`app/agent/`)
— planlama (fine-tune intent modeli), RAG/arama (Qdrant + Ollama/llama3.1:8b ile
**projenin ilk gerçek LLM cevap üretimi**), tool/rezervasyon ve politika doğrulama
agent'ları `StateGraph` ile bağlandı; ungrounded cevaplarda top_k artırılarak 1 retry,
sonra dürüst "bulamadım" fallback'i var. 61 test geçiyor (34 yeni). Canlı doğrulamada
çarpıcı bir bulgu: intent modeli "Filodaki uçak tiplerinin teknik özellikleri nedir?"
sorusunu yanlış sınıflandırdı (düşük güvenle, %50.6) ama RAG/doğrulama katmanı bunu
yakalayıp doğru şekilde reddetti — tek bir sınıflandırıcıya güvenmemenin (çok-agent
mimarinin) somut kanıtı.

**Adım 7 tamamlandı** (30 Temmuz 2026): Guardrail katmanı (`app/guardrails/`) — injection
tespiti, iş kuralı kontrolü, sayısal halüsinasyon kontrolü. Asıl değer üç ÖLÇÜLMÜŞ deneyde:
(1) retrieval-skor eşiği denendi, kontrol/kapsam-dışı dağılımları tamamen iç içe çıktı,
kullanılmadı; (2) intent güven eşiği zayıf ama gerçek bir sinyal, sert blokaj yerine
gözlem bayrağına çevrildi; (3) llama3.1:8b'yi ikinci bir LLM-judge olarak faithfulness
kontrolünde kullanma denemesi 5 prompt varyasyonuyla test edildi, güvenilir bulunmadı
(doğru bir paraphrase cevabı bile reddediyor ya da açık bir uydurmayı onaylıyor) — bu
yüzden bloklayıcı değil sadece gözlem amaçlı tutuldu, asıl kontrol deterministik sayı
karşılaştırması oldu (`docs/adr/0003-...md`). 19 senaryoluk güvenlik test seti, "onay
istemeden iptal et" gibi bir denemenin iş kuralını gerçekten bypass edemediğinin kanıtı
dahil.

**Aynı gün ikinci tur (Deney 4-5):** Faithfulness judge'ı düzeltmek için whole-answer
değerlendirme yerine "önce iddialara böl, her iddiayı ayrı doğrula" (claim decomposition)
tekniği denendi — doğruluk ~%50 (rastgele) seviyesinden **%90**'a çıktı, artık bloklayıcı.
Prompt-injection dedektörüne embedding-tabanlı bir anlamsal katman eklendi — bilinen bir
parafraz boşluğunu kapattı ama stres testinde iki yeni sınırlama ortaya çıktı: roleplay/
hikaye-çerçeveli dolaylı saldırılar hâlâ kaçabiliyor, masum bir meta-soru yanlışlıkla
flaglenebiliyor — ikisi de gizlenmeden test setine eklendi. 95 test geçiyor (34 yeni).

**Adım 8 tamamlandı** (30 Temmuz 2026): Human-in-the-loop onay kuyruğu
(`app/human_in_the_loop/`) — rezervasyon iptali/tarih değişikliği artık HEMEN
çalışmıyor, `approval_queue.submit()` ile bekletiliyor; gerçek mutasyon yalnızca
`approval_flow.approve()` çağrıldığında oluyor, `reject()` store'u hiç değiştirmiyor.
Entegrasyon sırasında kritik bir regresyon bulundu: Adım 7'nin injection dedektörü,
anchor cümlelerindeki tasarım hatası yüzünden sistemin kendi meşru iptal talebini
("SYN3C4D rezervasyonumu iptal etmek istiyorum") yanlışlıkla blokluyordu — yalnızca
gerçek uçtan uca testte ortaya çıktı, izole birim testleri yakalayamadı. Anchor'lar
düzeltilip 7 yeni "sistemin kendi iş talebi" negatif örneğiyle yeniden ölçüldü
(`docs/adr/0003-...md` Deney 5b). 111 test geçiyor (16 yeni).

**Adım 9 tamamlandı** (30 Temmuz 2026): Gözlemlenebilirlik katmanı (`app/observability/`)
— JSON yapılandırılmış loglama (`request_id` korelasyonuyla), Langfuse'a gerçek LLM/agent
trace (her node bir span, her Ollama çağrısı token/latency'li bir generation), Prometheus
metrikleri (intent bazlı istek sayısı, guardrail blok sayısı, node gecikmesi, bekleyen
onay sayısı) ve dosya tabanlı audit log (`actor`: `system` vs `human_approver`). Langfuse
(self-hosted, Postgres ile), Prometheus ve Grafana gerçekten Docker'da ayağa kaldırılıp
canlı 5 senaryoyla doğrulandı: Langfuse'ta her turun tam trace ağacı (node span'leri +
LLM generation'ları) görüldü, Prometheus gerçek zamanlı veri topladı, Grafana'daki
provisioned dashboard bu veriyi gösterdi (ekran görüntüsüyle doğrulandı), audit log hem
sistem hem insan onayı kayıtlarını doğru ayırt etti. 119 test geçiyor (8 yeni).

**Adım 10 tamamlandı** (30 Temmuz 2026): `/chat` endpoint'i (`app/main.py`) tüm agent
katmanını tek bir HTTP çağrısında birleştiriyor; `/approvals/*` endpoint'leri Adım 8'in
onay kuyruğunu HTTP'den kullanılabilir yapıyor. Dockerfile gerçekten build edilip
`docker compose up` ile çalıştırıldı — bu sırada iki gerçek sorun bulunup düzeltildi:
(1) `QdrantRetriever`'ın `localhost:6333`'e hard-code'lu olması (container'da kendi
kendine bağlanmaya çalışıyordu) `QDRANT_URL` ortam değişkenine çevrildi; (2) model
ağırlıkları (422MB) GitHub'ın 100MB dosya sınırını aştığı için `.gitignore`'a alındı,
o modeli kullanan testler `@pytest.mark.live` ile CI dışı bırakıldı. Proje git'e alınıp
public bir depoya (yukarıdaki linke) push edildi ve **GitHub Actions CI gerçekten
çalışıp yeşil geçti** (ruff lint + 108 test). 129 test (108'i CI'da).

**Sonradan tamamlanan düzeltme:** İlk sürümde container'lı `app`, host'ta native
çalışan Ollama.app'e `host.docker.internal` üzerinden ulaşamıyordu (macOS Docker
Desktop nüansı) — bu, projeyi başka bir ekibe/makineye verildiğinde de tekrar
çıkabilecek, PORTATİF olmayan bir bağımlılıktı. Çözüm: Ollama artık `app` ile AYNI
docker-compose ağının bir parçası (`ollama` servisi, modeli kendi içine çekiyor) —
`app` host'a hiç çıkmadan `ollama:11434` adresine ulaşıyor. Bu, projeyi teslim alan
HERKESİN (Ollama kurulu olsun olmasın) tek komutla (`docker compose up`) çalışan bir
sistem elde etmesini sağlıyor. Bunu doğrularken **ayrı, gerçek bir kısıt ölçüldü**:
llama3.1:8b yüklendiğinde ~5.2GB RAM kullanıyor; Docker Desktop'ın varsayılan bellek
ayarı (bu makinede 7.75GB) tüm servisler (Qdrant/Postgres/Langfuse/Prometheus/Grafana/
MLflow/app) + modelin YÜKLENME anındaki ek bellek ihtiyacıyla birlikte yetmeyip bir
OOM (bellek yetersizliği) hatası verdi. Docker Desktop'a daha fazla bellek ayırmak
(10-12GB+) ya da kısıtlı makinelerde mlflow/prometheus/grafana'yı geçici durdurmak
bunu çözüyor — bu, herhangi bir LLM'i yerelde container'da çalıştıran her kurulum
için standart, beklenen bir kaynak gereksinimi (bkz. `docker/docker-compose.yml`
`ollama` servisi notu).

Detaylı adım planı için bkz. proje kök klasöründeki `p1_proje_plani.md`.

## Mimari

```
Kullanıcı → FastAPI (/chat) → Intent Sınıflandırma (HF fine-tuned)
        ├─ RAG Retrieval (Qdrant + embeddings)        ─┐
        └─ Tool Çağırma (mock uçuş/rezervasyon API)    ├─ LangGraph Orchestrator
                                                         │  (Planlama / Arama-RAG /
                                                         │   Rezervasyon-Tool /
                                                         │   Politika Doğrulama agent'ları)
                                                         ↓
                                            Guardrail Doğrulama (şema, iş kuralı,
                                            zorunlu kaynak atfı, sayısal şablon enjeksiyonu)
                                                         │
                                    düşük risk ──────────┼────────── kritik işlem
                                        │                              │
                                        ↓                    İnsan Onayı (onay/red arayüzü)
                                        └────────────→ Yanıt + Audit Log ←┘
                                                    (Langfuse trace, Prometheus/Grafana)
```

Neden bu şekilde: ilan "planlama, arama, politika sorumluluklarını ayrı uzman agent'lara
bölerek çalıştırma" (multi-agent mimari) istiyor; bu yüzden LangGraph içinde tek bir
monolitik node yerine dört uzmanlaşmış alt-agent var. Politika Doğrulama agent'ı ayrıca
faithfulness/groundedness kontrolünü ve belirsiz sorularda netleştirme mantığını çalıştırır
(bkz. proje planı, İlke 4).

## Teknoloji seçimleri ve gerekçesi (hepsi ücretsiz/yerel)

| Katman | Seçim | Neden |
|---|---|---|
| LLM | Ollama (Llama 3.1 / Qwen2.5 / Phi-3) | `openai` SDK ile OpenAI-uyumlu yerel API; ücretsiz |
| Embeddings | Hugging Face sentence-transformers | Yerel/CPU, ücretsiz |
| Vector DB | Qdrant (self-hosted Docker) | Ücretsiz, ilanın açık gereksinimi |
| Orkestrasyon | LangGraph | State/routing/retry/multi-agent, ilanın ana kanıtı |
| Fine-tuning | Hugging Face Transformers/Datasets | Yerel/CPU, ücretsiz |
| Deney takibi (Adım 4) | MLflow (self-hosted, Docker) | Ücretsiz; ilan matrisinde ML/AI rolü için "Ana" önemde istenen bir gereksinim |
| Trace/observability | Langfuse (self-hosted) + Prometheus/Grafana | LangSmith'in ücretli katmanı yerine açık kaynak alternatif |
| Deployment | Docker → kind/minikube → OpenShift Local (CRC) | Gerçek cloud hesabı açmadan Kubernetes/OpenShift kanıtı |
| CI | GitHub Actions (ücretsiz katman) | Lint + test otomasyonu |

## Klasör yapısı

```
app/
  agent/            LangGraph state, node'lar, multi-agent routing
  rag/               chunking, embedding, Qdrant retriever, RAG evaluation
  intent/            kural tabanlı router + HF fine-tuned sınıflandırıcı
  tools/             mock uçuş/rezervasyon/check-in/politika API'leri
  guardrails/        şema doğrulama, iş kuralı, kaynak atfı, sayısal şablon enjeksiyonu
  human_in_the_loop/ onay kuyruğu ve akışı
  observability/     structured logging, audit log, Langfuse/Prometheus entegrasyonu
data/
  policies/          RAG kaynak belgeleri (gerçek SHY-YOLCU + Turkish Airlines metinleri)
  mock_flights.json  gerçek rota ağına dayanan mock uçuş verisi
evaluation/          RAG ve intent değerlendirme scriptleri, test setleri
docker/              Dockerfile, docker-compose.yml, prometheus.yml, grafana provisioning
.github/workflows/   CI (lint + test) — bkz. yukarıdaki durum rozeti
tests/               pytest test paketi
```

## Çalıştırma (yerel makinede)

**Sistem gereksinimi:** Docker Desktop'a en az **10-12GB RAM** ayırın (Settings →
Resources → Memory) — llama3.1:8b container'da yüklendiğinde tek başına ~5.2GB
kullanıyor, geri kalan servislerle (Qdrant/Langfuse/Prometheus/Grafana/MLflow)
birlikte varsayılan ayarlarda (genelde ~8GB) OOM hatası verebilir (bkz. `docker/
docker-compose.yml`'deki `ollama` servisi notu — bu makinede ölçülüp doğrulandı).

```bash
git clone https://github.com/Slmnbal/p1-ai-passenger-assistant.git
cd p1-ai-passenger-assistant
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # yerel Ollama/Qdrant/Langfuse adreslerini kontrol edin

# Qdrant + MLflow + Ollama + Langfuse (+Postgres) + Prometheus + Grafana
docker compose -f docker/docker-compose.yml up -d

uvicorn app.main:app --reload --port 8001   # 8000 değil — bkz. not aşağıda
```

**Not (port 8001):** Bu geliştirme makinesinde 8000 portu ilgisiz başka bir projenin
sürecine ait olabilir; bu yüzden P1'i 8001'de çalıştırıyoruz —
`docker/prometheus/prometheus.yml`'deki scrape hedefi de buna göre ayarlı. Kendi
makinenizde 8000 boşsa doğrudan onu da kullanabilirsiniz (bu durumda
`prometheus.yml`'i güncelleyin).

**Erişim noktaları (hepsi ayakta olduğunda):**
- API: `http://localhost:8001` (`/health`, `/metrics`, `/tools/*`)
- Qdrant: `http://localhost:6333` · MLflow: `http://localhost:5001`
- Langfuse: `http://localhost:3000` (kullanıcı: `selman@local.dev` / `p1-local-dev-password`)
- Prometheus: `http://localhost:9090` · Grafana: `http://localhost:3001` (`admin`/`admin`, "P1" klasöründe hazır dashboard)

## İlgili belgeler

- `p1_proje_plani.md` — 14 adımlık detaylı build planı, ilkeler ve kısıtlar
- `data/policies/*.md` — RAG kaynak belgeleri (kaynak URL + erişim tarihi ile)
