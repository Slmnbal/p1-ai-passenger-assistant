"""LangGraph orkestrasyon katmanı — multi-agent mimari (Adım 6, tamamlandı).

- state.py: konuşma geçmişi, aktif intent, tool sonuçları, onay durumu, oturum içi
  kullanıcı tercihleri (agent memory)
- planning_agent.py: fine-tuned intent sınıflandırıcıyla (models/intent_full_10ep)
  akışı belirleyen planlama agent'ı
- retrieval_agent.py: Qdrant + Ollama/llama3.1:8b ile politika sorularını yanıtlayan
  arama/RAG agent'ı (projenin ilk gerçek LLM cevap üretme adımı)
- tool_agent.py: mock uçuş/rezervasyon/check-in API'lerini çağıran tool agent'ı
- policy_verification_agent.py: temel groundedness kontrolü ve çakışan kaynaklarda
  netleştirici soru döndüren politika doğrulama agent'ı (bkz. p1_proje_plani.md İlke 4 —
  tam guardrail Adım 7'nin işi, burada yalnızca bir ön kontrol var)
- graph.py: StateGraph routing + retry/reflection loop (`build_graph()`, `run()`)
"""
