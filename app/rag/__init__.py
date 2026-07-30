"""RAG katmanı: chunking, embedding, Qdrant retriever ve RAG evaluation.

Durum:
- chunking.py: tamamlandı, sandbox'ta gerçek 3 belgeyle test edildi (26 chunk üretti).
- tfidf_retriever.py: tamamlandı, baseline retriever, sandbox'ta test edildi. Gerçek
  bulgular: doğru bölümü bazen genel başlık chunk'ının gerisinde sıralıyor; kapsam dışı
  sorularda (örn. evcil hayvan — corpus'ta yok) düşük skor veriyor (beklenen davranış).
- embeddings.py: yazıldı (HF sentence-transformers) ama internet gerektirdiği için
  sandbox'ta test edilemedi; yerel makinede çalıştırılıp TF-IDF ile karşılaştırılacak.

Sıradaki (Adım 3): vector_store.py (Qdrant client), evaluation.py (Recall@k, MRR,
faithfulness) — TF-IDF vs embedding karşılaştırması burada yapılacak.

İlke 1 (bkz. p1_proje_plani.md): en az 20-30 gerçek belge, ~150-300 chunk,
50-100 soruluk etiketli değerlendirme seti, negatif/kapsam dışı örnekler.
"""
