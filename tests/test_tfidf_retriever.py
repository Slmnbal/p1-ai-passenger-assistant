import glob

import pytest

from app.rag.chunking import chunk_markdown_file
from app.rag.tfidf_retriever import TfidfRetriever, tokenize


def _load_real_policy_chunks():
    chunks = []
    for path in sorted(glob.glob("data/policies/*.md")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        chunks.extend(chunk_markdown_file(path, text))
    return chunks


@pytest.fixture(scope="module")
def retriever():
    chunks = _load_real_policy_chunks()
    assert len(chunks) > 0, "data/policies/*.md bulunamadı veya boş"
    return TfidfRetriever(chunks)


def test_tokenize_removes_stopwords_and_short_tokens():
    tokens = tokenize("Bu bir ve ile test metnidir")
    assert "bu" not in tokens
    assert "ve" not in tokens
    assert "test" in tokens


def test_delay_question_returns_delay_section_first(retriever):
    results = retriever.search("Uçuşum 4 saat gecikirse ne hakkım var?", top_k=3)
    top_titles = [c.section_title for c, _ in results]
    assert any("gecikme" in t.lower() or "tehir" in t.lower() for t in top_titles)


def test_checkin_question_returns_checkin_content(retriever):
    results = retriever.search("Online check-in ne zaman açılıyor?", top_k=3)
    combined_text = " ".join(c.text.lower() for c, _ in results)
    assert "check-in" in combined_text


def test_out_of_scope_question_has_low_confidence(retriever):
    """İlke 4: kapsam dışı bir soru (kargo taşımacılığı — corpus'ta yok) yüksek skor almamalı.

    Not: Bu test iki kez negatif kontrol sorusunu değiştirmek zorunda kaldı — önce "evcil
    hayvanla seyahat", sonra "seyahat sigortası" — çünkü corpus genişledikçe (Adım 1'in
    zorunlu genişletmesi) her ikisi de gerçekten kapsam içine girdi. Bu bir hata değil,
    negatif kontrolün korpusla birlikte güncellenmesi gerektiğinin kanıtı. Turkish Cargo
    (hava kargo taşımacılığı) THY'nin ayrı bir iş kolu; bu proje yalnızca yolcu
    politikalarını kapsıyor, bu yüzden hâlâ geçerli bir negatif kontrol.
    """
    results = retriever.search("Uluslararası kargo gönderimi için nasıl fiyat teklifi alabilirim?", top_k=1)
    top_score = results[0][1] if results else 0.0
    assert top_score < 0.3, (
        f"Kapsam dışı soru beklenenden yüksek skor aldı ({top_score:.3f}); "
        "bu ya corpus genişledi (iyi) ya da tokenizer yanlış pozitif üretiyor demektir."
    )
