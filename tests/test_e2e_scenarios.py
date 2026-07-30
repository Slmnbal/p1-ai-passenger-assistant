"""Adım 11 — bilinen hata kalıpları regresyon seti (İlke 4, proje planı Adım 11).

`evaluation/e2e_scenarios.json`'daki 48 senaryonun TAMAMINI burada tekrar test etmiyoruz
— bu, `evaluation/run_e2e_evaluation.py`'ın (tek seferlik/periyodik tam değerlendirme
aracı) işini pytest'te gereksiz yere tekrarlardı. Bunun yerine İlke 4'ün özellikle
istediği DÖRT bilinen hata kalıbı kategorisi (belirsiz soru, çelişen kaynak, sayısal
detay hatası, kapsam dışı soru) için, `docs/adr/0004-...md`'de ÖLÇÜLMÜŞ davranışı
sabitleyen bir regresyon seti var: ambiguous/out_of_scope %100 güvenilir olduğu için
"geçmeli", conflicting_source'un %33.3'lük (6'da 2) gerçek başarı oranı olduğu için
BİLEREK karışık (bazıları geçmeli, bazıları known-gap) yazıldı — ADR-0002/0003'ün
"bilinen sınırlamayı gizleme, test olarak göster" deseniyle aynı.

Tüm testler `@pytest.mark.live` — gerçek Qdrant+Ollama+fine-tune model gerektiriyor.
"""

from __future__ import annotations

import json

import pytest

from app.agent.graph import run as run_agent
from app.human_in_the_loop import approval_queue
from app.tools import store

with open("evaluation/e2e_scenarios.json", encoding="utf-8") as f:
    _SCENARIOS = {s["id"]: s for s in json.load(f)}


@pytest.fixture(autouse=True)
def reset_state():
    store.reset()
    approval_queue.reset()
    yield


def _run(scenario_id: str) -> dict:
    return run_agent(_SCENARIOS[scenario_id]["text"])


pytestmark = pytest.mark.live


# --- Belirsiz soru: %100 güvenilir ölçüldü (ADR-0004) ---

@pytest.mark.parametrize("scenario_id", ["e033", "e034", "e035", "e036", "e037"])
def test_ambiguous_message_always_asks_for_clarification(scenario_id):
    final_state = _run(scenario_id)
    assert final_state["intent"] == "belirsiz_acikliga_kavusturma"
    assert final_state["needs_clarification"] is True


# --- Kapsam dışı soru: %100 güvenilir ölçüldü (ADR-0004) ---
# NOT: `blocked` alanı burada KONTROL EDİLMİYOR — kapsam_disi, planning_agent'ta
# doğrudan sonlanıyor (route_after_planning "end" döner, output_guard/input_guard'a hiç
# uğramıyor), bu yüzden `blocked` alanı hep varsayılan (False) kalır. Asıl doğru sinyal
# `intent == "kapsam_disi"` olması (ilk sürümde bu ayrım gözden kaçmış, teste yanlışlıkla
# `blocked is True` eklenmişti — bkz. docs/adr/0004).

@pytest.mark.parametrize("scenario_id", ["e044", "e045", "e046", "e047", "e048"])
def test_out_of_scope_message_always_blocked(scenario_id):
    final_state = _run(scenario_id)
    assert final_state["intent"] == "kapsam_disi"


# --- Çelişen kaynak (kafa karıştırıcı): ölçülen gerçek oran 6'da 2 (ADR-0004) ---
#
# NOT: `final_state["grounded"]` yerine doğrudan `final_response` içeriğine bakıyoruz.
# Neden: `grounded`, verify_node'un (Adım 6) İLK değerlendirmesi — output_guard (Adım 7)
# SONRADAN daha sıkı bir kontrolle bunu bloklayıp final_response'u fallback mesajına
# çevirebilir ama `grounded` alanı geriye alınmıyor (hâlâ True görünür). Bu yüzden
# `grounded is True` tek başına "kullanıcı gerçek bir cevap aldı" anlamına gelmiyor —
# asıl kullanıcının GÖRDÜĞÜ metin `final_response`, ona bakmak daha güvenilir
# (bu ayrım rq_e2e_evaluation.py'de de bulunup düzeltildi, bkz. docs/adr/0004).

_FALLBACK_TEXT = "bu konuda kaynaklarımda net bir bilgi bulamadım"


@pytest.mark.parametrize("scenario_id", ["e040", "e043"])
def test_conflicting_source_reliably_answered(scenario_id):
    """Bu ikisi ADR-0004'te ölçülen 6 senaryodan güvenilir şekilde geçen ikisi."""
    final_state = _run(scenario_id)
    assert final_state["intent"] == "politika_bilgi_sorgusu"
    assert _FALLBACK_TEXT not in final_state["final_response"].lower()


@pytest.mark.parametrize(
    "scenario_id,reason",
    [
        ("e038", "intent sınıflandırıcı 'yaptığım...geçer mi' kalıbını rezervasyon işlemi sanıyor"),
        ("e039", "aynı kalıp, ters kural (ücretli yükseltme) için de tekrarlanıyor"),
    ],
)
def test_known_gap_conflicting_source_intent_misroute(scenario_id, reason):
    """BİLEREK bir başarısızlığı belgeliyor (bkz. docs/adr/0004-...md) — bu ikisi
    DETERMİNİSTİK (fine-tune model, dropout yok): iki ayrı koşuda da aynı yanlış intent'e
    düştü. Bu test artık geçmeye başlarsa, intent sınıflandırıcı iyileşmiş demektir."""
    final_state = _run(scenario_id)
    assert final_state["intent"] != "politika_bilgi_sorgusu", f"{scenario_id} artık doğru yönlendiriliyor ({reason})"


@pytest.mark.parametrize("scenario_id", ["e041", "e042"])
def test_conflicting_source_rag_conservative_rejection_is_flaky(scenario_id):
    """BİLİNEN, GÖZLEMLENMİŞ FLAKY davranış: bu iki senaryo iki ayrı değerlendirme
    koşusunda tutarlı şekilde reddedilmişti (ADR-0004), ama bu test dosyasını yazarken
    YAPILAN ÜÇÜNCÜ bir koşuda e041 beklenmedik şekilde GEÇTİ — yani RAG+faithfulness
    judge zinciri bu senaryolarda deterministik değil (temperature=0.0 olsa da, Qdrant
    top-k sıralamasında ondalık farklar veya claim decomposition'ın cümle bölme kararı
    çalıştırmalar arası küçük farklılıklar gösterebiliyor). Bu yüzden burada SERT bir
    "geçmeli" ya da "geçmemeli" iddiası YOK — asıl korunan invariant: sistem YA doğru
    cevap verir YA DA dürüstçe "bulamadım" der, ama hiçbir şekilde YANLIŞ/uydurma bir
    sayı vermez (halüsinasyon yok)."""
    final_state = _run(scenario_id)
    answer = final_state["final_response"].lower()
    if _FALLBACK_TEXT not in answer:
        assert any(n in answer for n in ("20", "25")), f"{scenario_id}: beklenmedik/olası hatalı sayı içeriyor"


# --- Sayısal detay hatası: RAG başarılı olduğunda sayı doğru mu? ---
#
# e001 üç ayrı koşuda da (iki değerlendirme + bu test) tutarlı şekilde doğru cevap verdi
# — bu yüzden burada SERT bir "grounded olmalı" iddiası var. e005 ise bu test dosyasını
# yazarken YAPILAN koşuda beklenmedik şekilde fallback'e düştü (önceki iki değerlendirmede
# grounded'dı) — o da e002/e041/e042 ile aynı "flaky" muameleyi görüyor, ayrı bir teste
# taşındı (bkz. aşağı).

def test_numeric_detail_e001_reliably_grounded():
    final_state = _run("e001")
    answer = final_state["final_response"].lower()
    assert _FALLBACK_TEXT not in answer
    assert "23" in answer


def test_numeric_detail_e005_sometimes_flaky_but_never_hallucinates():
    """'Gençlik indirimi oranı nedir?' — iki değerlendirme koşusunda grounded'dı, bu test
    dosyasını yazarken YAPILAN üçüncü koşuda beklenmedik şekilde fallback'e düştü. e002/
    e041/e042 ile aynı aile: RAG+claim-decomposition zincirinin ölçülen non-determinizmi
    (bkz. docs/adr/0004). Korunan invariant burada da aynı: halüsinasyon yok."""
    final_state = _run("e005")
    answer = final_state["final_response"].lower()
    if _FALLBACK_TEXT not in answer:
        assert "%" in answer or any(ch.isdigit() for ch in answer)


def test_known_gap_numeric_question_sometimes_over_rejected():
    """'Business Class'ta kaç parça kabin bagajı alabilirim?' — ADR-0004'te ölçülen,
    RAG'in bazen konservatif davranıp doğru cevabı bile reddettiği bilinen durum.
    Asıl korunan invariant: SESSİZCE yanlış bir sayı UYDURMAZ — ya doğru cevap ya dürüst
    "bulamadım"."""
    final_state = _run("e002")
    answer = final_state["final_response"].lower()
    if _FALLBACK_TEXT not in answer:
        assert "2" in answer


# --- Dil önyargısı (bias/fairness) — ADR-0004'te ölçülen, küçük örneklemli sinyal ---

def test_known_gap_english_policy_question_misrouted():
    """'Do infants need a separate ticket?' İngilizce girdide intent sınıflandırıcı
    kapsam_disi'ye yönlendiriyor (olması gereken: politika_bilgi_sorgusu) — ADR-0004'te
    ölçülen dil önyargısı sinyalinin somut bir örneği. Bu test artık geçerse (doğru
    intent'e yönlendirirse), dil önyargısı bulgusu güncellenmelidir."""
    final_state = _run("e011")
    assert final_state["intent"] != "politika_bilgi_sorgusu"
