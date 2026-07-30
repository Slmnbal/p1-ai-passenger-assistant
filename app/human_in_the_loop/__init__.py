"""Human-in-the-loop onay katmanı (Adım 8, tamamlandı).

- approval_queue.py: `ApprovalRequest` + in-memory onay kuyruğu (submit/list_pending/
  get/update/reset) — `app/tools/store.py` ile aynı desen
- approval_flow.py: `approve()`/`reject()` — onaylanan talebin ERTELENMİŞ gerçek tool
  çağrısını (cancel/change_date) o anda çalıştırır, reddedilen talep store'u hiç etkilemez

Kural: check-in, ekstra bagaj ve uçuş/fiyat sorgulama gibi geri döndürülebilir/düşük
riskli işlemler `tool_agent.py`'da otomatik tamamlanır; iptal/tarih değişikliği gibi
kritik işlemler artık HEMEN çalıştırılmaz, bu kuyruğa girer (bkz.
`data/policies/checkin_reservation_policy.md` ve `app/tools/reservation.py`'deki
`requires_human_approval` iş kuralı).
"""
