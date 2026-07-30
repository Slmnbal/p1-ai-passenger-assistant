"""Mock araçlar (tool/API) katmanı.

Doldurulacağı adım: Adım 5
- flight_search.py: uçuş arama mock endpoint'i (data/mock_flights.json üzerinden)
- reservation.py: rezervasyon değişikliği/iade mock endpoint'i (data/mock_reservations.json)
- checkin.py: check-in mock endpoint'i
- policy_lookup.py: politika sorgulama (RAG'e köprü)
- schemas.py: her tool için girdi/çıktı JSON şeması

Her tool bağımsız olarak pytest ile test edilebilir olacak (bkz. tests/).
"""
