"""Intent siniflandirma icin ornek mesaj veri setini uretir.

Bu veri seti GERCEK kullanici trafigi degildir (P1 henuz canliya alinmadi) — yazar
tarafindan, projenin gercek 51 belgelik RAG korpusundaki konulara ve gercek sistem
mimarisindeki routing kararlarina (bkz. README mimari semasi) dayanarak olusturulmus
gercekci ornek cumlelerdir. Bu, NLU sistemlerinde standart bir "cold-start" yontemidir;
data/intent/DATASET_CARD.md'de acikca belirtilir.

Calistirma: python data/intent/build_dataset.py
Cikti: data/intent/messages.json
"""

from __future__ import annotations

import json

# Her ogenin (metin, dil) seklinde. dil: "tr", "en", "mixed"

POLITIKA_BILGI_SORGUSU = [
    ("Ekonomi sınıfında bagaj hakkım kaç kilo?", "tr"),
    ("Kabin bagajının boyut sınırı nedir?", "tr"),
    ("Uçuşum 3 saat gecikirse tazminat alabilir miyim?", "tr"),
    ("Evcil hayvanla seyahat edebilir miyim?", "tr"),
    ("Kaç yaşındaki çocuk refakatsiz uçabilir?", "tr"),
    ("Hamileyken uçağa binebilir miyim?", "tr"),
    ("Online check-in ne zaman açılıyor?", "tr"),
    ("Mil ile ödül bilet nasıl alınır?", "tr"),
    ("Şehit yakınları indirimi var mı?", "tr"),
    ("Öğrenci indirimi kaç para?", "tr"),
    ("Kredi kartımla taksitli ödeme yapabilir miyim?", "tr"),
    ("Kayıp bagajım için ne yapmalıyım?", "tr"),
    ("Business class'ta kaç parça bagaj alabilirim?", "mixed"),
    ("Rehber köpekle uçabilir miyim?", "tr"),
    ("Do infants need a separate ticket?", "en"),
    ("What's the baggage allowance for economy class?", "en"),
    ("How many hours before can I check in online?", "en"),
    ("Is there a discount for military families?", "en"),
    ("Kaç yaşından itibaren bebek yolcu sayılır?", "tr"),
    ("Uçakta wifi ücretli mi?", "tr"),
    ("Fiyat sabitleme nasıl çalışıyor?", "tr"),
    ("TK Cüzdan nedir?", "tr"),
    ("Seyahat sigortası kim sağlıyor?", "tr"),
    ("Miles&Smiles üyeliği ücretli mi?", "mixed"),
    ("Acil çıkış koltuğuna kimler oturabilir?", "tr"),
    ("Özel yemek talebi nasıl yapılır?", "tr"),
    ("Golf ekipmanımı nasıl taşırım?", "tr"),
    ("Uçuş paketleri arasındaki fark ne?", "tr"),
    ("Elite kart avantajları neler?", "mixed"),
    ("Dubai vizesi almam gerekiyor mu?", "tr"),
]

UCUS_SORGULAMA = [
    ("THY1982 sefer sayılı uçuşum rötarda mı?", "tr"),
    ("İstanbul-Ankara arası bugün kaç uçuş var?", "tr"),
    ("Uçağım hangi kapıdan kalkıyor?", "tr"),
    ("Yarınki İzmir uçuşunun saati kaç?", "tr"),
    ("Uçuşumun durumunu öğrenebilir miyim?", "tr"),
    ("PNR ABC123 ile uçuş bilgilerimi görebilir miyim?", "tr"),
    ("Is my flight delayed?", "en"),
    ("What time does flight TK1234 depart?", "en"),
    ("Uçağım hangi tipte, geniş gövdeli mi?", "tr"),
    ("Bu hafta İstanbul-Londra uçuşları dolu mu?", "tr"),
    ("Uçuşumda boş koltuk var mı?", "tr"),
    ("Rezervasyonumu nasıl görüntülerim?", "tr"),
    ("Check-in durumumu kontrol edebilir miyim?", "mixed"),
    ("Uçağım ne zaman iniyor?", "tr"),
    ("Bagajım hangi bantta çıkacak?", "tr"),
    ("Flight status for TK123 please", "en"),
    ("Kaç dakika gecikme var şu an?", "tr"),
    ("Uçuşum hâlâ zamanında mı?", "tr"),
    ("Hangi terminalden kalkıyorum?", "tr"),
    ("Uçağımın kapı numarası değişti mi?", "tr"),
    ("Bugünkü tüm İstanbul çıkışlı uçuşları görebilir miyim?", "tr"),
    ("Uçuşumla ilgili bir değişiklik var mı?", "tr"),
    ("Rezervasyon kodum ile bilet detaylarımı sorgula", "tr"),
    ("Kaç yolcu kapasiteli bir uçakla gidiyorum?", "tr"),
    ("Aktarma uçuşum ne kadar sürede?", "tr"),
    ("Bu PNR'da kaç yolcu var?", "mixed"),
    ("Uçuş numaramı unuttum, nasıl bulurum?", "tr"),
    ("Havalimanına ne zaman gelmeliyim?", "tr"),
    ("Uçağım kalkmadan önce son durumu öğrenebilir miyim?", "tr"),
    ("Bilet numaramla uçuşumu sorgulayabilir miyim?", "tr"),
]

REZERVASYON_ISLEM_TALEBI = [
    ("Biletimi iptal etmek istiyorum", "tr"),
    ("Uçuş tarihimi değiştirebilir miyim?", "tr"),
    ("Rezervasyonumu iade almak istiyorum", "tr"),
    ("Biletimin adını düzeltmem gerekiyor", "tr"),
    ("Koltuğumu değiştirmek istiyorum", "tr"),
    ("Ekstra bagaj satın almak istiyorum", "tr"),
    ("Business class'a yükseltme yapabilir miyim?", "mixed"),
    ("Uçuşumu başka bir tarihe erteleyebilir miyim?", "tr"),
    ("I want to cancel my reservation", "en"),
    ("Can I get a refund for my ticket?", "en"),
    ("Rezervasyonuma bebek eklemek istiyorum", "tr"),
    ("Biletimi başka bir güzergaha çevirebilir miyim?", "tr"),
    ("Özel yemek talebimi eklemek istiyorum", "tr"),
    ("Koltuk numaramı değiştir", "tr"),
    ("Uçuşumu iptal edip parayı geri istiyorum", "tr"),
    ("Rezervasyonuma yolcu eklemek istiyorum", "tr"),
    ("Biletimi TK Cüzdan ile ödemek istiyorum", "mixed"),
    ("Mille kabin yükseltme yapmak istiyorum", "tr"),
    ("Uçuşumu değiştirmek için ne yapmalıyım", "tr"),
    ("Cancel my flight please", "en"),
    ("I need to change my flight date", "en"),
    ("Rezervasyonumdaki hatalı bilgiyi düzeltmek istiyorum", "tr"),
    ("Fiyat sabitleme yapmak istiyorum", "tr"),
    ("Ödül bilet almak istiyorum", "tr"),
    ("Uçuşuma refakatçi eklemek istiyorum", "tr"),
    ("Vefat nedeniyle biletimi iptal etmek istiyorum", "tr"),
    ("Rezervasyonumu gruba dönüştürmek istiyorum", "tr"),
    ("Ekstra diz mesafeli koltuk satın almak istiyorum", "tr"),
    ("Biletimi hediye kartla ödemek istiyorum", "tr"),
    ("Uçuşumu iptal edersem kesinti olur mu, iptal etmek istiyorum", "tr"),
]

CHECKIN_TALEBI = [
    ("Check-in yapmak istiyorum", "mixed"),
    ("Biniş kartımı almak istiyorum", "tr"),
    ("Online check-in yapabilir miyim?", "mixed"),
    ("Kiosktan check-in nasıl yapılır?", "mixed"),
    ("Uçuşuma check-in yaptım mı kontrol edebilir miyim?", "mixed"),
    ("I want to check in for my flight", "en"),
    ("Mobil biniş kartı alabilir miyim?", "tr"),
    ("Check-in için hangi belgeler gerekiyor?", "mixed"),
    ("Koltuğumu check-in sırasında seçebilir miyim?", "mixed"),
    ("Bagajımı check-in sırasında teslim etmek istiyorum", "mixed"),
    ("Grubum için check-in yapmak istiyorum", "mixed"),
    ("Çocuğumun check-in işlemini yapmak istiyorum", "mixed"),
    ("Can I check in online now?", "en"),
    ("Check-in işlemimi iptal edebilir miyim?", "mixed"),
    ("Biniş kartımı yazdırmam gerekiyor mu?", "tr"),
    ("Uçuşuma daha check-in yapmadım, şimdi yapabilir miyim?", "mixed"),
    ("Kontuardan check-in yapmak istiyorum", "mixed"),
    ("Check-in sırasında acil çıkış koltuğu seçebilir miyim?", "mixed"),
    ("Refakatsiz çocuğum için check-in nasıl yapılır?", "mixed"),
    ("Check-in yaptıktan sonra koltuk değiştirebilir miyim?", "mixed"),
    ("Uçağa check-in yapmam gerekiyor, yardımcı olur musunuz?", "mixed"),
    ("Please check me in for tomorrow's flight", "en"),
    ("Check-in kontuarı nerede?", "mixed"),
    ("Biletimle check-in yapamıyorum, sorun ne olabilir?", "mixed"),
    ("Check-in son ne zamana kadar açık, hemen yapmak istiyorum", "mixed"),
]

BELIRSIZ_ACIKLIGA_KAVUSTURMA = [
    ("Bagaj hakkım ne kadar?", "tr"),
    ("Check-in ne zaman açılıyor?", "mixed"),
    ("Kabin yükseltme yapmak istiyorum", "tr"),
    ("İndirim var mı?", "tr"),
    ("Koltuk seçimi ücretsiz mi?", "tr"),
    ("Bileti iptal edince ne kadar iade alırım?", "tr"),
    ("Kaç kilo bagaj hakkım var?", "tr"),
    ("Uçuşum gecikirse ne olur?", "tr"),
    ("Yardım lazım", "tr"),
    ("Bir sorum var", "tr"),
    ("Bilet almak istiyorum ama emin değilim", "tr"),
    ("Uçuş paketi hangisi bana uygun?", "tr"),
    ("Ne yapmalıyım?", "tr"),
    ("Bagaj sınırını aştım galiba, ne olur?", "tr"),
    ("Çocuğum için indirim var mı?", "tr"),
    ("How much luggage can I bring?", "en"),
    ("Salonu kullanabilir miyim?", "tr"),
    ("İptal etmek istiyorum", "tr"),
    ("Ücretsiz mi bu?", "tr"),
    ("Uygun mudur?", "tr"),
    ("Bu durumda ne yapabilirim?", "tr"),
    ("Hakkım var mı?", "tr"),
    ("Fiyat ne kadar?", "tr"),
    ("Değişiklik yapabilir miyim?", "tr"),
    ("Kurallar nedir?", "tr"),
]

KAPSAM_DISI = [
    ("Uluslararası kargo göndermek istiyorum", "tr"),
    ("Kabin memuru olmak için nasıl başvururum?", "tr"),
    ("THY hisse senedi ne kadar?", "tr"),
    ("Pilot olmak istiyorum, ne yapmalıyım?", "tr"),
    ("Uçağın teknik bakımı ne sıklıkla yapılır?", "tr"),
    ("Yatırımcı ilişkileri ile görüşmek istiyorum", "tr"),
    ("Filodaki uçak tiplerini öğrenebilir miyim?", "tr"),
    ("THY'nin sürdürülebilirlik raporunu istiyorum", "tr"),
    ("Turkish Cargo ile iletişime geçmek istiyorum", "mixed"),
    ("Şirketinizde staj imkanı var mı?", "tr"),
    ("Bugün hava durumu nasıl?", "tr"),
    ("En yakın restoran nerede?", "tr"),
    ("Bana bir şaka yapar mısın?", "tr"),
    ("Kredi kartı borcumu nasıl öderim?", "tr"),
    ("Do you sell tickets to the moon?", "en"),
    ("THY'nin genel merkezi nerede?", "tr"),
    ("İş ortaklığı teklifim var", "tr"),
    ("Basın bültenlerine nasıl ulaşabilirim?", "tr"),
    ("Reklam vermek istiyorum", "tr"),
    ("Sosyal medya hesabınız nedir?", "tr"),
    ("Uçak motoru nasıl çalışır?", "tr"),
    ("Havayolu sektöründe kariyer tavsiyesi verir misiniz?", "tr"),
    ("What's the capital of France?", "en"),
    ("Kurumsal sponsorluk başvurusu yapmak istiyorum", "tr"),
    ("Uçuş simülatörü deneyimi sunuyor musunuz?", "tr"),
]

CATEGORIES = {
    "politika_bilgi_sorgusu": POLITIKA_BILGI_SORGUSU,
    "ucus_sorgulama": UCUS_SORGULAMA,
    "rezervasyon_islem_talebi": REZERVASYON_ISLEM_TALEBI,
    "checkin_talebi": CHECKIN_TALEBI,
    "belirsiz_acikliga_kavusturma": BELIRSIZ_ACIKLIGA_KAVUSTURMA,
    "kapsam_disi": KAPSAM_DISI,
}


def main() -> None:
    records = []
    idx = 1
    for intent, examples in CATEGORIES.items():
        for text, lang in examples:
            records.append({
                "id": f"m{idx:04d}",
                "text": text,
                "intent": intent,
                "language": lang,
                "char_len": len(text),
                "word_len": len(text.split()),
            })
            idx += 1

    with open("data/intent/messages.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Toplam ornek: {len(records)}")
    for intent, examples in CATEGORIES.items():
        print(f"  {intent}: {len(examples)}")


if __name__ == "__main__":
    main()
