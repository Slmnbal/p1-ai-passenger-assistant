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
    # Kosullu cumle yapilari ("X yaparsa/ise") — Adim 11'in buldugu darboğaz icin
    ("Uçağım rötar yaparsa tazminat alabilir miyim?", "tr"),
    ("Bagajım kaybolursa ne yapmam gerekir?", "tr"),
    ("Uçuşum iptal olursa param ne olur?", "tr"),
    ("Vize başvurum reddedilirse biletimi iade edebilir miyim?", "tr"),
    ("Hastalanırsam biletimi değiştirebilir miyim, kuralı nedir?", "tr"),
    ("Uçağa yetişemezsem ne olur, bileti kaybeder miyim?", "tr"),
    ("Bagaj fazla çıkarsa ne kadar ücret öderim?", "tr"),
    ("Uçuşum ertelenirse otelde konaklama hakkım var mı?", "tr"),
    ("Grev nedeniyle uçuşum iptal olursa haklarım nedir?", "tr"),
    ("Kötü hava koşulları yüzünden uçuşum gecikirse tazminat var mı?", "tr"),
    ("Uçuşuma yetişemezsem bir sonraki uçuşa binebilir miyim?", "tr"),
    ("Pasaportum kaybolursa gümrükte ne olur?", "tr"),
    ("Evcil hayvanım uçakta hastalanırsa ne yapılır?", "tr"),
    ("Koltuk değişikliği yaparsam ek ücret çıkar mı?", "tr"),
    ("Uçağım teknik arıza yaparsa yolcu hakları ne diyor?", "tr"),
    # Ingilizce ornekler — dil dengesizligini azaltmak icin
    ("What happens if my flight is delayed more than 3 hours?", "en"),
    ("Can I bring a service animal on board?", "en"),
    ("What is the refund policy if I cancel within 24 hours?", "en"),
    ("How much does an extra checked bag cost?", "en"),
    ("Is travel insurance included in the ticket price?", "en"),
    ("What are the rules for traveling with an infant?", "en"),
    ("Can I get compensation if my baggage is lost?", "en"),
    ("What documents do I need for check-in?", "en"),
    ("Is Wi-Fi available on international flights?", "en"),
    ("How does the Miles&Smiles loyalty program work?", "en"),
    ("What happens to my ticket if my visa application is rejected?", "en"),
    ("Am I entitled to a hotel stay if my flight is cancelled overnight?", "en"),
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
    # Kosullu cumle yapilari
    ("Uçağım rötar yaparsa yeni saat ne zaman belli olur?", "tr"),
    ("Fırtına çıkarsa uçuşum iptal mi olur, durumunu öğrenebilir miyim?", "tr"),
    ("Aktarma uçuşuma yetişemezsem hangi uçakta olduğumu görebilir miyim?", "tr"),
    # Ingilizce ornekler
    ("Has flight TK123 been delayed?", "en"),
    ("Which gate does my flight depart from?", "en"),
    ("When does boarding start for flight TK456?", "en"),
    ("Is there a seat available on tonight's flight to Izmir?", "en"),
    ("Can you check the status of my connecting flight?", "en"),
    ("What time does my flight land?", "en"),
    ("Has the departure time for TK789 changed?", "en"),
    ("How long is the layover on my itinerary?", "en"),
    ("Can you look up my flight using my booking reference?", "en"),
    ("Is my flight on time today?", "en"),
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
    # Kosullu cumle yapilari
    ("Biletimi değiştirirsem fark ücreti öder miyim, değiştirmek istiyorum", "tr"),
    ("Rezervasyonumu iptal edersem param geri gelir mi, iptal etmek istiyorum", "tr"),
    ("Tarihi değiştirirsem ek ücret çıkar mı, yine de değiştirmek istiyorum", "tr"),
    # Ingilizce ornekler
    ("I would like to change my seat assignment", "en"),
    ("Please cancel my ticket and process a refund", "en"),
    ("I need to upgrade to business class", "en"),
    ("Can you add an extra checked bag to my booking?", "en"),
    ("I want to correct the spelling of my name on the ticket", "en"),
    ("I'd like to add a wheelchair request to my reservation", "en"),
    ("Please change my flight to next Friday", "en"),
    ("I want to convert my booking into a group reservation", "en"),
    ("I'd like to add a special meal request to my booking", "en"),
    ("Can I downgrade from business to economy and get a partial refund?", "en"),
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
    # Ingilizce ornekler
    ("I'd like to check in for my flight tomorrow", "en"),
    ("Can I get my boarding pass now?", "en"),
    ("What documents are required for check-in?", "en"),
    ("Is kiosk check-in available at this airport?", "en"),
    ("Can I select my seat during check-in?", "en"),
    ("I haven't checked in yet, can I do it now?", "en"),
    ("I'd like to check in my whole group at once", "en"),
    ("Do I need to print my boarding pass or is mobile fine?", "en"),
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
    # Belirsiz ama kosullu yapili (gercekten hangi konuya bagli oldugu belli degil)
    ("Bir şey olursa ne yapmalıyım?", "tr"),
    ("Böyle bir durumda hakkım var mı?", "tr"),
    ("Bu olursa ücret öder miyim?", "tr"),
    ("Ne kadar sürer?", "tr"),
    ("Kurallara uygun mu bu?", "tr"),
    ("Bunu nasıl yapabilirim?", "tr"),
    ("Uçuşum etkilenir mi?", "tr"),
    ("Bu işlem ücretli mi?", "tr"),
    ("Hakkımı nasıl kullanırım?", "tr"),
    # Ingilizce ornekler
    ("What should I do?", "en"),
    ("Is this allowed?", "en"),
    ("How much would that cost?", "en"),
    ("Can you help me?", "en"),
    ("Is there a fee for that?", "en"),
    ("What are my options?", "en"),
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
    ("Uçak motoru tamiri ne kadar sürer?", "tr"),
    ("THY'nin yıllık cirosu nedir?", "tr"),
    ("Sizin için çalışmak istiyorum, CV nereye gönderirim?", "tr"),
    ("Hangi bankayla anlaşmalısınız?", "tr"),
    ("Uçak bileti fiyatları neden bu kadar dalgalı, ekonomik analiz ister misiniz?", "tr"),
    # Ingilizce ornekler
    ("Do you offer flying lessons?", "en"),
    ("What's the weather like today?", "en"),
    ("Can you recommend a good restaurant nearby?", "en"),
    ("How do I apply for a job as a pilot?", "en"),
    ("What's your company's annual revenue?", "en"),
    ("Can you tell me a joke?", "en"),
    ("Do you sponsor sports teams?", "en"),
    ("How does a jet engine work?", "en"),
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
