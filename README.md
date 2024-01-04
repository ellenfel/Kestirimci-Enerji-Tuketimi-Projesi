# Enerji Analizörü Veri Açıklamaları

Bu README dosyası, enerji analizöründen alınan değerleri ve anlamlarını açıklamaktadır. Bu değerler, genellikle üç fazlı bir elektrik sisteminin ölçümlerini temsil eder.

## Değerler ve Açıklamaları

1. **VL12, VL23, VL13:**
   - Voltaj değerleri, üç farklı faz arasındaki voltajları gösterir.

2. **IL1, IL2, IL3:**
   - Akım değerleri, üç farklı fazdaki akımları temsil eder.

3. **Cosφ1, Cosφ2, Cosφ3:**
   - Güç faktörleri, farklı fazlardaki güç faktörlerini ifade eder.

4. **Frekans:**
   - Sistemin çalıştığı frekansı (Hertz) gösterir.

5. **P Imp, P Exp:**
   - P Imp, gerçek güç tüketimini; P Exp, gerçek güç ihracını temsil eder.

6. **Q Ind, Q Kap:**
   - Endüktif ve kapasitif reaktif güçleri ifade eder.

7. **S:**
   - Görünür gücü (VA) temsil eder.

8. **AE Imp, AE Exp:**
   - AE Imp, birim zamandaki aktif enerji tüketimini; AE Exp, birim zamandaki aktif enerji ihracını ifade eder.

9. **Ind Reak, Kap Reakt:**
   - Endüktif ve kapasitif reaktif güçleri belirtir.

10. **Günlük Tük, Aylık Tük:**
    - Günlük ve aylık enerji tüketimini gösterir.



# Enerji Telemetri Forecasting Projesi

Bu proje, IoT (Internet of Things) tabanlı bir sistem kullanarak enerji telemetri verilerini analiz etmek, gelecekteki enerji tüketimini tahmin etmek ve enerji kullanımını optimize etmek amacını taşımaktadır.

## Amaç

Bu projenin temel amacı iki aşamada ele alınmıştır:

### Aşama 1: Telemetri Forecasting - Enerji Kullanımı Öngörümü

1. **Veri Toplama ve Hazırlık:**
   - Üç fazlı sistemdeki voltaj, akım, güç faktörü ve diğer enerji verileri IoT cihazları aracılığıyla sürekli olarak toplanır.
   - Toplanan veriler, enerji tüketimini etkileyen faktörleri belirlemek için analiz edilir.

2. **Telemetri Veri Analizi ve Model Geliştirme:**
   - Geçmiş veriler üzerinde zaman serisi analizi veya makine öğrenimi modelleri kullanılarak enerji tüketimini tahmin etmek için bir model geliştirilir.

3. **Gelecekteki Enerji Tüketiminin Tahmini:**
   - Geliştirilen model, gelecekteki enerji tüketimini tahmin eder. Bu tahmin, belirli bir zaman diliminde beklenen enerji talebini içerir.

4. **Sonuçların Sunumu:**
   - Tahmin edilen enerji tüketimleri, kullanıcı arayüzü veya raporlar aracılığıyla paydaşlara sunulur.
   - Kullanıcılara enerji kullanımını planlama ve optimize etme konusunda rehberlik sağlanır.

### Aşama 2: Anormallik Tespiti, Ceza Oranı ve Optimizasyonlar

5. **Anormallik Tespiti ve Uyarılar:**
   - Güç faktörü (Cosφ) ve diğer enerji parametreleri üzerinde anormallikleri tespit etmek için bir anormallik tespit modeli geliştirilir.
   - Anormallik durumlarında sistem operatörlerine veya otomasyon sistemlerine uyarılar gönderilir.

6. **Ceza Oranı ve Optimizasyon:**
   - Güç faktörü düşüklüğüne bağlı olarak uygulanacak ceza oranlarını belirleyen bir optimizasyon algoritması kullanılır.
   - Güç faktörünün düzenlenmesiyle ceza oranını azaltmak ve enerji verimliliğini artırmak için otomatik düzeltme mekanizmaları devreye alınır.

7. **Enerji Kullanımı Optimizasyonu:**
   - Gelecekteki enerji talebini tahmin ederek, enerji tüketimini optimize etmeye yönelik öneriler sunulur.
   - Bu öneriler, kullanıcılara enerji tasarrufu sağlamak ve işletme maliyetlerini minimize etmek için rehberlik eder.

## Kullanım

