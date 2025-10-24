# ING Hubs Türkiye Datathon 2025: Müşteri Kaybı (Churn) Tahmin Modeli

Bu repository, ING Hubs Türkiye Datathon 2024 yarışması için geliştirilen müşteri kaybı (churn) tahmin modelinin kodlarını ve metodolojisini içermektedir. Proje, panel veri (zaman serisi) formatındaki bankacılık verilerini kullanarak, bir müşterinin önümüzdeki 6 ay içinde churn edip etmeyeceğini tahminlemeyi amaçlamaktadır.

Elde edilen model, Public Leaderboard'da 1.24035 skoru ile 30. sırada, Private Leaderboard'da ise 1.23491 skoru ile 49. sırada yer almıştır.

## Projenin Amacı ve Zorlukları

Amaç: Müşterilerin demografik bilgilerini ve aylık işlem geçmişlerini (panel veri) kullanarak, her bir müşteri için belirlenen referans tarihinden sonraki 6 ay içinde churn (bankayı terk etme) olasılığını tahminlemektir.

Temel Zorluklar:

Panel Veri Yapısı: Her müşterinin zaman içinde değişen bir işlem geçmişi vardır. Bu yapıyı, veri sızıntısına (data leakage) yol açmadan, modelin anlayabileceği tek bir satıra dönüştürmek gerekiyordu.

Özel Değerlendirme Metriği: Başarı, standart AUC veya F1-skoru yerine, Gini, Recall@10% ve Lift@10% metriklerinin ağırlıklı ortalamasıyla ölçüldü. Bu, modelin özellikle en yüksek riskli %10'luk dilimi yakalamada ne kadar başarılı olduğuna odaklanmasını gerektirdi.

Veri Kayması (Data Drift): Public (1.24035) ve Private (1.23491) leaderboard skorları arasındaki fark, büyük olasılıkla eğitim/public verisi ile private test verisi arasındaki zaman serisi tabanlı bir veri kaymasından (data drift) kaynaklanmaktadır. Bu, modelin zaman içindeki davranış değişikliklerine ne kadar hassas olduğunu göstermektedir.

## Geliştirilen Pipeline (Model Mimarisi)

Kazanma potansiyeli en yüksek modeli oluşturmak için aşağıdaki adımlardan oluşan metodolojik bir pipeline izlenmiştir:

# 1. Kapsamlı Özellik Mühendisliği (Feature Engineering)

Modelin başarısındaki en kritik adım, ham panel verisinden anlamlı sinyaller türetmekti. Toplamda 200'den fazla "akıllı" özellik üretilmiştir:

Zaman Bazlı Agregasyonlar: Her müşteri için 1, 3, 6, 9, 12, 18 ve 24 aylık zaman pencerelerinde işlem tutarı, işlem sayısı ve aktif ürün sayısı gibi metriklerin mean, median, sum, std gibi istatistikleri hesaplandı.

Trend ve Büyüme Özellikleri: Müşterinin davranışsal momentumunu yakalamak için farklı zaman pencereleri arasındaki oranlar ve farklar hesaplandı (örn: ratio_amt_mean_3m_vs_6m).

RFM ve Müşteri Profili:

Recency (Yenilik): days_since_last_transaction

Frequency (Sıklık): transaction_frequency_ratio, inactive_months_count

Monetary (Parasal Değer): avg_monetary_value_all

Davranışsal Segmentasyon (KMeans): Müşteriler RFM metriklerine göre 5 farklı loyalty_tier (sadakat segmenti) kümesine ayrıldı.

Müşteri Sağlık Skoru: Müşterinin transaction_frequency_ratio (pozitif) ve days_since_last_transaction (negatif) gibi kritik metrikleri birleştirilerek, her müşteri için genel bir "sağlık skoru" (customer_health_score) oluşturuldu.

Anomali Tespiti (Z-Score): Müşterinin son aydaki davranışının, kendi 3 ve 6 aylık ortalamasından ne kadar saptığını ölçen Z-score özellikleri eklendi.

Etkileşim Özellikleri: age_group_x_work_sector, religion_x_age_group, province_x_loyalty_tier gibi demografik, mesleki ve davranışsal özelliklerin birbirleriyle olan karmaşık etkileşimleri modele sunuldu.

# 2. Akıllı Ön İşleme (Preprocessing)

Eksik Veri Yönetimi: work_sector gibi sütunlardaki eksik veriler, work_type gibi diğer sütunlara bakılarak (Student -> Student olarak) mantıksal olarak dolduruldu. Kalan tüm kategorik eksiklikler, "eksik bilgi"nin kendisini bir özellik olarak modellemek için "-" olarak etiketlendi.

Nadir Kategori Birleştirme: Modelin aşırı öğrenmesini engellemek için, eğitim setinde 30'dan az görünen kategorik seviyeler ("province", "work_sector" vb.) tek bir "rare" kategorisi altında toplandı.

Çoklu Doğrusallık Temizliği: Birbirine %96'dan daha fazla benzeyen ve gereksiz bilgi tekrarı yaratan özelliklerden, hedefle olan korelasyonu daha düşük olanı otomatik olarak modelden çıkarıldı.

# 3. Modelleme ve Eğitim Stratejisi

Model: Kategorik verileri işlemedeki üstün başarısı ve hızı nedeniyle CatBoostClassifier tek model olarak seçildi.

Hiperparametre Optimizasyonu (Optuna): En iyi model parametrelerini (depth, learning_rate vb.) bulmak için Optuna kütüphanesi ile 50 denemelik bir optimizasyon süreci çalıştırıldı.

K-Fold İçi Target Encoding: Veri sızıntısını (leakage) kesin olarak önlemek için, tüm kategorik özellikler StratifiedKFold'un her bir döngüsü içinde ayrı ayrı Target Encoding (Hedef Kodlama) işlemine tabi tutuldu. Bu, modelin her bir kategorinin churn olasılığı üzerindeki etkisini güvenli bir şekilde öğrenmesini sağladı.

Eğitim: En iyi parametreler bulunduktan sonra, nihai model 7-Katlı StratifiedKFold ile eğitilerek hem tüm veriden faydalanıldı hem de modelin stabilitesi artırıldı.

# 4. Olasılık Kalibrasyonu

Modelin ham olasılık tahminleri (oof_preds) toplanarak, bu tahminler üzerinde bir IsotonicRegression modeli eğitildi. Bu kalibrasyon adımı, modelin olasılıklarını gerçek dünyadaki dağılıma daha iyi uyacak şekilde düzeltti ve özellikle sıralama bazlı metriklerde (Lift ve Recall) kritik bir iyileşme sağladı.

# Sonuç

Bu kapsamlı özellik mühendisliği, akıllı ön işleme ve rafine modelleme teknikleri sayesinde, modelimiz Public Leaderboard'da 1.24035'lik bir skora ulaşarak 30. sırada yer almıştır 
