# Faz 1 — TrOCR El Yazısı Tanıma PoC

> **Arşiv (2026-10-06):** Bu repo tamamlanmış başlangıç PoC'udur ve salt
> okunurdur. Aktif geliştirme (koordinat/homografi, etiketleme, ince ayar,
> toplu çalıştırma, doğal dil sorgu katmanı) `SHapeloglu/trocr-faz1`
> reposunda sürüyor. Sonuç için bkz. [§5 Faz 1 Sonucu](#5-faz-1-sonucu).

El yazısı form alanlarını (hasta adı, tanı kodu, ilaç adı vb.) otomatik
tanımak için Microsoft'un TrOCR modelini kullanan, containerize edilmiş
bir kanıt-of-konsept (PoC) pipeline'ı. Amaç, tam bir sisteme yatırım
yapmadan önce TrOCR'ın Türkçe el yazısındaki ham (fine-tune öncesi)
performansını ölçmek.

---

## 1. Üçüncü Parti Bileşenler

Bu proje kendi kodunun yanında aşağıdaki dış bileşenlere dayanıyor.
Üzerinde geliştirme yapacak kişinin bunların **hangisi bizim kodumuz,
hangisi hazır bileşen** olduğunu net ayırt etmesi önemli.

| Bileşen | Kaynak | Ne işe yarıyor | Nerede |
|---|---|---|---|
| **TrOCR modeli** | `microsoft/trocr-base-handwritten` (Hugging Face Hub) | El yazısı satır görüntüsünü metne çeviren önceden eğitilmiş model | İlk çalıştırmada indirilir, `./models` altında önbelleklenir (~1.3 GB) |
| **transformers** | Hugging Face, `pip` paketi (v4.44.2) | TrOCR modelini yükleyip çalıştırmak için kullanılan kütüphane | `Dockerfile` içinde sabitlenmiş |
| **PyTorch (CPU)** | `pytorch.org`, `pip` paketi (v2.3.1) | Modelin alt seviye tensor/inference motoru | `Dockerfile` içinde sabitlenmiş, CUDA'sız CPU sürümü |
| **OpenCV (`cv2`)** | `pip` paketi | `hucre_kes.py` içinde form görüntüsünden hücre/kutu tespiti | Sadece hücre kesme aşamasında kullanılıyor |
| **Docker / docker-compose** | Sistem paketi | Tüm pipeline'ı izole, tekrarlanabilir bir ortamda çalıştırmak | `Dockerfile`, `docker-compose.yml` |

**Önemli:** `models/` klasörünün içeriği (TrOCR ağırlıkları) **bizim
kodumuz değil** — Hugging Face Hub'dan otomatik iner. Bu yüzden bu
klasör Git deposuna dahil edilmez (bkz. `.gitignore`); herkes kendi
makinesinde `docker compose run --rm trocr` çalıştırdığında aynı model
otomatik olarak inecektir.

### Neden bu bileşenler seçildi

- **TrOCR (`trocr-base-handwritten`)**: El yazısı için özel olarak
  fine-tune edilmiş, hazır kullanılabilir en bilinen açık modellerden
  biri. Sıfırdan bir OCR modeli eğitmek yerine "bu hazır model bizim
  senaryomuzda ne kadar iyi?" sorusuna hızlı cevap almak için seçildi.
- **CPU sürümü PyTorch**: VPS'te GPU olmadığı için imaj boyutunu
  küçültmek (~700 MB vs ~1.5 GB) amacıyla bilinçli tercih.
- **Docker**: Model sürümü, kütüphane sürümleri ve sistem
  bağımlılıklarının (`libgl1` vb.) her ortamda birebir aynı kalması,
  "bende çalışıyordu" sorununu önlemek için.

---

## 2. Pipeline Mimarisi (Bizim Kodumuz)

```
formlar/*.png|jpg (ham form taramaları)
      │
      ▼
hucre_kes.py  ──────────────► data/giris/FORMID__alan_adi.png
(OpenCV ile hücre tespiti,        (kırpılmış, satır bazlı alan görüntüleri)
 opsiyonel alan_haritasi.json
 ile alan isimlendirme)
      │
      ▼
trocr_calistir.py ──────────► data/cikti/tahminler/FORMID.json
(TrOCR modeli ile her kırpıntıyı     (alan_adi → okunan metin + güven skoru)
 tek tek metne çevirir; --gt-taslak
 modunda etiketleme taslağı üretir)
      │
      ▼
degerlendir.py ─────────────► faz1_sonuc.csv
(ground_truth/ ile tahminler/'i          (CER/WER, alan doğruluğu,
 karşılaştırır)                           sözlük katkısı, otomasyon oranı)
```

Üç script tamamen bağımsız ve birbirini dosya üzerinden (JSON/PNG)
besliyor. Bu, herhangi bir aşamayı tek başına değiştirip test
edebilmeyi kolaylaştırıyor.

### Dosya sözlüğü

| Dosya | Sorumluluk |
|---|---|
| `hucre_kes.py` | Form görüntüsünden hücreleri otomatik kesme, `alan_haritasi.json` ile hücre → alan adı eşlemesi |
| `trocr_calistir.py` | Kırpıntıları TrOCR ile okuma; `--gt-taslak` ile boş etiketleme şablonu üretme |
| `degerlendir.py` | Ground truth ile tahminleri karşılaştırıp CER/WER, alan doğruluğu, sözlük katkısı ve otomasyon oranı metriklerini hesaplama |
| `alan_haritasi_ornek.json` | Hücre konumu (`r1_c2` gibi satır/sütun) → anlamlı alan adı (`hasta_adi`) eşleme şeması örneği |
| `alan_haritasi.json` | Sentetik EK-2 formu (`F0001`) için kullanılan gerçek harita (11 alan) |
| `formlar/` | Hücre kesmeye girecek form taramaları (repoda sentetik `F0001.png`) |
| `data/giris/` | `hucre_kes.py` çıktısı / `trocr_calistir.py` girdisi olan kırpıntılar |
| `data/onizleme/` | `hucre_kes.py`'nin hücre sınırlarını çizdiği kontrol görselleri |
| `data/cikti/` | `ground_truth/` (elle doldurulan) ve `tahminler/` (model çıktısı) JSON'ları |
| `OCR.html` | Paydaşlara yönelik süreç anlatımı ("El Yazılı Formlar Nasıl Bilgisayara Aktarılıyor?") |
| `Dockerfile` / `docker-compose.yml` | Ortamın tanımı ve çalıştırma komutları |
| `KURULUM.md` | Adım adım kurulum ve çalıştırma talimatı |

---

## 3. Üzerine Nasıl Geliştirme Yapılır

### 3.1 Yeni bir alan tipi eklemek

`alan_haritasi_ornek.json` şemasını takip ederek kendi `alan_haritasi.json`
dosyanızı oluşturun:

```json
{
  "r1_c1": null, "r1_c2": "hasta_adi",
  "r7_c2": "yeni_alan_adiniz"
}
```

`hucre_kes.py` bu haritayı okuyup kırpıntıları otomatik doğru isimle
kaydeder. Kod değişikliği gerekmez — sadece harita dosyası genişletilir.

### 3.2 Farklı bir OCR modeli denemek (TrOCR alternatifi)

`trocr_calistir.py` içindeki `MODEL_ADI` sabitini değiştirerek
Hugging Face Hub'daki başka bir model denenebilir:

```python
MODEL_ADI = "microsoft/trocr-base-handwritten"   # → başka bir model adıyla değiştirin
```

`KURULUM.md`'de belirtildiği gibi Faz 1 sonucunda CER > %40 çıkarsa
zaten alternatif olarak PaddleOCR-VL gibi modellerin denenmesi
öneriliyor — bu değişiklik büyük ihtimalle `trocr_calistir.py`'nin
model yükleme kısmının ayrı bir modüle çıkarılmasını (adapter pattern)
gerektirecektir.

### 3.3 Sözlük / son işleme (post-processing) katmanı eklemek

`degerlendir.py` zaten `deger_sozluk_sonrasi` alanını (opsiyonel)
destekleyecek şekilde tasarlanmış. Fuzzy-match tabanlı bir sözlük
düzeltmesi eklemek isterseniz:

1. `trocr_calistir.py` çıktısına ek bir alan (`deger_sozluk_sonrasi`)
   ekleyin.
2. Bilinen değer listeleri (örn. ilaç adları, doktor isimleri) için
   bir fuzzy-match kütüphanesi (`rapidfuzz` gibi) ekleyin.
3. `degerlendir.py` otomatik olarak sözlük öncesi/sonrası doğruluk
   farkını raporlayacaktır — kod değişikliği gerekmez.

### 3.4 Fine-tuning'e geçiş (Faz 2)

Faz 1 sonucu CER %15–35 aralığındaysa (`KURULUM.md`'deki tablo),
sıradaki adım TrOCR'ı kendi el yazısı verinizle fine-tune etmektir:

- `ground_truth/*.json` dosyaları zaten etiketlenmiş veri seti olarak
  kullanılabilir — bu formatın eğitim script'i için `Dataset` sınıfına
  dönüştürülmesi yeterli.
- Fine-tuning ayrı bir `egitim/` klasöründe, ayrı bir Dockerfile ile
  yapılması önerilir (GPU gerekebilir, bu proje CPU-only tasarlanmıştır).

### 3.5 Yeni bir değerlendirme metriği eklemek

`degerlendir.py` bağımlılıksız (yalnızca Python standart kütüphanesi)
tasarlandığı için yeni bir metrik eklemek düşük risklidir. Mevcut
metrik hesaplama fonksiyonlarının yanına yeni bir fonksiyon eklemek ve
CSV çıktısına yeni bir sütun olarak dahil etmek yeterlidir.

### Geliştirme yaparken dikkat edilmesi gerekenler

- **`models/` klasörünü asla commit etmeyin** — `.gitignore`'da
  olmalı, herkesin makinesinde otomatik iner.
- **Sürümleri sabit tutun**: `Dockerfile`'daki `torch==`,
  `transformers==` gibi sürüm numaralarını değiştirirken
  `docker-compose.yml`'deki `image: poc-trocr:0.X` etiketini de
  artırın — tekrarlanabilirlik bu projenin temel prensibi.
- **Kırpıntı adlandırma kuralına uyun**: `FORMID__alan_adi.png`
  (çift alt çizgi) formatı hem `hucre_kes.py` çıktısı hem
  `trocr_calistir.py` girdisi için sözleşme niteliğinde; bu kural
  bozulursa pipeline'ın her iki ucu da kırılır.

---

## 4. Hızlı Başlangıç

Detaylı adımlar için bkz. [`KURULUM.md`](./KURULUM.md). Özet:

```bash
# (opsiyonel) form taramalarından hücre kes — ana makinede, Docker dışında:
pip install opencv-python numpy
python3 hucre_kes.py --giris formlar/ --cikti data/giris/ --harita alan_haritasi.json
#   → data/giris/FORMID__alan.png + data/onizleme/FORMID_onizleme.png
#   parametreler: --harita yoksa dosyalar FORMID__r<satır>_c<sütun>.png adını alır;
#   --ic-pay (vars. 4) hücre kenarlığından içeri kırpma payı (piksel)

docker compose build
docker compose run --rm trocr --gt-taslak   # etiketleme taslağı üret
# ground_truth/ dosyalarını elle doldurun
docker compose run --rm trocr               # tahmin üret
python3 degerlendir.py --gt data/cikti/ground_truth \
    --tahmin data/cikti/tahminler --csv faz1_sonuc.csv
```

`hucre_kes.py` Docker imajında **yer almaz** (imaj yalnızca
`trocr_calistir.py`'yi içerir, OpenCV kurulu değildir); bu yüzden ana
makinede çalıştırılır. `degerlendir.py` yalnızca standart kütüphane
kullanır, o da ana makinede çalışır.

---

## 5. Faz 1 Sonucu

PoC, sentetik EK-2 formu `F0001` üzerindeki 9 el yazısı alanıyla
(`data/giris/`) çalıştırıldı. İnce ayarsız `trocr-base-handwritten`
Türkçe el yazısını okuyamadı: çıktılar anlamsız İngilizce cümleler
(ör. nabız alanı için *"the CEC was able to be able to…"*), güven
skorları 0,10–0,42 aralığında (`data/cikti/tahminler/F0001.json`).

Bu, `KURULUM.md` §7'deki tabloya göre **"> %40 — Türkçe uyum sorunu
ciddi"** senaryosudur; sayısal ham CER kaydedilmedi. Karar: erken ince
ayara geçildi. Devamı `trocr-faz1` reposunda — v3 eval CER %44,1 → v4
eval CER **%12,43**; sağlıklı formlarda %98,8 tam eşleşme.

**Not:** `data/cikti/ground_truth/F0001.json` ve `faz1_sonuc.csv`
içindeki `DOLDUR` değerleri etiketleme formatını gösteren **örnek
taslaklardır**; gerçek ölçüm değildir.
