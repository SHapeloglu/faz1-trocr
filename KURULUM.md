# Faz 1 — TrOCR Hızlı Testi: Kurulum ve Çalıştırma

Amaç: pipeline'ı kurmadan önce en büyük riski ölçmek — TrOCR'ın Türkçe
el yazısındaki out-of-box performansı. 20-30 kırpıntı yeterli.

## 0. Ön koşul (VPS'te bir kez)

```bash
# Docker kurulu değilse:
apt-get update && apt-get install -y docker.io docker-compose-v2
```

## 1. Paketi VPS'e taşı ve aç

```bash
scp faz1-trocr.tar.gz root@SUNUCU:/root/
ssh root@SUNUCU
tar xzf faz1-trocr.tar.gz && cd faz1-trocr
```

## 2. İmajı kur (bir kez, ~5-10 dk)

```bash
docker compose build
```

## 3. Kırpıntıları hazırla

Anonim/sentetik formlardan 20-30 el yazısı alanını kırpın (ekran
görüntüsü aracı bile yeterli). Adlandırma kuralı — çift alt çizgi:

```
data/giris/F0001__hasta_adi.png
data/giris/F0001__ilac_1.png
data/giris/F0002__tani_kodu.jpg
```

İpucu: kırpıntı tek satır metin içermeli (TrOCR satır bazlı çalışır);
kutu kenarlıklarını mümkünse dışarıda bırakın.

## 4. Etiketleme taslaklarını üret ve doldur

```bash
docker compose run --rm trocr --gt-taslak
nano data/cikti/ground_truth/F0001.json   # "DOLDUR" yazan yerleri doldurun
```

## 5. Tanımayı çalıştır

```bash
docker compose run --rm trocr
```

- İlk çalıştırmada ~1.3 GB model ağırlığı iner (./models'a önbelleklenir,
  sonraki çalıştırmalar internetsiz de olur)
- CPU'da alan başına ~2-5 sn bekleyin; 30 kırpıntı ≈ 1-3 dk
- Çıktılar: `data/cikti/tahminler/*.json`

## 6. Değerlendir (ana makinede, Docker'sız)

```bash
python3 degerlendir.py \
    --gt data/cikti/ground_truth \
    --tahmin data/cikti/tahminler \
    --csv faz1_sonuc.csv
```

## 7. Sonucu yorumla (out-of-box, fine-tune ÖNCESİ)

| Ortalama CER | Anlamı | Sonraki adım |
|---|---|---|
| < %15 | Beklentinin üstünde | Faz 2'ye (tam pipeline) geç |
| %15-35 | Beklenen aralık | Faz 2'ye geç; sözlük + fine-tuning hedefe taşır |
| > %40 | Türkçe uyum sorunu ciddi | Durup strateji gözden geçir: erken fine-tuning veya PaddleOCR-VL denemesi |

Sayıları ve imaj sürümünü (poc-trocr:0.1) benchmarking.md'ye not edin.

## Sorun giderme

- **OOM / container öldü**: compose'daki mem_limit 5g → 6g yapın, başka
  servisleri geçici durdurun
- **Model inmiyor**: VPS'in huggingface.co'ya erişimi olmalı (yalnızca ilk
  çalıştırmada)
- **Çok yavaş**: normaldir; Faz 1'de hız önemli değil, ONNX int8
  optimizasyonu Faz 2'de eklenecek
