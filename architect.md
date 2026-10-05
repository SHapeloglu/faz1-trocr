# architect.md — Faz 1 — TrOCR El Yazısı Tanıma PoC Mimari Referansı

Bu dosya projenin yapısının hızlı-referans özetidir. Kod değiştikçe güncel tutun.

## Genel Bakış

El yazısı form alanlarını (hasta adı, tanı kodu, ilaç adı vb.) otomatik tanımak için Microsoft'un TrOCR modelini kullanan, containerize edilmiş bir kanıt-of-konsept (PoC) pipeline'ı. Amaç, tam bir sisteme yatırım yapmadan önce TrOCR'ın Türkçe el yazısındaki ham (fine-tune öncesi) performansını ölçmek. ---

## Teknoloji Yığını

- Docker / docker compose
- Python

## Dizin Yapısı

```
.gitignore
Dockerfile
KURULUM.md
OCR.html
README.md
alan_haritasi.json
alan_haritasi_ornek.json
data/
degerlendir.py
docker-compose.yml
faz1_sonuc.csv
formlar/
  F0001.png
hucre_kes.py
trocr_calistir.py
```

## Modüller / Kaynak Dosyalar

- `degerlendir.py` — degerlendir.py — El yazılı form dijitalleştirme PoC değerlendirme aracı
- `hucre_kes.py` — hucre_kes.py — Tablo çizgili formlardan hücreleri otomatik kesme
- `trocr_calistir.py` — trocr_calistir.py — Faz 1 hızlı testi: TrOCR ile el yazısı kırpıntı tanıma

## Giriş Noktaları ve Yapılandırma

- `Dockerfile`
- `docker-compose.yml`

## Dağıtım / Çalışma Ortamı

- GitHub: https://github.com/SHapeloglu/faz1-trocr
- Sunucu (Contabo): /root/faz1-trocr

## Diğer Dokümanlar

- `KURULUM.md`
- `README.md`

## Mimari Kararlar

_Önemli tasarım kararlarını ve gerekçelerini buraya ekleyin (ör. "X yerine Y seçildi çünkü ...")._
