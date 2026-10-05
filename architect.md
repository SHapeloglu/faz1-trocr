# architect.md — Faz 1 TrOCR PoC Mimarisi

> Ayrıntı: `README.md` §2. Güncel sistem mimarisi: `/root/faz1-trocr-main/ARCHITECTURE.md`.

```
formlar/*.png (form taraması)
   │ hucre_kes.py — OpenCV ile tablo hücresi tespiti; alan_haritasi.json (r<satır>_c<sütun> → alan adı)
   ▼
data/giris/FORMID__<alan>.png
   │ trocr_calistir.py — TrOCR (VisionEncoderDecoder, CPU) her kırpıntı → metin + güven; --gt-taslak modu
   ▼
data/cikti/tahminler/FORMID.json   +   data/cikti/ground_truth/FORMID.json (elle)
   │ degerlendir.py — CER/WER, ham vs sözlük sonrası doğruluk, otomasyon oranı
   ▼
faz1_sonuc.csv (form_id, alan_adi, alan_tipi, sozluk, zorluk, doktor, tarama, ref, ham, sozluk_sonrasi, cer, wer, dogru_ham, dogru_sozluk, guven)
```

## Bileşenler

| Bizim kodumuz | Üçüncü parti |
|---|---|
| `hucre_kes.py`, `trocr_calistir.py`, `degerlendir.py`, `alan_haritasi*.json`, `Dockerfile`, `docker-compose.yml` | TrOCR ağırlıkları (HF Hub), transformers, PyTorch CPU, OpenCV, Docker |

## Mimari Kararlar

- **Aşamalar arası dosya arayüzü** (PNG/JSON/CSV): her adım bağımsız değiştirilip test edilebilsin.
- **CPU-only Docker imajı**: GPU'suz VPS; imaj ~700 MB.
- **Önce ölç, sonra yatırım yap**: ince ayar (Faz 2+) kararı bu PoC'un ham CER sonuçlarına göre verildi → `trocr-faz1`.
