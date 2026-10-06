# CLAUDE.md — Faz 1 TrOCR PoC (ilk kanıt çalışması)

EK-2 işe giriş / periyodik muayene formlarındaki el yazısı alanları için `microsoft/trocr-base-handwritten`'ın **ince ayarsız** Türkçe performansını ölçen containerize PoC. Üç bağımsız adım dosya üzerinden birbirini besler: hücre kesme (OpenCV) → TrOCR tahmini → değerlendirme (CER/WER, alan doğruluğu, sözlük katkısı).

- GitHub: https://github.com/SHapeloglu/faz1-trocr (2026-07-20; 09-24'te `OCR.html` eklendi) — **2026-10-06'da arşivlendi (salt okunur)**
- **Aktif proje: `/root/faz1-trocr-main` (repo `SHapeloglu/trocr-faz1`)** — koordinat/homografi, etiketleme, ince ayar (v4, CER %12,43), toplu çalıştırma ve Faz 8 doğal dil sorgu katmanı orada. Bu repo başlangıç PoC'u ve referans.
- Sentetik test formları: **SyntheticFormGenerator_v3.2**.
- Ayrıntılı geliştirici rehberi: `README.md` (üçüncü parti bileşenler, pipeline, genişletme adımları) · `KURULUM.md`
- Mimari: `architect.md` · Görevler: `task.md` · Fikirler: `backlog.md` · Günlük: `session.md`

## Komutlar

```bash
docker compose build                              # image poc-trocr:0.1, CPU torch 2.3.1, transformers 4.44.2
docker compose run --rm trocr --gt-taslak         # data/cikti/ground_truth/ için etiketleme taslağı
# ground_truth JSON'larını elle doldur
docker compose run --rm trocr                     # data/cikti/tahminler/*.json
python3 degerlendir.py --gt data/cikti/ground_truth --tahmin data/cikti/tahminler --csv faz1_sonuc.csv
python3 hucre_kes.py --giris formlar/ --cikti data/giris/ --harita alan_haritasi.json
                                                  # ana makinede (opencv-python + numpy); önizleme → data/onizleme/
```

Model ağırlıkları (~1,3 GB) ilk çalıştırmada `./models`'e iner (gitignore'da). Container `mem_limit: 5g`, `cpus: 4`.

## Kurallar

- **Gerçek hasta/çalışan formu bu repoya konmaz** — sağlık verisi (KVKK özel nitelikli). Repodaki `F0001` sentetik örnektir.
- Repo tamamlanmış PoC olarak GitHub'da arşivli; yeni geliştirme `trocr-faz1` reposunda. Bu repoya push için önce GitHub'da arşivden çıkarmak gerekir — kullanıcıya sormadan yapma.
- Sürüm sabitleri (`Dockerfile`, `image: poc-trocr:0.X`) benchmark tekrarlanabilirliği için; kütüphane yükseltirsen imaj etiketini de artır.
- `OCR.html` paydaşlara yönelik süreç anlatımı ("El Yazılı Formlar Nasıl Bilgisayara Aktarılıyor?").
- Oturum sonunda `session.md`'ye kayıt düş, `task.md`'yi güncelle.
