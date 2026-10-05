# CLAUDE.md

Bu dosya, bu proje üzerinde çalışırken Claude'un (Claude Code dahil) izlemesi gereken bağlamı ve kuralları içerir.

## Proje

**Faz 1 — TrOCR El Yazısı Tanıma PoC** — El yazısı form alanlarını (hasta adı, tanı kodu, ilaç adı vb.) otomatik tanımak için Microsoft'un TrOCR modelini kullanan, containerize edilmiş bir kanıt-of-konsept (PoC) pipeline'ı. Amaç, tam bir sisteme yatırım yapmadan önce TrOCR'ın Türkçe el yazısındaki ham (fine-tune öncesi) performansını ölçmek. ---

- GitHub: https://github.com/SHapeloglu/faz1-trocr
- Sunucu (Contabo): /root/faz1-trocr

## Teknoloji Yığını

- Docker / docker compose
- Python

## Önemli Dosyalar

- `Dockerfile`
- `docker-compose.yml`

Mimari ayrıntılar için bkz. `architect.md`.

## Sık Kullanılan Komutlar

```bash
docker compose up -d --build
docker compose logs -f
```

## Kurallar

- `.env`, parola, token ve API anahtarlarını asla commit etme.
- Her çalışma oturumunun sonunda `session.md`ye kısa kayıt düş; görev durumunu `task.md`de güncelle.
- Önceliklendirilmemiş fikirleri `backlog.md`ye yaz; somutlaşınca `task.md`ye taşı.

## Çalışma Dosyaları

| Dosya | Amaç |
|---|---|
| `architect.md` | Mimari ve dizin yapısı referansı |
| `task.md` | Aktif / devam eden / tamamlanan görevler |
| `backlog.md` | Önceliklendirilmemiş fikir ve teknik borç havuzu |
| `session.md` | Oturum günlüğü — her oturum sonunda güncellenir |
