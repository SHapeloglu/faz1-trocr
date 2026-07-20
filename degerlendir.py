#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
degerlendir.py — El yazılı form dijitalleştirme PoC değerlendirme aracı

Kullanım:
    python3 degerlendir.py --gt ground_truth/ --tahmin tahminler/ [--esik 0.85] [--csv sonuc.csv]

Girdi formatı:
  - ground_truth/ : Her form için F0001.json gibi ground truth dosyaları
    (bkz. ground_truth_ornek.json şeması)
  - tahminler/    : Aynı şemada model çıktıları. Her alanda ek olarak
    şu anahtarlar bulunabilir:
       "deger"                : modelin ham (sözlük öncesi) okuması
       "deger_sozluk_sonrasi" : fuzzy-match sonrası düzeltilmiş değer (opsiyonel)
       "guven"                : 0-1 arası güven skoru (opsiyonel)

Hesaplanan metrikler:
  1. CER / WER            : karakter ve kelime hata oranı (ham HTR kalitesi)
  2. Alan doğruluğu       : alanın tamamen doğru okunma oranı (asıl iş metriği)
  3. Sözlük katkısı       : sözlük öncesi vs sonrası alan doğruluğu farkı
  4. Güven skoru analizi  : eşik altı alanların gerçekten hatalı olma oranı,
                            eşik üstü kalan (yakalanamayan) hata oranı
  5. Otomasyon oranı      : insana düşmeyen (eşik üstü) alan yüzdesi
  Kırılımlar              : zorluk, alan tipi, doktor, tarama kalitesi bazında

Bağımlılık: yok (yalnızca Python 3 standart kütüphanesi).
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------- yardımcılar

def normalize(s: str) -> str:
    """Karşılaştırma öncesi normalizasyon: unicode NFC, kırp, çoklu boşlukları
    tekle. Türkçe karakterler korunur; büyük/küçük harf KORUNUR (tıbbi
    kısaltmalar için önemli olabilir — gerekirse --kucuk-harf ile kapatılır)."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = " ".join(s.split())
    return s


def levenshtein(a: str, b: str) -> int:
    """Standart Levenshtein mesafesi (iki satırlı DP, O(len(a)*len(b)))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # silme
                cur[j - 1] + 1,       # ekleme
                prev[j - 1] + (ca != cb),  # değiştirme
            ))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Karakter hata oranı. ref boşsa tanımsız → None döner."""
    if len(ref) == 0:
        return None
    return levenshtein(ref, hyp) / len(ref)


def wer(ref: str, hyp: str) -> float:
    """Kelime hata oranı."""
    rw, hw = ref.split(), hyp.split()
    if len(rw) == 0:
        return None
    # kelime düzeyinde levenshtein: kelimeleri tek karakterlere eşle
    vocab = {}
    def enc(words):
        return "".join(chr(0xE000 + vocab.setdefault(w, len(vocab))) for w in words)
    return levenshtein(enc(rw), enc(hw)) / len(rw)


def pct(x, n):
    return f"{100.0 * x / n:5.1f}%" if n else "  n/a"


# ---------------------------------------------------------------- veri okuma

def load_dir(path: Path) -> dict:
    """Dizindeki tüm .json dosyalarını form_id -> veri sözlüğü olarak yükler."""
    forms = {}
    for f in sorted(path.glob("*.json")):
        if f.name.startswith("_") or f.name == "ground_truth_ornek.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"UYARI: {f.name} okunamadı ({e}), atlanıyor", file=sys.stderr)
            continue
        fid = data.get("form_id") or f.stem
        forms[fid] = data
    return forms


# ---------------------------------------------------------------- ana analiz

def evaluate(gt_forms, pred_forms, esik, kucuk_harf=False):
    rows = []          # alan bazlı satırlar (CSV için)
    missing_forms = [] # tahmini olmayan formlar

    def norm(s):
        s = normalize(s)
        return s.lower() if kucuk_harf else s

    for fid, gt in sorted(gt_forms.items()):
        pred = pred_forms.get(fid)
        if pred is None:
            missing_forms.append(fid)
            continue
        pred_alanlar = {a.get("alan_adi"): a for a in pred.get("alanlar", [])}

        for ga in gt.get("alanlar", []):
            ad = ga.get("alan_adi")
            if ga.get("okunamadi"):
                continue  # insan bile okuyamadıysa metrik dışı
            ref = norm(ga.get("deger"))
            pa = pred_alanlar.get(ad, {})
            hyp_ham = norm(pa.get("deger"))
            hyp_soz = norm(pa.get("deger_sozluk_sonrasi")) if pa.get("deger_sozluk_sonrasi") is not None else hyp_ham
            guven = pa.get("guven")

            rows.append({
                "form_id": fid,
                "alan_adi": ad,
                "alan_tipi": ga.get("alan_tipi", "?"),
                "sozluk": ga.get("sozluk") or "-",
                "zorluk": gt.get("zorluk", "?"),
                "doktor": gt.get("doktor_kodu", "?"),
                "tarama": gt.get("tarama_kalitesi", "?"),
                "ref": ref,
                "ham": hyp_ham,
                "sozluk_sonrasi": hyp_soz,
                "cer": cer(ref, hyp_ham),
                "wer": wer(ref, hyp_ham),
                "dogru_ham": ref == hyp_ham,
                "dogru_sozluk": ref == hyp_soz,
                "guven": guven,
            })

    return rows, missing_forms


def report(rows, missing_forms, esik):
    n = len(rows)
    if n == 0:
        print("Hiç karşılaştırılabilir alan bulunamadı.", file=sys.stderr)
        sys.exit(1)

    L = []
    L.append("=" * 62)
    L.append(" PoC DEĞERLENDİRME RAPORU")
    L.append("=" * 62)
    if missing_forms:
        L.append(f"UYARI: {len(missing_forms)} formun tahmini yok: "
                 + ", ".join(missing_forms[:10])
                 + (" ..." if len(missing_forms) > 10 else ""))
    L.append(f"Toplam alan sayısı           : {n}")

    # 1. CER / WER
    cers = [r["cer"] for r in rows if r["cer"] is not None]
    wers = [r["wer"] for r in rows if r["wer"] is not None]
    L.append("")
    L.append("-- 1. Ham HTR kalitesi (sözlük ÖNCESİ) " + "-" * 22)
    L.append(f"Ortalama CER                 : {100*sum(cers)/len(cers):5.1f}%")
    L.append(f"Ortalama WER                 : {100*sum(wers)/len(wers):5.1f}%")

    # 2-3. Alan doğruluğu, sözlük katkısı
    d_ham = sum(r["dogru_ham"] for r in rows)
    d_soz = sum(r["dogru_sozluk"] for r in rows)
    L.append("")
    L.append("-- 2. Alan doğruluğu (tam eşleşme) " + "-" * 27)
    L.append(f"Sözlük öncesi                : {pct(d_ham, n)}  ({d_ham}/{n})")
    L.append(f"Sözlük sonrası               : {pct(d_soz, n)}  ({d_soz}/{n})")
    L.append(f"Sözlük katkısı               : {100*(d_soz-d_ham)/n:+5.1f} puan")

    soz_rows = [r for r in rows if r["sozluk"] != "-"]
    if soz_rows:
        sn = len(soz_rows)
        sh = sum(r["dogru_ham"] for r in soz_rows)
        ss = sum(r["dogru_sozluk"] for r in soz_rows)
        L.append(f"  (yalnız sözlüklü alanlar)  : {pct(sh, sn)} → {pct(ss, sn)}  (n={sn})")

    # 4-5. Güven skoru + otomasyon
    g_rows = [r for r in rows if r["guven"] is not None]
    L.append("")
    L.append(f"-- 3. Güven skoru analizi (eşik = {esik}) " + "-" * 20)
    if not g_rows:
        L.append("Güven skoru verisi yok — pipeline 'guven' alanı üretince tekrar çalıştırın.")
    else:
        alti = [r for r in g_rows if r["guven"] < esik]     # insana düşen
        ustu = [r for r in g_rows if r["guven"] >= esik]    # otomatik geçen
        hata_ustu = sum(not r["dogru_sozluk"] for r in ustu)
        hata_alti = sum(not r["dogru_sozluk"] for r in alti)
        toplam_hata = hata_ustu + hata_alti
        L.append(f"Otomasyon oranı (eşik üstü)  : {pct(len(ustu), len(g_rows))}  ({len(ustu)}/{len(g_rows)})")
        L.append(f"Yakalanamayan hata           : {pct(hata_ustu, len(ustu))} of otomatik geçenler  ({hata_ustu} alan)")
        if toplam_hata:
            L.append(f"Hata yakalama oranı          : {pct(hata_alti, toplam_hata)}  (hataların insana düşen kısmı)")
        if alti:
            L.append(f"İnsana düşenlerin isabeti    : {pct(hata_alti, len(alti))} gerçekten hatalı  ({hata_alti}/{len(alti)})")

    # kırılımlar
    def breakdown(key, title):
        L.append("")
        L.append(f"-- Kırılım: {title} " + "-" * (48 - len(title)))
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r)
        for g in sorted(groups):
            rs = groups[g]
            d = sum(r["dogru_sozluk"] for r in rs)
            c = [r["cer"] for r in rs if r["cer"] is not None]
            L.append(f"  {g:<12} alan doğruluğu {pct(d, len(rs))}   CER {100*sum(c)/len(c):5.1f}%   (n={len(rs)})")

    breakdown("zorluk", "zorluk")
    breakdown("alan_tipi", "alan tipi")
    breakdown("tarama", "tarama kalitesi")
    breakdown("doktor", "doktor")

    # en kötü 10 alan
    L.append("")
    L.append("-- En yüksek CER'li 10 alan (hata analizi için) " + "-" * 14)
    worst = sorted((r for r in rows if r["cer"] is not None),
                   key=lambda r: r["cer"], reverse=True)[:10]
    for r in worst:
        L.append(f"  {r['form_id']}/{r['alan_adi']:<15} CER {100*r['cer']:5.1f}%  "
                 f"GT='{r['ref'][:30]}'  Model='{r['ham'][:30]}'")

    L.append("=" * 62)
    return "\n".join(L)


def write_csv(rows, path):
    cols = ["form_id", "alan_adi", "alan_tipi", "sozluk", "zorluk", "doktor",
            "tarama", "ref", "ham", "sozluk_sonrasi", "cer", "wer",
            "dogru_ham", "dogru_sozluk", "guven"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})


def main():
    ap = argparse.ArgumentParser(description="PoC değerlendirme aracı")
    ap.add_argument("--gt", required=True, help="Ground truth JSON dizini")
    ap.add_argument("--tahmin", required=True, help="Model çıktısı JSON dizini")
    ap.add_argument("--esik", type=float, default=0.85,
                    help="Güven skoru eşiği (varsayılan 0.85)")
    ap.add_argument("--csv", default=None, help="Alan bazlı sonuçları CSV'ye yaz")
    ap.add_argument("--kucuk-harf", action="store_true",
                    help="Karşılaştırmayı büyük/küçük harf duyarsız yap")
    args = ap.parse_args()

    gt = load_dir(Path(args.gt))
    pred = load_dir(Path(args.tahmin))
    if not gt:
        print("Ground truth dizini boş.", file=sys.stderr)
        sys.exit(1)

    rows, missing = evaluate(gt, pred, args.esik, args.kucuk_harf)
    print(report(rows, missing, args.esik))
    if args.csv:
        write_csv(rows, args.csv)
        print(f"\nAlan bazlı detaylar CSV'ye yazıldı: {args.csv}")


if __name__ == "__main__":
    main()
