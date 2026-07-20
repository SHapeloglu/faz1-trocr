#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hucre_kes.py — Tablo çizgili formlardan hücreleri otomatik kesme
Kullanım:
    python3 hucre_kes.py --giris formlar/ --cikti data/giris/
    python3 hucre_kes.py --giris formlar/ --cikti data/giris/ --harita alan_haritasi.json
Çıktı: FORMID__r1_c2.png (harita yoksa) / FORMID__hasta_adi.png (harita varsa)
       + onizleme/FORMID_onizleme.png (hücre sınırları ve etiketler çizili)
"""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np

UZANTILAR = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def hucreleri_bul(gri, min_hucre_oran=0.0005, max_hucre_oran=0.5):
    h, w = gri.shape
    ikili = cv2.adaptiveThreshold(gri, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 25, 15)
    yatay_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(w // 40, 10), 1))
    dikey_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(h // 40, 10)))
    yatay = cv2.morphologyEx(ikili, cv2.MORPH_OPEN, yatay_k, iterations=2)
    dikey = cv2.morphologyEx(ikili, cv2.MORPH_OPEN, dikey_k, iterations=2)
    izgara = cv2.add(yatay, dikey)
    izgara = cv2.dilate(izgara, np.ones((3, 3), np.uint8), iterations=1)
    ters = cv2.bitwise_not(izgara)
    sayi, etiketler, istatistik, _ = cv2.connectedComponentsWithStats(ters, 8)
    alan_toplam = h * w
    kutular = []
    for i in range(1, sayi):
        x, y, cw, ch, alan = istatistik[i]
        if alan < alan_toplam * min_hucre_oran:
            continue
        if alan > alan_toplam * max_hucre_oran:
            continue
        if cw < 20 or ch < 12:
            continue
        if cw > w * 0.85 and ch > h * 0.85:
            continue
        kutular.append((x, y, cw, ch))
    return kutular


def satirlara_grupla(kutular, tolerans_oran=0.5):
    if not kutular:
        return []
    kutular = sorted(kutular, key=lambda b: (b[1], b[0]))
    ort_h = np.median([b[3] for b in kutular])
    tol = ort_h * tolerans_oran
    satirlar = []
    for b in kutular:
        yerlesti = False
        for s in satirlar:
            if abs(b[1] - s[0][1]) < tol:
                s.append(b)
                yerlesti = True
                break
        if not yerlesti:
            satirlar.append([b])
    satirlar.sort(key=lambda s: s[0][1])
    sonuc = []
    for ri, s in enumerate(satirlar, 1):
        for ci, b in enumerate(sorted(s, key=lambda b: b[0]), 1):
            sonuc.append((ri, ci, b))
    return sonuc


def kes(form_yolu, cikti_dizin, onizleme_dizin, harita, ic_pay=4):
    form_id = form_yolu.stem
    img = cv2.imread(str(form_yolu))
    if img is None:
        print(f"HATA: {form_yolu.name} okunamadı", file=sys.stderr)
        return 0
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hucreler = satirlara_grupla(hucreleri_bul(gri))
    if not hucreler:
        print(f"UYARI: {form_id}: hiç hücre bulunamadı (çizgiler soluk olabilir)",
              file=sys.stderr)
        return 0
    onizleme = img.copy()
    kaydedilen = 0
    for ri, ci, (x, y, w, h) in hucreler:
        anahtar = f"r{ri}_c{ci}"
        alan_adi = harita.get(anahtar) if harita else anahtar
        renk = (0, 200, 0) if alan_adi is not None else (0, 0, 220)
        cv2.rectangle(onizleme, (x, y), (x + w, y + h), renk, 2)
        cv2.putText(onizleme, anahtar, (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, renk, 2)
        if alan_adi is None:
            continue
        p = min(ic_pay, w // 4, h // 4)
        kirp = img[y + p: y + h - p, x + p: x + w - p]
        if kirp.size == 0:
            continue
        cv2.imwrite(str(cikti_dizin / f"{form_id}__{alan_adi}.png"), kirp)
        kaydedilen += 1
    on_yol = onizleme_dizin / f"{form_id}_onizleme.png"
    cv2.imwrite(str(on_yol), onizleme)
    print(f"{form_id}: {len(hucreler)} hücre bulundu, {kaydedilen} kırpıntı "
          f"kaydedildi  (önizleme: {on_yol.name})")
    return kaydedilen


def main():
    ap = argparse.ArgumentParser(description="Tablo hücrelerini otomatik kes")
    ap.add_argument("--giris", required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--harita", default=None)
    ap.add_argument("--ic-pay", type=int, default=4)
    args = ap.parse_args()
    giris, cikti = Path(args.giris), Path(args.cikti)
    onizleme = cikti.parent / "onizleme"
    cikti.mkdir(parents=True, exist_ok=True)
    onizleme.mkdir(parents=True, exist_ok=True)
    harita = {}
    if args.harita:
        harita = json.loads(Path(args.harita).read_text(encoding="utf-8"))
    formlar = [f for f in sorted(giris.iterdir()) if f.suffix.lower() in UZANTILAR]
    if not formlar:
        print(f"{giris} altında görüntü yok.", file=sys.stderr)
        sys.exit(1)
    toplam = 0
    for f in formlar:
        toplam += kes(f, cikti, onizleme, harita, args.ic_pay)
    print(f"\nToplam {toplam} kırpıntı → {cikti}")
    print(f"Önizlemeleri kontrol edin: {onizleme}/")
    if not args.harita:
        print("İpucu: önizlemeye bakıp alan_haritasi.json yazarsanız "
              "kırpıntılar gerçek alan adlarıyla kaydedilir.")


if __name__ == "__main__":
    main()
