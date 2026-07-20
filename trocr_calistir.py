#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trocr_calistir.py — Faz 1 hızlı testi: TrOCR ile el yazısı kırpıntı tanıma

Kırpıntı adlandırma kuralı (çift alt çizgi ile):
    FORMID__ALANADI.png        örn:  F0001__hasta_adi.png
                                     F0001__ilac_1.jpg
                                     F0002__tani_kodu.png

Girdi : /data/giris/   (png/jpg/jpeg/tif/bmp kırpıntılar)
Çıktı : /data/cikti/tahminler/FORMID.json   (degerlendir.py şeması)

Ek mod:
    --gt-taslak  → aynı kırpıntılardan /data/cikti/ground_truth/ altına
                   etiketleme taslakları üretir ("deger" alanları boş);
                   siz doğru metinleri elle doldurursunuz.

Güven skoru: üretilen token'ların ortalama olasılığı (0-1). Faz 1'de
kabaca yönlendirme kalitesini görmek için yeterli; Faz 2'de rafine edilir.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

GIRIS = Path("/data/giris")
CIKTI = Path("/data/cikti")
UZANTILAR = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
MODEL_ADI = "microsoft/trocr-base-handwritten"


def kirpintilari_bul():
    """Giriş dizinindeki kırpıntıları form_id -> [(alan_adi, yol)] olarak grupla."""
    formlar = defaultdict(list)
    hatali = []
    for f in sorted(GIRIS.iterdir()):
        if f.suffix.lower() not in UZANTILAR:
            continue
        if "__" not in f.stem:
            hatali.append(f.name)
            continue
        form_id, alan_adi = f.stem.split("__", 1)
        formlar[form_id].append((alan_adi, f))
    if hatali:
        print(f"UYARI: adlandırma kuralına uymayan {len(hatali)} dosya atlandı "
              f"(FORMID__ALANADI.png bekleniyor): {', '.join(hatali[:5])}",
              file=sys.stderr)
    return formlar


def gt_taslak_uret(formlar):
    """Etiketleme için boş ground truth taslakları üret."""
    hedef = CIKTI / "ground_truth"
    hedef.mkdir(parents=True, exist_ok=True)
    for form_id, alanlar in formlar.items():
        dosya = hedef / f"{form_id}.json"
        if dosya.exists():
            print(f"  {dosya.name} zaten var, üzerine yazılmadı")
            continue
        veri = {
            "form_id": form_id,
            "zorluk": "DOLDUR: kolay|orta|zor",
            "doktor_kodu": "DOLDUR",
            "tarama_kalitesi": "DOLDUR: iyi|orta|kotu",
            "alanlar": [
                {
                    "alan_adi": alan_adi,
                    "alan_tipi": "el_yazisi",
                    "sozluk": None,
                    "deger": "DOLDUR: görüntüde yazan doğru metin",
                }
                for alan_adi, _ in sorted(alanlar)
            ],
        }
        dosya.write_text(json.dumps(veri, ensure_ascii=False, indent=2),
                         encoding="utf-8")
        print(f"  {dosya.name} oluşturuldu ({len(alanlar)} alan)")
    print(f"\nTaslaklar hazır: {hedef}")
    print("Şimdi 'DOLDUR' yazan yerleri gerçek değerlerle doldurun.")


def tani(formlar):
    """TrOCR ile tüm kırpıntıları tanı, form bazlı JSON üret."""
    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    torch.set_num_threads(4)

    print(f"Model yükleniyor: {MODEL_ADI}")
    print("(ilk çalıştırmada ~1.3 GB ağırlık indirilir, /models'a önbelleklenir)")
    t0 = time.time()
    processor = TrOCRProcessor.from_pretrained(MODEL_ADI)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ADI)
    model.eval()
    print(f"Model hazır ({time.time()-t0:.0f} sn)\n")

    hedef = CIKTI / "tahminler"
    hedef.mkdir(parents=True, exist_ok=True)

    toplam = sum(len(a) for a in formlar.values())
    i = 0
    sureler = []

    for form_id, alanlar in sorted(formlar.items()):
        sonuc = {"form_id": form_id, "model": MODEL_ADI, "alanlar": []}
        for alan_adi, yol in sorted(alanlar):
            i += 1
            t0 = time.time()
            try:
                img = Image.open(yol).convert("RGB")
                pixel_values = processor(images=img,
                                         return_tensors="pt").pixel_values
                with torch.no_grad():
                    cikti = model.generate(
                        pixel_values,
                        max_new_tokens=64,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                metin = processor.batch_decode(
                    cikti.sequences, skip_special_tokens=True)[0].strip()

                # güven: üretilen token'ların ortalama olasılığı
                probs = []
                for adim, skorlar in enumerate(cikti.scores):
                    token_id = cikti.sequences[0, adim + 1]
                    p = torch.softmax(skorlar[0], dim=-1)[token_id].item()
                    probs.append(p)
                guven = round(sum(probs) / len(probs), 4) if probs else 0.0
            except Exception as e:
                print(f"  HATA {yol.name}: {e}", file=sys.stderr)
                metin, guven = "", 0.0

            dt = time.time() - t0
            sureler.append(dt)
            sonuc["alanlar"].append({
                "alan_adi": alan_adi,
                "deger": metin,
                "guven": guven,
            })
            print(f"[{i}/{toplam}] {form_id}/{alan_adi:<20} "
                  f"{dt:4.1f}sn  güven={guven:.2f}  → '{metin}'")

        (hedef / f"{form_id}.json").write_text(
            json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")

    if sureler:
        print(f"\nBitti: {toplam} alan, ortalama {sum(sureler)/len(sureler):.1f} "
              f"sn/alan, toplam {sum(sureler)/60:.1f} dk")
    print(f"Tahminler: {hedef}")
    print("\nSonraki adım (ana makinede):")
    print("  python3 degerlendir.py --gt data/cikti/ground_truth "
          "--tahmin data/cikti/tahminler --csv faz1_sonuc.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-taslak", action="store_true",
                    help="Tanıma yerine etiketleme taslakları üret")
    args = ap.parse_args()

    if not GIRIS.exists():
        print(f"Giriş dizini yok: {GIRIS} — kırpıntıları data/giris/ altına "
              "koyun (FORMID__ALANADI.png)", file=sys.stderr)
        sys.exit(1)

    formlar = kirpintilari_bul()
    if not formlar:
        print("data/giris/ altında uygun kırpıntı bulunamadı.", file=sys.stderr)
        sys.exit(1)

    print(f"{len(formlar)} form, "
          f"{sum(len(a) for a in formlar.values())} kırpıntı bulundu.\n")

    if args.gt_taslak:
        gt_taslak_uret(formlar)
    else:
        tani(formlar)


if __name__ == "__main__":
    main()
