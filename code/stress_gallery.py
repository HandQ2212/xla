"""Render explanatory cases; all scored images remain in the CSV denominator."""
import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

import external_data as data
import pipeline as pl
from stress_eval import transform

CASES = {"000031": "sealed seam", "000052": "cover + patch", "000061": "expansion joint",
         "000064": "paint + vehicle shadow", "000066": "faint crack", "000073": "thin-filter regression"}
WIDTH, HEIGHT = 320, 300


def panel(image, title, footer):
    output = Image.new("RGB", (WIDTH, HEIGHT), "white")
    picture = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    picture.thumbnail((WIDTH, HEIGHT - 45))
    output.paste(picture, (0, 23))
    draw = ImageDraw.Draw(output)
    draw.text((5, 4), title, fill="black")
    draw.text((5, HEIGHT - 17), footer, fill="black")
    return output


def main():
    manifest = json.loads(data.MANIFEST.read_text(encoding="utf-8"))
    output = data.ROOT / "docs" / "benchmarks" / "external"
    rows = list(csv.DictReader((data.ROOT / "results/external-discovery/per_image.csv").open(encoding="utf-8")))
    table = {(r["id"], r["transform"], r["variant"]): r for r in rows}
    canvas = Image.new("RGB", (WIDTH * 5, HEIGHT * len(CASES)), "white")
    for index, (sid, description) in enumerate(CASES.items()):
        case = next(c for c in manifest["cases"] if c["id"] == sid)
        image, label = data.read_case(case)
        gt = np.isin(label, [30, 150])
        annotation = image.copy()
        for value, color in [(30, (0, 255, 0)), (150, (255, 255, 0)), (90, (0, 255, 255)),
                             (120, (255, 0, 255)), (180, (255, 0, 0)), (210, (0, 128, 255))]:
            region = label == value
            annotation[region] = (annotation[region] * .35 + np.array(color) * .65).astype(np.uint8)
        panels = [panel(image, sid + ": " + description, "Original photograph"),
                  panel(annotation, "Published expert labels", "green/cyan=open; yellow=sealed")]
        for variant in ("legacy", "reconstruct", "thin"):
            path = data.ROOT / "results/external-discovery/predictions/original" / variant / (sid + ".png")
            prediction = cv2.imread(str(path), 0) > 127
            row = table[sid, "original", variant]
            note = f"Dice={float(row['dice']):.3f}" if gt.any() else f"FALSE ALARM, area={float(row['area_pct']):.2f}%"
            panels.append(panel(pl.overlay_red(image, prediction), variant + " prediction (red)", note))
        for column, item in enumerate(panels):
            canvas.paste(item, (WIDTH * column, HEIGHT * index))
    canvas.save(output / "gallery.jpg", quality=90)
    canvas = Image.new("RGB", (WIDTH * 4, HEIGHT * 2), "white")
    for index, sid in enumerate(("000064", "000066")):
        image, label = data.read_case(next(c for c in manifest["cases"] if c["id"] == sid))
        shaded, _ = transform(image, label, "shadow")
        panels = [panel(image, sid + ": original", "Same original GT and source group"),
                  panel(shaded, "Synthetic smooth shadow", "illumination multiplier 0.35-1")]
        for name, img in [("original", image), ("shadow", shaded)]:
            path = data.ROOT / "results/external-discovery/predictions" / name / "legacy" / (sid + ".png")
            prediction = cv2.imread(str(path), 0) > 127
            panels.append(panel(pl.overlay_red(img, prediction), "P: " + name, "red = predicted region"))
        for column, item in enumerate(panels):
            canvas.paste(item, (WIDTH * column, HEIGHT * index))
    canvas.save(output / "shadow-gallery.jpg", quality=90)


if __name__ == "__main__":
    main()
