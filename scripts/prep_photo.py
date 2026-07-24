import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "source-prepped.png"


def prep(source_path: Path) -> None:
    source = Image.open(source_path).convert("RGB")
    subject = remove(source).convert("RGBA")
    rgba = np.asarray(subject)
    alpha = rgba[:, :, 3]

    points = cv2.findNonZero((alpha > 20).astype(np.uint8))
    if points is None:
        raise RuntimeError("Background removal did not find a subject")

    x, y, width, height = cv2.boundingRect(points)
    # IMG_2962 is a long portrait. Keep the complete head and upper body so the
    # face receives enough ASCII cells to retain the eyes, nose, ears, and mouth.
    side_margin = int(width * 0.08)
    top_margin = int(height * 0.04)
    left = max(0, x - side_margin)
    top = max(0, y - top_margin)
    right = min(subject.width, x + width + side_margin)
    bottom = min(subject.height, y + int(height * 0.64))
    rgba = rgba[top:bottom, left:right]

    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.08, beta=5)

    alpha = rgba[:, :, 3].astype(np.float32) / 255
    composited = gray.astype(np.float32) * alpha + 255 * (1 - alpha)
    Image.fromarray(composited.astype(np.uint8), mode="L").save(OUTPUT)
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/prep_photo.py IMG_2962.jpg")
    prep(Path(sys.argv[1]))
