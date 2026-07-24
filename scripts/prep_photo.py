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

    # IMG_2962-specific head-and-shoulders crop. This keeps every facial feature
    # while giving the face most of the available ASCII cells.
    left, top, right, bottom = 650, 1350, 2650, 3650
    rgba = rgba[top:bottom, left:right]

    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.08, beta=5)
    edges = cv2.Canny(gray, 42, 105)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    gray = np.clip(gray.astype(np.float32) - edges.astype(np.float32) * 0.24, 0, 255).astype(np.uint8)

    alpha = rgba[:, :, 3].astype(np.float32) / 255
    composited = gray.astype(np.float32) * alpha + 255 * (1 - alpha)
    Image.fromarray(composited.astype(np.uint8), mode="L").save(OUTPUT)
    print(f"Wrote {OUTPUT.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/prep_photo.py IMG_2962.jpg")
    prep(Path(sys.argv[1]))
