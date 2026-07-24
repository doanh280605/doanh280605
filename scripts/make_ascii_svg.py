from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "IMG_2962.jpg"
OUTPUT = ROOT / "avi-ascii.svg"

COLS = 112
ROWS = 104
CELL_W = 6
CELL_H = 10
RAMP = " .`:-=+*cs#%@"


def portrait_grid() -> list[str]:
    image = Image.open(SOURCE).convert("RGB")

    # A portrait-first crop keeps the architecture from competing with the face.
    image = image.crop((650, 1250, 2600, 3650))
    image = ImageOps.grayscale(image)
    image = ImageOps.autocontrast(image, cutoff=(0.5, 1))
    image = ImageEnhance.Contrast(image).enhance(1.12)
    image = image.filter(ImageFilter.UnsharpMask(radius=1.4, percent=145, threshold=2))
    image = image.resize((COLS, ROWS), Image.Resampling.LANCZOS)

    values = np.asarray(image, dtype=np.float32)
    local_mean = np.asarray(image.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32)
    values = np.clip(local_mean + (values - local_mean) * 1.75, 0, 255)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    x = (xx + 0.5) / COLS
    y = (yy + 0.5) / ROWS

    # Keep the complete head, ears, neck, and shoulders while clearing the scene.
    head = ((x - 0.515) / 0.285) ** 2 + ((y - 0.405) / 0.285) ** 2 <= 1
    neck = (y >= 0.57) & (y <= 0.73) & (np.abs(x - 0.515) <= 0.16)
    half_width = np.clip(0.27 + (y - 0.64) * 0.82, 0.27, 0.49)
    torso = (y >= 0.63) & (np.abs(x - 0.515) <= half_width)
    mask = head | neck | torso
    values[~mask] = 255

    # Lift deep shadows slightly so facial detail survives instead of becoming a blob.
    values = 255 * np.power(values / 255, 0.9)
    indices = np.clip(((255 - values) / 255 * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)
    return ["".join(RAMP[i] for i in row).rstrip() for row in indices]


def build_svg(lines: list[str]) -> str:
    width = COLS * CELL_W
    height = ROWS * CELL_H
    defs = []
    rows = []

    for index, line in enumerate(lines):
        y = (index + 1) * CELL_H
        clip_id = f"row-{index}"
        begin = index * 0.035
        duration = 0.42
        defs.append(
            f'<clipPath id="{clip_id}"><rect x="0" y="{index * CELL_H}" width="0" '
            f'height="{CELL_H + 1}"><animate attributeName="width" from="0" to="{width}" '
            f'dur="{duration}s" begin="{begin:.3f}s" fill="freeze"/></rect></clipPath>'
        )
        rows.append(
            f'<text x="0" y="{y}" clip-path="url(#{clip_id})">{escape(line)}</text>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">ASCII portrait of Doanh Phung</title>
  <desc id="desc">A monochrome portrait rendered as animated ASCII characters.</desc>
  <defs>{''.join(defs)}</defs>
  <g fill="#8b949e" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="10px" xml:space="preserve">
    {''.join(rows)}
  </g>
</svg>
'''


if __name__ == "__main__":
    OUTPUT.write_text(build_svg(portrait_grid()), encoding="utf-8")
    print(f"Wrote {OUTPUT.name}")
