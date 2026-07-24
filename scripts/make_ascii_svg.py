from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "source-prepped.png"
OUTPUT = ROOT / "avi-ascii.svg"

COLS = 140
CELL_W = 6
CELL_H = 10
RAMP = " .`:-=+*cs#%@"


def portrait_grid() -> list[str]:
    image = Image.open(SOURCE).convert("L")
    rows = max(1, round(COLS * image.height / image.width * CELL_W / CELL_H))
    image = image.resize((COLS, rows), Image.Resampling.LANCZOS)
    values = np.asarray(image, dtype=np.float32)

    # White is the leading space; progressively darker pixels use denser glyphs.
    indices = np.clip(
        ((255 - values) / 255 * (len(RAMP) - 1)).astype(int),
        0,
        len(RAMP) - 1,
    )
    return ["".join(RAMP[index] for index in row).rstrip() for row in indices]


def build_svg(lines: list[str]) -> str:
    width = COLS * CELL_W
    height = len(lines) * CELL_H
    defs: list[str] = []
    rows: list[str] = []
    cursors: list[str] = []

    for index, line in enumerate(lines):
        top = index * CELL_H
        baseline = (index + 1) * CELL_H
        clip_id = f"row-{index}"
        begin = index * 0.035
        duration = 0.42
        defs.append(
            f'<clipPath id="{clip_id}"><rect x="0" y="{top}" width="0" '
            f'height="{CELL_H + 1}"><animate attributeName="width" from="0" '
            f'to="{width}" dur="{duration}s" begin="{begin:.3f}s" fill="freeze"/>'
            f"</rect></clipPath>"
        )
        rows.append(
            f'<text x="0" y="{baseline}" clip-path="url(#{clip_id})">'
            f"{escape(line)}</text>"
        )
        cursors.append(
            f'<rect x="0" y="{top + 1}" width="5" height="{CELL_H - 2}" opacity="0">'
            f'<animate attributeName="opacity" values="0;1;1;0" dur="{duration}s" '
            f'begin="{begin:.3f}s" fill="freeze"/><animate attributeName="x" '
            f'from="0" to="{width}" dur="{duration}s" begin="{begin:.3f}s" '
            f'fill="freeze"/></rect>'
        )

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
  <title id="title">ASCII portrait of Doanh Phung</title>
  <desc id="desc">A monochrome portrait rendered as animated ASCII characters.</desc>
  <defs>{''.join(defs)}</defs>
  <g fill="#8b949e" font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="10px" xml:space="preserve">
    {''.join(rows)}
    {''.join(cursors)}
  </g>
</svg>
'''


if __name__ == "__main__":
    OUTPUT.write_text(build_svg(portrait_grid()), encoding="utf-8")
    print(f"Wrote {OUTPUT.name}")
