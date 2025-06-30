"""Generate the VXDF layout diagram for the paper.

Saves a PDF to figures/vxdf_layout.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # Define layout blocks
    blocks = [
        {"label": "Header (Magic, Version)", "size": 8, "color": "#d1e7dd"},
        {"label": "Chunk 1 (Text, Meta, Vec)", "size": 40, "color": "#cfe2ff"},
        {"label": "Chunk 2", "size": 40, "color": "#cfe2ff"},
        {"label": "...", "size": 5, "color": "#f8f9fa"},
        {"label": "Chunk N", "size": 40, "color": "#cfe2ff"},
        {"label": "Index (JSON)", "size": 20, "color": "#fff3cd"},
        {"label": "Footer (Index Addr, Checksum)", "size": 15, "color": "#f8d7da"},
        {"label": "End Marker", "size": 8, "color": "#d1e7dd"},
    ]

    y_pos = 0
    heights = [b["size"] for b in blocks]
    total_height = sum(heights)

    for block in reversed(blocks):
        ax.barh(
            y=y_pos + block["size"] / 2,
            width=1,
            height=block["size"],
            color=block["color"],
            edgecolor="black",
            linewidth=0.8,
        )
        ax.text(
            0.5,
            y_pos + block["size"] / 2,
            block["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontfamily="monospace",
        )
        y_pos += block["size"]

    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_height)
    ax.axis("off")

    fig.tight_layout(pad=0)
    output_path = Path(__file__).parent / "figures" / "vxdf_layout.pdf"
    output_path.parent.mkdir(exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
