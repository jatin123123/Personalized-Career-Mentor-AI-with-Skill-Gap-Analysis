"""
Generate icon16.png, icon48.png, icon128.png for the Chrome extension.
Uses only Python standard library — no pip installs needed.
"""
import struct, zlib, os

OUT = os.path.join(os.path.dirname(__file__), "icons")
os.makedirs(OUT, exist_ok=True)

def make_png(size):
    """Create a square PNG with an indigo-violet gradient and a white document icon."""
    # ── Build raw RGBA pixels ──────────────────────────────────────────────────
    pixels = []
    for y in range(size):
        row = []
        for x in range(size):
            # Gradient: top-left indigo → bottom-right violet
            t = (x + y) / (2 * (size - 1))
            r = int(99  + t * (224 - 99))   # 99→224
            g = int(91  + t * (64  - 91))   # 91→64
            b = int(255 + t * (251 - 255))  # 255→251
            a = 255

            # Rounded-rect mask (clip corners)
            corner = size * 0.2
            cx, cy = min(x, size-1-x), min(y, size-1-y)
            if cx < corner and cy < corner:
                dx, dy = corner - cx - 1, corner - cy - 1
                if dx*dx + dy*dy > corner*corner:
                    a = 0   # transparent corner

            # White document shape in center
            pad   = size * 0.16
            doc_w = size - pad * 2
            doc_h = doc_w * 1.3
            dx    = (size - doc_w) / 2
            dy    = (size - doc_h) / 2
            fold  = doc_w * 0.28

            in_doc = (dx <= x <= dx + doc_w) and (dy <= y <= dy + doc_h)

            # Fold corner triangle
            fold_x = dx + doc_w - fold
            in_fold = (x >= fold_x) and (y <= dy + fold) and ((x - fold_x) + (y - dy) <= fold)

            if in_doc and a:
                if in_fold:
                    # Slightly tinted fold
                    r2 = max(0, r - 40)
                    g2 = max(0, g - 40)
                    b2 = 255
                    row.extend([r2, g2, b2, 200])
                else:
                    # White document body
                    row.extend([245, 245, 255, 240])

                # Text lines inside doc
                line_x  = dx + doc_w * 0.15
                line_w  = doc_w * 0.58
                line_h  = max(1, doc_h * 0.06)
                for pct in [0.48, 0.61, 0.74, 0.86]:
                    ly = dy + doc_h * pct
                    lw = line_w if pct < 0.8 else line_w * 0.65
                    if (line_x <= x <= line_x + lw) and (ly <= y <= ly + line_h) and not in_fold:
                        row[-4:] = [120, 100, 230, 200]
                        break
            else:
                row.extend([r, g, b, a])

        pixels.append(row)

    # ── Encode as PNG ───────────────────────────────────────────────────────────
    def chunk(name, data):
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw = b""
    for row in pixels:
        raw += b"\x00" + bytes(row)   # filter type 0 (None) per row

    png  = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    return png


for sz in [16, 48, 128]:
    path = os.path.join(OUT, f"icon{sz}.png")
    with open(path, "wb") as f:
        f.write(make_png(sz))
    print(f"[OK] Created {path}")

print("\nDone! All icons created in extension/icons/")
