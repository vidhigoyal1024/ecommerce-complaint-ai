"""
Generates a small set of synthetic "authentic product photo" images so the
image-authenticity CNN has something to train on out of the box.

These are simple procedurally-rendered product-style images (boxes, labels,
gradients, textures) — NOT real photographs. They exist purely so the demo
pipeline is runnable end-to-end without needing you to source a real photo
dataset first.

For a real deployment, replace data/images/real/ with genuine, verified-
authentic customer/product photos before training.

Usage:
    python scripts/generate_sample_images.py --count 25
"""
import os
import argparse
import random

from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "images", "real")


def random_color():
    return tuple(random.randint(30, 225) for _ in range(3))


def make_product_photo(size=(300, 300)):
    """Draws a plausible 'product on a background' style synthetic photo."""
    bg = random_color()
    img = Image.new("RGB", size, bg)
    draw = ImageDraw.Draw(img)

    # soft background gradient
    for y in range(size[1]):
        shade = int(20 * (y / size[1]))
        draw.line([(0, y), (size[0], y)], fill=tuple(max(0, c - shade) for c in bg))

    # a "product" shape (box, bottle, or blob)
    shape_type = random.choice(["box", "circle", "rounded"])
    fg = random_color()
    margin_x = random.randint(40, 80)
    margin_y = random.randint(40, 80)
    box = (margin_x, margin_y, size[0] - margin_x, size[1] - margin_y)

    if shape_type == "box":
        draw.rectangle(box, fill=fg, outline=tuple(max(0, c - 40) for c in fg), width=3)
    elif shape_type == "circle":
        draw.ellipse(box, fill=fg, outline=tuple(max(0, c - 40) for c in fg), width=3)
    else:
        draw.rounded_rectangle(box, radius=25, fill=fg, outline=tuple(max(0, c - 40) for c in fg), width=3)

    # a "label" rectangle with some texture lines
    label_w = (box[2] - box[0]) * 0.5
    label_h = (box[3] - box[1]) * 0.3
    lx0 = box[0] + (box[2] - box[0] - label_w) / 2
    ly0 = box[1] + (box[3] - box[1] - label_h) / 2
    label_box = (lx0, ly0, lx0 + label_w, ly0 + label_h)
    draw.rectangle(label_box, fill=(245, 245, 245))
    for i in range(3):
        ly = ly0 + 8 + i * 10
        draw.line([(lx0 + 8, ly), (lx0 + label_w - 8, ly)], fill=(180, 180, 180), width=2)

    # mild blur + noise to look more photo-like and give JPEG compression something to bite into
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(7)

    for i in range(args.count):
        img = make_product_photo()
        path = os.path.join(OUT_DIR, f"sample_real_{i:03d}.jpg")
        img.save(path, "JPEG", quality=92)

    print(f"Wrote {args.count} synthetic sample photos to {OUT_DIR}")
    print("Replace these with real, verified-authentic photos for production use.")


if __name__ == "__main__":
    main()
