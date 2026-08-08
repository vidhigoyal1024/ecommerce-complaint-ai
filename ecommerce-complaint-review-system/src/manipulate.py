"""
Synthetic tampering generator.

We don't have a labeled "real vs. fake complaint photo" dataset lying around,
so this module manufactures one automatically from a folder of authentic
("real") images: for each real image it produces one or more tampered
("fake") copies using classic forgery techniques. Training the CNN on
ELA(real) vs. ELA(tampered) teaches it the visual signature of local
recompression / splicing artifacts described in image_utils.py.

Techniques implemented:
  - copy_move_forgery: duplicate a patch of the image elsewhere in itself
    (the classic "clone stamp" forgery used to hide/duplicate objects).
  - splice_forgery: paste a patch from a *different* image in ("this photo
    contains an object copied from another photo").
  - noise_patch_forgery: locally blur/re-render a region to simulate
    an edited-out defect or edited-in damage.
  - recompress_forgery: resave at a different JPEG quality after a small
    localized edit, simulating a screenshot/edited-then-reshared photo.
"""
import random
import numpy as np
from PIL import Image, ImageFilter


def _random_patch_box(w, h, min_frac=0.15, max_frac=0.35):
    pw = int(w * random.uniform(min_frac, max_frac))
    ph = int(h * random.uniform(min_frac, max_frac))
    x0 = random.randint(0, max(1, w - pw - 1))
    y0 = random.randint(0, max(1, h - ph - 1))
    return x0, y0, x0 + pw, y0 + ph


def copy_move_forgery(image: Image.Image) -> Image.Image:
    img = image.copy()
    w, h = img.size
    box = _random_patch_box(w, h)
    patch = img.crop(box)

    pw, ph = patch.size
    dest_x = random.randint(0, max(1, w - pw - 1))
    dest_y = random.randint(0, max(1, h - ph - 1))
    img.paste(patch, (dest_x, dest_y))
    return img


def splice_forgery(image: Image.Image, donor_image: Image.Image) -> Image.Image:
    img = image.copy()
    w, h = img.size
    donor = donor_image.convert("RGB").resize((w, h))
    box = _random_patch_box(w, h)
    patch = donor.crop(box)
    img.paste(patch, (box[0], box[1]))
    return img


def noise_patch_forgery(image: Image.Image) -> Image.Image:
    img = image.copy()
    w, h = img.size
    box = _random_patch_box(w, h)
    region = img.crop(box)
    region = region.filter(ImageFilter.GaussianBlur(radius=random.uniform(2, 5)))

    arr = np.array(region).astype(np.int16)
    noise = np.random.normal(0, 18, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    region = Image.fromarray(arr)

    img.paste(region, (box[0], box[1]))
    return img


def recompress_forgery(image: Image.Image) -> Image.Image:
    """Simulates a locally edited-then-resaved photo (double JPEG compression)."""
    img = copy_move_forgery(image)  # small structural edit first
    import io
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=random.choice([40, 55, 70]))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


FORGERY_FUNCS = [copy_move_forgery, noise_patch_forgery, recompress_forgery]


def generate_fake(image: Image.Image, donor_pool=None) -> Image.Image:
    """Applies a randomly chosen forgery technique. Uses splicing if a donor pool is given."""
    funcs = list(FORGERY_FUNCS)
    if donor_pool:
        funcs.append(lambda img: splice_forgery(img, random.choice(donor_pool)))
    func = random.choice(funcs)
    return func(image)
