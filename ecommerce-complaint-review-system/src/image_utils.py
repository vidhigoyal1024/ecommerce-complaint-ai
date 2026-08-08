"""
Image forensics preprocessing: Error Level Analysis (ELA).

ELA is a well-established technique for spotting digitally altered regions
in a JPEG image. The idea: re-save the image at a known JPEG quality, then
take the pixel-wise difference between the original and the re-saved copy.
Untouched, uniformly-compressed regions settle to a low, even error level.
Regions that were pasted in, redrawn, or edited after the original save
were compressed a *different* number of times than the rest of the image,
so they show up as noticeably brighter patches in the ELA output.

This gives the CNN a meaningful, well-understood signal to learn from,
rather than trying to learn "real vs fake" from raw pixels directly.
"""
import io
from PIL import Image, ImageChops
import numpy as np

ELA_QUALITY = 90
IMAGE_SIZE = 128


def compute_ela(image: Image.Image, quality: int = ELA_QUALITY) -> Image.Image:
    """Returns an RGB PIL image representing the amplified compression-error map."""
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)

    diff = ImageChops.difference(image, resaved)
    diff_arr = np.array(diff).astype(np.float32)

    max_diff = float(diff_arr.max())
    if max_diff == 0:
        max_diff = 1.0
    scale = 255.0 / max_diff
    ela_arr = np.clip(diff_arr * scale, 0, 255).astype(np.uint8)

    return Image.fromarray(ela_arr)


def preprocess_for_model(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Full preprocessing: compute ELA, resize, scale to [0, 1], return CHW numpy array."""
    ela_img = compute_ela(image).resize((size, size))
    arr = np.array(ela_img).astype(np.float32) / 255.0  # HWC in [0,1]
    arr = arr.transpose(2, 0, 1)  # CHW
    return arr


def load_image_any_format(path_or_file) -> Image.Image:
    return Image.open(path_or_file).convert("RGB")
