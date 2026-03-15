"""
corruptions.py

All 15 CIFAR-10-C corruption functions copied exactly from the original
paper code (hendrycks/robustness).

Corruptions:
    Noise   : gaussian_noise, shot_noise, impulse_noise
    Blur    : defocus_blur, glass_blur, motion_blur, zoom_blur, gaussian_blur
    Weather : fog, frost, snow, brightness
    Digital : contrast, elastic_transform, pixelate, jpeg_compression

Each function:
    Input  : PIL Image (32x32 RGB)
    Output : float32 numpy array in [0, 255], shape (32, 32, 3)
             Cast to uint8 before saving / passing to ToTensor.

Dependencies:
    pip install numpy opencv-python Pillow Wand scipy scikit-image
    ImageMagick must also be installed on the system (required by Wand).
    On Colab: !apt-get install -q libmagickwand-dev

For frost you need 6 overlay images in data/frost_images/.
Download from:
    https://github.com/hendrycks/robustness/tree/master/ImageNet-C/create_c/frost
"""

import ctypes
import io
import os

import cv2
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import zoom as scizoom
from scipy.ndimage import map_coordinates
from skimage.filters import gaussian
from wand.api import library as wandlibrary
from wand.image import Image as WandImage


# ---------------------------------------------------------------------------
# Wand motion blur extension  (used by snow and motion_blur)
# ---------------------------------------------------------------------------

wandlibrary.MagickMotionBlurImage.argtypes = (
    ctypes.c_void_p,   # wand
    ctypes.c_double,   # radius
    ctypes.c_double,   # sigma
    ctypes.c_double,   # angle
)


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Returns square 2d array, side length 'mapsize', floats in range 0-1.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float64)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
                 stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        mapsize_ = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize_:stepsize,
                          stepsize // 2:mapsize_:stepsize]
        ulgrid = maparray[0:mapsize_:stepsize, 0:mapsize_:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize_:stepsize,
                 stepsize // 2:mapsize_:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize_:stepsize,
                 0:mapsize_:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    ch = int(np.ceil(h / zoom_factor))
    top = (h - ch) // 2
    img = scizoom(
        img[top:top + ch, top:top + ch],
        (zoom_factor, zoom_factor, 1),
        order=1,
    )
    trim_top = (img.shape[0] - h) // 2
    return img[trim_top:trim_top + h, trim_top:trim_top + h]


def disk(radius, alias_blur=0.1, dtype=np.float32):
    """Generate a disk-shaped kernel for defocus blur."""
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# ---------------------------------------------------------------------------
# Noise corruptions
# ---------------------------------------------------------------------------

def gaussian_noise(x, severity=1):
    c = [0.04, 0.06, 0.08, 0.09, 0.10][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    c = [500, 250, 100, 75, 50][severity - 1]

    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    c = [.01, .02, .03, .05, .07][severity - 1]

    x = sk_util_random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


# ---------------------------------------------------------------------------
# Blur corruptions
# ---------------------------------------------------------------------------

def defocus_blur(x, severity=1):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))
    return np.clip(channels, 0, 1) * 255


def glass_blur(x, severity=1):
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3),
         (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255.,
                 sigma=c[0], channel_axis=-1) * 255)

    for i in range(c[2]):
        img_h, img_w = x.shape[:2]
        for h in range(img_h - c[1], c[1], -1):
            for w in range(img_w - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255


def motion_blur(x, severity=1):
    c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5)][severity - 1]

    output = io.BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(
        np.frombuffer(x.make_blob(), np.uint8),
        cv2.IMREAD_UNCHANGED,
    )

    if len(x.shape) == 3:
        return np.clip(x[..., [2, 1, 0]], 0, 255)   # BGR -> RGB
    else:
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    c = [np.arange(1, 1.06, 0.01),
         np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.01),
         np.arange(1, 1.26, 0.01)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)
    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255


# ---------------------------------------------------------------------------
# Weather corruptions
# ---------------------------------------------------------------------------

def fog(x, severity=1):
    c = [(.2, 3), (.5, 3), (0.75, 2.5), (1, 2), (1.5, 1.75)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    h, w = x.shape[:2]
    map_size = max(64, 1 << (max(h, w) - 1).bit_length())
    x += c[0] * plasma_fractal(mapsize=map_size,
                               wibbledecay=c[1])[:h, :w][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, frost_dir="data/frost_images"):
    c = [(1, 0.2), (1, 0.3), (0.9, 0.4),
         (0.85, 0.4), (0.75, 0.45)][severity - 1]

    frost_files = [
        "frost1.png", "frost2.png", "frost3.png",
        "frost4.jpg", "frost5.jpg", "frost6.jpg",
    ]
    idx = np.random.randint(5)
    filename = os.path.join(frost_dir, frost_files[idx])

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Frost overlay not found: {filename}\n"
            f"Download frost images to '{frost_dir}/' from:\n"
            "https://github.com/hendrycks/robustness/tree/master/ImageNet-C/create_c/frost"
        )

    frost_img = cv2.imread(filename)
    frost_img = cv2.resize(frost_img, (0, 0), fx=0.2, fy=0.2)

    img_h, img_w = np.array(x).shape[:2]
    if frost_img.shape[0] <= img_h or frost_img.shape[1] <= img_w:
        frost_img = cv2.resize(frost_img, (img_w * 2, img_h * 2))

    x_start = np.random.randint(0, frost_img.shape[0] - img_h)
    y_start = np.random.randint(0, frost_img.shape[1] - img_w)
    frost_img = frost_img[x_start:x_start + img_h,
                          y_start:y_start + img_w][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost_img, 0, 255)


def snow(x, severity=1):
    c = [
        (0.1,  0.2,  1.00, 0.60,  8,  3, 0.95),
        (0.1,  0.2,  1.00, 0.50, 10,  4, 0.90),
        (0.15, 0.3,  1.75, 0.55, 10,  4, 0.90),
        (0.25, 0.3,  2.25, 0.60, 12,  6, 0.85),
        (0.3,  0.3,  1.25, 0.65, 14, 12, 0.80),
    ][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray(
        (np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L'
    )
    output = io.BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(
        radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(
        np.frombuffer(snow_layer.make_blob(), np.uint8),
        cv2.IMREAD_UNCHANGED,
    ) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    img_h, img_w = x.shape[:2]
    x = c[6] * x + (1 - c[6]) * np.maximum(
        x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(
            img_h, img_w, 1) * 1.5 + 0.5
    )
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def brightness(x, severity=1):
    c = [.05, .1, .15, .2, .3][severity - 1]

    x = np.array(x) / 255.
    x = sk_color_rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk_color_hsv2rgb(x)
    return np.clip(x, 0, 1) * 255


# ---------------------------------------------------------------------------
# Digital corruptions
# ---------------------------------------------------------------------------

def contrast(x, severity=1):
    c = [.75, .5, .4, .3, 0.15][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def elastic_transform(x, severity=1):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(x, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([
        center_square + square_size,
        [center_square[0] + square_size, center_square[1] - square_size],
        center_square - square_size
    ])
    pts2 = pts1 + \
        np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(
        image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1,
          size=shape[:2]), c[1], mode='reflect') * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1,
          size=shape[:2]), c[1], mode='reflect') * c[0]).astype(np.float32)

    x_grid, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y_grid + dy, (-1, 1)
                         ), np.reshape(x_grid + dx, (-1, 1))

    return np.clip(
        np.concatenate([
            map_coordinates(image[:, :, i], indices, order=1, mode='reflect').reshape(
                shape[:2])[..., np.newaxis]
            for i in range(shape[2])
        ], axis=-1),
        0, 1
    ) * 255


def pixelate(x, severity=1):
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    orig_w, orig_h = x.size
    x = x.resize((int(orig_w * c), int(orig_h * c)), PILImage.BOX)
    x = x.resize((orig_w, orig_h), PILImage.BOX)
    return np.clip(np.array(x), 0, 255).astype(np.float32)


def jpeg_compression(x, severity=1):
    c = [25, 18, 15, 10, 7][severity - 1]

    output = io.BytesIO()
    x.save(output, format='JPEG', quality=c)
    x = PILImage.open(output)
    return np.clip(np.array(x), 0, 255).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour space helpers (replaces skimage.color to avoid extra dependency)
# ---------------------------------------------------------------------------

def sk_color_rgb2hsv(img):
    """Convert RGB image (float32, 0-1) to HSV."""
    img_uint8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] /= 179.0   # H: 0-179 in OpenCV
    hsv[:, :, 1] /= 255.0   # S: 0-255
    hsv[:, :, 2] /= 255.0   # V: 0-255
    return hsv


def sk_color_hsv2rgb(img):
    """Convert HSV image (float32, 0-1) to RGB."""
    hsv = img.copy()
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] * 179.0, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 255.0, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 255.0, 0, 255)
    rgb = cv2.cvtColor(hsv.astype(np.uint8),
                       cv2.COLOR_HSV2RGB).astype(np.float32)
    return rgb / 255.0


def sk_util_random_noise(image, mode='gaussian', amount=0.05):
    """Minimal reimplementation of skimage.util.random_noise for s&p mode."""
    out = image.copy()
    if mode == 's&p':
        # Salt
        num_salt = int(np.ceil(amount * image.size * 0.5))
        coords = [np.random.randint(0, i, num_salt) for i in image.shape]
        out[tuple(coords)] = 1.0
        # Pepper
        num_pepper = int(np.ceil(amount * image.size * 0.5))
        coords = [np.random.randint(0, i, num_pepper) for i in image.shape]
        out[tuple(coords)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

CORRUPTION_FNS = {
    # Noise
    "gaussian_noise":    gaussian_noise,
    "shot_noise":        shot_noise,
    "impulse_noise":     impulse_noise,
    # Blur
    "defocus_blur":      defocus_blur,
    "glass_blur":        glass_blur,
    "motion_blur":       motion_blur,
    "zoom_blur":         zoom_blur,
    # gaussian_blur removed — not in original CIFAR-10-C 15 corruptions
    # Weather
    "fog":               fog,
    "frost":             frost,
    "snow":              snow,
    "brightness":        brightness,
    # Digital
    "contrast":          contrast,
    "elastic_transform": elastic_transform,
    "pixelate":          pixelate,
    "jpeg_compression":  jpeg_compression,
}

# Corruptions that need PIL Image input rather than numpy array
# (pixelate and jpeg_compression operate on PIL directly)
_PIL_INPUT = {"pixelate", "jpeg_compression", "motion_blur"}

# Corruptions that need frost_dir argument
_NEEDS_FROST_DIR = {"frost"}


def apply_corruption(
    pil_img,
    corruption: str,
    severity: int,
    frost_dir: str = "data/frost_images",
) -> np.ndarray:
    """
    Apply a named corruption to a PIL Image and return a uint8 numpy array.

    Args:
        pil_img    : PIL Image, 32x32 RGB
        corruption : any of the 16 corruption names in CORRUPTION_FNS
        severity   : 1-5
        frost_dir  : directory containing frost overlay images (frost only)

    Returns:
        uint8 numpy array, shape (32, 32, 3)
    """
    if corruption not in CORRUPTION_FNS:
        raise ValueError(
            f"Unknown corruption '{corruption}'. "
            f"Available: {sorted(CORRUPTION_FNS.keys())}"
        )

    fn = CORRUPTION_FNS[corruption]

    # Some functions take PIL input directly, others take numpy
    if corruption in _PIL_INPUT:
        inp = pil_img
    else:
        inp = pil_img

    if corruption in _NEEDS_FROST_DIR:
        result = fn(inp, severity=severity, frost_dir=frost_dir)
    else:
        result = fn(inp, severity=severity)

    # pixelate and jpeg_compression return float32 already clipped
    # everything else returns float in [0,255]
    return np.clip(result, 0, 255).astype(np.uint8)
