""" Utility functions to read or encode images. """

from io import BytesIO
from pathlib import Path
from typing import Tuple

from PIL import Image


def encode_image(image: Image, format: str = "png", **params) -> bytes:
    """Compress (aka encode) an 'image' into 'format'
    Args:
      image: PIL.Image (or numpy array) to be saved
      format: image format to use for saving
      params: optional named parameters passed to PIL. See for reference:
        https://pillow.readthedocs.io/en/3.1.x/handbook/image-file-formats.html?highlight=quality#fully-supported-formats
    Returns:
        encoded_image: bytes containing the encoded/compressed image
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, format=format, **params)
    return buffer.getvalue()


def read_image(file_path: Path) -> bytes:
    """ Read image from disk and return it """
    with file_path.open("rb") as f:
        return f.read()


def read_and_resize_image(
    file_path: Path, target_size: Tuple[int, int]
) -> Tuple[bytes, Tuple[int, int]]:
    """Read the from disk and resize it to target size.

    Parameters
    ----------
    file_path:
      path to the image file
    target_size:
      tuple of (height, width)

    Returns
    -------
    image:
      resized image encoded in the original image format
    original_size:
      the image size before resizing
    """
    img = Image.open(file_path)
    owidth, oheight = img.size

    theight, twidth = target_size
    img = img.resize((twidth, theight))

    return img, (oheight, owidth)
