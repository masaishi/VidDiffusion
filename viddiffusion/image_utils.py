from PIL import Image


def load_image(image_path: str) -> Image:
    """
    Load an image
    :param image_path: path to input image
    :return: image
    """
    return Image.open(image_path).convert('RGB')


def shrink_image(image: Image, size: int) -> Image:
    """
    Shrink an image by keep aspect ratio
    :param image: input image
    :param size: after shrink image size (width * height)
    :return: shrunk image
    """
    width, height = image.size
    i = 0
    while (width * height) > size:
        i += 1
        width = int(width * 0.99)
        height = int(height * 0.99)
    width = width - (width % 64)
    height = height - (height % 64)
    return image.resize((width, height), Image.ANTIALIAS)