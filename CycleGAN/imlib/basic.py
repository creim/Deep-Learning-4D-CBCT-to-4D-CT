import numpy as np
import skimage.io as iio

from imlib import dtype


def imread(path, as_gray=False, **kwargs):
    """Return a float64 image in [-1.0, 1.0]."""
    image = iio.imread(path, as_gray, **kwargs)
    if image.dtype == np.uint8:
        image = image / 127.5 - 1
    elif image.dtype == np.uint16:
        image = image / 32767.5 - 1
    elif image.dtype in [np.float32, np.float64]:
        image = image * 2 - 1.0
    else:
        raise Exception("Inavailable image dtype: %s!" % image.dtype)
    return image


def imwrite(image, path, quality=95, **plugin_args):
    """Save a [-1.0, 1.0] image Why ??""" #Rather safe a 0,1 grayscale img as png
    #Normalizing between 0 and 1
    img_min, img_max = np.amin(image), np.amax(image)
    img = image * (1/np.sqrt(np.square(img_max-img_min)))
    iio.imsave(path, dtype.im2uint(img), **plugin_args)


def imshow(image):
    """Show a [-1.0, 1.0] image."""
    iio.imshow(dtype.im2uint(image))


show = iio.show
