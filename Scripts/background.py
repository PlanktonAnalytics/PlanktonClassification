#!/usr/bin/env python3

# Good enough background subtraction?

import cv2
import numpy as np
import tifffile as tiff
from io import BytesIO
import logging


def background_correction(image_bytes):

    # Read the TIFF image

    stream = BytesIO(image_bytes)
    image = tiff.imread(stream)
    tiff.imwrite("original.tiff", image)

    height, width, channels = image.shape
    print(f"Image shape {height}, {width}, {channels}")

    # The background line occurs after the main tiff file as an
    # additional line. We assume it's `width` wide and consists of BGR
    # triplets

    ptr = stream.tell()
    print(f"Pointer {ptr}")

    bg = image_bytes[ptr:ptr + width * 3]
    bg = np.frombuffer(bg, dtype=np.uint8)
    bg = 255 - bg # invert
    bg = bg.reshape(1,width,3)

    # Make a full sized background image by stacking rows
    bg = np.vstack([bg] * height)

    tiff.imwrite("background.tiff", bg)

    # Subtract the background from the image
    d = (image.astype(int) - bg.astype(int)).clip(0, 255).astype(np.uint8)

    tiff.imwrite("corrected.tiff", d)


def main():
    # Load a PI10 TIFF file example

    f = open('../data/examples/pia1.2023-05-10.1100+N00005127.tif', 'rb')
    ba = bytearray(f.read())
    background_correction(ba)


if __name__ == "__main__":
    main()
