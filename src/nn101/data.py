"""Data helpers: synthetic examples now, MNIST loading to follow."""

import numpy as np


def make_digit_one(size=28):
    """A synthetic slanted '1' on a 28x28 grid, values 0 to 1."""
    img = np.zeros((size, size))
    for r in range(4, 24):
        c = int(round(18 - (r - 4) * 0.45))
        img[r, c] = 1.0
        img[r, c - 1] = 0.6
        img[r, c + 1] = 0.6
    return img
