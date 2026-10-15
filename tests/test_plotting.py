import matplotlib

matplotlib.use("Agg")

import numpy as np

from nn101.data import make_digit_one
from nn101.plotting import plot_mlp


def test_make_digit_one_shape_and_range():
    img = make_digit_one()
    assert img.shape == (28, 28)
    assert img.min() >= 0 and img.max() <= 1
    assert (img > 0).any()


def test_plot_mlp_draws_without_activations():
    fig, ax = plot_mlp((784, 16, 16, 10))
    assert fig is ax.figure


def test_plot_mlp_highlights_the_argmax_output():
    sizes = (784, 16, 16, 10)
    acts = [np.zeros(n) for n in sizes]
    acts[-1][7] = 1.0
    _, ax = plot_mlp(sizes, activations=acts, input_image=make_digit_one())
    assert any(t.get_text() == "prediction: 7" for t in ax.texts)
