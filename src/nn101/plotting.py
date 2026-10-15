"""Plotting helpers for visualising neural networks."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_mlp(layer_sizes=(784, 16, 16, 10), activations=None, input_image=None,
             input_shown=20, output_labels=None, title=None, ax=None):
    """Draw a multilayer perceptron. The first layer is abbreviated to `input_shown` neurons
    with an ellipsis. Neuron fill shows activation (0 = light, 1 = dark)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6.5))
    else:
        fig = ax.figure
    cmap = plt.get_cmap("Blues")
    edge_col = "#1F3A5F"
    line_col = "#9AA5B1"
    n_layers = len(layer_sizes)
    xs = np.linspace(0, 1, n_layers)

    # y positions per layer; the input layer shows the top and bottom halves with a gap
    ys, shown_idx = [], []
    for li, n in enumerate(layer_sizes):
        if li == 0 and n > input_shown:
            half = input_shown // 2
            top = np.linspace(1, 0.12, half)
            bot = np.linspace(-0.12, -1, half)
            ys.append(np.concatenate([top, bot]))
            shown_idx.append(list(range(half)) + list(range(n - half, n)))
        else:
            span = min(1.0, 0.055 * n)
            ys.append(np.linspace(span, -span, n))
            shown_idx.append(list(range(n)))

    # edges
    for li in range(n_layers - 1):
        for y0 in ys[li]:
            for y1 in ys[li + 1]:
                ax.plot([xs[li], xs[li + 1]], [y0, y1], color=line_col, lw=0.35, alpha=0.35, zorder=1)

    # neurons
    for li, n in enumerate(layer_sizes):
        if activations is not None:
            vals = np.asarray(activations[li])[shown_idx[li]]
        else:
            vals = np.zeros(len(shown_idx[li]))
        cols = cmap(0.08 + 0.82 * np.clip(vals, 0, 1))
        size = 90 if li == 0 else 260
        ax.scatter(np.full(len(ys[li]), xs[li]), ys[li], s=size, c=cols,
                   edgecolors=edge_col, linewidths=0.9, zorder=3)

    # input ellipsis and count
    if layer_sizes[0] > input_shown:
        ax.text(xs[0], 0, "⋮", ha="center", va="center", fontsize=16, color=edge_col)
        ax.annotate(f"{layer_sizes[0]} inputs", xy=(xs[0], -1.08), ha="center", va="top",
                    fontsize=10, color="#3B4652")
    for li in range(1, n_layers - 1):
        ax.text(xs[li], -1.08, f"{layer_sizes[li]} neurons", ha="center", va="top",
                fontsize=10, color="#3B4652")
    ax.text(xs[-1], -1.08, f"{layer_sizes[-1]} outputs", ha="center", va="top",
            fontsize=10, color="#3B4652")

    # output labels and the winner
    if output_labels is None:
        output_labels = [str(i) for i in range(layer_sizes[-1])]
    for lab, y in zip(output_labels, ys[-1]):
        ax.text(xs[-1] + 0.045, y, lab, va="center", ha="left", fontsize=11, color="#1F2933")
    if activations is not None:
        k = int(np.argmax(activations[-1]))
        y = ys[-1][k]
        ax.add_patch(Rectangle((xs[-1] - 0.035, y - 0.05), 0.11, 0.1, fill=False,
                               edgecolor="#C77C02", lw=1.6, zorder=4))
        ax.text(xs[-1] + 0.12, y, f"prediction: {output_labels[k]}", va="center",
                fontsize=10, color="#C77C02")

    # input image inset
    if input_image is not None:
        inset = ax.inset_axes([-0.24, 0.56, 0.2, 0.36])
        inset.imshow(input_image, cmap="gray_r", vmin=0, vmax=1)
        inset.set_xticks([]); inset.set_yticks([])
        for s in inset.spines.values():
            s.set_edgecolor(edge_col)
        inset.set_title(f"{input_image.shape[0]}x{input_image.shape[1]} pixels", fontsize=9, color="#3B4652")

    ax.set_xlim(-0.26 if input_image is not None else -0.1, 1.32)
    ax.set_ylim(-1.2, 1.15)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=13, color="#1F2933", pad=12)
    return fig, ax
