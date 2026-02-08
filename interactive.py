import forward_sim
import numpy as np
from config import config
from zern import generate_abberations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def f(x, y):
    x = int(x)
    y = int(y)

    r_in = np.zeros((config.N, config.N))
    r_in[x, y] = 1.0

    r_in_prop = np.abs(forward_sim.input_path(r_in))

    r_reflected = np.abs(forward_sim.input_path(r_in) * forward_sim.obj)

    abberations = np.angle(generate_abberations(forward_sim.c_in))

    return (
        r_in,
        r_in_prop,
        r_reflected,
        np.abs(forward_sim.propagate_r_r(r_in)),
        forward_sim.obj,
        abberations,
    )


def f_k(x, y):
    x = int(x)
    y = int(y)

    k_in = np.zeros((config.N, config.N))
    k_in[x, y] = 1.0

    r_incident = forward_sim.input_path_k(k_in)
    r_reflected = r_incident * forward_sim.obj

    abberations = np.angle(forward_sim.generate_abberations(forward_sim.c_in))

    return (
        k_in,
        np.abs(r_incident),
        np.abs(r_reflected),
        np.abs(forward_sim.propagate_k_r(k_in)),
        forward_sim.obj,
        abberations,
    )


def slider_interactive():
    k_space = False
    init_x = 12
    init_y = 31

    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    titles = ["r_in", "r_incident", "r_reflected", "r_out", "obj", "abberations"]

    for ax, title in zip(axes, titles):
        ax.axis("off")
        ax.set_title(title)

    datas = f_k(init_x, init_y) if k_space else f(init_x, init_y)

    imgs = [axes[i].imshow(datas[i]) for i in range(len(axes))]

    for _, (im, ax) in enumerate(zip(imgs, axes)):
        fig.colorbar(im, ax=ax)

    fig.subplots_adjust(left=0.25, bottom=0.25)

    ax_x = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
    x_slider = Slider(
        ax=ax_x,
        label="X",
        valmin=0,
        valmax=config.N - 1,
        valinit=init_x,
        orientation="vertical",
    )
    ax_x.invert_yaxis()

    ax_y = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    y_slider = Slider(
        ax=ax_y,
        label="Y",
        valmin=0,
        valmax=config.N - 1,
        valinit=init_y,
    )

    resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button = Button(resetax, "swap space", hovercolor="0.975")

    def update(val):
        if k_space:
            axes[0].set_title("k_in")
        else:
            axes[0].set_title("r_in")
        new_datas = (
            f_k(x_slider.val, y_slider.val)
            if k_space
            else f(x_slider.val, y_slider.val)
        )
        for im, new_data in zip(imgs, new_datas):
            im.set_data(new_data)
            im.set_clim(vmin=new_data.min(), vmax=new_data.max())
        fig.canvas.draw_idle()

    def swap_space(val):
        nonlocal k_space
        k_space = not k_space
        update(val)

    x_slider.on_changed(update)
    y_slider.on_changed(update)
    button.on_clicked(swap_space)

    plt.show()


if __name__ == "__main__":
    slider_interactive()
