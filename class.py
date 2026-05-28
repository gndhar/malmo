import numpy as np
from config import config
from class_utils import (
    get_dk_mapping,
    apply_T,
    apply_TH,
    power_iteration,
    image_reconstruction,
    class_algorithm,
)
from plot import imshow_phase, imshow_mag

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import forward_sim
    import reflection_matrix
    import obj

    os.makedirs("class_imgs", exist_ok=True)

    N = config.N
    s_in, s_out = forward_sim.simulate_batched()
    R_k = reflection_matrix.generate_R_k(s_in, s_out)
    final_image, ab_in, ab_out = class_algorithm(
        R_k, N, max_iteration_number=10, max_PI_num=6, kfilter=6
    )

    # ==========================================
    # Figure 1: Image Comparison (CLASS vs Truth)
    # ==========================================
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))

    imshow_mag(final_image, axs1[0], title="Reconstructed image")
    imshow_mag(obj.obj, axs1[1], title="Ground-truth object")

    fig1.tight_layout()
    fig1.savefig(os.path.join("class_imgs", "image_comparison.png"), dpi=150)
    plt.show()

    # ==========================================
    # Figure 2: Phase Aberration Comparison
    # ==========================================
    N2 = N // 2
    N2N = N2 + N

    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))

    imshow_phase(ab_in.conj(), axs2[0, 0], title="Input aberration (estimated θ_in)")
    imshow_phase(ab_out.conj(), axs2[0, 1], title="Output aberration (estimated θ_out)")
    imshow_phase(
        forward_sim.input_abberations[N2:N2N, N2:N2N],
        axs2[1, 0],
        title="True input aberration",
    )
    imshow_phase(
        forward_sim.output_abberations[N2:N2N, N2:N2N],
        axs2[1, 1],
        title="True output aberration",
    )

    fig2.tight_layout()
    fig2.savefig(os.path.join("class_imgs", "phase_comparison.png"), dpi=150)
    plt.show()
