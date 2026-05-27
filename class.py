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

if __name__ == "__main__":
    import forward_sim
    import reflection_matrix
    import obj

    N = config.N
    s_in, s_out = forward_sim.simulate_batched()
    R_k = reflection_matrix.generate_R_k(s_in, s_out)
    # R_k = reflection_matrix.RM_fft(R)
    final_image, ab_in, ab_out = class_algorithm(
        R_k, N, max_iteration_number=10, max_PI_num=6, kfilter=6
    )

    print(final_image)

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the target directory exists
    os.makedirs("class_imgs", exist_ok=True)

    # ==========================================
    # Figure 1: Image Comparison (CLASS vs Truth)
    # ==========================================
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))

    axs1[0].imshow(np.abs(final_image))
    axs1[0].set_title("Reconstructed image")

    axs1[1].imshow(np.abs(obj.obj))
    axs1[1].set_title("Ground-truth object")

    fig1.tight_layout()
    fig1.savefig(os.path.join("class_imgs", "image_comparison.png"), dpi=150)
    plt.show()

    N2 = N // 2
    N2N = N2 + N

    # ==========================================
    # Figure 2: Phase Aberration Comparison
    # ==========================================
    fig2, axs2 = plt.subplots(2, 2, figsize=(10, 10))

    # Top Row: Estimated
    axs2[0, 0].imshow(-np.angle(ab_in), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs2[0, 0].set_title("Input aberration (estimated θ_in)")

    axs2[0, 1].imshow(-np.angle(ab_out), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axs2[0, 1].set_title("Output aberration (estimated θ_out)")

    # Bottom Row: Ground Truth
    axs2[1, 0].imshow(
        np.angle(forward_sim.input_abberations[N2:N2N, N2:N2N]),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs2[1, 0].set_title("True input aberration")

    axs2[1, 1].imshow(
        np.angle(forward_sim.output_abberations[N2:N2N, N2:N2N]),
        cmap="twilight",
        vmin=-np.pi,
        vmax=np.pi,
    )
    axs2[1, 1].set_title("True output aberration")

    fig2.tight_layout()
    fig2.savefig(os.path.join("class_imgs", "phase_comparison.png"), dpi=150)
    plt.show()
