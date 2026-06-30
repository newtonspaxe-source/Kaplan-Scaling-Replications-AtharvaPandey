import numpy as np
import matplotlib.pyplot as plt

def generate_scaling_plots(N_arr, L_N_arr, D_arr, L_D_arr):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].loglog(N_arr, L_N_arr, 'o-b')
    ax[0].set_title('Loss vs Parameters (N)')
    ax[0].set_xlabel('No. OF Non-Embedding Parameters')
    ax[0].set_ylabel('Val. Cross- Entropy Loss')
    ax[0].grid(True, which="both", ls="--", alpha=0.5)
    ax[1].loglog(D_arr, L_D_arr, 'o-g')
    ax[1].set_title('Loss vs Dataset Size (D)')
    ax[1].set_xlabel('No. OF Training Size')
    ax[1].set_ylabel('Val. Cross- Entropy Loss')
    ax[1].grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('scaling_laws.png', dpi=300)
    plt.show()
