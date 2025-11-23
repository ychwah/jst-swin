import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

NUM_COLORS = 20
cm = plt.get_cmap('tab20')


def plot(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    regularization = df["regularization"].to_numpy()
    residual_error = df["residual_error"].to_numpy()
    ref_regularization = df["ref_regul"].to_numpy()
    psnr = df["psnr"].to_numpy()
    psnr_2 = df["psnr_2"].to_numpy()
    psnr_max = np.max(psnr)
    psnr_max_2 = np.nanmax(psnr_2)
    error_min = np.min(residual_error)
    iterations_ = np.arange(len(regularization))

    fig, ax = plt.subplots(3)

    # Regularization
    ax[0].plot(iterations_, regularization, label="R(u,v)")
    ax[0].plot(iterations_, ref_regularization, label="R(u0, v0)")
    ax[0].legend(prop={'size': 6}, loc="upper right")

    # error
    ax[1].plot(iterations_, residual_error, "r-", label=f"Residual error, min={error_min:2.4f}")
    ax[1].legend(prop={'size': 6}, loc="upper right")
    ax[1].set_yscale('log')
    ax[2].plot(iterations_, psnr, "g-", label=f"psnr, max={psnr_max:2.4f}")
    ax[2].plot(iterations_, psnr_2, "b-", label=f"psnr_z, max={psnr_max_2:2.4f}")
    ax[2].legend(prop={'size': 6}, loc="lower right")
    fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    plt.close()


def plot_2(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    error = df["relative_error"].to_numpy()
    psnr = df["psnr"].to_numpy()
    psnr_2 = df["psnr_2"].to_numpy()
    psnr_max = np.max(psnr)
    psnr_max_2 = np.nanmax(psnr_2)
    error_min = np.min(error)
    iterations_ = np.arange(len(error))

    fig, ax = plt.subplots(2)

    # error
    ax[0].plot(iterations_, error, "r-", label=f"Relative error, min={error_min:2.4f}")
    ax[0].legend(prop={'size': 6}, loc="upper right")
    ax[0].set_yscale('log')
    ax[1].plot(iterations_, psnr, "g-", label=f"psnr, max={psnr_max:2.4f}")
    ax[1].plot(iterations_, psnr_2, "b-", label=f"psnr_z, max={psnr_max_2:2.4f}")
    ax[1].legend(prop={'size': 6}, loc="lower right")
    fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    plt.close()


def plot3(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    regularization = df["regularization"].to_numpy()
    ref_regularization = df["ref_regul"].to_numpy()
    psnr = df["psnr"].to_numpy()
    psnr_max = np.max(psnr)
    iterations_ = np.arange(len(regularization))

    fig, ax = plt.subplots(2)

    # Regularization
    ax[0].plot(iterations_, regularization, label="R(u,v)")
    ax[0].plot(iterations_, ref_regularization, label="R(u0, v0)")
    ax[0].legend(prop={'size': 6}, loc="upper right")

    # error
    ax[1].plot(iterations_, psnr, "g-", label=f"psnr, max={psnr_max:2.4f}")
    ax[1].legend(prop={'size': 6}, loc="lower right")
    fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    plt.close()


def plot4(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    regularization = df["regularization"].to_numpy()
    residual_error = df["residual_error"].to_numpy()
    ref_regularization = df["ref_regul"].to_numpy()
    psnr = df["psnr"].to_numpy()
    psnr_2 = df["psnr_2"].to_numpy()
    psnr_max = np.max(psnr)
    psnr_max_2 = np.nanmax(psnr_2)
    error_min = np.min(residual_error)
    iterations_ = np.arange(len(regularization))
    structure_psnr = df["s_psnr"].to_numpy()
    texture_psnr = df["t_psnr"].to_numpy()

    fig, ax = plt.subplots(3, 2)
    x1, y1 = ax[0, 1]._position.__array__()
    x1[1] -= 0.10
    y1[1] -= 0.10
    x1, y1 = ax[1, 1]._position.__array__()
    x1[1] -= 0.2
    y1[1] -= 0.2
    ax[2, 1].set_visible(False)

    # Regularization
    ax[0, 0].plot(iterations_, regularization, label="R(u,v)")
    ax[0, 0].plot(iterations_, ref_regularization, label="R(u0, v0)")
    ax[0, 0].legend(prop={'size': 6}, loc="upper right")

    # error
    ax[1, 0].plot(iterations_, residual_error, "r-", label=f"Residual error, min={error_min:2.4f}")
    ax[1, 0].legend(prop={'size': 6}, loc="upper right")
    ax[1, 0].set_yscale('log')
    ax[2, 0].plot(iterations_, psnr, "g-", label=f"psnr, max={psnr_max:2.4f}")
    ax[2, 0].plot(iterations_, psnr_2, "b-", label=f"psnr_z, max={psnr_max_2:2.4f}")
    ax[2, 0].legend(prop={'size': 6}, loc="lower right")

    # cartoon
    ax[0, 1].plot(iterations_, structure_psnr, "c-", label=f"structure psnr, max={np.nanmax(structure_psnr):2.4f}")
    ax[0, 1].legend(prop={'size': 6}, loc="lower right")
    ax[0, 1].set_title("structure")
    ax[1, 1].plot(iterations_, texture_psnr, "k-", label=f"texture psnr, max={np.nanmax(texture_psnr):2.4f}")
    ax[1, 1].legend(prop={'size': 6}, loc="lower right")
    ax[1, 1].set_title("texture")

    fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    plt.close()


def plot_param(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    mu = df["mu"].to_numpy()
    proj_t = df["proj_t"].to_numpy()

    fig, ax = plt.subplots()

    ax.scatter(proj_t, mu)
    for i in range(len(mu)):
        ax.annotate(f"{i}", (proj_t[i], mu[i]))
    ax.set_xlabel("proj_t")
    ax.set_ylabel("$\\mu$")
    ax.grid()
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 3.0)
    fig.savefig(f"{output_folder}/param_plot_{output_id}.pdf", bbox_inches="tight")


def plot_fista(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    regularization = df["regularization"].to_numpy()
    regularization_line_search = df["regularization_line_search"].to_numpy()
    regularization_fista = df["regularization_fista"].to_numpy()
    ref_regularization = df["ref_regul"].to_numpy()

    residual_error = df["residual_error"].to_numpy()
    residual_error_line_search = df["residual_error_line_search"].to_numpy()
    residual_error_fista = df["residual_error_fista"].to_numpy()

    psnr = df["psnr"].to_numpy()
    psnr_line_search = df["psnr_line_search"].to_numpy()
    psnr_fista = df["psnr_fista"].to_numpy()

    iterations_ = np.arange(len(regularization))

    fig, ax = plt.subplots(4)

    # Regularization
    ax[0].plot(iterations_, regularization, label="R(u,v)_line_search")
    ax[0].plot(iterations_, regularization_line_search, label="R(u,v)_line_search")
    ax[0].plot(iterations_, regularization_fista, label="R(u,v)_fista")
    ax[0].plot(iterations_, ref_regularization, label="R($u_0, v_0$)")
    ax[0].legend(prop={'size': 6}, loc="upper right")
    ax[0].set_ylim(0, 10.0)

    # error
    ax[1].plot(iterations_, psnr, label=f"psnr, max={np.max(psnr):2.4f}")
    ax[1].plot(iterations_, psnr_line_search, label=f"psnr_line_search, max={np.max(psnr_line_search):2.4f}")
    ax[1].plot(iterations_, psnr_fista, label=f"psnr_fista, max={np.max(psnr_fista):2.4f}")
    ax[1].legend(prop={'size': 6}, loc="lower right")

    # residual_error
    ax[2].plot(iterations_, residual_error, label=f"dist($z$,$C_f$) , min={np.min(residual_error):2.4f}")
    ax[2].plot(iterations_, residual_error_line_search, label="dist($z_{line}$,$C_f$), "+f"min={np.min(residual_error_line_search):2.4f}")
    ax[2].plot(iterations_, residual_error_fista, label="dist($z_{fista}$,$C_f$), "+f"min={np.min(residual_error_fista):2.4f}")
    ax[2].legend(prop={'size': 6}, loc="upper right")
    ax[2].set_yscale('log')

    # fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    # plt.close()
    #
    # fig, ax = plt.subplots(1)

    lipschitz_constant = df["lipschitz_constant"].to_numpy()
    lipschitz_constant_line_search = df["lipschitz_constant_line_search"].to_numpy()
    lipschitz_constant_fista = df["lipschitz_constant_fista"].to_numpy()

    ax[3].plot(iterations_, lipschitz_constant, label="Lipschitz constant")
    ax[3].plot(iterations_, lipschitz_constant_line_search, label="Lipschitz constant: line search")
    ax[3].plot(iterations_, lipschitz_constant_fista, label="Lipschitz constant: FISTA")
    ax[3].legend(prop={'size': 6}, loc="upper right")
    ax[3].set_ylim(0, 2.0)

    fig.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')

    fig, ax = plt.subplots(1)
    ax.plot(iterations_, lipschitz_constant, label="Lipschitz constant")
    ax.plot(iterations_, lipschitz_constant_line_search, label="Lipschitz constant: line search")
    ax.plot(iterations_, lipschitz_constant_fista, label="Lipschitz constant: FISTA")
    ax.legend(prop={'size': 6}, loc="upper right")
    ax.set_ylim(0, 2.0)
    fig.savefig(f"{output_folder}/lipschitz_cst_{output_id}.pdf", bbox_inches='tight')
    plt.close()


def plot_paper(csv_path, output_folder, output_id):
    df = pd.read_csv(csv_path, index_col=0)
    regularization = df["regularization"].to_numpy()
    residual_error = df["residual_error"].to_numpy()
    ref_regularization = df["ref_regul"].to_numpy()
    psnr = df["psnr"].to_numpy()
    psnr_2 = df["psnr_2"].to_numpy()
    psnr_max = np.max(psnr)
    psnr_max_2 = np.nanmax(psnr_2)
    error_min = np.min(residual_error)
    iterations_ = np.arange(len(regularization))
    structure_psnr = df["s_psnr"].to_numpy()
    texture_psnr = df["t_psnr"].to_numpy()

    # fig, ax = plt.subplots(3, 2)
    # x1, y1 = ax[0, 1]._position.__array__()
    # x1[1] -= 0.10
    # y1[1] -= 0.10
    # x1, y1 = ax[1, 1]._position.__array__()
    # x1[1] -= 0.2
    # y1[1] -= 0.2
    # ax[2, 1].set_visible(False)
    # fig, ax = plt.subplots(1, 5)
    plt.rcParams.update({'font.size': 20})

    # Regularization
    plt.plot(iterations_, regularization, label="$R(u,v)$")
    plt.plot(iterations_, ref_regularization, label="$R(u_{0}, v_{0})$")
    plt.ylim(0, 10)
    plt.xlim(0, 30)
    plt.legend(prop={'size': 18}, loc="upper right")
    plt.savefig(f"{output_folder}/plots_{output_id}_{0}.pdf", bbox_inches='tight')
    plt.close()

    # error
    plt.plot(iterations_, residual_error, "r-", label=f"Residual error")
    plt.legend(prop={'size': 18}, loc="upper right")
    plt.yscale('log')
    plt.xlim(0, 30)
    plt.savefig(f"{output_folder}/plots_{output_id}_{1}.pdf", bbox_inches='tight')
    plt.close()

    # psnr
    plt.plot(iterations_, psnr, "g-", label=f"PSNR($x_n$, $x_{0}$)\n max: {psnr_max:2.4f}")
    plt.plot(iterations_, psnr_2, "b-", label=f"PSNR($y_n$, $x_{0}$)\n max: {psnr_max_2:2.4f}")
    plt.ylim(0, 45)
    plt.xlim(0, 30)
    plt.legend(prop={'size': 18}, loc="lower right")
    plt.savefig(f"{output_folder}/plots_{output_id}_{2}.pdf", bbox_inches='tight')
    plt.close()

    # cartoon
    plt.plot(iterations_, structure_psnr, "c-", label=f"PSNR($u_n$, $u_{0}$)\n max: {np.nanmax(structure_psnr):2.4f}")
    plt.legend(prop={'size': 18}, loc="lower right")
    plt.ylim(0, 45)
    plt.xlim(0, 30)
    plt.title("structure")
    plt.savefig(f"{output_folder}/plots_{output_id}_{3}.pdf", bbox_inches='tight')
    plt.close()

    plt.plot(iterations_, texture_psnr, "k-", label=f"PSNR($v_n$, $v_{0}$)\n max: {np.nanmax(texture_psnr):2.4f}")
    plt.legend(prop={'size': 18}, loc="lower right")
    plt.ylim(0, 45)
    plt.xlim(0, 30)
    plt.title("texture")
    plt.savefig(f"{output_folder}/plots_{output_id}_{4}.pdf", bbox_inches='tight')
    plt.close()

    # plt.savefig(f"{output_folder}/plots_{output_id}.pdf", bbox_inches='tight')
    # plt.close()


if __name__ == "__main__":

    for idx in [47, 55, 490]:
        csv_file = f"../plots/data_{idx}.csv"

        plot_paper(csv_file, "/home/aguennecjacq/Documents/these/article_pnp/plots/", idx)
