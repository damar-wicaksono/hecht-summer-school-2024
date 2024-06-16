import numpy as np
import matplotlib.pyplot as plt
import minterpy as mp

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_grid(grid: mp.Grid):
    fig = plt.figure(facecolor="white")
    if grid.spatial_dimension == 1:
        ax = plt.axes(frameon=False)
        ax.get_xaxis().tick_bottom()
        ax.axes.get_yaxis().set_visible(False)
        ax.scatter(grid.unisolvent_nodes.reshape(-1), np.zeros(len(grid.multi_index)), marker="x", color="black")
        ax.set_ylim([-0.01, 0.01])
        ax.hlines(0, -1, 1, linewidth=0.5, linestyle="--", color="black")
        ax.set_title("Interpolation grid");
        plt.show()
    elif grid.spatial_dimension == 2:
        ax = plt.gca()
        ax.scatter(grid.unisolvent_nodes[:, 0], grid.unisolvent_nodes[:, 1], marker='x', color="black")
        ax.set_xlabel("$x_1$");
        ax.set_ylabel("$x_2$");
        ax.set_title("Interpolation grid");
        plt.show()
    else:
        raise ValueError("Cannot plot grid of dimension higher than 2")
        
def plot_poly(poly: mp.NewtonPolynomial, ori_fun = None):

    if poly.spatial_dimension == 1:
        fig = plt.figure(facecolor="white")
        ax = plt.gca()
        xx_test = np.linspace(-1, 1, 1000)
        ax.plot(xx_test, poly(xx_test), label="interpolation")
        unisolvent_nodes = poly.grid.unisolvent_nodes.reshape(-1)
        lag_coeffs = poly(unisolvent_nodes)
        ax.scatter(unisolvent_nodes, lag_coeffs, marker="x", label="interpolation nodes")
        if ori_fun is not None:
            plt.plot(xx_test, ori_fun(xx_test), label="original")
        plt.legend(loc="upper right");
    elif poly.spatial_dimension == 2:
        # --- Create 2D data
        xx_1d = np.linspace(-1, 1, 1000)[:, np.newaxis]
        mesh_2d = np.meshgrid(xx_1d, xx_1d)
        xx_2d = np.array(mesh_2d).T.reshape(-1, 2)
        yy_2d = poly(xx_2d)

        # --- Create a series of plots
        fig = plt.figure(figsize=(15, 5))


        # Surface
        axs_2 = plt.subplot(132, projection='3d')
        axs_2.plot_surface(
            mesh_2d[0],
            mesh_2d[1],
            yy_2d.reshape(1000, 1000).T,
            cmap="plasma",
            linewidth=0,
            antialiased=False,
            alpha=0.5
        )
        axs_2.set_xlabel("$x_1$", fontsize=14)
        axs_2.set_ylabel("$x_2$", fontsize=14)
        axs_2.set_zlabel("$\mathcal{M}(x_1, x_2)$", fontsize=14)
        axs_2.set_title("Surface plot", fontsize=14)

        # Contour
        axs_3 = plt.subplot(133)
        cf = axs_3.contourf(
            mesh_2d[0], mesh_2d[1], yy_2d.reshape(1000, 1000).T, cmap="plasma"
        )
        axs_3.set_xlabel("$x_1$", fontsize=14)
        axs_3.set_ylabel("$x_2$", fontsize=14)
        axs_3.set_title("Contour plot", fontsize=14)
        divider = make_axes_locatable(axs_3)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(cf, cax=cax, orientation='vertical')
        axs_3.axis('scaled')

        fig.tight_layout(pad=3.0)
        plt.gcf().set_dpi(150);

    else:
        raise ValueError("Cannot plot polynomial more than dimension 2")

        
def plot_multi_index(mi: mp.MultiIndexSet):
    """Plot the exponents of 2D multi-index set and the corresponding grid."""
    if mi.spatial_dimension != 2:
        raise ValueError("Can only plot multi-index set of dimension 2")
        
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Exponents
    axs[0].scatter(mi.exponents[:, 0], mi.exponents[:, 1], marker='x', color="black")
    axs[0].set_xlabel("m = 1 (poly. degree)");
    axs[0].set_ylabel("m = 2 (poly. degree)");
    axs[0].set_xticks(np.arange(0, 3 + 1));
    axs[0].set_yticks(np.arange(0, 3 + 1));
    axs[0].set_title("Multi-index set exponents");

    # Grid points
    grd = mp.Grid(mi)
    axs[1].scatter(grd.unisolvent_nodes[:, 0], grd.unisolvent_nodes[:, 1], marker='x', color="black")
    axs[1].set_xlabel("$x_1$");
    axs[1].set_ylabel("$x_2$");
    axs[1].set_title("Interpolation grid");