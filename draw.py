import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations
from matplotlib import cm
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from numpy import ndarray, sin, cos, mgrid, abs, sum, array
from os.path import join
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'

def plot_cube(ax):
    # draw cube
    r = [-1, 1]
    for s, e in combinations(array(list(product(r, r, r))), 2):
        if sum(abs(s-e)) == r[1]-r[0]:
            ax.plot3D(*zip(s, e), color="k", linestyle="--", alpha=0.5)
    ax.set_xlim(r)
    ax.set_ylim(r)
    ax.set_zlim(r)
    ax.set_xlabel( r"$x^1$" )
    ax.set_ylabel( r"$x^2$" )
    ax.set_zlabel( r"$x^3$" )

def plot_sphere(ax):
    # draw sphere
    u, v = mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = cos(u)*sin(v)
    y = sin(u)*sin(v)
    z = cos(v)
    # ax.plot_wireframe(x, y, z, color="r")
    ax.plot_surface(x, y, z, color='m', alpha=0.3, linewidth=2, antialiased=False, rstride=1, cstride=1)
    # ax.plot_trisurf(x.flatten(), y.flatten(), z.flatten(), color='m', alpha=0.3)# cmap=cm.coolwarm)
    # draw a point
    ax.scatter([0], [0], [0], color="r", s=20)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

def plot_arrow(ax, vec, anno=r"$z_1$"):
    x0, y0, z0 = vec[:3]
    a = Arrow3D([0, x0], [0, y0], [0, z0], mutation_scale=20,
                lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)
    ax.text(x0, y0, z0, anno, size=20, zorder=1,  color='k') 

def vis_3d(vecs_embed: ndarray, dir_embed="embed"):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_box_aspect(aspect=(1, 1, 1))
    plot_cube(ax)
    plot_sphere(ax)
    plot_arrow(ax, vecs_embed[0], anno=r"$z_1$")
    plt.savefig(join(dir_embed, "embed1.png"), bbox_inches="tight")
    plot_arrow(ax, vecs_embed[1], anno=r"$z_2$")
    plt.savefig(join(dir_embed, "embed2.png"), bbox_inches="tight")
    plot_arrow(ax, vecs_embed[2], anno=r"$z_3$")
    plt.savefig(join(dir_embed, "embed3.png"), bbox_inches="tight")

if __name__ == "__main__":
    from embed import create_similar, match_category, embed_vec
    from numpy import genfromtxt
    path_hierarchy = join("hierarchy", "simple_case.csv") # "final_match.csv"
    similar, list_category_hierarchy = create_similar(path_hierarchy)
    dict_category = dict(genfromtxt(join("label", "category.txt"), delimiter=" ", dtype=None, encoding=None))
    list_category = list(dict_category.keys())
    category_missing, category_redundant, map_category = match_category(list_category, list_category_hierarchy)
    dict_vec_hierarchy = embed_vec(similar, list_category_hierarchy, category_redundant)