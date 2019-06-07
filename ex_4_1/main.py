import numpy as np
import porepy as pp

from data import add_data
from create_grid import create_grid
from solvers import run_flow

# ------------------------------------------------------------------------------#

def homo_rt0(g):
    return {"scheme": pp.RT0("flow"), "dof": {"cells": 1, "faces": 1}, "label": "homo_rt0"}

# ------------------------------------------------------------------------------#

def homo_tpfa(g):
    return {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "homo_tpfa"}

# ------------------------------------------------------------------------------#

def homo_mpfa(g):
    return {"scheme": pp.Mpfa("flow"), "dof": {"cells": 1}, "label": "homo_mpfa"}

# ------------------------------------------------------------------------------#

def homo_mvem(g):
    return {"scheme": pp.MVEM("flow"), "dof": {"cells": 1, "faces": 1}, "label": "homo_mvem"}

# ------------------------------------------------------------------------------#

def hete1(g):
    if g.dim == 2:
        scheme = {"scheme": pp.RT0("flow"), "dof": {"cells": 1, "faces": 1}, "label": "hete1"}
    else:
        scheme = {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "hete1"}
    return scheme

# ------------------------------------------------------------------------------#

def hete2(g):
    if g.dim == 2:
        scheme = {"scheme": pp.MVEM("flow"), "dof": {"cells": 1, "faces": 1}, "label": "hete2"}
    else:
        scheme = {"scheme": pp.Tpfa("flow"), "dof": {"cells": 1}, "label": "hete2"}
    return scheme

# ------------------------------------------------------------------------------#

def homo_mortar(g):
    return {"scheme": pp.RT0("flow"), "dof": {"cells": 1, "faces": 1}, "label": "homo_mortar"}

# ------------------------------------------------------------------------------#

def main(mesh_size, discr, flow_dir, is_coarse, refine_1d, folder):

    # set the geometrical tolerance
    tol = 1e-6
    # create the gb
    gb, partition = create_grid(mesh_size, is_coarse, refine_1d, tol)
    # set the scheme for each grid
    for g, d in gb:
        d["discr"] = discr(g)
    # add the problem data
    add_data(gb, flow_dir, tol)
    # solve the darcy problem
    run_flow(gb, partition, folder)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    # the second parameter is requested in case of coasened grid, only when MVEM is applied to the 2d
    # the third parameter is related to the mortar
    solver_list = {"tpfa":   (homo_tpfa,   False, False),
                   "mpfa":   (homo_mpfa,   False, False),
                   "mvem":   (homo_mvem,   True,  False),
                   "rt0":    (homo_rt0,    False, False),
                   "hete1":  (hete1,       False, False),
                   "hete2":  (hete2,       True,  False),
                   "mortar": (homo_mortar, False, True)}

    flow_dirs = ["top_to_bottom", "left_to_right"]
    mesh_sizes = [0.0125, 0.025, 0.06]

    for solver_name, (solver, is_coarse, refine_1d) in solver_list.items():
        for flow_dir in flow_dirs:
            for mesh_size in mesh_sizes:
                folder = solver_name + "_" + flow_dir + "_" + str(mesh_size)
                main(mesh_size, solver, flow_dir, is_coarse, refine_1d, folder)
