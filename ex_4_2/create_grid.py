"""
Create the grid (bucket) used in the simulations.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import porepy as pp

# Function declaration
def make_grid(mesh_size=2.0, L=[100.0, 10.0]):
    """
    Create an unstructured triangular mesh using Gmsh.

    Parameters:
        mesh_size (scalar): (approximated) size of triangular elements [-]
        L (array): length of the domain for each dimension [m]

    Returns:
        gb (PorePy object): PorePy grid bucket object containing all the grids
                            In this case we only have one grid.
    """

    domain = {"xmin": 0.0, "xmax": L[0], "ymin": 0.0, "ymax": L[1]}
    network_2d = pp.FractureNetwork2d(None, None, domain)
    target_h_bound = target_h_fracture = target_h_min = mesh_size

    mesh_args = {
        "mesh_size_bound": target_h_bound,
        "mesh_size_frac": target_h_fracture,
        "mesh_size_min": target_h_min,
    }

    gb = network_2d.mesh(mesh_args)

    return gb
