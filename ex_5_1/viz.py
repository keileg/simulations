import porepy as pp
import numpy as np


def split_variables(gb, variables, names):
    dof_start = 0
    for g, d in gb:
        dof_end = dof_start + g.num_cells
        for i, var in enumerate(variables):
            if isinstance(var, pp.ad.Ad_array):
                var = var.val
            if d.get(pp.STATE) is None:
                d[pp.STATE] = dict()
            d[pp.STATE][names[i]] = var[dof_start:dof_end]
        dof_start = dof_end


def store_avg_concentration(gb, t, var_name, out_file):
    area = 0
    concentration = 0
    for g, d in gb:
        if g.dim==(gb.dim_max() - 1):
            concentration += np.sum(d[pp.STATE][var_name] * g.cell_volumes)
            area += np.sum(g.cell_volumes)

    avg_c = concentration / area
    out_file.write("{}, ".format(t) + "{}\n".format(avg_c))
