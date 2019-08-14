import porepy as pp


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
