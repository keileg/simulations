import porepy as pp
import numpy as np

import scipy.sparse as sps


class ProjectionOperator:
    def __init__(self, g, cond=None):
        self.g = g
        if cond is None:
            self.cond = lambda g: True
        else:
            self.cond = cond

    def dofs(self, g, dof_type):
        if dof_type == "cell":
            return g.num_cells
        elif dof_type == "face":
            return g.num_faces
        elif dof_type == "node":
            return g.num_nodes
        else:
            raise ValueError(
                "Unknown type of dof: "
                + dof_type
                + ". Should be 'cell', 'face' or 'node'"
            )

    def local_to_global(self, gb, dof_type):
        R = []
        if "mortar_grid" in self.g.name:
            for e, d in gb.edges():
                if self.cond(d["mortar_grid"]):
                    g = d["mortar_grid"]
                    self._append_matrix(R, g, dof_type)
        else:
            for g, d in gb:
                if self.cond(g):
                    self._append_matrix(R, g, dof_type)
        return sps.vstack(R)

    def global_to_local(self, gb, dof_type):
        return self.local_to_global(gb, dof_type).T

    def _append_matrix(self, R, g, dof_type):
        if g == self.g:
            mat = sps.eye(self.dofs(g, dof_type))
        else:
            mat = sps.csc_matrix((self.dofs(g, dof_type), self.dofs(self.g, dof_type)))
        R.append(mat)


def restriction_operator(gb, edges=False, cond=None):
    if cond is None:
        cond = lambda g: True

    if edges == False:
        raise NotImplementedError("Not implemented for graph nodes")
    else:
        matrices = np.empty(gb.num_graph_edges(), dtype=np.object)
        mask = np.ones(len(matrices), dtype=np.bool)
        for _, d in gb.edges():
            if cond(d["mortar_grid"]):
                matrices[d["edge_number"]] = sps.eye(d["mortar_grid"].num_cells)
            else:
                mask[d["edge_number"]] = False
        return sps.block_diag(matrices[mask])


def mixed_dim_projections(gb, cond=None):
    """
    Define the global mixed dimensional projection operators. This is equivalent to the
    local MortarGrid.master_to_mortar and MortarGrid.master_to_slave operators, but
    concatinated to one large operator giving the mapping from all mortar grids to all
    slave/master grids. The lower dimension is considered the slave while the higher
    dimension is considered the master.

    master   slave
     ----    
    | 2d | | |
    |    | | |
     ----
           ^
        mortar

    Arguments:
    gb : (GridBucket)
    cond : Optional. Function that takes a mortar grid as argument and returns True
        if the mortar grid is a mixed dimensional mortar
    
    Returns:
    master2mortar: Projection from faces of master to cells of mortar
    slave2mortar: Projection from cells of slave to cells of mortar
    mortar2master: Projection from cells of mortar to faces of master
    mortar2slave: Projection from cells of mortar to cells of slave
    """
    # Define mortar projections
    if cond is None:
        cond = lambda g: True
    num_mortar_cells = gb.num_mortar_cells(cond)

    master2mortar = sps.csc_matrix((num_mortar_cells, gb.num_faces()))
    slave2mortar = sps.csc_matrix((num_mortar_cells, gb.num_cells()))
    mortar2master = sps.csc_matrix((gb.num_faces(), num_mortar_cells))
    mortar2slave = sps.csc_matrix((gb.num_cells(), num_mortar_cells))
    for e, d in gb.edges():
        gs, gm = gb.nodes_of_edge(e)
        mg = d["mortar_grid"]
        if not cond(mg):
            continue
        Ps = ProjectionOperator(gs)
        Pm = ProjectionOperator(gm)
        Pmort = ProjectionOperator(d["mortar_grid"], cond)

        Pg = Pmort.local_to_global(gb, "cell")
        Pl = Pmort.global_to_local(gb, "cell")
        Pm_l2g = Pm.local_to_global(gb, "face")
        Ps_l2g = Ps.local_to_global(gb, "cell")
        Pm_g2l = Pm.global_to_local(gb, "face")
        Ps_g2l = Ps.global_to_local(gb, "cell")

        mortar2master += Pm_l2g * mg.mortar_to_master_int() * Pl
        mortar2slave += Ps_l2g * mg.mortar_to_slave_int() * Pl
        master2mortar += Pg * mg.master_to_mortar_avg() * Pm_g2l
        slave2mortar += Pg * mg.slave_to_mortar_avg() * Ps_g2l

    return master2mortar, slave2mortar, mortar2master, mortar2slave


def cells2faces_avg(gb):
    """
        Averageing from cells to faces
    """
    avg = []
    for g, d in gb:
        boundary = g.get_all_boundary_faces()
        weight_array = 0.5 * np.ones(g.num_faces)
        weight_array[boundary] = 1
        weights = sps.dia_matrix((weight_array, 0), shape=(g.num_faces, g.num_faces))
        avg.append(weights * np.abs(g.cell_faces))
    return sps.block_diag(avg)


def faces2cells(gb):
    """
    Projection from faces to cells. This is equivalent to the g.cell_faces.T map
    on each grid in the grid bucket, and also pp.fvutils.scalar_divergence(g)
    """
    div = []
    for g, d in gb:
        div.append(g.cell_faces.T)
    return sps.block_diag(div)


def edge_assemble(
    gb, keyword, name, dof_type, side, grid_name="mortar_grid", transpose=False
):

    if transpose:
        matrix = np.empty((gb.num_graph_edges(), gb.num_graph_nodes()), dtype=np.object)
    else:
        matrix = np.empty((gb.num_graph_nodes(), gb.num_graph_edges()), dtype=np.object)

    edge_mask = np.zeros(gb.num_graph_edges(), dtype=np.bool)

    if dof_type == "face":
        for g, d in gb:
            d["dof"] = g.num_faces
        for _, d_e in gb.edges():
            if not d_e.get(grid_name):
                continue
            edge_mask[d_e["edge_number"]] = True
            d_e["dof"] = d_e[grid_name].num_cells

    elif dof_type == "cell":
        for g, d in gb:
            d["dof"] = g.num_cells
        for _, d_e in gb.edges():
            if not d_e.get(grid_name):
                continue
            edge_mask[d_e["edge_number"]] = True
            d_e["dof"] = d_e[grid_name].num_cells
    else:
        raise ValueError("Unknown dof type")

    # Initiate zero mat to ensure correct dimensions
    for e, d in gb.edges():
        col = d["edge_number"]
        if not edge_mask[col]:
            continue
        col_dof = d["dof"]
        break

    for g, d in gb:
        row = d["node_number"]
        if transpose:
            matrix[col, row] = sps.coo_matrix((col_dof, d["dof"]))
        else:
            matrix[row, col] = sps.coo_matrix((d["dof"], col_dof))

    for _, d_e in gb.edges():
        col = d_e["edge_number"]
        if not edge_mask[col]:
            continue
        if transpose:
            matrix[col, row] = sps.coo_matrix((d_e["dof"], d["dof"]))
        else:
            matrix[row, col] = sps.coo_matrix((d["dof"], d_e["dof"]))

    for e, d_e in gb.edges():
        (g_l, g_h) = gb.nodes_of_edge(e)
        ind_c = d_e["edge_number"]
        if not edge_mask[ind_c]:
            continue
        if side == "master":
            d = gb.node_props(g_h)
            ind_r = d["node_number"]
            if transpose:
                matrix[ind_c, ind_r] = d_e[pp.DISCRETIZATION_MATRICES][keyword][name]
            else:
                matrix[ind_r, ind_c] = d_e[pp.DISCRETIZATION_MATRICES][keyword][name]
        if side == "slave":
            d = gb.node_props(g_l)
            ind_r = d["node_number"]
            if transpose:
                matrix[ind_c, ind_r] = d_e[pp.DISCRETIZATION_MATRICES][keyword][name]
            else:
                matrix[ind_r, ind_c] = d_e[pp.DISCRETIZATION_MATRICES][keyword][name]
        if side != "slave" and side != "master":
            raise ValueError("Unkown dimension keyword")

    if transpose:
        matrix = matrix[edge_mask]
    else:
        matrix = matrix[:, edge_mask]

    return sps.bmat(matrix)
