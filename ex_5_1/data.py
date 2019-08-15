import numpy as np
import porepy as pp


class Data(object):
    """ Data class for copuled flow and temperature transport.
    """

    def __init__(self, mesh_args):
        """
        Parameters:
        mesh_args(dictionary): Dictionary containing meshing parameters.
        """
        self.gb = None
        self.domain = None
        self.mesh_args = mesh_args

        self.tol = 1e-3
        self.flow_keyword = "flow"
        self.transport_keyword = "transport"

        self.param = {
            "km": 10 ** -3 * pp.METER ** 2,
            "kf": 10 ** 0 * pp.METER ** 2,
            "kn": 10 ** 0 * pp.METER ** 2,
            "Dm": 10 ** -4 * pp.METER ** 2 / pp.SECOND,
            "Df": 10 ** -4 * pp.METER ** 2 / pp.SECOND,
            "Dn": 10 ** -4 * pp.METER ** 2 / pp.SECOND,
            "aperture": 0.01 * pp.METER,
            "porosity": 0.2,
        }

        self.create_gb()

    # ------------------------------------------------------------------------ #

    def create_gb(self):
        """ Load fractures and create grid bucket
        """
        file_name = self.mesh_args['fracture_file_name']
        fracture_network = pp.fracture_importer.network_3d_from_csv(file_name)
        self.fracture_network = fracture_network
        self.gb = fracture_network.mesh(self.mesh_args)

        g_max = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        (xmin, ymin, zmin) = np.min(g_max.nodes, axis=1)
        (xmax, ymax, zmax) = np.max(g_max.nodes, axis=1)
        self.domain = {
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "zmin": zmin,
            "zmax": zmax,
        }

    def swap_fracture_grids(self, mesh_args):
        """
        Remesh the domain and swap out lower-dimensional meshes with the new meshes.

        Parameters:
        mesh_args (Dictionary) : Mesharguments for new mesh
        """
        gb_fracs = self.fracture_network.mesh(mesh_args, dfn=False)
        g_map = dict()
        for g in self.gb.grids_of_dimension(self.gb.dim_max() - 1):
            for g_f in gb_fracs.grids_of_dimension(g.dim):
                if g.frac_num == g_f.frac_num:
                    g_map[g] = g_f
                    break
        pp.mortars.replace_grids_in_bucket(self.gb, g_map)
        rem_nodes = []
        for g, _ in self.gb:
            if g.dim <= self.gb.dim_max() - 2:
                rem_nodes.append(g)
        for g in rem_nodes:
            self.gb.remove_node(g)

        new_grids = []
        for g, _ in gb_fracs:
            if g.dim <= self.gb.dim_max() - 2:
                new_grids.append(g)
        self.gb.add_nodes(new_grids)

        for e, d in gb_fracs.edges():
            if d["mortar_grid"].dim <= self.gb.dim_max() - 2:
                gl, gh = gb_fracs.nodes_of_edge(e)
                self.gb.add_edge((gl, gh), d["face_cells"])
                new_d = self.gb.edge_props((gl, gh))
                new_d["mortar_grid"] = d["mortar_grid"]

    def viscosity(self, c):
        """ Return the viscosity as a function of temperature
        """
        return pp.ad.exp(1 * c)

    def add_data(self):
        """ Add data to the GridBucket
        """
        self.add_flow_data()
        self.add_transport_data()

    def add_flow_data(self):
        """ Add the flow data to the grid bucket
        """
        keyword = self.flow_keyword
        # Iterate over nodes and assign data
        for g, d in self.gb:
            param = {}
            # Shorthand notation
            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential permeability
            if g.dim == self.gb.dim_max():
                kxx = self.param["km"] * unity
            else:
                kxx = self.param["kf"] * specific_volume * unity

            perm = pp.SecondOrderTensor(kxx)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)

            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]
                # Find faces on right and left side of domain
                north = bound_face_centers[1, :] > self.domain["ymax"] - self.tol
                south = bound_face_centers[1, :] < self.domain["ymin"] + self.tol

                labels = np.array(["neu"] * bound_faces.size)
                # Add dirichlet condition to left and right faces
                labels[south + north] = "dir"
                # Add boundary condition values
                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[north]] = 1

                param["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Loop over edges and set coupling parameters
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            kn = self.param["kn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": kn}
            pp.initialize_data(e, d, keyword, param)

    def add_transport_data(self):
        """ Add the transport data to the grid bucket
        """
        keyword = self.transport_keyword
        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = {}
            d["is_tangential"] = True

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            # Specific volume.
            specific_volume = np.power(
                self.param["aperture"], self.gb.dim_max() - g.dim
            )
            param["specific_volume"] = specific_volume
            # Tangential diffusivity
            if g.dim == 3:
                kxx = self.param["Dm"] * unity
            else:
                kxx = self.param["Df"] * specific_volume * unity
            perm = pp.SecondOrderTensor(kxx)
            param["second_order_tensor"] = perm

            # Source term
            param["source"] = zeros

            # Mass weight
            param["mass_weight"] = specific_volume * self.param["porosity"] * unity

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)
            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                north = bound_face_centers[1, :] > self.domain["ymax"] - self.tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[north] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[north]] = 1

                param["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Normal diffusivity
        for e, d in self.gb.edges():
            # Get higher dimensional grid
            g_h = self.gb.nodes_of_edge(e)[1]
            param_h = self.gb.node_props(g_h, pp.PARAMETERS)
            mg = d["mortar_grid"]
            specific_volume_h = np.ones(mg.num_cells) * param_h[keyword]["specific_volume"]
            dn = self.param["Dn"] * specific_volume_h / (self.param["aperture"] / 2)
            param = {"normal_diffusivity": dn}
            pp.initialize_data(e, d, keyword, param)
