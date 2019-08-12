import numpy as np
import porepy as pp

class Data(object):
    """ Data class for copuled flow and temperature transport.
    """
    def __init__(self,num_fracs, param):
        """
        Parameters:
        num_fracs (int): Number of fractures to include in the model
        param (dictionary): Dictionary containing paramters.
        """
        self.gb = None
        self.domain = None
        self.param = param

        self.tol = 1e-3
        self.flow_keyword = "flow"
        self.transport_keyword = "transport"

        self.create_gb(num_fracs)

    # ------------------------------------------------------------------------ #

    def create_gb(self, num_fracs):
        """ Load the pickled grid bucket
        """
        np.random.seed(1)

        fracs = []
        Lx = 2
        Ly = 2
        Lz = 1
        domain = {'xmin': 0, 'xmax': Lx,
                  'ymin': 0, 'ymax': Ly,
                  'zmin': 0, 'zmax': Lz,
        }
        cc_offset = np.linspace(0, Ly, num_fracs)
        for i in range(num_fracs):
            cc = np.random.rand(3) * np.array([Lx, Ly, Lz])
            cc[1] = cc_offset[i]
            R1 = 0.5 + 0.5 * np.random.randn(1)
            R2 = 0.5 + 0.5 * np.random.randn(1)
            R1 = min(R1, 1) + max(0.25, R1)
            R2 = min(R2, 1) + max(0.25, R2)
            angle = np.random.rand(1) * np.pi
            strike = np.random.rand(1) * np.pi
            dip = np.random.rand(1) * np.pi
            fracs.append(pp.EllipticFracture(cc, R1, R2, angle, strike, dip))

        fracture_network = pp.FractureNetwork3d(fracs, domain)
        self.fracture_network = fracture_network
        self.gb = fracture_network.mesh(self.param['mesh_args'])

        g_max = self.gb.grids_of_dimension(self.gb.dim_max())[0]
        (xmin, ymin, zmin) = np.min(g_max.nodes, axis=1)
        (xmax, ymax, zmax) = np.max(g_max.nodes, axis=1)
        self.domain = {'xmin': xmin, 'xmax': xmax,
                       'ymin': ymin, 'ymax': ymax,
                       'zmin': zmin, 'zmax': zmax,
                       }

    def swap_fracture_grids(self, mesh_args):
        """
        Remesh the domain and swap out lower-dimensional meshes with the new meshes.

        Parameters:
        mesh_args (Dictionary) : Meshargumetns for new mesh
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
            if d['mortar_grid'].dim <= self.gb.dim_max() - 2:
                gl, gh = gb_fracs.nodes_of_edge(e)
                self.gb.add_edge((gl, gh), d['face_cells'])
                new_d = self.gb.edge_props((gl, gh))
                new_d['mortar_grid'] = d['mortar_grid']


    def viscosity(self, c):
        """ Return the viscosity as a function of temperature
        """

        return pp.ad.exp(1*c)

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

            # Tangential permeability
            if g.dim == self.gb.dim_max():
                kxx = self.param["km"] * unity
                perm = pp.SecondOrderTensor(3, kxx)
            else:
                kxx = self.param["kf"] * unity
                perm = pp.SecondOrderTensor(3, kxx)

            param["second_order_tensor"] = perm

            # Crossectional area.
            aperture = np.power(self.param["aperture"], self.gb.dim_max() -
                                g.dim)
            param["aperture"] = aperture * unity

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
                north = bound_face_centers[1, :] > self.domain["ymax"] - \
                                                   self.tol
                south = bound_face_centers[1, :] < self.domain['ymin'] + self.tol

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
            # Get lower dimensional grid
            g_l = self.gb.nodes_of_edge(e)[0]
            mg = d['mortar_grid']
            check_P = mg.slave_to_mortar_avg()

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[keyword][
                "aperture"]
            gamma = check_P * np.power(aperture, 1. / (self.gb.dim_max() -
                                                       g_l.dim))
            kn = self.param['kn'] * np.ones(mg.num_cells) / (gamma / 2)

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

            # Tangential permeability
            if g.dim == 3:
                kxx = self.param["Dm"] * unity
            else:
                kxx = self.param["Df"] * unity
            perm = pp.SecondOrderTensor(3, kxx)

            param["second_order_tensor"] = perm

            # Aperture
            aperture = np.power(self.param["aperture"], self.gb.dim_max() -
                                g.dim)
            param["aperture"] = aperture * unity
            # Source term
            param["source"] = zeros

            # Boundaries
            bound_faces = g.get_boundary_faces()
            bc_val = np.zeros(g.num_faces)
            if bound_faces.size == 0:
                param["bc"] = pp.BoundaryCondition(g, empty, empty)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                south = bound_face_centers[1, :] < self.domain['ymin'] + self.tol
                north = bound_face_centers[1, :] > self.domain["ymax"] - \
                                                   self.tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[north] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[north]] = 1

                param["bc"] = pp.BoundaryCondition(g, bound_faces, labels)

            param["bc_values"] = bc_val

            pp.initialize_data(g, d, keyword, param)

        # Normal permeability
        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]
            mg = d['mortar_grid']
            check_P = mg.slave_to_mortar_avg()

            kxx = self.param["Dn"]
            
            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[keyword][
                "aperture"]
            gamma = check_P * np.power(aperture, 1. / (self.gb.dim_max() -
                                                       g_l.dim))
            kn = kxx * np.ones(mg.num_cells) / (gamma / 2)

            param = {"normal_diffusivity": kn}

            pp.initialize_data(e, d, keyword, param)

    # ------------------------------------------------------------------------ #

    def print_setup(self):
        print(" ------------------------------------------------------------- ")
        print(" -------------------- PROBLEM SETUP -------------------------- ")

        table = [["Mesh size", self.param["mesh_size"]],
                 ["Aperture", self.param["aperture"]],
                 ["Km", self.param["km"]],
                 ["Kf", self.param["kf"]],
                 ["Kn", self.param["kn"]]]


        print(" ------------------------------------------------------------- ")

    # ------------------------------------------------------------------------ #

