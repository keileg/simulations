"""
Main script to solve Mandel's problem in a quarter domain using
MPFA/MPSA-FV in PorePy.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# %% Importing modules
import numpy as np
import scipy.sparse as sps
import pickle
import sys

import porepy as pp

from mandel_analytical import extract_mandel_data

# %% Function definitions


def make_grid(N=[40, 40], L=[100, 10]):
    """
    Creates a structured cartesian grid.

    Parameters:
        N (array): number of cells for each dimension [-]
        L (array): length of the domain for each dimension [m]

    Returns:
        gb (PorePy object): PorePy grid bucket object containing all the grids
                            In this case we only have one grid.
    """

    gb = pp.meshing.cart_grid([], N, physdims=L)  # create cartesian grid
    gb.compute_geometry()  # compute the grid geometry
    gb.assign_node_ordering()  # assign node ordering in the grid bucket

    return gb


def set_time_parameters(data_dictionary,
                        parameter_keyword_flow):
    """
    Sets the time parameters for the coupled problem.

    Parameters:
        d (Dictionary): Model's data dictionary
        parameter_keyword_flow (String): Keyword for the flow parameter

    Note: This function does not return any value, but rather updates the data
          dictionary. The data is stored in the time parameters.
    """

    # Renaming input data
    d = data_dictionary
    kw_f = parameter_keyword_flow

    # Declaring time parameters (These parameters must be set by the user)
    initial_simulation_time = 0     # [s]
    final_simulation_time = 50000   # [s]
    time_step = 10                  # [s]
    time_values = np.arange(initial_simulation_time,
                            final_simulation_time+time_step,
                            time_step)

    # Storing in the dictionary
    d[pp.PARAMETERS][kw_f]["initial_time"] = initial_simulation_time
    d[pp.PARAMETERS][kw_f]["final_time"] = final_simulation_time
    d[pp.PARAMETERS][kw_f]["time_step"] = time_step
    d[pp.PARAMETERS][kw_f]["time_values"] = time_values


def set_model_data(data_dictionary,
                   parameter_keyword_flow,
                   parameter_keyword_mechanics):
    """
    Declaration of the model's data.

    Parameters:
        data_dictionary (Dictionary)        : Data dictionary
        parameter_keyword_flow (String)     : Keyword for the flow parameters.
        parameter_keyword_mechanics (String): Keyword for the mech parameters.

    Note: This function does not return any value, but rather updates the data
          dictionary.
    """

    # Renaming input data
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics

    # Declaring model data (These parameters must be set by the user)
    # In this example we use the data from
    # https://link.springer.com/article/10.1007/s10596-013-9393-8

    mu_s = 2.475E+09                        # [Pa] First Lame parameter
    lambda_s = 1.65E+09                     # [Pa] Second Lame parameter
    K_s = (2/3) * mu_s + lambda_s           # [Pa] Bulk modulus
    E_s = mu_s * ((9*K_s)/(3*K_s+mu_s))     # [Pa] Young's modulus
    nu_s = (3*K_s-2*mu_s)/(2*(3*K_s+mu_s))  # [-] Poisson's coefficient
    k_s = 0.1 * 9.869233E-13                # [m^2] Permeabiliy
    alpha_biot = 1.                         # [-] Biot's coefficient

    mu_f = 1.0E-3                           # [Pa s] Dynamic viscosity

    S_m = 1/1.65E10                         # [1/Pa] Specific Storage
    K_u = K_s + (alpha_biot**2)/S_m         # [Pa] Undrained bulk modulus
    B = alpha_biot / (S_m * K_u)            # [-] Skempton's coefficient
    nu_u = ((3*nu_s + B*(1-2*nu_s))         # [-] Undrained Poisson's ratio
            / (3-B*(1-2*nu_s)))
    c_f = ((2*k_s*(B**2)*mu_s*(1-nu_s)*(1+nu_u)**2)  # m^2/s Fluid diffusivity
           / (9*mu_f*(1-nu_u)*(nu_u-nu_s)))

    F = 6.0E8                               # [N/m] Applied load

    # Storing in the dictionary
    d[pp.PARAMETERS][kw_f]["viscosity"] = mu_f
    d[pp.PARAMETERS][kw_f]["alpha_biot"] = alpha_biot
    d[pp.PARAMETERS][kw_f]["specific_storage"] = S_m
    d[pp.PARAMETERS][kw_f]["fluid_diffusivity"] = c_f
    d[pp.PARAMETERS][kw_f]["permeability"] = k_s

    d[pp.PARAMETERS][kw_m]["lame_mu"] = mu_s
    d[pp.PARAMETERS][kw_m]["lame_lambda"] = lambda_s
    d[pp.PARAMETERS][kw_m]["bulk_modulus"] = K_s
    d[pp.PARAMETERS][kw_m]["young_modulus"] = E_s
    d[pp.PARAMETERS][kw_m]["poisson_coefficient"] = nu_s
    d[pp.PARAMETERS][kw_m]["alpha_biot"] = alpha_biot
    d[pp.PARAMETERS][kw_m]["skempton_coefficient"] = B
    d[pp.PARAMETERS][kw_m]["undrained_poisson_coefficient"] = nu_u
    d[pp.PARAMETERS][kw_m]["applied_load"] = F


def set_boundary_conditions(grid_object,
                            data_dictionary,
                            parameter_keyword_flow,
                            parameter_keyword_mechanics,
                            mandel_dictionary):
    """
    Sets the boundary conditions for the coupled problem.

    Parameters:
        grid_object (PorePy object)         : PorePy grid object.
        data_dictionary (Dictionary)        : Model's data dictionary.
        parameter_keyword_flow (String)     : Keyword for the flow parameters.
        parameter_keyword_mechanics (String): Keyword for the mech parameters.
        mandel_dictionary (Dictionary)      : Containing the solutions and
                                              top boundary condition
    Returns:
        bc_dictionary (dictionary): containing the bc objects and values for
                                    the flow and mechanics problems.
    """

    # Renaming input data
    g = grid_object
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics
    d_m = mandel_dictionary
    times = d[pp.PARAMETERS][kw_f]["time_values"]

    # Retrieving data from the grid object
    [Nx, Ny] = g.cart_dims      # Number of cells in each direction
    a = g.bounding_box()[1][0]  # [m] This is the same as Lx
    b = g.bounding_box()[1][1]  # [m] This is the same as Ly

    # Getting the boundary faces
    b_faces = g.tags['domain_boundary_faces'].nonzero()[0]

    # Extracting indices of boundary faces w.r.t g
    x_min = b_faces[g.face_centers[0, b_faces] < 0.0001]
    x_max = b_faces[g.face_centers[0, b_faces] > 0.9999*a]
    y_min = b_faces[g.face_centers[1, b_faces] < 0.0001]
    y_max = b_faces[g.face_centers[1, b_faces] > 0.9999*b]

    # Extracting indices of boundary faces w.r.t b_faces
    west = np.in1d(b_faces, x_min).nonzero()
    east = np.in1d(b_faces, x_max).nonzero()
    south = np.in1d(b_faces, y_min).nonzero()
    north = np.in1d(b_faces, y_max).nonzero()

    # Set flow boundary conditions [Time-independent Boundary Condition]

    # Setting the tags at each boundary side
    labels_flow = np.array([None]*b_faces.size)
    labels_flow[west] = 'neu'   # no flow
    labels_flow[east] = 'dir'   # constant pressure
    labels_flow[south] = 'neu'  # no flow
    labels_flow[north] = 'neu'  # no flow

    # Constructing the (scalar) bc object
    bc_flow = pp.BoundaryCondition(g, b_faces, labels_flow)

    # Constructing the boundary values array
    bc_flow_values = np.zeros(g.num_faces)

    # West side boundary condition
    bc_flow_values[x_min] = 0  # [m^3/s]

    # East side boundary condition
    bc_flow_values[x_max] = 0  # [Pa]

    # South side boundary condition
    bc_flow_values[y_min] = 0  # [m^3/s]

    # North side boundary condition
    bc_flow_values[y_max] = 0  # [m^3/s]

    # Set mechanics boundary conditions [Time-dependent Boundary Condition]

    # Applied displacement (time-dependent boundary condition)
    u_top = d_m["top_bc_values"]

    # Setting the tags at each boundary side for the mechanics problem
    labels_mech = np.array([None]*b_faces.size)
    labels_mech[west] = 'dir_x'   # roller
    labels_mech[east] = 'neu'     # traction free
    labels_mech[south] = 'dir_y'  # roller
    labels_mech[north] = 'dir_y'  # roller (with non-zero uy)

    # Constructing the bc object for the mechanics problem
    bc_mech = pp.BoundaryConditionVectorial(g, b_faces, labels_mech)

    # Constructing the boundary values array for the mechanics problem
    bc_mech_values = np.zeros((len(times), g.num_faces*g.dim,))

    for t in range(len(times)):

        # West side boundary conditions
        bc_mech_values[t][2*x_min] = 0           # [m]
        bc_mech_values[t][2*x_min+1] = 0         # [Pa]

        # East side boundary conditions
        bc_mech_values[t][2*x_max] = 0           # [Pa]
        bc_mech_values[t][2*x_max+1] = 0         # [Pa]

        # South Side boundary conditions
        bc_mech_values[t][2*y_min] = 0           # [Pa]
        bc_mech_values[t][2*y_min+1] = 0         # [m]

        # North Side boundary conditions
        bc_mech_values[t][2*y_max] = 0           # [Pa]
        bc_mech_values[t][2*y_max+1] = u_top[t]  # [m]

    # Saving boundary conditions in a dictionary
    bc_dictionary = dict()

    bc_dictionary[kw_f] = {"bc": bc_flow,
                           "bc_values": bc_flow_values}

    bc_dictionary[kw_m] = {"bc": bc_mech,
                           "bc_values": bc_mech_values}

    return bc_dictionary


def assign_data(grid_object,
                data_dictionary,
                boundary_conditions_dictionary,
                parameter_keyword_flow,
                parameter_keyword_mechanics):
    """
    Assign data to the model, which will later be used to discretize
    the coupled problem.

    Parameters:
        grid_object (PorePy object)                 : PorePy grid object
        data_dictionary (Dictionary)                : Model's data dictionary
        boundary_conditions_dictionary (Dicitionary): Boundary conditions
        parameter_keyword_flow (String)             : Keyword for the flow par.
        parameter_keyword_mechanics (String)        : Keyword for the mech par.

    Note: This function does not return any output. Instead, the data dict
          is updated with the proper fields.
    """

    # Renaming input data
    g = grid_object
    d = data_dictionary
    bc_dict = boundary_conditions_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics

    # Assing flow data

    # Retrieve data for the flow problem
    k = d[pp.PARAMETERS][kw_f]["permeability"]
    alpha_biot = d[pp.PARAMETERS][kw_f]["alpha_biot"]
    S_m = d[pp.PARAMETERS][kw_f]["specific_storage"]
    dt = d[pp.PARAMETERS][kw_f]["time_step"]

    bc_flow = bc_dict[kw_f]["bc"]
    bc_flow_values = bc_dict[kw_f]["bc_values"]

    # Create second order tensor object
    perm = pp.SecondOrderTensor(g.dim,
                                k * np.ones(g.num_cells))

    # Create specified parameters dicitionary
    specified_parameters_flow = {"second_order_tensor": perm,
                                 "biot_alpha": alpha_biot,
                                 "bc": bc_flow,
                                 "bc_values": bc_flow_values,
                                 "time_step": dt,
                                 "mass_weight": S_m * np.ones(g.num_cells)}

    # Initialize the flow data
    d = pp.initialize_default_data(g, d, kw_f, specified_parameters_flow)

    # Assign mechanics data

    # Retrieve data for the mechanics problem
    mu = d[pp.PARAMETERS][kw_m]["lame_mu"]
    lmbda = d[pp.PARAMETERS][kw_m]["lame_lambda"]
    alpha_biot = d[pp.PARAMETERS][kw_m]["alpha_biot"]

    bc_mech = bc_dict[kw_m]["bc"]
    bc_mech_values = bc_dict[kw_m]["bc_values"][1]

    # Create fourth order tensor
    constit = pp.FourthOrderTensor(g.dim,
                                   mu * np.ones(g.num_cells),
                                   lmbda * np.ones(g.num_cells))

    # Create specified parameters dicitionary
    specified_parameters_mechanics = {"fourth_order_tensor": constit,
                                      "biot_alpha": alpha_biot,
                                      "bc": bc_mech,
                                      "bc_values": bc_mech_values}

    # Initialize the mechanics
    d = pp.initialize_default_data(g, d, kw_m, specified_parameters_mechanics)

    # Save boundary conditions in d[pp.STATE]
    pp.set_state(
            d,
            {kw_m: {"bc_values": bc_dict[kw_m]["bc_values"][0]}}
            )


def initial_condition(data_dictionary,
                      variable_flow,
                      variable_mechanics,
                      mandel_dictionary):
    """
    Establishes initial condition.

    Parameters:
        data_dictionary (Dictionary)  : Model's data dictionary
        variable_flow (String)        : Keyword for the flow primary variable
        variable_mechanics (String)   : Keyword for the mechs primary variable
        mandel_dictionary (Dictionary): Containing the solutions and the
                                        top boundary condition.
    """

    d_0 = mandel_dictionary["displacement"][0]
    p_0 = mandel_dictionary["pressure"][0]

    state = {variable_m: d_0, variable_f: p_0}

    pp.set_state(data_dictionary, state)

    return d_0, p_0


# %% Setting up the grid, defining keywords and creating data dictionary

# Define a grid (User defined)
gb = make_grid()
g = gb.grids_of_dimension(2)[0]

# Defining keywords and primary variable names
kw_f = "flow"
kw_m = "mechanics"
variable_f = "pressure"
variable_m = "displacement"

# Create data dicitionary and initialize
data = gb.node_props(g)
data = pp.initialize_default_data(g, data, kw_f)
data = pp.initialize_default_data(g, data, kw_m)

# %% Setting parameters, assigning data and computing initial condition

set_time_parameters(data, kw_f)  # set time parameters (User-defined)

set_model_data(data, kw_f, kw_m)  # set model data (User-defined)

mandel_dict = extract_mandel_data(g, data, kw_f, kw_m)   # extracting data

bc_dict = set_boundary_conditions(g, data, kw_f, kw_m, mandel_dict)  # set bc

assign_data(g, data, bc_dict, kw_f, kw_m)  # assigning data

d_0, p_0 = initial_condition(data,
                             variable_f,
                             variable_m,
                             mandel_dict)  # get initial condition

# %% Discretize

# Now we discretize the problem using the Biot's class, which uses
# the MPSA discretization for the mechanics problem and MPFA for the
# flow problem

biot_discretizer = pp.Biot(kw_m, kw_f, variable_m, variable_f)
biot_discretizer._discretize_mech(g, data)  # discretize mech problem
biot_discretizer._discretize_flow(g, data)  # discretize flow problem

# %% Assembly equations

# First, we modify some classes to account for the implicit Euler
# discretization in time (see Biot's tutorial for a detailed explanation)


class ImplicitMassMatrix(pp.MassMatrix):

    def __init__(self, keyword="flow", variable="pressure"):
        """
        Set the discretization, with the keyword used for storing various
        information associated with the discretization. The time discretisation
        also requires the previous solution, thus the variable needs to be
        specified.

        Paramemeters:
            keyword (str) : Identifier of all parameters used for this
                            discretization.
            variable (str): Name of the variable the discretization is applied.
        """
        super().__init__(keyword)
        self.variable = variable

    def assemble_rhs(self, g, data):
        """
        Overwrite MassMatrix method to
        Return the correct rhs for an IE time discretization
        of the Biot problem.
        """

        matrix_dictionary = data[pp.DISCRETIZATION_MATRICES][self.keyword]
        previous_pressure = data[pp.STATE][self.variable]

        return matrix_dictionary["mass"] * previous_pressure


class ImplicitMpfa(pp.Mpfa):

    def assemble_matrix_rhs(self, g, data):
        """
        Overwrite MPFA method to be consistent with Biot's
        time discretization
        """

        viscosity = data[pp.PARAMETERS][self.keyword]["viscosity"]
        a, b = super().assemble_matrix_rhs(g, data)
        dt = data[pp.PARAMETERS][self.keyword]["time_step"]
        return a * (1/viscosity) * dt, b * (1/viscosity) * dt


# Variable names
v_0 = variable_m
v_1 = variable_f

# Names of the five terms of the equation + additional stabilization term.
#                                        Term in the Biot equation:
term_00 = "stress_divergence"          # div symmetric grad u
term_01 = "pressure_gradient"          # alpha grad p
term_10 = "displacement_divergence"    # d/dt alpha div u
term_11_0 = "fluid_mass"               # d/dt beta p
term_11_1 = "fluid_flux"               # div (rho g - K grad p)
term_11_2 = "stabilization"            #

# Store in the data dictionary d and specify discretization objects.
data[pp.PRIMARY_VARIABLES] = {v_0: {"cells": g.dim}, v_1: {"cells": 1}}
data[pp.DISCRETIZATION] = {
    v_0: {term_00: pp.Mpsa(kw_m)},
    v_1: {
        term_11_0: ImplicitMassMatrix(kw_f, v_1),
        term_11_1: ImplicitMpfa(kw_f),
        term_11_2: pp.BiotStabilization(kw_f, v_1),
    },
    v_0 + "_" + v_1: {term_01: pp.GradP(kw_m)},
    v_1 + "_" + v_0: {term_10: pp.DivD(kw_m, v_0)},
}

# %% Solving the linear system

time_values = data[pp.PARAMETERS][kw_f]["time_values"]

# Let's create a dictionary to store the solutions
sol = {"pressure": np.zeros((len(time_values), g.num_cells)),
       "displacement": np.zeros((len(time_values), g.dim*g.num_cells))}
sol["pressure"][0] = p_0
sol["displacement"][0] = d_0

pressure = p_0
displacement = d_0
assembler = pp.Assembler(gb)    # assembly equation (note that the structure
                                # of the linear system is time independent)

for t in range(len(time_values)-1):

    # Update data for current time
    pp.set_state(data, {variable_m: displacement, variable_f: pressure})
    pp.set_state(data, {kw_m: {"bc_values": bc_dict[kw_m]["bc_values"][t]}})
    data[pp.PARAMETERS][kw_m]["bc_values"] = bc_dict[kw_m]["bc_values"][t+1]

    # Assemble and solve
    A, b = assembler.assemble_matrix_rhs()
    x = sps.linalg.spsolve(A, b)

    # Distribute primary variables
    assembler.distribute_variable(x)
    displacement = data[variable_m]
    pressure = data[variable_f]

    # Save in solution dictionary
    sol["pressure"][t+1] = pressure
    sol["displacement"][t+1] = displacement

    # Print progress on console
    sys.stdout.write("\rSimulation progress: %d%%" %
                     (np.ceil((t/(len(time_values)-2))*100)))
    sys.stdout.flush()

# %% Exporting solutions

# Time levels at which the solutions will be exporte
plot_levels = [1, 5, 10, 100, 500, 800, 1000, 2000, 3000, 5000]

export_sol = {
        "analytical": {
                "pressure": mandel_dict["pressure"][plot_levels],
                "displacement": mandel_dict["displacement"][plot_levels]
                },
        "numerical": {
                "pressure": sol["pressure"][plot_levels],
                "displacement": sol["displacement"][plot_levels]
                },
        "data": {
                "plot_levels": plot_levels,
                "applied_load": data[pp.PARAMETERS][kw_m]["applied_load"],
                "time_values": time_values
                },
        "grid": g
        }

f = open("results.pkl", "wb")  # open pickle
pickle.dump(export_sol, f)     # dumping dictionary
f.close()                      # closing pickle
