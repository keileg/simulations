"""
Sets the model data. Contains the following functions:
    * set_time_parameters
    * set_physical_parameters
    * set_boundary_conditions
    * assign_parameters
    * initial_condition

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import numpy as np
import porepy as pp

# Functions declaration
def set_time_parameters(data_dictionary, parameter_keyword_flow, time_parameters):
    """
    Sets the time parameters for the coupled problem.

    Parameters:
        data_dictionary (Dictionary): Model's data dictionary
        parameter_keyword_flow (String): Keyword for the flow parameter
        time_parameters (Dictionary): Containing the keys ["initial_time"],
                                      ["final_time"] and ["time_step"]
    """

    # Renaming input data
    d = data_dictionary
    kw_f = parameter_keyword_flow

    # Declaring time parameters
    time_values = np.arange(
        time_parameters["initial_time"],
        time_parameters["final_time"] + time_parameters["time_step"],
        time_parameters["time_step"],
    )

    # Storing in the dictionary
    d[pp.PARAMETERS][kw_f]["initial_time"] = time_parameters["initial_time"]
    d[pp.PARAMETERS][kw_f]["final_time"] = time_parameters["final_time"]
    d[pp.PARAMETERS][kw_f]["time_step"] = time_parameters["time_step"]
    d[pp.PARAMETERS][kw_f]["time_values"] = time_values


def set_physical_parameters(
    data_dictionary, parameter_keyword_flow, parameter_keyword_mechanics
):
    """
    Declaration of the physical parameters

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

    # Declaring model data. In this example we use the data from:
    # https://link.springer.com/article/10.1007/s10596-013-9393-8

    # Nomenclature regarding parameters' declaration
    # Variable          Units           Description
    # mu_s              Pa              First Lame parameter
    # lambda_s          Pa              Second Lame parameter
    # K_s               Pa              Bulk modulus
    # E_s               Pa              Young's modulus
    # nu_s              -               Poisson's coefficient
    # k_s               m^2             Intrinsic permeability
    # alpha_biot        -               Biot's coefficient
    # mu_f              Pa s            Dynamic viscosity
    # S_m               Pa^{-1}         Specific storage
    # K_u               K_u             Undrained bulk modulus
    # B                 -               Skempton's coefficient
    # nu_u              -               Undrained poisson coefficient
    # c_f               m^2 s^{-1}      Fluid diffusivity
    # F                 N m^{-1}        Applied load

    mu_s = 2.475e09
    lambda_s = 1.65e09
    K_s = (2 / 3) * mu_s + lambda_s
    E_s = mu_s * ((9 * K_s) / (3 * K_s + mu_s))
    nu_s = (3 * K_s - 2 * mu_s) / (2 * (3 * K_s + mu_s))
    k_s = 0.1 * 9.869233e-13
    alpha_biot = 1.0
    mu_f = 1.0e-3
    S_m = 1 / 1.65e10
    K_u = K_s + (alpha_biot ** 2) / S_m
    B = alpha_biot / (S_m * K_u)
    nu_u = (3 * nu_s + B * (1 - 2 * nu_s)) / (3 - B * (1 - 2 * nu_s))
    c_f = (2 * k_s * (B ** 2) * mu_s * (1 - nu_s) * (1 + nu_u) ** 2) / (
        9 * mu_f * (1 - nu_u) * (nu_u - nu_s)
    )
    F = 6.0e8

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


def set_boundary_conditions(
    grid_object,
    data_dictionary,
    parameter_keyword_flow,
    parameter_keyword_mechanics,
    exact_data,
):
    """
    Sets the boundary conditions for the coupled problem.

    Parameters:
        grid_object (PorePy object)         : PorePy grid object.
        data_dictionary (Dictionary)        : Model's data dictionary.
        parameter_keyword_flow (String)     : Keyword for the flow parameters.
        parameter_keyword_mechanics (String): Keyword for the mech parameters.
        exact_data (Dictionary)             : Containing the solutions and
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
    times = d[pp.PARAMETERS][kw_f]["time_values"]

    # Retrieving data from the grid object
    a = g.bounding_box()[1][0]  # [m] This is the same as Lx
    b = g.bounding_box()[1][1]  # [m] This is the same as Ly

    # Getting the boundary faces
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]

    # Extracting indices of boundary faces w.r.t g
    x_min = b_faces[g.face_centers[0, b_faces] < 1e-8]
    x_max = b_faces[g.face_centers[0, b_faces] > a - 1e-8]
    y_min = b_faces[g.face_centers[1, b_faces] < 1e-8]
    y_max = b_faces[g.face_centers[1, b_faces] > b - 1e-8]

    # Extracting indices of boundary faces w.r.t b_faces
    west = np.in1d(b_faces, x_min).nonzero()
    east = np.in1d(b_faces, x_max).nonzero()
    south = np.in1d(b_faces, y_min).nonzero()
    north = np.in1d(b_faces, y_max).nonzero()

    # Set flow boundary conditions [Time-independent Boundary Condition]

    # Setting the tags at each boundary side
    labels_flow = np.array([None] * b_faces.size)
    labels_flow[west] = "neu"  # no flow
    labels_flow[east] = "dir"  # constant pressure
    labels_flow[south] = "neu"  # no flow
    labels_flow[north] = "neu"  # no flow

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
    u_top = exact_data["top_bc_values"]

    # Setting the tags at each boundary side for the mechanics problem
    labels_mech = np.array([None] * b_faces.size)
    labels_mech[west] = "dir_x"  # roller
    labels_mech[east] = "neu"  # traction free
    labels_mech[south] = "dir_y"  # roller
    labels_mech[north] = "dir_y"  # roller (with non-zero uy)

    # Constructing the bc object for the mechanics problem
    bc_mech = pp.BoundaryConditionVectorial(g, b_faces, labels_mech)

    # Constructing the boundary values array for the mechanics problem
    bc_mech_values = np.zeros((len(times), g.num_faces * g.dim))

    for t in range(len(times)):

        # West side boundary conditions
        bc_mech_values[t][2 * x_min] = 0  # [m]
        bc_mech_values[t][2 * x_min + 1] = 0  # [Pa]

        # East side boundary conditions
        bc_mech_values[t][2 * x_max] = 0  # [Pa]
        bc_mech_values[t][2 * x_max + 1] = 0  # [Pa]

        # South Side boundary conditions
        bc_mech_values[t][2 * y_min] = 0  # [Pa]
        bc_mech_values[t][2 * y_min + 1] = 0  # [m]

        # North Side boundary conditions
        bc_mech_values[t][2 * y_max] = 0  # [Pa]
        bc_mech_values[t][2 * y_max + 1] = u_top[t]  # [m]

    # Saving boundary conditions in a dictionary
    bc_dictionary = dict()
    bc_dictionary[kw_f] = {"bc": bc_flow, "bc_values": bc_flow_values}
    bc_dictionary[kw_m] = {"bc": bc_mech, "bc_values": bc_mech_values}

    return bc_dictionary


def assign_parameters(
    grid_object,
    data_dictionary,
    parameter_keyword_flow,
    parameter_keyword_mechanics,
    boundary_conditions_dictionary,
):
    """
    Assign data to the model, which will later be used to discretize
    the coupled problem.

    Parameters:
        grid_object (PorePy object):           PorePy grid object
        data_dictionary (Dict):                Model's data dictionary
        parameter_keyword_flow (String):       Keyword for the flow parameter
        parameter_keyword_mechanics (String):  Keyword for the mechanics parameter
        boundary_conditions_dictionary (Dict): Dictionary containing boundary conditions
    """

    # Renaming input data
    g = grid_object
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics
    bc_dict = boundary_conditions_dictionary

    # Assing flow data

    # Retrieve data for the flow problem
    k = d[pp.PARAMETERS][kw_f]["permeability"]
    alpha_biot = d[pp.PARAMETERS][kw_f]["alpha_biot"]
    S_m = d[pp.PARAMETERS][kw_f]["specific_storage"]
    dt = d[pp.PARAMETERS][kw_f]["time_step"]

    bc_flow = bc_dict[kw_f]["bc"]
    bc_flow_values = bc_dict[kw_f]["bc_values"]

    # Create second order tensor object
    kxx = k * np.ones(g.num_cells)
    perm = pp.SecondOrderTensor(kxx)

    # Create specified parameters dicitionary
    specified_parameters_flow = {
        "second_order_tensor": perm,
        "biot_alpha": alpha_biot,
        "bc": bc_flow,
        "bc_values": bc_flow_values,
        "time_step": dt,
        "mass_weight": S_m * np.ones(g.num_cells),
    }

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
    mu_lame = mu * np.ones(g.num_cells)
    lambda_lame = lmbda * np.ones(g.num_cells)
    constit = pp.FourthOrderTensor(mu_lame, lambda_lame)

    # Create specified parameters dicitionary
    specified_parameters_mechanics = {
        "fourth_order_tensor": constit,
        "biot_alpha": alpha_biot,
        "bc": bc_mech,
        "bc_values": bc_mech_values,
    }

    # Initialize the mechanics
    d = pp.initialize_default_data(g, d, kw_m, specified_parameters_mechanics)

    # Save boundary conditions in d[pp.STATE]
    pp.set_state(d, {kw_m: {"bc_values": bc_dict[kw_m]["bc_values"][0]}})


def initial_condition(data_dictionary, variable_flow, variable_mechanics, exact_data):
    """
    Establishes initial condition.

    Parameters:
        data_dictionary (Dictionary)  : Model's data dictionary
        variable_flow (String)        : Primary variable of the flow problem
        variable_mechanics (String)   : Primary varibale of the mechanics problem
        exact_data (Dictionary)       : Containing the solutions and the
                                        top boundary condition.
    """

    d_0 = exact_data[variable_mechanics][0]
    p_0 = exact_data[variable_flow][0]

    state = {variable_mechanics: d_0, variable_flow: p_0}
    pp.set_state(data_dictionary, state)
