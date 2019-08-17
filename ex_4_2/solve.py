"""
Solves the linear system.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import numpy as np
import porepy as pp
import scipy.sparse as sps
import sys

# Function declaration
def solve_mandel(
    grid_bucket,
    data_dictionary,
    parameter_keyword_flow,
    parameter_keyword_mechanics,
    variable_flow,
    variable_mechanics,
    assembler,
    boundary_conditions_dictionary,
):
    """
    The problem is soved by looping through all the time levels. The ouput
    is a dictionary containing the solutions for pressure and displacement
    for all discrete times.

    Parameters:
        grid_bucket (PorePy object):           Grid bucket
        data_dictionary (Dict):                Model's data dictionary
        parameter_keyword_flow (String):       Keyword for the flow parameter
        parameter_keyword_mechanics (String):  Keyword for the mechanics parameter
        variable_flow (String):                Primary variable of the flow problem
        variable_mechanics (String):           Primary variable of the mechanics problem
        assembler (PorePy object):             Assembler containing discretization
        boundary_conditions_dictionary (Dict): Dictionary containing boundary conditions

    Output
        sol (Dict):                            Dictionary containing numerial solutions
    """

    # Renaming variables
    gb = grid_bucket
    g = gb.grids_of_dimension(2)[0]
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics
    variable_f = variable_flow
    variable_m = variable_mechanics
    bc_dict = boundary_conditions_dictionary

    # Retrieving time values from the data dicitionary
    time_values = d[pp.PARAMETERS][kw_f]["time_values"]

    # Create a dictionary to store the solutions
    sol = {
        variable_f: np.zeros((len(time_values), g.num_cells)),
        variable_m: np.zeros((len(time_values), g.dim * g.num_cells)),
    }
    sol[variable_f][0] = d[pp.STATE][variable_f]
    sol[variable_m][0] = d[pp.STATE][variable_m]

    # For convience, create pressure and displacement variables
    pressure = d[pp.STATE][variable_f]
    displacement = d[pp.STATE][variable_m]

    # Assemble equations
    # NOTE: The structure of the linear system is time-independent
    assembler = pp.Assembler(gb)

    # Time loop
    for t in range(len(time_values) - 1):

        # Update data for current time
        pp.set_state(d, {variable_m: displacement, variable_f: pressure})
        pp.set_state(d, {kw_m: {"bc_values": bc_dict[kw_m]["bc_values"][t]}})
        d[pp.PARAMETERS][kw_m]["bc_values"] = bc_dict[kw_m]["bc_values"][t + 1]

        # Assemble matrix and rhs and solve
        A, b = assembler.assemble_matrix_rhs()
        x = sps.linalg.spsolve(A, b)

        # Distribute primary variables
        assembler.distribute_variable(x)
        displacement = d[pp.STATE][variable_m]
        pressure = d[pp.STATE][variable_f]

        # Save in solution dictionary
        sol["pressure"][t + 1] = pressure
        sol["displacement"][t + 1] = displacement

        # Print progress on console
        sys.stdout.write(
            "\rSimulation progress: %d%%"
            % (np.ceil((t / (len(time_values) - 2)) * 100))
        )
        sys.stdout.flush()

    sys.stdout.write("\nThe simulation has ended without any errors!\n")

    return sol
