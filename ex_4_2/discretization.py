"""
Discretize the problem using MPFA/MPSA-FV method.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import porepy as pp

from porepy.utils.derived_discretizations.implicit_euler import ImplicitMassMatrix

# Function declaration
def discretize(
    grid_bucket,
    data_dictionary,
    parameter_keyword_flow,
    parameter_keyword_mechanics,
    variable_flow,
    variable_mechanics,
):
    """
    Discretize the problem.

    Parameters:
        grid_bucket (PorePy object):          Grid bucket
        data_dictionary (Dict):               Model's data dictionary
        parameter_keyword_flow (String):      Keyword for the flow parameter
        parameter_keyword_mechanics (String): Keyword for the mechanics parameter
        variable_flow (String):               Primary variable of the flow problem
        variable_mechanics (String):          Primary variable of the mechanics problem

    Output:
        assembler (PorePy object):            Assembler containing discretization
    """

    # The Mpfa discretization assumes unit viscosity. Hence we need to
    # overwrite the class to include it.
    class ImplicitMpfa(pp.Mpfa):
        def assemble_matrix_rhs(self, g, d):
            """
            Overwrite MPFA method to be consistent with Biot's
            time discretization and inclusion of viscosity in Darcy's law
            """
            viscosity = d[pp.PARAMETERS][self.keyword]["viscosity"]
            a, b = super().assemble_matrix_rhs(g, d)
            dt = d[pp.PARAMETERS][self.keyword]["time_step"]
            return a * (1 / viscosity) * dt, b * (1 / viscosity) * dt

    # Redefining input parameters
    gb = grid_bucket
    g = gb.grids_of_dimension(2)[0]
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics
    v_0 = variable_mechanics
    v_1 = variable_flow

    # Discretize the subproblems using Biot's class, which employs
    # MPSA for the mechanics problem and MPFA for the flow problem
    biot_discretizer = pp.Biot(kw_m, kw_f, v_0, v_1)
    biot_discretizer._discretize_mech(g, d)  # discretize mech problem
    biot_discretizer._discretize_flow(g, d)  # discretize flow problem

    # Names of the five terms of the equation + additional stabilization term.
    ####################################### Term in the Biot equation:
    term_00 = "stress_divergence"  ######## div symmetric grad u
    term_01 = "pressure_gradient"  ######## alpha grad p
    term_10 = "displacement_divergence"  ## d/dt alpha div u
    term_11_0 = "fluid_mass"  ############# d/dt beta p
    term_11_1 = "fluid_flux"  ############# div (rho g - K grad p)
    term_11_2 = "stabilization"  ##########

    # Store in the data dictionary and specify discretization objects.
    d[pp.PRIMARY_VARIABLES] = {v_0: {"cells": g.dim}, v_1: {"cells": 1}}
    d[pp.DISCRETIZATION] = {
        v_0: {term_00: pp.Mpsa(kw_m)},
        v_1: {
            term_11_0: ImplicitMassMatrix(kw_f, v_1),
            term_11_1: ImplicitMpfa(kw_f),
            term_11_2: pp.BiotStabilization(kw_f, v_1),
        },
        v_0 + "_" + v_1: {term_01: pp.GradP(kw_m)},
        v_1 + "_" + v_0: {term_10: pp.DivU(kw_m, kw_f, v_0)},
    }

    assembler = pp.Assembler(gb)
    # Discretize the flow and accumulation terms - the other are already handled
    # by the biot_discretizer
    assembler.discretize(term_filter=[term_11_0, term_11_1])

    return assembler
