"""
Main script to solve Mandel's problem in a quarter domain using
MPFA/MPSA-FV in PorePy.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import porepy as pp

import analytical
import create_grid
import data
import discretization
import solve
import viz

# Constructing the grid
mesh_size = 2.0  # [m]
domain_size = [100.0, 10.0]  # [m]
gb = create_grid.make_grid(mesh_size, domain_size)
g = gb.grids_of_dimension(2)[0]  # extracting 2D grid from the GridBucket

# Define keywords and primary variables
kw_f = "flow"  ######## Keyword for the flow problem
kw_m = "mechanics"  ### Keyword for the mechanics problem
v_f = "pressure"  ##### Primary variable of the flow problem
v_m = "displacement"  # Primary varibale of the mechanics problem

# Create data dicitionary and initialize
d = gb.node_props(g)
d = pp.initialize_default_data(g, d, kw_f)  # initialize the flow problem
d = pp.initialize_default_data(g, d, kw_m)  # initialize the mechanics problem

# Set time parameters
time_parameters = {
    "initial_time": 0,  ### [s]
    "final_time": 50000,  # [s]
    "time_step": 10,  ##### [s]
}
data.set_time_parameters(d, kw_f, time_parameters)

# Set physical parameters
data.set_physical_parameters(d, kw_f, kw_m)

# Extract exact data from analytical solution
exact_data = analytical.extract_exact_data(g, d, kw_f, kw_m)

# Set boundary conditions
bc_dict = data.set_boundary_conditions(g, d, kw_f, kw_m, exact_data)

# Assign parameters
data.assign_parameters(g, d, kw_f, kw_m, bc_dict)

# Compute initial condition
data.initial_condition(d, v_f, v_m, exact_data)

# Perform discretization using MPFA/MPSA-FV
assembler = discretization.discretize(gb, d, kw_f, kw_m, v_f, v_m)

# Solve the model and colect the results in the "sol" dictionary
sol = solve.solve_mandel(gb, d, kw_f, kw_m, v_f, v_m, assembler, bc_dict)

# Export results
# Speciy the times you wish to export the solutions. Note that this should
# be in accordance with the "time_parameters" dictionary
plot_times = [10, 50, 100, 1000, 5000, 8000, 10000, 20000, 30000, 50000]
viz.generate_csv(gb, d, kw_f, kw_m, v_f, v_m, plot_times, sol, exact_data)
