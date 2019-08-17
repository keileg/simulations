"""
Export the results in CSV format.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# Importing modules
import numpy as np
import itertools
import os
import porepy as pp

# Function declarations
def generate_csv(
    grid_bukcket,
    data_dictionary,
    parameter_keyword_flow,
    parameter_keyword_mechanics,
    variable_flow,
    variable_mechanics,
    plot_times,
    solution_dictionary,
    exact_data_dictionary,
):

    # Renaming variables
    gb = grid_bukcket
    g = gb.grids_of_dimension(2)[0]
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics
    v_f = variable_flow
    v_m = variable_mechanics
    sol_numer = solution_dictionary
    sol_exact = exact_data_dictionary

    # Retrieving data
    F_load = d[pp.PARAMETERS][kw_m]["applied_load"]
    a = g.bounding_box()[1][0]  # [m] This is the same as Lx
    b = g.bounding_box()[1][0]  # [m] This is the same as Ly
    xc = g.cell_centers[0]  # [m] Horizontal position of the cell centers
    time_values = d[pp.PARAMETERS][kw_f]["time_values"]
    time_levels = [np.where(time_values == x)[0][0] for x in plot_times]
    p_numer = sol_numer[v_f][time_levels]  # numerical pressure
    u_numer = sol_numer[v_m][time_levels]  # numerical displacement
    p_exact = sol_exact[v_f][time_levels]  # exact pressure
    u_exact = sol_exact[v_m][time_levels]  # exact displacement

    # Select the cells that we are going to use for plotting
    # In this case, the closest horizontal cells w.r.t the bottom of the domain
    half_max_diam = np.max(g.cell_diameters()) / 2
    xc_eval = np.arange(0, a, half_max_diam)
    closest_cells = g.closest_cell(np.array([xc_eval, np.zeros_like(xc_eval)]))
    _, idx = np.unique(closest_cells, return_index=True)
    xc_plot = closest_cells[np.sort(idx)]

    # Preparing for exporting
    x_dimless = xc[xc_plot] / a  # dimensionless horizontal length
    pn_dimless = np.zeros([len(time_levels), len(xc_plot)])
    pe_dimless = np.zeros([len(time_levels), len(xc_plot)])
    uxn_dimless = np.zeros([len(time_levels), len(xc_plot)])
    uxe_dimless = np.zeros([len(time_levels), len(xc_plot)])
    for t, _ in enumerate(time_levels):
        pn_dimless[t] = p_numer[t][xc_plot] * a / F_load  # dimless numerical p
        pe_dimless[t] = p_exact[t][xc_plot] * a / F_load  # dimless exact p
        uxn_dimless[t] = u_numer[t][::2][xc_plot] / a  # dimless numerical u_x
        uxe_dimless[t] = u_exact[t][::2][xc_plot] / a  # dimless numerical u_x

    # Building the CSV
    delimiter = ","
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    time_list = []
    for time in plot_times:
        time_list.append("t = " + np.str(time) + " [s]")

    header_list = ["x_dimless [-]"] + time_list
    header = delimiter.join(header_list)

    xdim_col = np.reshape(x_dimless, np.array([len(x_dimless), 1]))

    # Times
    pn_csv = np.hstack((xdim_col, pn_dimless.T))
    np.savetxt(
        output_folder + "/" + "times.csv",
        plot_times,
        delimiter=delimiter,
        header="times [s]",
    )

    # Numerical pressure
    pn_csv = np.hstack((xdim_col, pn_dimless.T))
    np.savetxt(
        output_folder + "/" + "p_numerical.csv",
        pn_csv,
        delimiter=delimiter,
        header=header,
    )

    # Analytical pressure
    pe_csv = np.hstack((xdim_col, pe_dimless.T))
    np.savetxt(
        output_folder + "/" + "p_exact.csv", pe_csv, delimiter=delimiter, header=header
    )

    # Numerical horizontal displacement
    uxn_csv = np.hstack((xdim_col, uxn_dimless.T))
    np.savetxt(
        output_folder + "/" + "ux_numerical.csv",
        uxn_csv,
        delimiter=delimiter,
        header=header,
    )

    # Analytical horizontal displacement
    uxe_csv = np.hstack((xdim_col, uxe_dimless.T))
    np.savetxt(
        output_folder + "/" + "ux_exact.csv",
        uxe_csv,
        delimiter=delimiter,
        header=header,
    )
