"""
Auxiliary function to extract necessary data from the exact solution, i.e.:
pressure and displacement solutions, and boundary conditions.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Data: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# %% Importing modules
import numpy as np
import scipy.optimize as opt

import porepy as pp


# %% Function definitions
def extract_mandel_data(grid_object, data_dictionary,
                        parameter_keyword_flow,
                        parameter_keyword_mechanics):
    """
    Computes the exact solution of Mandel's problem.

    The function assumes that all the necessary fields
    are present inside the data dicitionary,

    Parameters:
        grid_object (PorePy object)         : PorePy grid object
        data_dictionary (Dictionary)        : Model's data dictionary
        parameter_keyword_flow (String)     : Keyword for the flow parameters
        parameter_keyword_mechanics (String): Keyword for the mechs parameters

    Returns:
        mandel_dict (Dictionary): Containing exact solutions and top bc
    """

    # Renaming input data
    g = grid_object
    d = data_dictionary
    kw_f = parameter_keyword_flow
    kw_m = parameter_keyword_mechanics

    # Retrieving data from the grid object
    [Nx, Ny] = g.cart_dims        # Number of cells in each direction
    a = g.bounding_box()[1][0]    # [m] This is the same as Lx
    b = g.bounding_box()[1][1]    # [m] This is the same as Ly

    xc = g.cell_centers[0]        # [m] x-cell-centers
    yc = g.cell_centers[1]        # [m] y-cell-centers
    yf = g.face_centers[1]        # [m] y-face-centers

    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]   # boundary faces
    y_max = b_faces[g.face_centers[1, b_faces] > 0.9999*b]   # faces top bc

    # Retrieving data from the data dictionary
    c_f = d[pp.PARAMETERS][kw_f]["fluid_diffusivity"]
    times = d[pp.PARAMETERS][kw_f]["time_values"]

    F = d[pp.PARAMETERS][kw_m]["applied_load"]
    mu_s = d[pp.PARAMETERS][kw_m]["lame_mu"]
    nu_s = d[pp.PARAMETERS][kw_m]["poisson_coefficient"]
    B = d[pp.PARAMETERS][kw_m]["skempton_coefficient"]
    nu_u = d[pp.PARAMETERS][kw_m]["undrained_poisson_coefficient"]

    # Create empty dictionary to store analytical solutions and boundary terms
    mandel_dict = {"pressure": np.zeros((len(times), g.num_cells)),
                   "displacement": np.zeros((len(times), g.dim*g.num_cells)),
                   "top_bc_values": np.zeros((len(times), len(y_max)))}

    # Compute analytical solution and top boundary condition

    # Numerical approximation to the the roots of f(x) = 0, where
    # f(x) = tan(x) - ((1-nu)/(nu_u-nu)) x.

    """
    Note that we have to solve the above equation numerically to get
    all the positive solutions to the equation. Later we will use them to
    compute the infinite series. Experience has shown that 200 roots are enough
    to achieve accurate results. Note: We find the roots using the bisection
    method, Newton method fails.
    """

    # Define algebraic function
    def f(x):
        y = np.tan(x) - ((1-nu_s)/(nu_u-nu_s))*x
        return y

    n_series = 200              # number of approimated roots
    a_n = np.zeros(n_series)    # initializing roots array
    x0 = 0                      # initial point
    for i in range(n_series):
        a_n[i] = opt.bisect(f,  # function
                            x0+np.pi/4,  # left point
                            x0+np.pi/2-10000000*2.2204e-16,  # right point
                            xtol=1e-30,  # absolute tolerance
                            rtol=1e-14)  # relative tolerance
        x0 += np.pi           # apply a phase change of pi to get the next root

    # Auxiliary (constant) terms necessary to compute the solutions
    p0 = (2*F*B*(1+nu_u)) / (3*a)

    ux0_1 = ((F*nu_s) / (2*mu_s*a))
    ux0_2 = -((F*nu_u) / (mu_s*a))
    ux0_3 = F/mu_s

    uy0_1 = (-F*(1-nu_s)) / (2*mu_s*a)
    uy0_2 = (F*(1-nu_u) / (mu_s*a))

    # Determine solutions for all the time steps (including initial condition)

    aa_n = a_n[:, np.newaxis]  # preparing to broadcast

    for t in range(len(times)):

        # Pressures
        p_sum = np.sum(((np.sin(aa_n))/(aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                       * (np.cos((aa_n*xc)/a) - np.cos(aa_n))
                       * np.exp((-(aa_n**2) * c_f * times[t])/(a**2)),
                       axis=0)
        mandel_dict["pressure"][t] = p0 * p_sum

        # Displacements
        ux_sum1 = np.sum((np.sin(aa_n)*np.cos(aa_n))
                         / (aa_n - np.sin(aa_n)*np.cos(aa_n))
                         * np.exp((-(aa_n**2) * c_f * times[t])/(a**2)),
                         axis=0)
        ux_sum2 = np.sum((np.cos(aa_n)/(aa_n - (np.sin(aa_n) * np.cos(aa_n))))
                         * np.sin((aa_n * xc)/a)
                         * np.exp((-(aa_n**2) * c_f * times[t])/(a**2)),
                         axis=0)
        uy_sum = np.sum(((np.sin(aa_n) * np.cos(aa_n))
                         / (aa_n - np.sin(aa_n) * np.cos(aa_n)))
                        * np.exp((-(aa_n**2) * c_f * times[t])/(a**2)),
                        axis=0)
        ux = (ux0_1 + ux0_2*ux_sum1) * xc + ux0_3 * ux_sum2
        uy = (uy0_1 + uy0_2*uy_sum) * yc
        mandel_dict["displacement"][t] = np.array((ux, uy)).ravel("F")

        # Time-dependent dirichlet boundary condition to be imposed at the top
        mandel_dict["top_bc_values"][t] = (uy0_1 + uy0_2*uy_sum) * yf[y_max]

    return mandel_dict
