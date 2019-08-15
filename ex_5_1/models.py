"""
Module to solve viscous flow. This module contains the function viscous_flow,
which takes a discretization class and a data class as arguments. The
discretization class is found at discretization.ViscouFlow and the data is found
at data.Data.
"""

import numpy as np
import porepy as pp
import porepy.ad as ad
import scipy.sparse as sps
import os

import projection
import viz


def viscous_flow(disc, data, time_step_param):
    """
    Solve the coupled problem of fluid flow and temperature transport, where the
    viscosity is depending on the temperature.
    Darcy's law and mass conservation is solved for the fluid flow:
    u = -K/mu(T) grad p,   div u = 0,
    where mu(T) is a given temperature depending viscosity.
    The temperature is advective and diffusive:
    \partial phi T /\partial t + div (cu) -div (D grad c) = 0.
    
    A darcy type coupling is assumed between grids of different dimensions:
    lambda = -kn/mu(T) * (p^lower - p^higher),
    and similar for the diffusivity:
    lambda_c = -D * (c^lower - c^higher).

    Parameters:
    disc (discretization.ViscousFlow): A viscous flow discretization class
    data (data.ViscousData): a viscous flow data class

    Returns:
    None

    The solution is exported to vtk.
    """
    # Get information from data
    gb = data.gb
    flow_kw = data.flow_keyword
    tran_kw = data.transport_keyword

    # define shorthand notation for discretizations
    # Flow
    flux = disc.mat[flow_kw]["flux"]
    bound_flux = disc.mat[flow_kw]["bound_flux"]
    trace_p_cell = disc.mat[flow_kw]["trace_cell"]
    trace_p_face = disc.mat[flow_kw]["trace_face"]
    bc_val_p = disc.mat[flow_kw]["bc_values"]
    kn = disc.mat[flow_kw]["kn"]
    mu = data.viscosity

    # Transport
    diff = disc.mat[tran_kw]["flux"]
    bound_diff = disc.mat[tran_kw]["bound_flux"]
    trace_c_cell = disc.mat[tran_kw]["trace_cell"]
    trace_c_face = disc.mat[tran_kw]["trace_face"]
    bc_val_c = disc.mat[tran_kw]["bc_values"]
    Dn = disc.mat[tran_kw]["dn"]

    # Define projections between grids
    master2mortar, slave2mortar, mortar2master, mortar2slave = projection.mixed_dim_projections(
        gb
    )
    # And between cells and faces
    avg = projection.cells2faces_avg(gb)
    div = projection.faces2cells(gb)

    # Assemble geometric values
    mass_weight = disc.mat[tran_kw]["mass_weight"]
    cell_volumes = gb.cell_volumes() * mass_weight
    mortar_volumes = gb.cell_volumes_mortar()
    # mortar_area = mortar_volumes * (master2mortar * avg * specific_volume)

    # Define secondary variables:
    q_func = lambda p, c, lam: (
        (flux * p + bound_flux * bc_val_p) * (mu(avg * c)) ** -1
        + bound_flux * mortar2master * lam
    )
    trace_p = lambda p, lam: trace_p_cell * p + trace_p_face * (lam + bc_val_p)
    trace_c = lambda c, lam_c: trace_c_cell * c + trace_c_face * (lam_c + bc_val_c)

    # Define discrete equations
    # Flow, conservation
    mass_conservation = lambda p, lam, q: (div * q - mortar2slave * lam)
    # Flow, coupling law
    coupling_law = lambda p, lam, c: (
        lam / kn / mortar_volumes
        + (slave2mortar * p - master2mortar * trace_p(p, mortar2master * lam))
        / mu((slave2mortar * c + master2mortar * avg * c) / 2)
    )

    # Transport
    # Define upwind and diffusive discretizations
    upwind = lambda c, lam, q: (
        div * disc.upwind(c, q)
        + disc.mortar_upwind(
            c, lam, div, avg, master2mortar, slave2mortar, mortar2master, mortar2slave
        )
    )

    diffusive = lambda c, lam_c: (
        div * (diff * c + bound_diff * (mortar2master * lam_c + bc_val_c))
        - mortar2slave * lam_c
    )
    # Diffusive copuling law
    coupling_law_c = lambda c, lam_c: (
        lam_c / Dn / mortar_volumes
        + slave2mortar * c
        - master2mortar * trace_c(c, mortar2master * lam_c)
    )
    # Tranpsort, conservation equation
    theta = 0.5
    transport = lambda lam, lam0, c, c0, lam_c, lam_c0, q, q0: (
        (c - c0) * (cell_volumes / dt)
        + theta * (upwind(c, lam, q) + diffusive(c, lam_c))
        + (1 - theta) * (upwind(c0, lam0, q0) + (1 - theta) * diffusive(c0, lam_c0))
    )
    # Define ad variables
    print("Solve for initial condition")
    # We solve for inital pressure and mortar flux by fixing the temperature
    # to the initial value.
    c0 = np.zeros(gb.num_cells())
    lam_c0 = np.zeros(gb.num_mortar_cells())
    # Initial guess for the pressure and mortar flux
    p0_init = np.zeros(gb.num_cells())
    lam0_init = np.zeros(gb.num_mortar_cells())
    # Define Ad variables and set up equations
    p0, lam0 = ad.initAdArrays([p0_init, lam0_init])
    eq_init = ad.concatenate(
        (mass_conservation(p0, lam0, q_func(p0, c0, lam0)), coupling_law(p0, lam0, c0))
    )
    # As the temperature is fixed, the system is linear, thus Newton's method converges
    # in one iteration
    sol_init = -sps.linalg.spsolve(eq_init.jac, eq_init.val)
    p0 = sol_init[: gb.num_cells()]
    lam0 = sol_init[gb.num_cells() :]

    # Now that we have solved for initial condition, initalize full problem
    p, lam, c, lam_c = ad.initAdArrays([p0, lam0, c0, lam_c0])
    sol = np.hstack((p.val, lam.val, c.val, lam_c.val))
    q = q_func(p, c, lam)

    # define dofs indices
    p_ix = slice(gb.num_cells())
    lam_ix = slice(gb.num_cells(), gb.num_cells() + gb.num_mortar_cells())
    c_ix = slice(
        gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + gb.num_mortar_cells(),
    )
    lam_c_ix = slice(
        2 * gb.num_cells() + gb.num_mortar_cells(),
        2 * gb.num_cells() + 2 * gb.num_mortar_cells(),
    )
    # solve system
    dt = time_step_param["dt"]
    t = 0
    k = 0
    exporter = pp.Exporter(
        gb, time_step_param["file_name"], time_step_param["folder_name"]
    )
    # Export initial condition
    viz.split_variables(gb, [p.val, c.val], ["pressure", "concentration"])
    exporter.write_vtk(["pressure", "concentration"], time_step=k)
    times = [0]
    # Store average concentration
    out_file_name = "res_avg_c/" + time_step_param["file_name"] + ".csv"
    os.makedirs(os.path.dirname(out_file_name), exist_ok=True)
    out_file = open(out_file_name, "w")
    out_file.write('time, average_c\n')
    viz.store_avg_concentration(gb, 0, "concentration", out_file)

    while t <= time_step_param["end_time"]:
        t += dt
        k += 1
        print("Solving time step: ", k, " dt: ", dt, " Time: ", t)
        p0 = p.val
        lam0 = lam.val
        c0 = c.val
        lam_c0 = lam_c.val
        q0 = q.val
        err = np.inf
        newton_it = 0
        sol0 = sol.copy()
        while err > 1e-9:
            newton_it += 1
            q = q_func(p, c, lam)
            equation = ad.concatenate(
                (
                    mass_conservation(p, lam, q),
                    coupling_law(p, lam, c),
                    transport(lam, lam0, c, c0, lam_c, lam_c0, q, q0),
                    coupling_law_c(c, lam_c),
                )
            )
            err = np.max(np.abs(equation.val))
            if err < 1e-9:
                break
            print("newton iteration number: ", newton_it - 1, ". Error: ", err)
            sol = sol - sps.linalg.spsolve(equation.jac, equation.val)
            p.val = sol[p_ix]
            lam.val = sol[lam_ix]

            c.val = sol[c_ix]
            lam_c.val = sol[lam_c_ix]

            if err != err or newton_it > 10 or err > 10e10:
                # Reset
                print("failed Netwon, reducing time step")
                t -= dt / 2
                dt = dt / 2
                p.val = p0
                lam.val = lam0

                c.val = c0
                lam_c.val = lam_c0

                sol = sol0
                err = np.inf
                newton_it = 0
            # print(err)
        print("Converged Newton in : ", newton_it - 1, " iterations. Error: ", err)
        if newton_it < 3 and dt < time_step_param["max_dt"]:
            dt = dt * 2
        elif newton_it < 7 and dt < time_step_param["max_dt"]:
            dt *= 1.1

        viz.split_variables(gb, [p.val, c.val], ["pressure", "concentration"])

        exporter.write_vtk(["pressure", "concentration"], time_step=k)
        viz.store_avg_concentration(gb, t, "concentration", out_file)
        times.append(t)
    exporter.write_pvd(timestep=np.array(times))
    out_file.close()
