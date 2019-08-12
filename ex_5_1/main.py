import porepy as pp

import random_network.data as data
import discretizations
import models

time_step_param = {
    'dt': 0.005 * pp.SECOND,
    'end_time': 20 * pp.SECOND,
    'max_dt': 1 * pp.SECOND,
    'file_name': 'matching',
    'folder_name': 'res_matching',
    }
mesh_args = {
    "mesh_size_frac": 0.5,
    'mesh_size_min': 0.1
}
param = {
    "km": 10**-3 * pp.METER**2,
    "kf": 10**0 * pp.METER**2,
    "kn": 10**0 * pp.METER**2,
    "Dm": 10**-4 * pp.METER**2 / pp.SECOND,
    "Df": 10**-4 * pp.METER**2 / pp.SECOND,
    "Dn": 10**-4 * pp.METER**2 / pp.SECOND,
    "aperture": 0.01 * pp.METER,
    "porosity": 0.2,
    "time_step_param": time_step_param,
    "mesh_args": mesh_args,
}
print("Solve with matching mesh")
print("mesh")
matching_data = data.Data(15, param)
print('Generate data')
matching_data.add_data()
print('Discretize')
matching_disc = discretizations.ViscousFlow(matching_data)
print('Call viscous flow model')
models.viscous_flow(matching_disc, matching_data)

print('Solved with matching mesh\n')
print('--------------------------------------------------------------\n')
print('Solve with non-matching mesh')

time_step_param['file_name'] = "non-matching"
time_step_param['folder_name'] = "res_non-matching"
mesh_args_frac = {"mesh_size_frac": 0.1, 'mesh_size_min': 0.01}
print("mesh")
nm_data = data.Data(15, param)
nm_data.swap_fracture_grids(mesh_args_frac)
print('Generate data')
nm_data.add_data()
print('Discretize')
nm_disc = discretizations.ViscousFlow(nm_data)
print('Call viscous flow model')
models.viscous_flow(nm_disc, nm_data)
