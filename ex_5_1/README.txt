This folder contains the setup and runscripts for Section 5.1 in the paper:

PorePy: Open-Source Software for Simulation of multiphysics processes in Fractured
Porous Media

by
Eirik Keilegavlen, Runar Berge, Alessio Fumagalli, Michele Starnoni, Jhabrial Verala, and
Inga Berre

We recomend to run the example in Docker, see Readme in root folder. To run this example,
type:

python main.py

in the Docker terminal. This will execute two simulations, one with a matching mesh, and
one with a non-matching mesh. Note that the simulations are rather time consuming, and
if you just want to get some quick results we recomend to reduce the number of fractures,
which can be modified in the main.py file.

This subfolder contains the following files:
main.py:	    Main runscript used to run the example.
discretizations.py: Contains wrappers for the discretization methods in PorePy.
projection.py:      Contains wrappers for projection operators between subdomains.
viz.py:             Utility-functions for plotting.
plot_results.py:    Utility function for plotting the average concentration in the
		    fractures for each time step. Can be called after main.py, and will
		    store the plot in the subfolder img/

In addition, there are three subfolders, res_matching, res_non_matching and res_avg_c.
After main.py is run, these folders will contain the results from the simulations.
res_matching will contain the vtk files from the simulation on a matching mesh, and
res_non-matching will contian the vtk files from the simulation on a non-mathcing mesh.
The subfolder res_avg_c will contain two csv files, matching.csv and non-matching.csv,
which contains the volume weigthed average concentration in the fractures at each time
step of the two simulations.
