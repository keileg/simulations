This folder contains the setup and runscripts for Section 4.1 in the paper:

PorePy: Open-Source Software for Simulation of multiphysics processes in Fractured
Porous Media

by
Eirik Keilegavlen, Runar Berge, Alessio Fumagalli, Michele Starnoni, Ivar Stefansson,
Jhabriel Varela, and Inga Berre

We recommend to run the example in Docker, see Readme in root folder. To run this example,
type:

python main.py

in the Docker terminal. This will execute the simulation using the setup from the
paper.

This subfolder contains the following files:

main.py:	    Main runscript used to run the example.
analytical.py       Generate exact data from the analytical solution.
create_grid.py      Create the grid for a given mesh and domain sizes.
data.py             Utility-function for parameter declaration/assignment.
discretization.py   Contains wrappers for the discretization methods in PorePy.
solve.py            Utility-function to solve and store the results.
export_results.py   Generates the .csv files inside the "results" subfolder.
plot_results.py     Utility-function for plotting the dimensionless pressure and
                    horizontal displacement, for both numerical and exact solutions.

Also, we include the plot from the paper in the "img/" subfolder, as a reference.

After running main.py, two mesh files (one .geo and one .mesh)
and one subfolder ("results/") containing the results of the simulation will be
generated. Inside "results/" subfolder, the following files should be present:

p_exact.csv
p_numerical.csv
times.csv
ux_exact.csv
ux_numerical.csv

In addition, you can plot the results (provided that you ran main.py) using
plot_results.py. This will generate the figure "mandel.pdf" inside the "img/" subfolder.
