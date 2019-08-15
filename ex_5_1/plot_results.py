"""
Plotting script for fully coupled flow and transport.
 
Author: jv; E-mail: jhabriel.varela@uib.no; Date: 06.08.2019
 
It is assumed that the folder res_avg_c contains:
(1) matching.csv
(2) non_matching.csv
 
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""
# %% Importing modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns
import itertools
import os
 
from matplotlib.ticker import FormatStrFormatter
 
# %% Folders and data
 
# Folders
folder = "res_avg_c/"
output_folder = "img/"
output_files = "matching-non_matching.pdf"
files = ["matching", "non_matching"]
extension = ".csv"
file_name_1 = folder + files[0] + extension
file_name_2 = folder + files[1] + extension
 
# Loading Data
matching = np.loadtxt(file_name_1, skiprows=1, delimiter=",", encoding="utf-8")
non_matching = np.loadtxt(file_name_2, skiprows=1, delimiter=",", encoding="utf-8")
 
# Number of floating points
mf = matplotlib.ticker.ScalarFormatter(useMathText=True)
mf.set_powerlimits((-4, 4))
 
# %% Plotting
 
# Preparing plot
sns.set_context("paper")  # set scale and size of figures
sns.set_palette("tab10", 10)
itertools.cycle(sns.color_palette())  # iterate if > 10 colors are needed
 
fig = plt.figure(1, constrained_layout=False)
gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.01, right=0.84)
gs2 = fig.add_gridspec(nrows=1, ncols=1, left=0.87, right=0.99)
 
with sns.axes_style("whitegrid"):  # assign the style
    ax1 = fig.add_subplot(gs1[0, 0])
ax2 = fig.add_subplot(gs2[0, 0])
ax2.axis("off")
 
# Plotting data
color_match = next(ax1._get_lines.prop_cycler)["color"]
ax1.plot(matching[:, 0], matching[:, 1], "-", color=color_match, linewidth=2)
color_non_match = next(ax1._get_lines.prop_cycler)["color"]
ax1.plot(
    non_matching[:, 0],
    non_matching[:, 1],
    "o",
    color=color_non_match,
    alpha=1,
    markersize=3,
)
 
# Setting axes labels
ax1.set_xlabel(r"$t \, [s]$")
ax1.set_ylabel(r"$\frac{1}{|\Omega_2|}\,\int_{\Omega_2} c \, \mathrm{d}V$")
 
# Plotting legend
ax2.plot([], [], "-", color=color_match, linewidth=2, label="Non-matching")
ax2.plot([], [], "o", color=color_non_match, markersize=3, label="Matching")
ax2.legend(loc="center left", frameon=False)
 
# Saving figure
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
 
fig.savefig(output_folder + output_files, bbox_inches="tight")
