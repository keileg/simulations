"""
Plotting script for Mandel's problem in a quarter domain. You can run the script
after main.py.

It is assumed that the subfolder "results" contains the following files:
    - p_exact.csv
    - p_numerical.csv
    - times.csv
    - ux_exact.csv
    - ux_numerical.csv

The script will generate a figure named "mandel.pdf inside" the "img" subfolder.

Author: Jhabriel Varela
E-mail: jhabriel.varela@uib.no
Date: 03.06.2019
Institution: Porous Media Group [https://pmg.w.uib.no/]
"""

# %% Importing modules
import numpy as np
import itertools
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

matplotlib.use("agg", warn=False, force=True)  # force non-GUI backend.

# %% Folders
folder = "results/"
output_folder = "img/"
output_files = "mandel.pdf"
files = ["p_exact", "p_numerical", "ux_exact", "ux_numerical", "times"]
extension = ".csv"
file_name_0 = folder + files[0] + extension
file_name_1 = folder + files[1] + extension
file_name_2 = folder + files[2] + extension
file_name_3 = folder + files[3] + extension
file_name_4 = folder + files[4] + extension

# %% Loading data
p_exact = np.loadtxt(file_name_0, skiprows=1, delimiter=",", encoding="utf-8")
p_numer = np.loadtxt(file_name_1, skiprows=1, delimiter=",", encoding="utf-8")
ux_exact = np.loadtxt(file_name_2, skiprows=1, delimiter=",", encoding="utf-8")
ux_numer = np.loadtxt(file_name_3, skiprows=1, delimiter=",", encoding="utf-8")
plot_times = np.loadtxt(file_name_4, skiprows=1, delimiter=",", encoding="utf-8")

# %% Setting style
sns.set_context("paper")  # set's the scale and size of figures
sns.set_palette("tab10", 10)  # color palette (up to 10 colors)
itertools.cycle(sns.color_palette())  # iterate if > 10 colors are needed

# %% Plot the results
# Create subplots windows
fig1 = plt.figure(constrained_layout=False)
gs1 = fig1.add_gridspec(nrows=1, ncols=1, left=0.01, right=0.40)
gs2 = fig1.add_gridspec(nrows=1, ncols=1, left=0.54, right=0.93)
gs3 = fig1.add_gridspec(nrows=1, ncols=1, left=0.95, right=0.99)

# Assign one frame to each plot
with sns.axes_style("whitegrid"):  # assign the style
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax2 = fig1.add_subplot(gs2[0, 0])
ax3 = fig1.add_subplot(gs3[0, 0])

# Legend Plot (Analytical and numerical)
ax3.plot([], [], "-k", linewidth=1, label="Analytical")
ax3.plot([], [], "ok", label="Mpsa/Mpfa", markersize=4)

# Pressure, displacement and legends
for t, times in enumerate(plot_times):

    color = next(ax1._get_lines.prop_cycler)["color"]

    # Pressure
    ax1.plot(p_exact[:, 0], p_exact[:, t + 1], color=color)
    ax1.plot(p_numer[:, 0], p_numer[:, t + 1], "o", markersize=3, color=color)

    # Horizontal displacement
    ax2.plot(ux_exact[:, 0], ux_exact[:, t + 1], color=color)
    ax2.plot(ux_numer[:, 0], ux_numer[:, t + 1], "o", markersize=3, color=color)

    # Legends
    ax3.plot([], [], color=color, label="t = {} s".format(np.int(times)))

# Labels and formatting
ax1.tick_params(axis="both", which="major")
ax1.set_xlabel(r"$x\,/\,a$")
ax1.set_ylabel(r"$a\,p\,/\,F$")

ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax2.tick_params(axis="both", which="major")
ax2.set_xlabel(r"$x\,/\,a$")
ax2.set_ylabel(r"$u_x\,/\,a$")

ax3.legend(loc="center left", frameon=False)
ax3.axis("off")

# Saving figure
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
fig1.savefig(output_folder + output_files, bbox_inches="tight")
