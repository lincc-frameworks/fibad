import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from matplotlib.colors import LogNorm


def plot_grid(data_list):
    """
    Plots an n x 4 grid of matplotlib plots.

    Parameters
    ----------
    data_list : list of tuples
        Each tuple in the list is (object id, rounded median distance to NN, file name)
    """

    num_cols = 4

    num_plots = len(data_list)
    num_rows = (num_plots + num_cols) // num_cols  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))

    for i, data in enumerate(data_list):
        row = i // num_cols
        col = i % num_cols
        ax = axes[row, col]
        plotter(ax, data)
        fig.patch.set_facecolor("darkslategrey")

    # Hide any unused subplots
    for j in range(num_plots, num_rows * num_cols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()


def plotter(ax, data_tuple):
    """Plot the R band image for a given object ID.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot the image on
    data_tuple : (int, float, str)
        Each tuple is (object id, rounded median distance to NN, file name)
    """
    # Read the FITS files
    object_id, dist, file_name = data_tuple

    fits_file = file_name + "_HSC-R.fits"
    data = fits.getdata(fits_file)

    # Normalize the data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))

    title = f"Obj ID: {object_id}\nMedian dist: {np.round(dist)}"

    # Display the image
    ax.imshow(data, origin="lower", norm=LogNorm(), cmap="Greys")
    ax.set_title(title, y=1.0, pad=-30)
    ax.axis("off")  # Hide the axis


def objects_in_range(all_embeddings, median_dist_all_nn, data_directory, min_thresh=0, max_thresh=100_000):
    """Find the objects that are within a certain threshold range."""

    # Find the indexes of the objects that are well separated from their nearest neighbors
    indexes = [(i, x) for i, x in enumerate(median_dist_all_nn) if min_thresh < x and x < max_thresh]

    print(f"Number of objects in threshold range: {len(indexes)}")

    # Use the indexes to gather metadata: object ID, rounded median distance, and file name
    data_directory = Path(data_directory).resolve()
    objects = []
    for indx in sorted(indexes, key=lambda x: x[1]):
        object_id = all_embeddings["ids"][indx[0]]

        found_files = glob.glob(f"{data_directory / object_id}*.fits")
        file_name = found_files[0][:-11]

        # print(f"{object_id} : {np.round(indx[1])} : {file_name}")
        objects.append((object_id, np.round(indx[1]), file_name))

    return objects
