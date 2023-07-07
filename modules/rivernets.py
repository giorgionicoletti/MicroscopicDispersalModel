import numpy as np
import networkx as nx
import csv

import matplotlib.pyplot as plt

import pysheds
from pysheds.grid import Grid

from numba import njit
import utils

###############################################
# Functions for generating & loading networks #
###############################################

def network_return(folder_path):
    '''
    Returns network and node attributes from network in the folder path.
    The network is a river network with nodes and edges obtained from a
    Digital Elevation Map (DEM) and flow accumulation. The node attributes
    are the flow accumulation and the location of the node in the DEM.
    The folder path should contain the following files:
    - directed_generated_network_adjmatrix.npz
    - accumul_attributes.npy
    - node_accumulations.csv
    - node_locations.csv

    Parameters
    ----------
    folder_path : string
        Path to the folder containing the network and node attributes.

    Returns
    -------
    generated_network : networkx graph
        Networkx graph of the river network.
    accumul_attributes : numpy array
        Array of the flow accumulation of each node.
    node_locations : dictionary
        Dictionary of the location of each node in the DEM.
    node_accumulation : dictionary
        Dictionary of the flow accumulation of each node.
    '''

    adj_matrix = np.load(folder_path+"directed_generated_network_adjmatrix.npz")["arr_0"]
    generated_network = nx.from_numpy_array(adj_matrix,create_using=nx.DiGraph)

    accumul_attributes = np.load(folder_path+"accumul_attributes.npy")
    node_accumulation_path = folder_path+"node_accumulations.csv"
    node_locations_path = folder_path+"node_locations.csv"

    node_accumulation_file = csv.reader(open(node_accumulation_path,"r"))
    node_locations_file = csv.reader(open(node_locations_path,"r"))

    initNodes = list(generated_network.nodes())
    node_accumulation = {}
    node_locations = {}
    node_mapping = {}
    counter = 0

    for row in node_locations_file:
        node_locations[int(row[0])] = (float(row[1]),float(row[2]))
        node_mapping[initNodes[counter]] = int(row[0])
        counter +=1
        
    for row in node_accumulation_file:
        node_accumulation[int(row[0])] = float(row[1])


    generated_network = nx.relabel_nodes(generated_network,node_mapping)

    return generated_network, np.array(accumul_attributes), node_locations, node_accumulation

def load_river_net(path):
    """
    Wrapper function for network_return. Returns the network, node attributes,
    node locations, node accumulations, and the average distance between nodes
    in the network.

    Parameters
    ----------
    path : string
        Path to the folder containing the network and node attributes.

    Returns
    -------
    net : networkx graph
        Networkx graph of the river network.
    accumul_attributes : numpy array
        Array of the flow accumulation of each node.
    pos : dictionary
        Dictionary of the location of each node in the DEM.
    node_accumulation : dictionary
        Dictionary of the flow accumulation of each node.
    dL : float
        Average distance between nodes in the network.
    xy : numpy array
        Array of the x and y coordinates of each node in the network.
    """
    net, accumul_attributes, pos, node_accumulation = network_return(path)
    net = net.to_undirected()

    xy = np.array(list(pos.values()))

    dx = abs(np.diff(xy, axis = 0))[:,0]
    dy = abs(np.diff(xy, axis = 0))[:,1]
    dx = np.mean(dx[dx != 0])
    dy = np.mean(dy[dy != 0])
    dL = np.sqrt(dx**2 + dy**2)

    return net, accumul_attributes, pos, node_accumulation, dL, xy

def generate_grid(original_network, pos, cut = True):
    """
    Generate a grid network in the same domain as the original network.
    If cut is True, the grid network will only contain nodes that are
    also present in the original network.

    Parameters
    ----------
    original_network : networkx graph
        Networkx graph of the original river network.
    pos : dictionary
        Dictionary of the location of each node in the DEM.
    cut : boolean, optional
        If True, the grid network will only contain nodes that are
        also present in the original network. The default is True.

    Returns
    -------
    grid_network : networkx graph
        Networkx graph of the grid network.
    grid_locs : dictionary
        Dictionary of the location of each node in the grid network.
    gridnodes : numpy array
        Array of the nodes in the grid network.
    """

    orig_nodes = np.array(list(original_network.nodes()))
    locs_array = np.array(list(pos.values()))

    longs = locs_array[:,0]
    lats = locs_array[:,1]

    Lx = np.max(longs) - np.min(longs)
    Ly = np.max(lats) - np.min(lats)

    dx = np.min(np.diff(np.sort(np.unique(longs))))
    dy = np.min(np.diff(np.sort(np.unique(lats))))

    Nx = int(Lx/dx) +1
    Ny = int(Ly/dy) +1

    grid_network = nx.grid_graph([Nx, Ny])

    grid_locs = {(j,i): np.array([np.min(longs) + i*dx,np.min(lats) + j*dy]) for i in range(Nx) for j in range(Ny)}
    gridnodes = np.array(list(grid_locs.keys()))

    if cut:
        selected_nodes = []
        grid_remapping = []

        for i in range(len(orig_nodes)):
            ognode = orig_nodes[i]
            ogloc = pos[ognode]

            start_dist = np.sqrt(Lx**2 + Ly**2)
            start_node = np.array([-1,-1])
            
            for j in range(len(gridnodes)):
                gnode = tuple(gridnodes[j])
                gloc = grid_locs[gnode]

                distx = gloc[0] - ogloc[0]
                disty = gloc[1] - ogloc[1]

                cur_dist = np.sqrt(distx**2 + disty**2)

                if(cur_dist<=start_dist):
                    start_dist = cur_dist
                    start_node[0] = gnode[0]
                    start_node[1] = gnode[1]
            selected_nodes.append(tuple(start_node))
            grid_remapping.append([ognode,start_node[0],start_node[1]])
    
        grid_network = grid_network.subgraph(selected_nodes)

    xy_grid = np.zeros((len(grid_network), 2))
    pos_grid = {}
    for i, node in enumerate(list(grid_network.nodes())):
        xy_grid[i] = grid_locs[node]
        pos_grid[node] = grid_locs[node]

    return grid_network, xy_grid, pos_grid




#################################
# Preprocessing & DEM functions #
#################################

def get_DEM(dem_file, clean = False):
    """
    Read a digital elevation map (DEM) from a raster file.

    Parameters
    ----------
    dem_file : string
        Path to the DEM raster file.
    clean : bool, optional
        Whether to clean the DEM by filling pits, depressions and resolving
        flats. Defaults to False.

    Returns
    -------
    dem : numpy.ndarray
        A 2D array of elevation values.
    grid : pysheds.grid.Grid
        A pysheds grid object.
    """
    grid = Grid.from_raster(dem_file, nodata=-1)
    dem = grid.read_raster(dem_file, nodata=-1)

    if clean:
        dem = grid.fill_pits(dem)
        dem = grid.fill_depressions(dem)
        dem = grid.resolve_flats(dem)

    return dem, grid

@njit
def get_pixel_slope(dem, i, j, Nrows, Ncols, to_nn):
    """
    Get the slope of the cell (i, j) in the DEM between neighbors.
    Take care of the boundaries by assigning a slope of infinity to
    the cells at the boundaries.

    Parameters
    ----------
    dem : numpy array
        DEM of the river network.
    i : int
        Row index of the cell.
    j : int
        Column index of the cell.
    Nrows : int
        Number of rows in the DEM.
    Ncols : int
        Number of columns in the DEM.
    to_nn : numpy array
        Array of the neighbors to consider.
        
    Returns
    -------
    slopes : numpy array
        Slopes of the cell (i, j) in the DEM between neighbors.
    """

    slopes = np.ones(len(to_nn), dtype = np.float64)*np.inf
    
    for idx, nn in enumerate(to_nn):
        if i + nn[0] >= 0 and i + nn[0] < Nrows and j + nn[1] >= 0 and j + nn[1] < Ncols:
            slopes[idx] = - dem[i, j] + dem[i + nn[0], j + nn[1]]        

    return slopes

@njit
def get_slopes(dem, to_nn = None):
    """
    Get the slope ratios of the cells in the DEM between neighbors.
    Take care of the boundaries by assigning a slope of infinity to
    the cells at the boundaries.

    Parameters
    ----------
    dem : numpy array
        DEM of the river network.
    to_nn : array, optional
        Array of the neighbors to consider. The default is None.
        If None, 8 neighbors with diagonal connections are considered.

    Returns
    -------
    slopes : numpy array
        Slopes of the cells in the DEM between neighbors.
    """
    Nrows, Ncols = dem.shape

    if to_nn is None:
        to_nn = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1],
                          [1, 1], [1, 0], [1, -1], [0, -1]])


    slopes = np.zeros((Nrows, Ncols, len(to_nn)), dtype = np.float64)

    for i in range(Nrows):
        for j in range(Ncols):
            slopes[i, j] = get_pixel_slope(dem, i, j, Nrows, Ncols, to_nn = to_nn)
    
    return slopes

@njit
def build_adjacency_elevation_exponential(dem, slopes, xmin, xmax, ymin, ymax, beta,
                                          to_nn = None):
    """
    Build the adjacency matrix of a DEM, taking into account the elevation
    difference between cells. Uses an exponential function to convert the 
    elevation difference into a weight.

    Parameters
    ----------
    dem : numpy array
        DEM of the river network.
    slopes : numpy array
        Slopes of the cells in the DEM between neighbors.
    xmin : float
        Minimum value of the elevation difference.
    xmax : float
        Maximum value of the elevation difference.
    ymin : float
        Minimum value of the weight.
    ymax : float
        Maximum value of the weight.
    beta : float
        Slope of the exponential function.
    to_nn : array, optional
        Array of the neighbors to consider. The default is None.
        If None, 8 neighbors with diagonal connections are considered.

    Returns
    -------
    Dmatrix : numpy array
        Adjacency matrix of the DEM.
    """
    Nrows, Ncols = dem.shape

    if to_nn is None:
        to_nn = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1],
                          [1, 1], [1, 0], [1, -1], [0, -1]])

    Dmatrix = np.zeros((Nrows*Ncols, Nrows*Ncols), dtype = np.float64)

    for i in range(Nrows):
        for j in range(Ncols):
            idx = utils.tuple_to_linear(i, j, dem.shape)

            for k, move in enumerate(to_nn):
                s = slopes[i, j, k]
                if s != np.inf:
                    idx2 = utils.tuple_to_linear(i + move[0], j + move[1], dem.shape)
                    Dmatrix[idx, idx2] = utils.exponential_range(s, xmin, xmax,
                                                                 ymin, ymax, beta)
    return Dmatrix

@njit
def build_adjacency_elevation(dem, slopes, beta, baseline = 1,
                              to_nn = None):
    """
    Build the adjacency matrix of a DEM, taking into account the elevation
    difference between cells. Uses an inverse sigmoid function to convert the 
    elevation difference into a weight.

    Parameters
    ----------
    dem : numpy array
        DEM of the river network.
    slopes : numpy array
        Slopes of the cells in the DEM between neighbors.
    beta : float
        Slope of the sigmoid function.
    baseline : float, optional
        Baseline value of the sigmoid function. The default is 1.
    to_nn : array, optional
        Array of the neighbors to consider. The default is None.
        If None, 8 neighbors with diagonal connections are considered.

    Returns
    -------
    Dmatrix : numpy array
        Adjacency matrix of the DEM.
    """
    Nrows, Ncols = dem.shape
    if to_nn is None:
        to_nn = np.array([[-1, -1], [-1, 0], [-1, 1], [0, 1],
                          [1, 1], [1, 0], [1, -1], [0, -1]])

    Dmatrix = np.zeros((Nrows*Ncols, Nrows*Ncols), dtype = np.float64)

    for i in range(Nrows):
        for j in range(Ncols):
            idx = utils.tuple_to_linear(i, j, dem.shape)

            for k, move in enumerate(to_nn):
                s = slopes[i, j, k]
                if s != np.inf:
                    idx2 = utils.tuple_to_linear(i + move[0], j + move[1], dem.shape)
                    Dmatrix[idx, idx2] = baseline*utils.sigmoid_inv(s, beta)

    return Dmatrix

def hill_DEM(Nx, Ny, centers, heights, sigmas, baseline_dem = None):
    """
    Returns a DEM with hills-like structures.

    Parameters
    ----------
    Nx : int
        Number of rows in the DEM.
    Ny : int
        Number of columns in the DEM.
    centers : list of tuples
        List of the centers of the hills.
    heights : list of floats
        List of the heights of the hills.
    sigmas : list of floats
        List of the wideness of the hills.
    baseline_dem : numpy array, optional
        Baseline DEM to which the hills are added.

    Returns
    -------
    dem : numpy array
        DEM with hills-like structures.
    """
    xspace = np.arange(Nx)
    yspace = np.arange(Ny)

    XX, YY = np.meshgrid(yspace, xspace)

    if baseline_dem is None:
        dem = np.ones((Nx, Ny))
    else:
        dem = baseline_dem

    for i, center in enumerate(centers):
        dem += np.exp(-((XX - center[0])**4 + (YY - center[1])**4)/(2*sigmas[i]**2))*heights[i]
    
    return dem

def euclidean_pos_3D(dem, dx, dy):
    """
    Computes the position of the cells in a DEM in 3D space.

    Parameters
    ----------
    dem : numpy array   
        A 2D array containing the elevations.
    dx : float
        Spacing between cells in the x direction.
    dy : float
        Spacing between cells in the y direction.

    Returns
    -------
    pos : numpy array
        A 3D array containing the positions of the cells in 3D space.
    """
    Nx, Ny = dem.shape
    pos = np.zeros((Nx, Ny, 3))
    for i in range(Nx):
        for j in range(Ny):
            pos[i, j] = np.array([i*dx, j*dy, dem[i, j]])
    return pos

def coarse_grain_non_overlapping(mat,reduction_level, redx, redy):
    """
    Coarsegraining a matrix by taking the mean of non-overlapping blocks of size
    redx*redy.

    Parameters
    ----------
    mat : numpy array
        Matrix to coarsegrain.
    reduction_level : int
        Number of times to coarsegrain the matrix.
    redx : int
        Size of the blocks in the x direction.
    redy : int
        Size of the blocks in the y direction.

    Returns
    -------
    coarse_mat : numpy array
        Coarse-grained matrix.
    """
    coarse_mat = mat.copy()
    
    for red_index in range(reduction_level):
        nx = coarse_mat.shape[0]
        ny = coarse_mat.shape[1]

        new_mat = np.zeros((nx//redx,ny//redy))
        for i in range(nx//redx):
            for j in range(ny//redy):
                block = coarse_mat[redx*i:redx*i+redx,redy*j:redy*j+redy]
                if((block ==0).all()):
                    new_mat[i,j] = 0
                new_mat[i,j] = np.mean(block)
        coarse_mat = new_mat.copy()
    
    return coarse_mat

from scipy.ndimage import uniform_filter
from scipy.ndimage import variance

def lee_filter(img, size):
    """
    Returns the Lee filter of an image.

    Parameters
    ----------
    img : numpy array
        Image to filter.
    size : int
        Size of the filter.

    Returns
    -------
    img_output : numpy array
        Filtered image.
    """
    img_mean = uniform_filter(img, (size, size))
    img_sqr_mean = uniform_filter(img**2, (size, size))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    return img_output

######################
# Plotting functions #
######################
cmap_statpop = utils.colors_to_color_map(['#121212', '#17507E', '#558D82', '#EBC247', '#fca626', '#f17d43', '#df6242', '#d53a3a'],
                                         nodes = [0, 0.15, 0.3, 0.55, 0.65, 0.75, 0.8, 1])

def plot_DEM(dem, grid, ax = None):
    """
    Plot a digital elevation map (DEM) with a colorbar.

    Parameters
    ----------
    dem : numpy.ndarray
        A 2D array of elevation values.
    grid : pysheds.grid.Grid
        A pysheds grid object.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes will
        be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure object.
    ax : matplotlib.axes.Axes
        A matplotlib axes object.
    """
    max_lat = grid.extent[3]
    min_lat = grid.extent[2]
    max_long = grid.extent[1]
    min_long = grid.extent[0]

    ratio = (max_lat - min_lat) / (max_long - min_long)

    if ax is None:
        fig, ax = plt.subplots(figsize = (5, 5 * ratio))
    else:
        fig = ax.get_figure()
        ax.set_aspect(ratio)

    im = ax.imshow(dem, extent = grid.extent, cmap = 'terrain', zorder = 1)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Digital elevation map')

    # Add a colorbar
    cbar = plt.colorbar(im, ax = ax, label = "Elevation (m)", fraction=0.046, pad=0.04)
    # rotate label
    cbar.ax.set_ylabel('Elevation (m)', rotation=270, labelpad=25)
    return fig, ax, cbar


def plot_DEM_statpop(dem, rho, vmin = None, vmax = None, ax = None):
    """
    Plot the DEM and the stationary population density side by side.

    Parameters
    ----------
    dem : numpy.ndarray
        A 2D array of elevation values.
    rho : numpy.ndarray
        A Nsteps x Nnodes array of population densities.
        Only the last step is plotted.
    vmin : float, optional
        Minimum value for the colorbar of rho.
    vmax : float, optional
        Maximum value for the colorbar of rho.
    ax : matplotlib.axes.Axes, optional
        A matplotlib axes object. If not provided, a new figure and axes will
        be created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        A matplotlib figure object.
    ax : matplotlib.axes.Axes
        A matplotlib axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize = (12, 5))

    im = ax[0].imshow(dem, cmap = 'terrain')
    ax[0].set_title('Digital elevation map')
    plt.colorbar(im, ax = ax[0], fraction=0.046, pad=0.04)

    if vmin is None:
        vmin = np.min(rho[-1])
    if vmax is None:
        vmax = np.max(rho[-1])

    im = ax[1].imshow(rho[-1].reshape(dem.shape), cmap = cmap_statpop, vmin = vmin, vmax = vmax)
    ax[1].set_title('Stationary population density')
    plt.colorbar(im, ax = ax[1], fraction=0.046, pad=0.04)

    plt.subplots_adjust(wspace = 0.4)

    if ax is None:
        return fig, ax
    else:
        return ax
    

#######################
# Animation functions #
#######################

import matplotlib.animation as animation
import sys

sys.path.insert(0, '../../')
import funPlots as fplot

def update_plot(frame, matrix_list, im):
    im.set_array(matrix_list[frame])

def create_animation(matrix_list, dem, grid, path, fps = 30, dpi = 150, bitrate = 5000,
                     label = None):
    fplot.master_format(ncols = 2, nrows = 1)
    vmin = 0
    vmax = np.max(matrix_list[-1])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    interval = 1000 / fps

    fig, ax = plt.subplots(ncols = 2, dpi = dpi, figsize = (12, 5))

    plot_DEM(dem, grid = grid, ax = ax[0])

    im = ax[1].imshow(matrix_list[0], cmap=cmap_statpop, interpolation = 'lanczos')
    ax[1].axis('off')
    ax[1].set_title("Settled population density", fontsize = 17)
    ax[0].set_title("Digital elevation map", fontsize = 17)

    im.set_norm(norm)
    cbar = plt.colorbar(im, ax = ax[1], fraction=0.046, pad=0.04)
    if label is not None:
        cbar.ax.set_ylabel(label, rotation=270, labelpad=25)
        
    plt.subplots_adjust(wspace = 0.4)
    
    animation_obj = animation.FuncAnimation(fig, update_plot, frames=len(matrix_list),
                                            fargs=(matrix_list, im), interval=interval)

    # Set the path to the FFmpeg executable
    ffmpeg_path = '/usr/bin/ffmpeg'  # Update with the correct path to the FFmpeg executable
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # Save the animation
    animation_obj.save(path, writer='ffmpeg', fps=fps,
                       bitrate=bitrate, extra_args=['-pix_fmt', 'yuv420p'])
    
    plt.show()

def update_plot_grouped(frame, all_rho, ims):
    for idx, im in enumerate(ims):
        im.set_array(all_rho[idx, frame])


def create_animation_grouped(all_rho, f_array, dem, grid, path,
                             fps = 30, dpi = 150, bitrate = 5000,
                             dem_xticks = None, dem_yticks = None):
    
    fplot.master_format(ncols = 3, nrows = 2)
    vmin = 0
    vmaxs = np.max(all_rho[:, -1], axis = (1,2))
    norms = [plt.Normalize(vmin=vmin, vmax=vmax) for vmax in vmaxs]
    interval = 1000 / fps

    fig, axs = plt.subplot_mosaic([['.', '.', 'f1', 'f1', 'f2', 'f2'],
                                   ['dem', 'dem', 'f1', 'f1', 'f2', 'f2'],
                                   ['dem', 'dem', 'f3', 'f3', 'f4', 'f4'],
                                   ['.', '.', 'f3', 'f3', 'f4', 'f4']],
                                   figsize = (12, 7), dpi = dpi)
    
    plt.subplots_adjust(wspace = 0.4, hspace = 0.5)

    _, _, cbar_dem = plot_DEM(dem, grid = grid, ax = axs['dem'])
    if dem_xticks is not None:
        axs['dem'].set_xticks(dem_xticks)
    if dem_yticks is not None:
        axs['dem'].set_yticks(dem_yticks)

    axs['dem'].set_position(axs['dem'].get_position().bounds + np.array([-0.05, 0.0, 0.0, 0.0]))
    pos_cbar = cbar_dem.ax.get_position()
    cbar_dem.ax.set_position([pos_cbar.x0 - 0.05, pos_cbar.y0, pos_cbar.width, pos_cbar.height])
    cbar_dem.set_ticks([dem.min() - 1, dem.max() - 1])
    cbar_dem.ax.set_ylabel('Elevation (m)', rotation=270, labelpad=-5)

    ims = []

    for idx_f, ax in enumerate([axs['f1'], axs['f2'], axs['f3'], axs['f4']]):
        im = ax.imshow(all_rho[idx_f, 0], cmap=cmap_statpop,
                       interpolation = 'lanczos')
        ax.axis('off')
        ax.set_title(f"$f = {f_array[idx_f]}$", fontsize = 14, pad = 5)

        im.set_norm(norms[idx_f])
        cbar = plt.colorbar(im, ax = ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.linspace(0, vmaxs[idx_f], 4))
        cbar.set_ticklabels(np.round(np.linspace(0, vmaxs[idx_f], 4), 2))
        ims.append(im)

    axs['f1'].text(-12, 105, "Settled population", fontsize = 17, rotation = 90)
        
    animation_obj = animation.FuncAnimation(fig, update_plot_grouped, frames=all_rho.shape[1],
                                            fargs=(all_rho, ims), interval=interval)

    # Set the path to the FFmpeg executable
    ffmpeg_path = '/usr/bin/ffmpeg'  # Update with the correct path to the FFmpeg executable
    plt.rcParams['animation.ffmpeg_path'] = ffmpeg_path

    # Save the animation
    animation_obj.save(path, writer='ffmpeg', fps=fps,
                       bitrate=bitrate, extra_args=['-pix_fmt', 'yuv420p'])
    
    plt.show()