import numpy as np

import sys
sys.path.insert(0, '../modules/')
import model
import utils
import rivernets

sys.path.insert(0, '../../')
import funPlots as fplot

import time as measure_time

path = '../data/RiverNet/Asia/'

f_array_plot = np.geomspace(0.9, 5, 50)
xi = 2

##### Prepare the DEM #####
dem, grid = rivernets.get_DEM(path + "Asia.tif")
dem = grid.fill_pits(dem)
dem = grid.fill_depressions(dem)
dem = grid.resolve_flats(dem)


##### Compute the adjacency matrix #####
to_nn = np.array([[-1, 0], [0, 1], [1, 0], [0, -1],
                  [-1, 1], [1, 1], [1, -1], [-1, -1]])

slopes = rivernets.get_slopes(dem, to_nn = to_nn)

xmin, xmax = np.min(slopes[slopes != np.inf]), np.max(slopes[slopes != np.inf])

ymin = 1e-9
ymax = 1
beta = 0.004

Dmatrix = rivernets.build_adjacency_elevation_exponential(dem, slopes,
                                                          xmin, xmax,
                                                          ymin, ymax,
                                                          beta, to_nn = to_nn)


##### Compute Laplacian and spectral decomposition #####
Laplacian = utils.find_laplacian_nb(Dmatrix)
Laplacian = Laplacian.astype(np.complex128)
L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

NDyn = Dmatrix.shape[0]


##### Simulations #####
Nsteps = 100000
dt = 1e-2
cvec = np.ones(NDyn)*1
evec = np.ones(NDyn)*1

Time = np.arange(0, Nsteps*dt, dt)

rho_0 = np.ones(NDyn)*0.2

path_data = path + 'data_Asia/'

for f in f_array_plot:
    print(f)
    t0 = measure_time.time()
    Ktemp = model.find_effective_kernel_nb(f, xi, Dmatrix, 
                                           Laplacian = Laplacian,
                                           L_eigvals = L_eigvals,
                                           V = V, V_inv = V_inv,
                                           undirected = False)
    print("\t", Ktemp.min(), Ktemp.max())
    np.save(path_data + f'Asia_K_f{f}.npy', Ktemp)
    print("\t time to compute K: ", measure_time.time() - t0)

    t0 = measure_time.time()
    lambdaM = model.find_connected_metapopulation_capacity_nb(Ktemp.astype(np.complex128))
    np.save(path_data + f'Asia_lambdaM_f{f}.npy', lambdaM)
    print("\t time to compute lambdaM: ", measure_time.time() - t0)
    
    t0 = measure_time.time()
    rho = model.simulate(NDyn, Nsteps, dt, Ktemp, cvec, evec, rho0 = rho_0)
    np.save(path_data + f'Asia_rhostat_f{f}.npy', rho[-1])
    print("\t time to simulate: ", measure_time.time() - t0)
    
np.save(path_data + 'Asia_f_array.npy', f_array_plot)