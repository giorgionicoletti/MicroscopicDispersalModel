import numpy as np

import sys
sys.path.insert(0, '../modules/')
import model
import utils
import rivernets

sys.path.insert(0, '../../')
import funPlots as fplot

import time as measure_time

import scipy 

path = '../data/RiverNet/Asia/'

alpha_array = np.linspace(15, 60, 12).round(2)

##### Prepare the DEM #####
dem, grid = rivernets.get_DEM(path + "Asia.tif")
dem = grid.fill_pits(dem)
dem = grid.fill_depressions(dem)
dem = grid.resolve_flats(dem)


##### Compute the distance matrix #####
to_nn = np.array([[-1, 0], [0, 1], [1, 0], [0, -1],
                  [-1, 1], [1, 1], [1, -1], [-1, -1]])

slopes = rivernets.get_slopes(dem, to_nn = to_nn)

dx = dy = np.median(abs(slopes[slopes != np.inf]).flatten())
pos_3D = rivernets.euclidean_pos_3D(dem, dx, dy)
distmatrix = scipy.spatial.distance_matrix(pos_3D.reshape(-1, 3), pos_3D.reshape(-1, 3))

NDyn = distmatrix.shape[0]

##### Simulations #####
Nsteps = 100000
dt = 1e-2
cvec = np.ones(NDyn)*1
evec = np.ones(NDyn)*1

Time = np.arange(0, Nsteps*dt, dt)

rho_0 = np.ones(NDyn)*0.2

path_data = path + 'data_Asia/'

for alpha in alpha_array:
    print(alpha)    
    t0 = measure_time.time()
    rho = model.simulate_Hanski(NDyn, Nsteps, dt, distmatrix, alpha, cvec, evec, rho0 = rho_0)
    np.save(path_data + f'Asia_Hanski_rhostat_alpha{alpha}.npy', rho[-1])
    print("\t time to simulate: ", measure_time.time() - t0)
    
np.save(path_data + 'Asia_Hanski_alpha_array.npy', alpha_array)