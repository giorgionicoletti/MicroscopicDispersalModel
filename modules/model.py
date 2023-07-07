import numpy as np
import networkx as nx
import igraph
import scipy.optimize as optimize

from numba import njit, prange

import utils


############################
# COMPUTE EFFECTIVE KERNEL #
############################

@njit
def find_effective_kernel_nb(f, xi, adj_matrix, Laplacian = None,
                             L_eigvals = None, V = None, V_inv = None,
                             undirected = True):
    """
    Finds the effective coupling between different patches assuming a Monod
    function for the explorer creation rate. Numba boosted.

    Parameters
    ----------
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    adj_matrix : numpy.ndarray
        Adjacency matrix of the network.
    Laplacian : numpy.ndarray
        Laplacian matrix of the network. Must be of type np.complex128.
        If None, it is computed from the network.
    L_eigvals : numpy.ndarray
        Eigenvalues of the Laplacian matrix. Must be of type np.complex128.
        If None, they are computed from the Laplacian matrix.
    V : numpy.ndarray
        Eigenvectors of the Laplacian matrix. Must be of type np.complex128.
        If None, they are computed from the Laplacian matrix.
    V_inv : numpy.ndarray
        Inverse of the eigenvectors of the Laplacian matrix. Must be of type
        np.complex128.
        If None, they are computed from the Laplacian matrix.

    Returns
    -------
    effective_coupling : numpy.ndarray
        Effective coupling between patches.
    """
    N = adj_matrix.shape[0]

    if Laplacian is None:
        Laplacian = utils.find_laplacian_nb(adj_matrix)
        Laplacian = Laplacian.astype(np.complex128)

    if L_eigvals is None or V is None or V_inv is None:
        if undirected:
            L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
            V = V.astype(np.complex128)
            V_inv = V_inv.astype(np.complex128)
            L_eigvals = L_eigvals.astype(np.complex128)
        else:
            L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    w_matrix = utils.create_diag_matrix(1/(1+f*L_eigvals))

    cNormCoupling = np.real(np.dot(V, np.dot(w_matrix, V_inv)))

    effective_coupling = np.dot(cNormCoupling, adj_matrix.T)*xi*f/(1+f)

    return effective_coupling

def find_effective_kernel(f, xi, network, Laplacian = None,
                          L_eigvals = None, V = None, V_inv = None,
                          undirected = True):
    """
    Wrapper for find_effective_kernel_nb.

    Parameters
    ----------
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    network : networkx.Graph
        Network of connected patches.
    Laplacian : numpy.ndarray
        Laplacian matrix of the network.
        If None, it is computed from the network.
    L_eigvals : numpy.ndarray
        Eigenvalues of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    V : numpy.ndarray
        Eigenvectors of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    V_inv : numpy.ndarray
        Inverse of the eigenvectors of the Laplacian matrix.
        If None, they are computed from the Laplacian matrix.
    undirected : bool
        If True, the network is considered undirected.
    q_regular : bool
        If True, the network is considered q-regular, where q is the degree.

    Returns
    -------
    effective_coupling : numpy.ndarray
        Effective coupling between patches.
    """

    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    if Laplacian is not None:
        Laplacian = Laplacian.astype(np.complex128)

    if L_eigvals is not None or V is not None or V_inv is not None:
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
        L_eigvals = L_eigvals.astype(np.complex128)

    K = find_effective_kernel_nb(f, xi, adj_matrix, Laplacian, L_eigvals, V, V_inv,
                                 undirected = undirected)

    return K

@njit
def dist_dependence_nb(distmatrix, unique_dist, effective_coupling):
    """
    Computes the distance dependence of the effective coupling, where distances
    are defined as the shortest path between patches. Numba boosted.

    Parameters
    ----------
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
    unique_dist : numpy.ndarray
        Array of unique distances between patches.
    effective_coupling : numpy.ndarray
        Effective coupling between patches.

    Returns
    -------
    coupling_distribution : list
        List of effective couplings for each distance.
    """

    N = distmatrix.shape[0]

    coupling_distribution = []

    avg_couplings = np.zeros(unique_dist.size, dtype = np.float64)
    std_couplings = np.zeros(unique_dist.size, dtype = np.float64)

    for idx_dist in prange(unique_dist.size):
        cdist = []
        for i in range(N):
            for j in range(N):
                if distmatrix[i,j] == unique_dist[idx_dist]:
                    cdist.append(effective_coupling[i,j])
                    avg_couplings[idx_dist] += effective_coupling[i,j]
                    std_couplings[idx_dist] += effective_coupling[i,j]**2
        coupling_distribution.append(cdist)

        avg_couplings[idx_dist] /= len(cdist)
        std_couplings[idx_dist] = np.sqrt(std_couplings[idx_dist]/len(cdist) - avg_couplings[idx_dist]**2)

    return avg_couplings, std_couplings, coupling_distribution

def dist_dependence(network, effective_coupling,
                    distmatrix = None, unique_dist = None, return_dist = False):
    """
    Wrapper for dist_dependence_nb. The distance matrix is computed
    with igraph if not provided.

    Parameters
    ----------
    network : networkx.Graph
        Network of connected patches.
    effective_coupling : numpy.ndarray
        Effective coupling between patches.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
        If None, it is computed from the network.
    unique_dist : numpy.ndarray
        Array of unique distances between patches.
        If None, it is computed from the network.
    return_dist : bool
        If True, the the array of unique distances is returned.

    Returns
    -------
    avg_c : numpy.ndarray
        Array of average effective couplings for each distance.
    std_c : numpy.ndarray
        Array of standard deviations of effective couplings for each distance.
    cdist : list
        List of effective couplings for each distance.
    unique_dist : numpy.ndarray
        Array of unique distances between patches.
        Only returned if return_dist is True or if distmatrix and unique_dist
        are None.
    """
    
    if distmatrix is None or unique_dist is None:
        distmatrix, unique_dist = utils.find_distance_matrix(network)
        return_dist = True
    
    avg_c, std_c, cdist = dist_dependence_nb(distmatrix, unique_dist, effective_coupling)

    if return_dist:
        return avg_c, std_c, cdist, unique_dist
    else:
        return avg_c, std_c, cdist

###################################
# COMPUTE METAPOPULATION CAPACITY #
###################################

@njit
def find_metapopulation_capacity_nb(K, adj, f, xi, undirected = True):
    """
    Compute the metapopulation capacity of a network.
    If the network has more than one connected component, the metapopulation
    capacity of each component is computed and the minimum is returned.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.
    adj : numpy.ndarray
        Adjacency matrix of the network.
        Used to check if the network is connected.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
    CComponents = utils.find_connected_components(adj)
    NComponents = len(CComponents)

    if NComponents == 1:
        return np.max(np.real(np.linalg.eigvals(K)))
    else:
        lambdaMax_components = np.zeros(NComponents, dtype = np.float64)

        for i in range(NComponents):
            module = CComponents[i]
            adj_sub = utils.extract_submatrix(adj, module)
            Ktemp = find_effective_kernel_nb(f, xi, adj_sub,
                                             undirected = undirected)
            lambdaMax_components[i] = np.max(np.real(np.linalg.eigvals(Ktemp)))

        return np.min(lambdaMax_components)

@njit
def find_connected_metapopulation_capacity_nb(K):
    """
    Compute the metapopulation capacity of a network. This function assumes that
    the network is connected, so use it only if you are sure that the network has
    only one connected component.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
 
    return np.max(np.real(np.linalg.eigvals(K)))


def find_metapopulation_capacity(K, network, f, xi, undirected = True,
                                 guaranteed_connected = False):
    """
    Wrapper for find_metapopulation_capacity_nb.

    Parameters
    ----------
    K : numpy.ndarray
        Effective coupling matrix.
    network : networkx.Graph
        Network of connected patches.
        Needed to check if the network is connected.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    undirected : bool
        If True, the network is considered undirected.
    guaranteed_connected : bool
        If True, the network is considered connected.
        No check is performed.

    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    K = K.astype(np.complex128)

    if guaranteed_connected:
        lambdaMax = find_connected_metapopulation_capacity_nb(K)
    else:
        lambdaMax = find_metapopulation_capacity_nb(K, adj_matrix, f, xi,
                                                    undirected = undirected)
    
    return lambdaMax

@njit
def find_metapopulation_capacity_Hanski_nb(adj, alpha, distmatrix):
    """
    Compute the metapopulation capacity of a network using Hanski kernel.
    If the network has more than one connected component, the metapopulation
    capacity of each component is computed and the minimum is returned.

    Parameters
    ----------
    network : networkx.Graph
        Network of connected patches.
    alpha : float
        Characteristic distance for the kernel.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
        
    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """

    CComponents = utils.find_connected_components(adj)
    NComponents = len(CComponents)

    if NComponents == 1:
        K = find_Hanski_kernel(distmatrix, alpha)
        np.fill_diagonal(K, 0)
        K = K.astype(np.complex128)

        return np.max(np.real(np.linalg.eigvals(K)))
    else:
        lambdaMax_components = np.zeros(NComponents, dtype = np.float64)

        for i in range(NComponents):
            module = CComponents[i]
            distsub = utils.extract_submatrix(distmatrix, module)
            Ktemp = find_Hanski_kernel(distsub, alpha)
            
            np.fill_diagonal(Ktemp, 0)
            Ktemp = Ktemp.astype(np.complex128)
            lambdaMax_components[i] = np.max(np.real(np.linalg.eigvals(Ktemp)))

        return np.min(lambdaMax_components)

def find_metapopulation_capacity_Hanski(network, alpha, distmatrix):
    """
    Compute the metapopulation capacity of a network using Hanski kernel.
    If the network has more than one connected component, the metapopulation
    capacity of each component is computed and the minimum is returned.

    Parameters
    ----------
    network : networkx.Graph
        Network of connected patches.
    alpha : float
        Characteristic distance for the kernel.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
        
    Returns
    -------
    lambdaMax : float
        Metapopulation capacity.
    """
    adj = nx.adjacency_matrix(network).toarray().astype(float)

    return find_metapopulation_capacity_Hanski_nb(adj, alpha, distmatrix)

@njit(parallel = True)
def find_all_metapop_nb(f_array, xi_array, adj, undirected = True,
                        guaranteed_connected = False):
    """
    Finds the metapopulation capacity for each pair of f and xi in the given
    network.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    adj : numpy.ndarray
        Adjacency matrix of the network.
    undirected : bool
        If True, the network is considered undirected.
    guaranteed_connected : bool
        If True, the network is considered connected.

    Returns
    -------
    lambdaMax_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi.
    """

    lambdaMax_array = np.zeros((len(xi_array), len(f_array)))

    Laplacian = utils.find_laplacian_nb(adj)
    Laplacian = Laplacian.astype(np.complex128)

    if undirected:
        L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
        L_eigvals = L_eigvals.astype(np.complex128)
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
    else:
        L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    for idx_f in prange(len(f_array)):
        f = f_array[idx_f]
        xi = 1.
        K = find_effective_kernel_nb(f, xi, adj,  Laplacian = Laplacian,
                                        L_eigvals = L_eigvals, V = V, V_inv = V_inv)
        K = K.astype(np.complex128)
        if guaranteed_connected:
            lambdaMax = find_connected_metapopulation_capacity_nb(K)
        else:
            lambdaMax = find_metapopulation_capacity_nb(K, adj, f, xi,
                                                        undirected = undirected)
        for idx_xi, xi in enumerate(xi_array):
            lambdaMax_array[idx_xi, idx_f] = lambdaMax*xi

    return lambdaMax_array

def find_all_metapop(f_array, xi_array, network, undirected = True,
                     guaranteed_connected = False):
    """
    Finds the metapopulation capacity for each pair of f and xi in the given
    network, and compares it with the metapopulation capacity of the same
    network with Hanski kernel.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    network : networkx.Graph
        Network of connected patches.
    undirected : bool
        If True, the network is considered undirected.
    guaranteed_connected : bool
        If True, the network is guaranteed to be connected.

    Returns
    -------
    lambdaMax_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi.
    lambdaMax_Hanski_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi using
        Hanski kernel.
    """
    adj = nx.adjacency_matrix(network).toarray().astype(float)

    return find_all_metapop_nb(f_array, xi_array, adj,
                               undirected = undirected,
                               guaranteed_connected = guaranteed_connected)


@njit(parallel = True)
def iterate_and_compare_nb(f_array, xi_array, adj, distmatrix, undirected = True):
    """
    Finds the metapopulation capacity for each pair of f and xi in the given
    network, and compares it with the metapopulation capacity of the same
    network with Hanski kernel.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    adj : numpy.ndarray
        Adjacency matrix of the network.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    lambdaMax_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi.
    lambdaMax_Hanski_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi using
        Hanski kernel.
    """

    lambdaMax_array = np.zeros((len(f_array), len(xi_array)), dtype = np.float64)
    lambdaMax_Hanski_array = np.zeros((len(f_array), len(xi_array)), dtype = np.float64)
    alphas = np.zeros((len(f_array), len(xi_array)), dtype = np.float64)

    Laplacian = utils.find_laplacian_nb(adj)
    Laplacian = Laplacian.astype(np.complex128)

    if undirected:
        L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
        L_eigvals = L_eigvals.astype(np.complex128)
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
    else:
        L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    for idx_xi in prange(len(xi_array)):
        xi = xi_array[idx_xi]
        for idx_f, f in enumerate(f_array):
            K = find_effective_kernel_nb(f, xi, adj, Laplacian = Laplacian,
                                         L_eigvals = L_eigvals, V = V, V_inv = V_inv)
            K = K.astype(np.complex128)
            lambdaMax = find_metapopulation_capacity_nb(K, adj, f, xi,
                                                        undirected = undirected)
            lambdaMax_array[idx_f, idx_xi] = lambdaMax

            alpha = get_alpha(K, distmatrix)
            alphas[idx_f, idx_xi] = alpha
            lambdaMax_Hanski = find_metapopulation_capacity_Hanski_nb(adj, alpha, distmatrix)
            lambdaMax_Hanski_array[idx_f, idx_xi] = lambdaMax_Hanski

    return lambdaMax_array, lambdaMax_Hanski_array, alphas

def iterate_and_compare(f_array, xi_array, network, undirected = True):
    """
    Finds the metapopulation capacity for each pair of f and xi in the given
    network, and compares it with the metapopulation capacity of the same
    network with Hanski kernel.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    network : networkx.Graph
        Network of connected patches.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    lambdaMax_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi.
    lambdaMax_Hanski_array : numpy.ndarray
        Array of metapopulation capacities for each pair of f and xi using
        Hanski kernel.
    """

    adj = nx.adjacency_matrix(network).toarray().astype(float)
    distmatrix, _ = utils.find_distance_matrix(network)

    return iterate_and_compare_nb(f_array, xi_array, adj, distmatrix, undirected = undirected)

def find_all_kernels(network_list, f, xi, undirected = True,
                     compute_metapop = False):
    """
    Find the effective kernel of a list of networks.

    Parameters
    ----------
    network_list : list
        List of networks.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
    undirected : bool, optional
        Whether the networks are undirected. The default is True.
    compute_metapop : bool, optional
        Whether to compute the metapopulation capacity of the networks. The
        default is False.

    Returns
    -------
    kernels : list
        List of effective kernels.
    lambdaMax : list
        List of metapopulation capacities.
        Only returned if compute_metapop = True.
    """
    is_list = True

    if not isinstance(network_list, list):
        network_list = [network_list]
        is_list = False

    kernels = []
    lambdaMax = []

    for idx, net in enumerate(network_list):
        K = find_effective_kernel(f, xi, net, undirected = undirected)
        kernels.append(K)

        if compute_metapop:
            lM = find_metapopulation_capacity(K, net, f, xi,
                                              undirected = undirected)
            lambdaMax.append(lM)

    if not is_list:
        kernels = kernels[0]
        lambdaMax = lambdaMax[0]

    if compute_metapop:
        return kernels, lambdaMax
    else:
        return kernels

@njit(parallel = True)
def network_analysis_nb(f_array, xi_array, adj_matrix, 
                        undirected = True, q_regular = False,
                        guaranteed_connected = False):
    """
    Compute the effective couplings and the metapopulation capacity
    for a range of f and xi values. Numba boosted.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of f values.
    xi_array : numpy.ndarray
        Array of xi values.
    adj_matrix : numpy.ndarray
        Adjacency matrix of the network.
    undirected : bool
        If True, the network is considered undirected.
        This is needed to use the appropriate function to find the eigenvalues.
    q_regular : bool
        If True, the network is considered q-regular, where q is the degree.
        In this case, the network is assumed to be connected and undirected.
    guaranteed_connected : bool
        If True, the network is assumed to be connected and undirected.
        
    Returns
    ------- 
    K : numpy.ndarray
        Array of effective couplings.
    max_eig : numpy.ndarray
        Array of maximum eigenvalues of the effective coupling matrix.
        Can be complex.
    """

    N = adj_matrix.shape[0]

    K = np.zeros((xi_array.size, f_array.size, N, N), dtype = np.complex128)
    max_eig = np.zeros((xi_array.size, f_array.size), dtype = np.complex128)

    Laplacian = utils.find_laplacian_nb(adj_matrix)
    Laplacian = Laplacian.astype(np.complex128)

    if undirected:
        L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
        L_eigvals = L_eigvals.astype(np.complex128)
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
    else:
        L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    for idx_xi in prange(xi_array.size):
        xi = xi_array[idx_xi]
        
        for idx_f in range(f_array.size):
            f = f_array[idx_f]
            K[idx_xi, idx_f] = find_effective_kernel_nb(f, xi, adj_matrix, Laplacian,
                                                        L_eigvals, V, V_inv,
                                                        undirected = undirected)
            
            if q_regular:
                evals = np.linalg.eigvalsh(K[idx_xi, idx_f])
                max_eig[idx_xi, idx_f] = evals[np.real(evals).argmax()]
            elif guaranteed_connected:
                lambdaMax = find_connected_metapopulation_capacity_nb(K[idx_xi, idx_f])
            else:
                lambdaMax = find_metapopulation_capacity_nb(K[idx_xi, idx_f],
                                                            adj_matrix, f, xi,
                                                            undirected = undirected)
                max_eig[idx_xi, idx_f] = lambdaMax
    
    return np.real(K), max_eig

def network_analysis(f_array, xi_array, network,
                     undirected = True, q_regular = False):
    """
    Compute the effective couplings and the metapopulation capacity
    for a range of f and xi values.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of f values.
    xi_array : numpy.ndarray
        Array of xi values.
    network : networkx.Graph
        Network of connected patches.
    undirected : bool
        If True, the network is considered undirected.
    q_regular : bool
        If True, the network is considered q-regular, where q is the degree.

    Returns
    ------- 
    K : numpy.ndarray
        Array of effective couplings.
    max_eig : numpy.ndarray
        Array of maximum eigenvalues of the effective coupling matrix.
    """

    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    K, max_eig = network_analysis_nb(f_array, xi_array, adj_matrix,
                                     undirected = undirected, q_regular = q_regular)

    if((np.imag(max_eig)**2 > 1e-8).any()):
        print("Imaginary eigenvalues! Something went wrong. Returning real part.")

    return K, np.real(max_eig)

@njit
def metapop_capacity_fragmented(adj_list, f, xi):
    """
    Compute the metapopulation capacity of a sequence of possibly
    fragmented networks.

    Parameters
    ----------
    adj_list : list
        List of adjacency matrices of the networks.
    f : float
        Explorers efficiency, defined as f = D / lambda where D is the baseline
        diffusion coefficient and lambda is the decay rate of explorers.
    xi : float
        Baseline creation rate of explorers at large f.
        
    Returns
    -------
    lambdaMax : numpy.ndarray
        Metapopulation capacity of the sequence of networks.
    """

    lambdaMax = np.zeros(len(adj_list), dtype = np.float64)

    for idx_adj, adj in enumerate(adj_list):
        CComponents = utils.find_connected_components(adj)
        NComponents = len(CComponents)

        lambdaMax_modules = np.zeros(NComponents, dtype = np.float64)

        for i in range(NComponents):
            module = CComponents[i]
            adj_sub = utils.extract_submatrix(adj, module)
            Ktemp = find_effective_kernel_nb(f, xi, adj_sub,
                                             undirected = True)
            lambdaMax_modules[i] = np.max(np.real(np.linalg.eigvals(Ktemp)))

        lambdaMax[idx_adj] = np.min(lambdaMax_modules)

    return lambdaMax

################################
# SIMULATE DYNAMICAL EVOLUTION #
################################

@njit
def simulate(N, Nsteps, dt, K, cvec, evec, rho0 = None,
             check_stationary = False):
    """
    Simulates the model for a given set of parameters.

    Parameters
    ----------
    N : int
        Number of patches.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    K : numpy.ndarray
        Effective kernel between patches.
    cvec : numpy.ndarray
        Vector of creation rates of explorers.
    evec : numpy.ndarray
        Vector of death rates of settled population.
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.
        If None, it is set to a random vector.
    check_stationary : bool
        If True, the simulation is stopped when the density of settled population
        is stationary.        
        
    Returns
    -------
    rho : numpy.ndarray
        Density of settled population in each patch at each time step.
    """

    rho = np.zeros((Nsteps, N), dtype=np.float64)

    if rho0 is None:
        rho0 = np.random.rand(N)

    rho[0] = rho0

    for t in range(Nsteps - 1):
        rho[t+1] = rho[t] + dt*(-rho[t]*evec + (1 - rho[t])*np.dot(K, rho[t]*cvec))
        if check_stationary:
            if np.linalg.norm(rho[t+1] - rho[t]) < 1e-8:
                for i in range(N):
                    rho[t+1:, i] = rho[t+1, i]
                break

    return rho

@njit
def simulate_Hanski(N, Nsteps, dt, distmatrix, alpha, cvec, evec, rho0 = None):
    """
    Simulate the dynamics of the Hanski model, with an exponential kernel in
    the network distance.

    Parameters
    ----------
    N : int
        Number of patches.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
    alpha : float
        Characteristic distance for the kernel.
    cvec : numpy.ndarray
        Vector of colonization rates.
    evec : numpy.ndarray
        Vector of extinction rates.
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.
        If None, it is set to a random vector.

    Returns
    -------
    rho : numpy.ndarray
        Density of settled population in each patch at each time step.
    """

    rho = np.zeros((Nsteps, N), dtype=np.float64)

    kernel = np.exp(-distmatrix/alpha)
    np.fill_diagonal(kernel, 0)

    if rho0 is None:
        rho0 = np.random.rand(N)

    rho[0] = rho0

    for t in range(Nsteps - 1):
        rho[t+1] = rho[t] + dt*(-rho[t]*evec + (1 - rho[t])*np.dot(kernel, rho[t]*cvec))

        for i in range(N):
            if rho[t+1, i] < 0:
                rho[t+1, i] = 0
            if rho[t+1, i] > 1:
                rho[t+1, i] = 1
    
    return rho

@njit(parallel = True)
def find_statpop_nb(f_array, xi_array, adj_matrix, Nsteps, dt, cvec, evec, rho0 = None,
                    undirected = True):
    """
    Compute the stationary population for a range of f and xi values.
    
    Parameters
    ----------
    f_array : numpy.ndarray
        Array of f values.
    xi_array : numpy.ndarray
        Array of xi values.
    adj_matrix : numpy.ndarray
        Adjacency matrix of the network.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    cvec : numpy.ndarray
        Vector of creation rates of explorers.
    evec : numpy.ndarray
        Vector of death rates of settled population.
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.
        If None, it is set to a random vector.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    statpop : numpy.ndarray
        Array of stationary population for each f and xi value.
    """

    N = adj_matrix.shape[0]
    Laplacian = utils.find_laplacian_nb(adj_matrix)
    Laplacian = Laplacian.astype(np.complex128)

    if undirected:
        L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
        L_eigvals = L_eigvals.astype(np.complex128)
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
    else:
        L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    statpop = np.zeros((xi_array.size, f_array.size, N), dtype = np.float64)

    for idx_xi in prange(xi_array.size):
        xi = xi_array[idx_xi]
        for idx_f in range(f_array.size):
            f = f_array[idx_f]

            K = find_effective_kernel_nb(f, xi, adj_matrix, Laplacian,
                                         L_eigvals, V, V_inv,
                                         undirected = undirected)
            
            rho = simulate(N, Nsteps, dt, K, cvec, evec, rho0 = rho0)

            #check that all rho have reached a constant value
            if np.abs(rho[-1] - rho[-2]).max() > 1e-3:
                print("Warning: simulation has not reached a steady state.")
                print("Stopping simulation, run with more timesteps")
                
                return statpop

            statpop[idx_xi, idx_f] = rho[-1]

    return statpop

def find_statpop(f_array, xi_array, network, Nsteps, dt, cvec, evec, rho0 = None,
                 undirected = True):
    """
    Compute the stationary population for a range of f and xi values.
    
    Parameters
    ----------
    f_array : numpy.ndarray
        Array of f values.
    xi_array : numpy.ndarray
        Array of xi values.
    network : networkx.Graph
        Networkx graph of the network.
    Nsteps : int
        Number of time steps.
    dt : float
        Time step.
    cvec : numpy.ndarray
        Vector of creation rates of explorers.
    evec : numpy.ndarray
        Vector of death rates of settled population.
    rho0 : numpy.ndarray
        Initial condition for the density of settled population in each patch.
        If None, it is set to a random vector.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    statpop : numpy.ndarray
        Array of stationary population for each f and xi value.
    """

    adj_matrix = nx.adjacency_matrix(network).toarray().astype(float)

    statpop = find_statpop_nb(f_array, xi_array, adj_matrix,
                              Nsteps, dt, cvec, evec, rho0 = rho0,
                              undirected = undirected)
    
    return statpop

##########################
# HANSKI MODEL FUNCTIONS #
##########################

@njit(parallel = True)
def find_all_alphas_nb(f_array, xi_array, adj, distmatrix, undirected = True):
    """
    Find the characteristic distance alpha for each pair of f and xi in the
    given network.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    adj : numpy.ndarray
        Adjacency matrix of the network.
    distmatrix : numpy.ndarray
        Matrix of distances between patches.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    alpha_array : numpy.ndarray
        Array of characteristic distances for each pair of f and xi.    
    """

    alpha_array = np.zeros((len(xi_array), len(f_array)))

    Laplacian = utils.find_laplacian_nb(adj)
    Laplacian = Laplacian.astype(np.complex128)

    if undirected:
        L_eigvals, V, V_inv = utils.diagonalize_matrix_sym(Laplacian.T)
        L_eigvals = L_eigvals.astype(np.complex128)
        V = V.astype(np.complex128)
        V_inv = V_inv.astype(np.complex128)
    else:
        L_eigvals, V, V_inv = utils.diagonalize_matrix(Laplacian.T)

    for idx_xi in prange(len(xi_array)):
        xi = xi_array[idx_xi]
        for idx_f, f in enumerate(f_array):
            K = find_effective_kernel_nb(f, xi, adj,  Laplacian = Laplacian,
                                         L_eigvals = L_eigvals, V = V, V_inv = V_inv)

            alpha = get_alpha(K, distmatrix)
            alpha_array[idx_xi, idx_f] = alpha

    return alpha_array

def find_all_alphas(f_array, xi_array, network, undirected = True):
    """
    Find the characteristic distance alpha for each pair of f and xi in the
    given network.

    Parameters
    ----------
    f_array : numpy.ndarray
        Array of explorers efficiency.
    xi_array : numpy.ndarray
        Array of baseline explorers creation rate.
    network : networkx.Graph
        Network of connected patches.
    undirected : bool
        If True, the network is considered undirected.

    Returns
    -------
    alpha_array : numpy.ndarray
        Array of characteristic distances for each pair of f and xi.
    """

    adj = nx.adjacency_matrix(network).toarray().astype(float)
    distmatrix, _ = utils.find_distance_matrix(network)

    return find_all_alphas_nb(f_array, xi_array, adj, distmatrix, undirected = undirected)

@njit
def find_Hanski_kernel(distmatrix, alpha):
    """
    Computes the hanski kernel from a distance matrix and a characteristic
    distance alpha.

    Parameters
    ----------
    distmatrix : numpy.ndarray
        Distance matrix.
    alpha : float
        Characteristic distance of the kernel.

    Returns
    -------
    K : numpy.ndarray
        Hanski kernel.
    
    Notes
    -----
    The Hanski kernel is defined as:
    K_ij = exp(-d_ij / alpha)
    where d_ij is the distance between patches i and j.

    References
    ----------
    Hanski, I. (1998). Metapopulation dynamics. Nature, 396(6706), 41-49.
    """

    K = np.exp(-distmatrix/alpha)
    np.fill_diagonal(K, 0)

    return K

@njit
def get_alpha(K, distmatrix, connected = True):
    """
    Compute the characteristic distance alpha of a kernel K,
    assuming an exponential shape.

    Parameters
    ----------
    K : numpy.ndarray
        Kernel matrix.
    distmatrix : numpy.ndarray
        Distance matrix.

    Returns
    -------
    alpha : float
        Characteristic distance of the kernel.
    """
    
    K_pos = K.copy()
    if connected:
        K_pos[np.where(K_pos == 0)[0]] = np.min(K_pos)

    avg_distance = np.mean(utils.off_diag(distmatrix))
    avg_logK = np.mean(np.log(utils.off_diag(np.abs(K_pos))))

    return - avg_distance / avg_logK