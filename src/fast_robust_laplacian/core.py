import numpy as np

import fast_robust_laplacian_bindings as rlb

def set_print_timing(print_timing):
    """
    Enable or disable timing output for laplacian computation.
    
    Parameters:
    -----------
    print_timing : bool
        If True, timing information will be printed during laplacian computation.
        If False, no timing information will be printed.
    """
    rlb.setPrintTiming(print_timing)

def mesh_laplacian(verts, faces, mollify_factor=1e-5):

    ## Validate input
    if type(verts) is not np.ndarray:
        raise ValueError("`verts` should be a numpy array")
    if (len(verts.shape) != 2) or (verts.shape[1] != 3):
        raise ValueError("`verts` should have shape (V,3), shape is " + str(verts.shape))
    
    if type(faces) is not np.ndarray:
        raise ValueError("`faces` should be a numpy array")
    if (len(faces.shape) != 2) or (faces.shape[1] != 3):
        raise ValueError("`faces` should have shape (F,3), shape is " + str(faces.shape))

    ## Call the main algorithm from the bindings
    L, M = rlb.buildMeshLaplacian(verts, faces, mollify_factor)

    ## Return the result
    return L, M

def point_cloud_laplacian(points, mollify_factor=1e-5, n_neighbors=30):

    ## Validate input
    if type(points) is not np.ndarray:
        raise ValueError("`points` should be a numpy array")
    if (len(points.shape) != 2) or (points.shape[1] != 3):
        raise ValueError("`points` should have shape (V,3), shape is " + str(points.shape))
    
    ## Call the main algorithm from the bindings
    L, M = rlb.buildPointCloudLaplacian(points, mollify_factor, n_neighbors)

    ## Return the result
    return L, M

def point_cloud_laplacian_batched(list_of_points, mollify_factor=1e-5, n_neighbors=30):
    """
    Compute Laplacians for multiple point clouds and return a single block-diagonal system.

    Parameters:
    - list_of_points: list of numpy arrays, each shaped `(V_i, 3)`
    - mollify_factor: float, regularization strength
    - n_neighbors: int, neighborhood size used to build local triangulations

    Returns:
    - (L, M): tuple of scipy.sparse matrices, block-diagonal concatenation across inputs
    """
    ## Validate input
    if type(list_of_points) is not list:
        raise ValueError("`list_of_points` should be a list of numpy arrays")
    if len(list_of_points) == 0:
        raise ValueError("`list_of_points` cannot be empty")
    for i, pts in enumerate(list_of_points):
        if type(pts) is not np.ndarray:
            raise ValueError(f"`list_of_points[{i}]` should be a numpy array")
        if (len(pts.shape) != 2) or (pts.shape[1] != 3):
            raise ValueError(f"`list_of_points[{i}]` should have shape (V,3), shape is {pts.shape}")

    ## Call the batched algorithm from the bindings
    # NOTE: pybind11 should convert list[np.ndarray] -> std::vector<Eigen::MatrixXd>
    # If bindings fail to cast, consider converting to a Python list of contiguous arrays.
    L, M = rlb.buildPointCloudLaplacianBatched(list_of_points, mollify_factor, n_neighbors)

    ## Return the result
    return L, M
