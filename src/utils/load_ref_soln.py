import numpy as np


def _trans_time_data_to_dataset(
    data,
    datapath,
    input_dim,
    output_dim,
):
    slice = (data.shape[1] - input_dim + 1) // output_dim
    assert (
        slice * output_dim == data.shape[1] - input_dim + 1
    ), "Data shape is not multiple of pde.output_dim"

    # Extract time points from file header
    with open(datapath, "r") as f:

        def extract_time(string):
            index = string.find("t=")
            if index == -1:
                return None
            return float(string[index + 2 :].split(" ")[0])

        t = None
        for line in f.readlines():
            if line.startswith("%") and line.count("@") == slice * output_dim:
                t = line.split("@")[1:]
                t = list(map(extract_time, t))
        if t is None or None in t:
            raise ValueError(
                "Reference Data not in Comsol format or does not contain time info"
            )
        t = np.array(t[::output_dim])

    # Extract spatial grids
    spatial_grids = [np.unique(data[:, i]) for i in range(input_dim - 1)]

    # Create solution array with shape (n_t, n_x1, n_x2, ...)
    solution_shape = [len(t)] + [len(grid) for grid in spatial_grids] + [output_dim]
    solution = np.zeros(solution_shape)

    # Fill solution array
    for i_t in range(len(t)):
        for i_out in range(output_dim):
            # Get the column index for this time and output
            col_idx = input_dim - 1 + i_t * output_dim + i_out
            # Reshape the data into n_x1 x n_x2 x ... array
            spatial_shape = [len(grid) for grid in spatial_grids]
            solution[i_t, ..., i_out] = data[:, col_idx].reshape(spatial_shape)

    # Also return the original flattened dataset for backward compatibility
    t_mesh, x0 = np.meshgrid(t, data[:, 0])
    list_x = [x0.reshape(-1)]
    for i in range(1, input_dim - 1):
        list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
    # Repeat time values to match spatial dimensions
    list_x.append(t_mesh.reshape(-1))  # Use t_mesh instead of t
    for i in range(output_dim):
        list_x.append(data[:, input_dim - 1 + i :: output_dim].reshape(-1))
    ref_data = np.stack(list_x).T

    return ref_data, t, spatial_grids, solution


def load_ref_data_raw(
    datapath: str,
    transform_fn=None,
):
    """Load raw reference data without time transposition."""
    ref_data = np.loadtxt(datapath, comments="%").astype(np.float32)
    if transform_fn is not None:
        ref_data = transform_fn(ref_data)
    return ref_data


def load_ref_data_time(
    datapath: str,
    input_dim: int,
    output_dim: int,
    transform_fn=None,
):
    """Load reference data with time transposition, returning grids and solution array."""
    ref_data = np.loadtxt(datapath, comments="%").astype(np.float32)
    ref_data, t, spatial_grids, solution = _trans_time_data_to_dataset(
        ref_data, datapath, input_dim, output_dim
    )
    if transform_fn is not None:
        ref_data = transform_fn(ref_data)
        # Note: transform_fn would need to be updated to handle solution array if needed
    return ref_data, t, spatial_grids, solution


def load_ref_data(
    datapath: str,
    input_dim: int,
    output_dim: int,
    transform_fn=None,
    t_transpose: bool = True,
):
    """Legacy function that combines both loading methods."""
    if t_transpose:
        return load_ref_data_time(datapath, input_dim, output_dim, transform_fn)
    else:
        return load_ref_data_raw(datapath, transform_fn)
