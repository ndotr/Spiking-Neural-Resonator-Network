import numpy as np

def load_data(path, frame_idx):
    """
    Loading BBM radar data from given filepath (../../../data.mat).

    Args:
        path (String): Path to matlab data file.
        frame_idx (int or tuple): Index of frame or tuple for frame range.

    Returns:
        numpy.array: radar data cube (frames, chirps, samples, antennas).
    """


    if isinstance(frame_idx, int):
        data = np.load(path + "/{0}.npy".format(frame_idx))
        data = np.swapaxes(data, axis1=-1, axis2=-2)
        data = np.expand_dims(data, axis=0)
    elif isinstance(frame_idx, tuple):
        data = []
        for i in range(frame_idx[0], frame_idx[1]):
            data.append(np.swapaxes(np.load(path + "/{0}.npy".format(i)), axis1=-1, axis2=-2))
        data = np.array(data)

    return data