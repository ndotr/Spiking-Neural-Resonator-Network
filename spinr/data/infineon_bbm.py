import numpy as np
import scipy.io

def load_data(path):
    """
    Loading BBM radar data from given filepath (../../../data.mat).

    Args:
        path (String): Path to matlab data file.

    Returns:
        numpy.array: radar data cube (frames, chirps, samples, antennas).
    """

    file_path =  path
    mat = scipy.io.loadmat(file_path)
    data = mat['data']
    data = np.swapaxes(data, 1,2)
    data = np.swapaxes(data, 0,1)

    return data.T

def load_targets(path):
    """
    Loading BBM target data from given filepath (../../../target.mat).

    Args:
        path (String): Path to matlab target file.

    Returns:
        list: targets (targets, frames) containg dictionaries.
    """

    file_path =  path
    mat = scipy.io.loadmat(file_path)
    targets = mat['targets']

    return targets

def load_config(path):
    """
    Loading BBM radar config from given directory path.

    Args:
        path (String): Path to directory where config file is located.

    Returns:
        dict: dictionary containing parameters of the simulated radar sensor.
    """

    file_path =  path + '/config.mat'
    mat = scipy.io.loadmat(file_path, struct_as_record=True)
    config = mat['mmic']
    keys = config.dtype.names
    vals = config[0][0]
    config = {}
    for k,v in zip(keys, vals):
        config[k] = v[0][0]

    return config

def distance_range(config, n_distances=None):
    """
    Return array of distance to map to frequency bins.

    Args:
        config (dict):      Configuration dictionary of radar sensor settings.
        n_distances (int):  Number of distance bins.

    Returns:
        (np.array):     Array containing distance values.
    """

    fs = 50e6 # Hard-coded sampling rate of ADC
    # Infineon BBM simulation cuts off samples by defining Nrange
    # => not full bandwidth is used
    # => effective bandwidth is needed
    eff_b = float(config['Nrange'])*config['bandwidth']/(fs*config['Tramp']/2+1)
    d_res = config['c0']/(2*eff_b)
    d_max = float(config['Nrange'])//2*d_res
    if n_distances is None:
        distance_range = np.linspace(0, d_max, config['Nrange'] // 2, endpoint=False)
    else:
        distance_range = np.linspace(0, d_max, n_distances, endpoint=False)

    return distance_range

def angle_range(n_angles):
    """
    Return array of angles to map to frequency bins.

    Args:
        n_angles (int):  Number of angle bins.

    Returns:
        (np.array):     Array containing angle values.
    """

    angles = np.zeros(n_angles) 
    for a in range(n_angles):
        angles[a] = -np.arcsin(2*(a-np.floor(n_angles/2))/n_angles)

    return angles

def targets_to_map_pertarget(targets, n_distances, n_angles, config, rcs_flag=False):
    """
    Read targets from BBM target structure and create binary or rcs range-angle map to match SNN/FFT output.

    Args:
        targets (list):     Targets from BBM structure.
        n_distances:        Number of distance bins.
        n_angles:           Number of angle bins.
        config (dict):      Configuration dictionary of BBM radar sensor settings.
        rcs_flag (bool):    Boolean to indicate, whether binary or rcs map.

    Return:
        (np.array):     Target Range-Angle map
    
    """

    d_range = distance_range(config, n_distances)
    a_range = angle_range(n_angles)

    target_map = np.zeros((len(targets)+1, n_distances, n_angles)).astype('float32')

    for i, target in enumerate(targets):
        # target is a tuple of name, pos, velo, rcs, (?)
        pos = target[1][0] # pos
        d = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
        a = np.arctan(pos[1]/pos[0])
        rcs = target[3][0]

        d_idx = (np.abs(d_range - d)).argmin()
        a_idx = (np.abs(a_range - a)).argmin()

        if rcs_flag:
            target_map[i+1, d_idx, a_idx] = rcs
            target_map[0, d_idx, a_idx] = rcs
        else:
            target_map[i+1, d_idx, a_idx] = (rcs > -100)
            target_map[0, d_idx, a_idx] = (rcs > -100)


    return target_map

def targets_to_maps(targets, n_distances, n_angles, config, rcs_flag):
    """
    Run over all frames to create target maps.

    Args:
        targets (list):     Targets from BBM structure.
        n_distances:        Number of distance bins.
        n_angles:           Number of angle bins.
        config (dict):      Configuration dictionary of BBM radar sensor settings.
        rcs_flag (bool):    Boolean to indicate, whether binary or rcs map.

    Return:
        (np.array):     Target Range-Angle map over all frames
    """

    target_maps = [] 
    target_maps_per_target = [] 
    n_targets, n_frames = targets.shape
    for frame in range(n_frames):
        target = targets[:,frame]
        target_map = targets_to_map_pertarget(target, n_distances, n_angles, config, rcs_flag = rcs_flag)
        target_maps_per_target.append(target_map)

    return np.swapaxes(np.array(target_maps_per_target), axis1=0, axis2=1)
