import numpy as np
import cupy as cp
import cupyx.scipy.signal

def create_ca_maps_gpu(data, kernel, normalize=True):
    """
    Create CA - CFAR Threshold Maps by performing averaging via convolve2D.
    This map CA can by used within the peak detection formula:
        data > factor * CA + offset

    Args:
        data (np.array):    Array of RA Maps with shape (n_frames, n_distances, n_angles).
        kernel (np.array):  Kernel for convolution.

    Returns:
        (np.array):     CA Map array with shape (n_frames, n_distances, n_angles).
    """

    if normalize:
        data = cp.einsum('ijk,i->ijk', data, 1/np.max(data,axis=(1,2)))

    ca_maps = cp.zeros_like(data)
    for i in range(data.shape[0]):
        ca_maps[i] = (cupyx.scipy.signal.convolve2d(cp.pad(data[i], 
                                                           pad_width=((kernel.shape[0], kernel.shape[0]), (kernel.shape[1], kernel.shape[1])),
                                                            mode='symmetric'), 
                                                    kernel, mode='same')/cp.sum(kernel))[kernel.shape[0]:-kernel.shape[0],kernel.shape[1]:-kernel.shape[1]]
    return ca_maps

def cacfar_detection_gpu(data, ca_maps, factor, offset, normalize=True, detection_area=None):
    """
    Create Peak Maps by performing CA-CFAR.
    -> data > factor * CA + offset

    Args:
        data (np.array):    Array of RA Maps with shape (n_frames, n_distances, n_angles).
        kernel (np.array):  Kernel for convolution.
        factor (int):       Factor.
        offset (int):       Offset.
        kernel (np.array):  Kernel to extend the peak maps.
        normalize (bool):   Whether to normalize the data.

    Returns:
        (np.array):     Peak Maps array with shape (n_frames, n_distances, n_angles).
    """

    # TODO: undirty it
    data = data[...,:200,:]
    ca_maps = ca_maps[...,:200,:]

    peak_maps = cp.zeros_like(data)

    if normalize:
        data = cp.einsum('ijk,i->ijk', data, 1/np.max(data,axis=(1,2)))

    # CA CFAR
    cfar_maps = (ca_maps+offset)*factor
    peak_maps = cp.array(data > (cfar_maps), dtype='float')

    if detection_area is not None:
        extended_peak_maps = cp.zeros_like(peak_maps)
        for i in range(peak_maps.shape[0]):
            extended_peak_maps[i] = cupyx.scipy.signal.convolve2d(peak_maps[i], detection_area, mode='same')>0
        peak_maps = extended_peak_maps

    return peak_maps