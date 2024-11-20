import numpy as np
import cupy as cp
import cupyx.scipy.signal
import scipy as sp

def snr_gpu(data, target_maps, kernel, tags):
    """
    Signal-to-Noise Ratio calculated with given target maps.
    SNR1: Averaged map-wise SNR.
    SNR2: Global SNR of all maps. 

    Args:
        data (np.array):            Range-Angle Maps
        target_maps (np.array):     Range-Angle Binary Target Maps
        kernel (np.array):          Kernel to extend the target maps
        tags (String):              Tag to name the list entry

    Return:
        (np.array):     Array of tag, target idx, SNR1, SNR2
    
    """
    
    # Cut-off uninformative region
    target_maps = target_maps[...,:200,:]
    data = data[...,:200,:]

    ret = np.zeros((target_maps.shape[0], 4))
    if kernel is not None:
        for target_idx in range(target_maps.shape[0]):
            for i in range(target_maps.shape[1]):
                target_maps[target_idx, i] = cupyx.scipy.signal.convolve2d(target_maps[target_idx, i], kernel, mode='same')

    sums = cp.repeat(cp.expand_dims(cp.sum(data, axis=(1,2)), axis=0), target_maps.shape[0], axis=0)
    signals = cp.sum(cp.einsum('ijk, tijk-> tijk', data, target_maps), axis=(2,3))#/np.sum(target_maps, axis=(1,2))
    ssr = cp.sum(signals/(sums+1e-8)/(cp.sum(target_maps, axis=(2,3))+1e-8), axis=1)/target_maps.shape[1]
    ssr2 = cp.sum(signals, axis=1)/cp.sum(sums, axis=1)

    for target_idx in range(target_maps.shape[0]):
        ret[target_idx,:] = np.array([tags, target_idx, ssr[target_idx].get(), ssr2[target_idx].get()])

    return ret
